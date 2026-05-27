use std::{
    collections::HashMap,
    env,
    path::PathBuf,
    process::Stdio,
    sync::{Mutex, OnceLock},
    time::Duration,
};

use serde_json::{json, Map, Value};
use sinew_app::AgentEvent;
use tokio::{io::AsyncWriteExt, process::Command, time};

const BRIDGE_RELATIVE_PATH: &str = ".vibe-island/bin/vibe-island-bridge";
const BRIDGE_SOURCE: &str = "claude";
const BRIDGE_TIMEOUT: Duration = Duration::from_secs(2);

static TOOL_NAMES: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();

pub(super) fn emit_agent_event_to_vibe_island(
    workspace_id: &str,
    conversation_id: &str,
    event: &AgentEvent,
) {
    if bridge_path().is_none() {
        return;
    }

    for payload in payloads_for_event(workspace_id, conversation_id, event) {
        spawn_bridge(payload);
    }
}

pub(super) fn emit_user_prompt_to_vibe_island(
    workspace_id: &str,
    conversation_id: &str,
    prompt: &str,
) {
    if bridge_path().is_none() {
        return;
    }

    let payload = base_payload(workspace_id, conversation_id, "UserPromptSubmit")
        .with("prompt", json!(prompt));
    spawn_bridge(payload);
}

fn payloads_for_event(
    workspace_id: &str,
    conversation_id: &str,
    event: &AgentEvent,
) -> Vec<Map<String, Value>> {
    match event {
        AgentEvent::TurnStarted => {
            vec![base_payload(workspace_id, conversation_id, "SessionStart")]
        }
        AgentEvent::ToolStarted { id, name } => {
            remember_tool_name(conversation_id, id, name);
            Vec::new()
        }
        AgentEvent::ToolReady {
            id, args_pretty, ..
        } => {
            let tool_name = remembered_tool_name(conversation_id, id).unwrap_or_else(|| id.clone());
            vec![base_payload(workspace_id, conversation_id, "PreToolUse")
                .with("tool_use_id", json!(id))
                .with("tool_name", json!(tool_name))
                .with("tool_input", parse_json_or_string(args_pretty))]
        }
        AgentEvent::ToolFinished {
            id,
            output,
            is_error,
            ..
        } => {
            let tool_name = forget_tool_name(conversation_id, id).unwrap_or_else(|| id.clone());
            vec![base_payload(workspace_id, conversation_id, "PostToolUse")
                .with("tool_use_id", json!(id))
                .with("tool_name", json!(tool_name))
                .with("tool_response", json!(truncate_for_bridge(output)))
                .with("is_error", json!(is_error))]
        }
        AgentEvent::Interrupted => {
            vec![base_payload(workspace_id, conversation_id, "StopFailure")
                .with("reason", json!("interrupted"))]
        }
        AgentEvent::Error { message } => {
            vec![base_payload(workspace_id, conversation_id, "StopFailure")
                .with("reason", json!(truncate_for_bridge(message)))]
        }
        AgentEvent::TurnFinished { .. } => {
            vec![base_payload(workspace_id, conversation_id, "Stop")]
        }
        AgentEvent::SubAgentEvent {
            id,
            agent_name,
            team_name,
            model,
            initial_message,
            event,
            ..
        } => {
            let sub_conversation_id = format!("{conversation_id}:{id}");
            let mut payloads = payloads_for_event(workspace_id, &sub_conversation_id, event);
            for payload in &mut payloads {
                payload.insert("parent_session_id".into(), session_id(conversation_id));
                payload.insert("agent_name".into(), json!(agent_name));
                payload.insert("model".into(), json!(model.name));
                if let Some(team_name) = team_name {
                    payload.insert("team_name".into(), json!(team_name));
                }
                if let Some(initial_message) = initial_message {
                    payload.insert("prompt".into(), json!(initial_message));
                }
                payload.insert("title".into(), json!(format!("Sinew · {agent_name}")));
                payload.insert("customTitle".into(), json!(format!("Sinew · {agent_name}")));
            }
            payloads
        }
        _ => Vec::new(),
    }
}

fn base_payload(
    workspace_id: &str,
    conversation_id: &str,
    hook_event_name: &str,
) -> Map<String, Value> {
    let mut payload = Map::new();
    payload.insert("hook_event_name".into(), json!(hook_event_name));
    payload.insert("session_id".into(), session_id(conversation_id));
    payload.insert("cwd".into(), json!(workspace_id));
    payload.insert("title".into(), json!("Sinew"));
    payload.insert("customTitle".into(), json!("Sinew"));
    payload.insert("sinew".into(), json!(true));
    payload
}

fn session_id(conversation_id: &str) -> Value {
    json!(format!("sinew:{conversation_id}"))
}

trait PayloadExt {
    fn with(self, key: &str, value: Value) -> Self;
}

impl PayloadExt for Map<String, Value> {
    fn with(mut self, key: &str, value: Value) -> Self {
        self.insert(key.into(), value);
        self
    }
}

fn spawn_bridge(payload: Map<String, Value>) {
    let Some(bridge) = bridge_path() else {
        return;
    };

    tauri::async_runtime::spawn(async move {
        let _ = time::timeout(BRIDGE_TIMEOUT, call_bridge(bridge, payload)).await;
    });
}

async fn call_bridge(bridge: PathBuf, payload: Map<String, Value>) -> std::io::Result<()> {
    let input = match serde_json::to_vec(&payload) {
        Ok(input) => input,
        Err(_) => return Ok(()),
    };

    let mut child = Command::new(bridge)
        .arg("--source")
        .arg(BRIDGE_SOURCE)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(&input).await?;
    }

    let _ = child.wait().await?;
    Ok(())
}

fn bridge_path() -> Option<PathBuf> {
    let home = env::var_os("HOME")?;
    let path = PathBuf::from(home).join(BRIDGE_RELATIVE_PATH);
    path.is_file().then_some(path)
}

fn tool_key(conversation_id: &str, tool_id: &str) -> String {
    format!("{conversation_id}:{tool_id}")
}

fn remember_tool_name(conversation_id: &str, tool_id: &str, name: &str) {
    let tools = TOOL_NAMES.get_or_init(Default::default);
    if let Ok(mut tools) = tools.lock() {
        tools.insert(tool_key(conversation_id, tool_id), name.to_string());
    }
}

fn remembered_tool_name(conversation_id: &str, tool_id: &str) -> Option<String> {
    TOOL_NAMES
        .get_or_init(Default::default)
        .lock()
        .ok()?
        .get(&tool_key(conversation_id, tool_id))
        .cloned()
}

fn forget_tool_name(conversation_id: &str, tool_id: &str) -> Option<String> {
    TOOL_NAMES
        .get_or_init(Default::default)
        .lock()
        .ok()?
        .remove(&tool_key(conversation_id, tool_id))
}

fn parse_json_or_string(input: &str) -> Value {
    serde_json::from_str(input).unwrap_or_else(|_| json!(input))
}

fn truncate_for_bridge(input: &str) -> String {
    const MAX_CHARS: usize = 8_000;
    let mut output = String::new();
    for (index, ch) in input.chars().enumerate() {
        if index >= MAX_CHARS {
            output.push_str("…");
            break;
        }
        output.push(ch);
    }
    output
}
