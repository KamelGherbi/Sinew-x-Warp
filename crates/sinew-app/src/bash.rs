use std::{
    collections::HashMap,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex as StdMutex,
    },
    thread,
    time::Duration,
};

use anyhow::{bail, Context, Result};
use portable_pty::{native_pty_system, ChildKiller, CommandBuilder, PtySize};
use serde::Deserialize;
use serde_json::{json, Value};
use sinew_core::ToolDescriptor;
use tokio::{
    sync::{mpsc, watch, Mutex},
    time::{sleep, timeout, Instant},
};

use crate::{
    tool_run::{diff_snapshots, snapshot_workspace, ToolRunResult, WorkspaceSnapshot},
    workspace::resolve_workspace_path,
};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_YIELD: Duration = Duration::from_millis(1_000);
const MAX_YIELD: Duration = Duration::from_secs(30);
const OUTPUT_LIMIT: usize = 64 * 1024;
const MAX_INTERACTIVE_SESSIONS: usize = 8;

#[derive(Clone)]
pub struct BashTool {
    workspace_root: PathBuf,
    timeout: Duration,
    sessions: Arc<Mutex<HashMap<u64, BashSession>>>,
    next_session_id: Arc<AtomicU64>,
}

impl BashTool {
    pub fn new(workspace_root: impl Into<PathBuf>) -> Self {
        Self {
            workspace_root: workspace_root.into(),
            timeout: DEFAULT_TIMEOUT,
            sessions: Arc::new(Mutex::new(HashMap::new())),
            next_session_id: Arc::new(AtomicU64::new(1)),
        }
    }

    pub fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            name: "bash".into(),
            description: "Run a shell command. Commands run in a temporary terminal session, so interactive prompts can stay open and be answered with bash_input.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "Shell command to execute." },
                    "description": { "type": "string", "description": "Short human-readable description of what this command does, in the user's language (e.g. \"Afficher la version de Node\", \"List project files\"). Shown in the UI card header." },
                    "cwd": { "type": "string", "description": "Optional working directory inside the workspace." },
                    "timeout_secs": { "type": "integer", "minimum": 1, "description": "Maximum lifetime for the command. Defaults to 120 seconds." },
                    "yield_time_ms": { "type": "integer", "minimum": 250, "description": "How long to wait for output or exit before returning a live session id. Defaults to 1000ms." }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        }
    }

    pub fn input_descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            name: "bash_input".into(),
            description: "Send text to an interactive bash session, poll its output, or stop it."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "integer", "minimum": 1, "description": "Session id returned by bash while a process is still running." },
                    "input": { "type": "string", "description": "Text to send to the process. Include a newline when submitting an answer. Leave empty to only poll output." },
                    "yield_time_ms": { "type": "integer", "minimum": 250, "description": "How long to wait for new output or process exit. Defaults to 1000ms." },
                    "kill": { "type": "boolean", "description": "Stop the session instead of sending input." }
                },
                "required": ["session_id"],
                "additionalProperties": false
            }),
        }
    }

    pub async fn run(&self, input: Value) -> ToolRunResult {
        let parsed: BashInput = match serde_json::from_value(input) {
            Ok(value) => value,
            Err(err) => {
                return ToolRunResult::err(format!("invalid bash input: {err}"), Vec::new());
            }
        };

        let cwd = match self.resolve_cwd(parsed.cwd.as_deref()) {
            Ok(path) => path,
            Err(err) => return ToolRunResult::err(err.to_string(), Vec::new()),
        };

        let max_lifetime = parsed
            .timeout_secs
            .map(Duration::from_secs)
            .unwrap_or(self.timeout)
            .min(self.timeout.max(Duration::from_secs(1)));
        let yield_time = parsed
            .yield_time_ms
            .map(Duration::from_millis)
            .unwrap_or(DEFAULT_YIELD)
            .clamp(Duration::from_millis(250), MAX_YIELD)
            .min(max_lifetime);

        let before = snapshot_workspace(&self.workspace_root);
        let mut session = match self.spawn_session(parsed.command, cwd, max_lifetime, before) {
            Ok(session) => session,
            Err(err) => return ToolRunResult::err(err.to_string(), Vec::new()),
        };

        let output = collect_output(&mut session, yield_time).await;
        self.finish_or_store(session, output).await
    }

    pub async fn run_input(&self, input: Value) -> ToolRunResult {
        let parsed: BashInputCommand = match serde_json::from_value(input) {
            Ok(value) => value,
            Err(err) => {
                return ToolRunResult::err(format!("invalid bash_input input: {err}"), Vec::new());
            }
        };

        let mut session = {
            let mut sessions = self.sessions.lock().await;
            match sessions.remove(&parsed.session_id) {
                Some(session) => session,
                None => {
                    return ToolRunResult::err(
                        format!("unknown bash session {}", parsed.session_id),
                        Vec::new(),
                    );
                }
            }
        };

        if parsed.kill {
            session.terminate();
            let output = collect_output(&mut session, Duration::from_millis(500)).await;
            return self.finish_killed(session, output).await;
        }

        if session.started_at.elapsed() >= session.max_lifetime {
            session.terminate();
            let output = collect_output(&mut session, Duration::from_millis(500)).await;
            return self.finish_timed_out(session, output).await;
        }

        if !parsed.input.is_empty() {
            if let Err(err) = session.write(parsed.input.as_bytes()) {
                return ToolRunResult::err(err, Vec::new());
            }
        }

        let remaining_lifetime = session
            .max_lifetime
            .saturating_sub(session.started_at.elapsed());
        let yield_time = parsed
            .yield_time_ms
            .map(Duration::from_millis)
            .unwrap_or(DEFAULT_YIELD)
            .clamp(Duration::from_millis(250), MAX_YIELD)
            .min(remaining_lifetime.max(Duration::from_millis(250)));

        let output = collect_output(&mut session, yield_time).await;
        if session.started_at.elapsed() >= session.max_lifetime && output.exited.is_none() {
            session.terminate();
            let extra = collect_output(&mut session, Duration::from_millis(500)).await;
            return self.finish_timed_out(session, output.join(extra)).await;
        }

        self.finish_or_store(session, output).await
    }

    fn resolve_cwd(&self, cwd: Option<&str>) -> Result<PathBuf> {
        match cwd.filter(|value| !value.trim().is_empty()) {
            Some(value) if Path::new(value).is_absolute() => {
                let absolute = PathBuf::from(value);
                let canonical = absolute
                    .canonicalize()
                    .with_context(|| format!("unable to resolve cwd {}", absolute.display()))?;
                if canonical.starts_with(&self.workspace_root) {
                    Ok(canonical)
                } else {
                    bail!("cwd must stay inside the workspace")
                }
            }
            Some(value) => resolve_workspace_path(&self.workspace_root, value),
            None => Ok(self.workspace_root.clone()),
        }
    }

    fn spawn_session(
        &self,
        command: String,
        cwd: PathBuf,
        max_lifetime: Duration,
        before: WorkspaceSnapshot,
    ) -> Result<BashSession> {
        let pty_system = native_pty_system();
        let pair = pty_system
            .openpty(PtySize {
                rows: 24,
                cols: 100,
                pixel_width: 0,
                pixel_height: 0,
            })
            .context("unable to open pty")?;

        let mut builder = CommandBuilder::new("/bin/bash");
        builder.arg("-lc");
        builder.arg(&command);
        builder.cwd(cwd.as_os_str());
        builder.env("TERM", "dumb");
        builder.env("NO_COLOR", "1");
        builder.env("PAGER", "cat");
        builder.env("GIT_PAGER", "cat");
        builder.env("GH_PAGER", "cat");

        let mut child = pair
            .slave
            .spawn_command(builder)
            .context("unable to spawn bash")?;
        drop(pair.slave);

        let mut reader = pair
            .master
            .try_clone_reader()
            .context("pty reader unavailable")?;
        let writer = Arc::new(StdMutex::new(
            pair.master
                .take_writer()
                .context("pty writer unavailable")?,
        ));
        let killer = Arc::new(StdMutex::new(child.clone_killer()));
        let (output_tx, output_rx) = mpsc::unbounded_channel();
        let (exit_tx, exit_rx) = watch::channel(None);

        thread::spawn(move || {
            let mut buffer = [0u8; 8192];
            loop {
                match reader.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => {
                        if output_tx.send(buffer[..n].to_vec()).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        thread::spawn(move || {
            let exit = match child.wait() {
                Ok(status) => {
                    let signal = status.signal().map(|value| value.to_string());
                    let code = status.exit_code();
                    let display = signal
                        .as_ref()
                        .map(|value| format!("signal {value}"))
                        .unwrap_or_else(|| code.to_string());
                    SessionExit {
                        display,
                        success: signal.is_none() && code == 0,
                    }
                }
                Err(err) => SessionExit {
                    display: err.to_string(),
                    success: false,
                },
            };
            let _ = exit_tx.send(Some(exit));
        });

        Ok(BashSession {
            id: self.next_session_id.fetch_add(1, Ordering::Relaxed),
            writer,
            killer,
            output_rx,
            exit_rx,
            before,
            started_at: Instant::now(),
            max_lifetime,
        })
    }

    async fn finish_or_store(
        &self,
        mut session: BashSession,
        output: CapturedOutput,
    ) -> ToolRunResult {
        if let Some(exit) = output.exited {
            return self.finish_exited(session, output.text, output.truncated, exit);
        }

        let mut sessions = self.sessions.lock().await;
        if sessions.len() >= MAX_INTERACTIVE_SESSIONS {
            drop(sessions);
            session.terminate();
            let extra = collect_output(&mut session, Duration::from_millis(500)).await;
            return self.finish_timed_out(session, output.join(extra)).await;
        }

        let session_id = session.id;
        let text = interactive_transcript(output.text, output.truncated, session_id);
        sessions.insert(session_id, session);
        ToolRunResult::ok(text, Vec::new())
    }

    fn finish_exited(
        &self,
        session: BashSession,
        text: String,
        truncated: bool,
        exit: SessionExit,
    ) -> ToolRunResult {
        let after = snapshot_workspace(&self.workspace_root);
        let file_changes = diff_snapshots(session.before.clone(), after);
        let transcript = final_transcript(text, truncated, &exit.display);
        if exit.success {
            ToolRunResult::ok(transcript, file_changes)
        } else {
            ToolRunResult::err(transcript, file_changes)
        }
    }

    async fn finish_killed(&self, session: BashSession, output: CapturedOutput) -> ToolRunResult {
        let after = snapshot_workspace(&self.workspace_root);
        let file_changes = diff_snapshots(session.before.clone(), after);
        let mut transcript = output.text;
        if output.truncated {
            transcript.push_str("\n...[output truncated]");
        }
        transcript.push_str("\n[process killed]");
        ToolRunResult::ok(transcript, file_changes)
    }

    async fn finish_timed_out(
        &self,
        session: BashSession,
        output: CapturedOutput,
    ) -> ToolRunResult {
        let after = snapshot_workspace(&self.workspace_root);
        let file_changes = diff_snapshots(session.before.clone(), after);
        let mut transcript = output.text;
        if output.truncated {
            transcript.push_str("\n...[output truncated]");
        }
        transcript.push_str(&format!(
            "\n[bash command timed out after {}s]",
            session.max_lifetime.as_secs()
        ));
        ToolRunResult::err(transcript, file_changes)
    }
}

#[derive(Debug, Deserialize)]
struct BashInput {
    command: String,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
    #[serde(default)]
    yield_time_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct BashInputCommand {
    session_id: u64,
    #[serde(default)]
    input: String,
    #[serde(default)]
    yield_time_ms: Option<u64>,
    #[serde(default)]
    kill: bool,
}

#[derive(Debug, Clone)]
struct SessionExit {
    display: String,
    success: bool,
}

struct BashSession {
    id: u64,
    writer: Arc<StdMutex<Box<dyn Write + Send>>>,
    killer: Arc<StdMutex<Box<dyn ChildKiller + Send + Sync>>>,
    output_rx: mpsc::UnboundedReceiver<Vec<u8>>,
    exit_rx: watch::Receiver<Option<SessionExit>>,
    before: WorkspaceSnapshot,
    started_at: Instant,
    max_lifetime: Duration,
}

impl BashSession {
    fn write(&self, bytes: &[u8]) -> std::result::Result<(), String> {
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| "bash session writer unavailable".to_string())?;
        writer.write_all(bytes).map_err(|err| err.to_string())?;
        writer.flush().map_err(|err| err.to_string())
    }

    fn terminate(&self) {
        if let Ok(mut killer) = self.killer.lock() {
            let _ = killer.kill();
        }
    }
}

impl Drop for BashSession {
    fn drop(&mut self) {
        if self.exit_rx.borrow().is_none() {
            self.terminate();
        }
    }
}

#[derive(Debug)]
struct CapturedOutput {
    text: String,
    truncated: bool,
    exited: Option<SessionExit>,
}

impl CapturedOutput {
    fn join(self, next: CapturedOutput) -> Self {
        Self {
            text: format!("{}{}", self.text, next.text),
            truncated: self.truncated || next.truncated,
            exited: next.exited.or(self.exited),
        }
    }
}

async fn collect_output(session: &mut BashSession, wait: Duration) -> CapturedOutput {
    let deadline = Instant::now() + wait;
    let mut bytes = Vec::with_capacity(4096);
    let mut truncated = false;
    let mut exit_seen = session.exit_rx.borrow().clone();

    loop {
        while let Ok(chunk) = session.output_rx.try_recv() {
            append_limited(&mut bytes, &mut truncated, &chunk);
        }

        exit_seen = exit_seen.or_else(|| session.exit_rx.borrow().clone());
        if exit_seen.is_some() {
            if let Ok(Some(chunk)) =
                timeout(Duration::from_millis(50), session.output_rx.recv()).await
            {
                append_limited(&mut bytes, &mut truncated, &chunk);
                continue;
            }
            while let Ok(chunk) = session.output_rx.try_recv() {
                append_limited(&mut bytes, &mut truncated, &chunk);
            }
            break;
        }

        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }

        tokio::select! {
            chunk = session.output_rx.recv() => {
                match chunk {
                    Some(chunk) => append_limited(&mut bytes, &mut truncated, &chunk),
                    None => {
                        let _ = timeout(Duration::from_millis(100), session.exit_rx.changed()).await;
                        exit_seen = session.exit_rx.borrow().clone();
                        break;
                    }
                }
            }
            changed = session.exit_rx.changed() => {
                if changed.is_err() {
                    break;
                }
                exit_seen = session.exit_rx.borrow().clone();
            }
            _ = sleep(remaining) => break,
        }
    }

    CapturedOutput {
        text: String::from_utf8_lossy(&bytes).into_owned(),
        truncated,
        exited: exit_seen,
    }
}

fn append_limited(bytes: &mut Vec<u8>, truncated: &mut bool, chunk: &[u8]) {
    let remaining = OUTPUT_LIMIT.saturating_sub(bytes.len());
    if remaining == 0 {
        *truncated = true;
        return;
    }
    let take = remaining.min(chunk.len());
    bytes.extend_from_slice(&chunk[..take]);
    if take < chunk.len() {
        *truncated = true;
    }
}

fn interactive_transcript(mut text: String, truncated: bool, session_id: u64) -> String {
    if truncated {
        text.push_str("\n...[output truncated]");
    }
    text.push_str(&format!(
        "\n[process still running: bash session {session_id}]\nUse bash_input with session_id {session_id} to send input or poll output. Include a newline when answering a prompt. Use kill=true to stop it."
    ));
    text
}

fn final_transcript(mut text: String, truncated: bool, status: &str) -> String {
    if truncated {
        text.push_str("\n...[output truncated]");
    }
    text.push_str(&format!("\n[exit status: {status}]"));
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[tokio::test]
    async fn interactive_session_accepts_input() {
        let root = temp_workspace("interactive_session_accepts_input");
        let tool = BashTool::new(&root);

        let started = tool
            .run(json!({
                "command": "printf 'Name: '; read name; printf 'Hello %s\\n' \"$name\"",
                "yield_time_ms": 500
            }))
            .await;

        assert!(!started.is_error, "{}", started.content);
        assert!(started.content.contains("process still running"));
        let session_id = parse_session_id(&started.content);

        let finished = tool
            .run_input(json!({
                "session_id": session_id,
                "input": "Ada\n",
                "yield_time_ms": 1_000
            }))
            .await;

        assert!(!finished.is_error, "{}", finished.content);
        assert!(
            finished.content.contains("Hello Ada"),
            "{}",
            finished.content
        );
        assert!(
            finished.content.contains("[exit status: 0]"),
            "{}",
            finished.content
        );

        let _ = fs::remove_dir_all(root);
    }

    fn parse_session_id(content: &str) -> u64 {
        content
            .split("bash session ")
            .nth(1)
            .and_then(|tail| tail.split(']').next())
            .and_then(|value| value.parse::<u64>().ok())
            .expect("session id should be present")
    }

    fn temp_workspace(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("sinew-bash-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create temp workspace");
        root
    }
}
