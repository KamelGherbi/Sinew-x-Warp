use std::{
    collections::{BTreeSet, HashMap},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use futures_util::future::join_all;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_core::{ChatMessage, ModelRef, Part, Provider, Role, ToolDescriptor};
use tokio::sync::{mpsc, Notify, RwLock, Semaphore};
use uuid::Uuid;

use crate::tool_run::{DiffLineKind, FileChange, FileChangeKind, ToolRunImage};
use crate::{
    run_turn, subagent_system_prompt, AgentEvent, AgentEventScope, AgentMode, ApplyPatchTool,
    BashTool, CreateImageTool, GlobTool, GoalWorkflowState, GrepTool, McpSettings, McpToolRegistry,
    ReadTool, SkillSettings, SkillTool, SubAgentConfig, SubAgentSettings, TodoListState,
    ToolRunResult, ToolSettings, TurnCancel, TurnContext, WebFetchTool, WebSearchTool,
};

const TEAM_RUN_TOOL: &str = "TeamRun";
const TEAM_CREATE_TOOL: &str = "TeamCreate";
const AGENT_TOOL: &str = "Agent";
const SEND_MESSAGE_TOOL: &str = "SendMessage";
const TEAM_STATUS_TOOL: &str = "TeamStatus";
const TEAM_STOP_TOOL: &str = "TeamStop";
const TASK_CREATE_TOOL: &str = "TaskCreate";
const TASK_LIST_TOOL: &str = "TaskList";
const TASK_UPDATE_TOOL: &str = "TaskUpdate";
const TEAM_SETTLE_GRACE_MS: u64 = 100;
const TEAM_RECENT_FILE_CHANGE_LIMIT: usize = 20;

#[derive(Debug, Default)]
pub struct TeamRuntime {
    scopes: HashMap<String, TeamScope>,
    agent_notifiers: HashMap<String, Arc<Notify>>,
    workspace_write_locks: HashMap<String, Arc<Semaphore>>,
}

#[derive(Debug, Default)]
struct TeamScope {
    active_team: Option<String>,
    teams: HashMap<String, TeamSession>,
    team_cancels: HashMap<String, Vec<TurnCancel>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamSession {
    pub name: String,
    pub description: Option<String>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    pub agents: HashMap<String, TeamAgent>,
    pub tasks: Vec<TeamTask>,
    pub next_task_id: u64,
    pub queued_messages: Vec<TeamQueuedMessage>,
    pub next_message_id: u64,
    #[serde(default)]
    pub pending_task_wakes: Vec<TeamTaskWake>,
    #[serde(default)]
    pub recent_file_changes: Vec<TeamRecentFileChange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamAgent {
    pub id: String,
    pub name: String,
    pub description: String,
    pub prompt: String,
    pub model: ModelRef,
    pub status: TeamAgentStatus,
    pub history: Vec<ChatMessage>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    pub last_summary: Option<String>,
    #[serde(default)]
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TeamAgentStatus {
    Idle,
    Running,
    Stopped,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamTask {
    pub id: u64,
    pub subject: String,
    pub description: Option<String>,
    pub status: TeamTaskStatus,
    pub owner: Option<String>,
    pub blocked_by: Vec<u64>,
    pub created_by: String,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    pub completed_at_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamQueuedMessage {
    pub id: u64,
    pub from: String,
    pub to: String,
    #[serde(default)]
    pub target: Option<String>,
    pub message: String,
    pub created_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamTaskWake {
    pub task_id: u64,
    pub owner: String,
    pub created_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamRecentFileChange {
    pub agent: String,
    pub tool: String,
    pub relative_path: String,
    pub kind: FileChangeKind,
    pub added: usize,
    pub removed: usize,
    pub created_at_ms: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TeamTaskStatus {
    #[serde(alias = "todo")]
    Pending,
    InProgress,
    Blocked,
    #[serde(alias = "done", alias = "resolved")]
    Completed,
}

#[derive(Clone)]
pub struct TeamTool {
    scope_id: String,
    workspace_root: PathBuf,
    system_prompt: String,
    providers: HashMap<String, Arc<dyn Provider>>,
    sub_agent_settings: SubAgentSettings,
    mcp_settings: McpSettings,
    tool_settings: ToolSettings,
    skill_settings: SkillSettings,
    default_model: ModelRef,
    max_tool_rounds: usize,
    runtime: Arc<RwLock<TeamRuntime>>,
    cancel: TurnCancel,
    current_agent: Option<TeamIdentity>,
}

#[derive(Clone)]
struct TeamIdentity {
    team_name: String,
    agent_name: String,
}

impl TeamTool {
    pub fn new(
        scope_id: String,
        workspace_root: PathBuf,
        system_prompt: String,
        providers: HashMap<String, Arc<dyn Provider>>,
        sub_agent_settings: SubAgentSettings,
        mcp_settings: McpSettings,
        tool_settings: ToolSettings,
        skill_settings: SkillSettings,
        default_model: ModelRef,
        max_tool_rounds: usize,
        runtime: Arc<RwLock<TeamRuntime>>,
        cancel: TurnCancel,
    ) -> Self {
        Self {
            scope_id,
            workspace_root,
            system_prompt,
            providers,
            sub_agent_settings: sub_agent_settings.normalized(),
            mcp_settings,
            tool_settings,
            skill_settings,
            default_model,
            max_tool_rounds,
            runtime,
            cancel,
            current_agent: None,
        }
    }

    fn for_agent(&self, team_name: String, agent_name: String) -> Self {
        let mut next = self.clone();
        next.current_agent = Some(TeamIdentity {
            team_name,
            agent_name,
        });
        next
    }

    fn with_cancel(&self, cancel: TurnCancel) -> Self {
        let mut next = self.clone();
        next.cancel = cancel;
        next
    }

    pub fn descriptors(&self) -> Vec<ToolDescriptor> {
        if self.current_agent.is_some() {
            Self::agent_descriptors_static()
        } else {
            Self::descriptors_static()
        }
    }

    pub fn descriptors_static() -> Vec<ToolDescriptor> {
        vec![
            ToolDescriptor {
                name: TEAM_RUN_TOOL.into(),
                description: "Launch an agent team. Use this for swarms/agent teams.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "objective": {
                            "type": "string",
                            "description": "Full user objective for a new team. Required unless relaunching an existing teammate with agent."
                        },
                        "agent": {
                            "type": "string",
                            "description": "Teammate name to relaunch in the active team. When set, provide only agent."
                        },
                        "agent_names": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 8,
                            "items": { "type": "string" },
                            "description": "Required teammate names when starting a new team."
                        },
                        "agent_profiles": {
                            "type": "array",
                            "description": "Optional profile assignments for teammates. Use this to make a team member inherit a configured sub-agent profile. Each agent must match agent_names; each profile is a configured sub-agent id or name.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "agent": {
                                        "type": "string",
                                        "description": "Teammate name from agent_names."
                                    },
                                    "profile": {
                                        "type": "string",
                                        "description": "Configured sub-agent id or name to use for that teammate."
                                    }
                                },
                                "required": ["agent", "profile"],
                                "additionalProperties": false
                            }
                        },
                        "agent_prompts": {
                            "description": "Optional teammate-specific launch prompts from the main agent. Each agent must match agent_names. These prompts are delivered only to that teammate alongside the shared objective.",
                            "oneOf": [
                                {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "agent": {
                                                "type": "string",
                                                "description": "Teammate name from agent_names."
                                            },
                                            "prompt": {
                                                "type": "string",
                                                "description": "Specific launch prompt for this teammate."
                                            }
                                        },
                                        "required": ["agent", "prompt"],
                                        "additionalProperties": false
                                    }
                                },
                                {
                                    "type": "object",
                                    "additionalProperties": { "type": "string" },
                                    "description": "Map from teammate name to that teammate's launch prompt."
                                }
                            ]
                        },
                        "tasks": {
                            "type": "array",
                            "description": "Optional initial shared task board for a new team. Prefer parallel, autonomous workstreams owned by different teammates. Avoid a fully sequential chain. Use blockedBy only for real dependency constraints (scaffold etc..).",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "subject": {
                                        "type": "string",
                                        "description": "Short task title."
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Detailed task instructions."
                                    },
                                    "owner": {
                                        "type": "string",
                                        "description": "Optional teammate name that should own this task."
                                    },
                                    "blockedBy": {
                                        "type": "array",
                                        "items": {
                                            "oneOf": [
                                                { "type": "integer", "minimum": 1 },
                                                { "type": "string" }
                                            ]
                                        },
                                        "description": "Initial task IDs this task waits on. IDs are assigned by task order starting at 1."
                                    },
                                    "blocked_by": {
                                        "type": "array",
                                        "items": {
                                            "oneOf": [
                                                { "type": "integer", "minimum": 1 },
                                                { "type": "string" }
                                            ]
                                        },
                                        "description": "Alias for blockedBy."
                                    }
                                },
                                "required": ["subject"],
                                "additionalProperties": false
                            }
                        }
                    },
                    "additionalProperties": false
                }),
            },
            ToolDescriptor {
                name: TEAM_STATUS_TOOL.into(),
                description: "Inspect the active agent team, teammates, queued messages, tasks, and latest summaries.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
            },
            ToolDescriptor {
                name: TEAM_STOP_TOOL.into(),
                description: "Stop one teammate or stop an entire agent team.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "Optional teammate name to stop. Omit to stop every teammate in the team."
                        }
                    },
                    "additionalProperties": false
                }),
            },
        ]
    }

    pub fn agent_descriptors_static() -> Vec<ToolDescriptor> {
        vec![
            ToolDescriptor {
                name: SEND_MESSAGE_TOOL.into(),
                description: "Send a live async message to a teammate by name, or broadcast to all teammates with to=\"*\". Peers receive messages at their next model turn.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "to": { "type": "string", "description": "Teammate name, without @. Use * for broadcast." },
                        "message": { "type": "string", "description": "Plain text peer message." }
                    },
                    "required": ["to", "message"],
                    "additionalProperties": false
                }),
            },
            ToolDescriptor {
                name: TASK_LIST_TOOL.into(),
                description: "Mutate the agent team's shared task board. The latest board is injected as a system reminder before every model call, so do not call this tool just to inspect tasks. Tasks with unresolved blockedBy dependencies stay blocked; update/delete are allowed, but status cannot become pending, in_progress, or completed until dependencies finish.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["create", "update", "delete", "claim"],
                            "description": "Operation to perform. claim assigns ownership only; use update with status=in_progress when actually starting work."
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "blocked", "completed"],
                            "description": "For action=update, new task status. Tasks with unresolved blockedBy dependencies cannot be set to pending, in_progress, or completed."
                        },
                        "taskId": {
                            "oneOf": [{ "type": "integer", "minimum": 1 }, { "type": "string" }],
                            "description": "Task ID for update/delete/claim."
                        },
                        "id": {
                            "oneOf": [{ "type": "integer", "minimum": 1 }, { "type": "string" }],
                            "description": "Alias for taskId."
                        },
                        "subject": {
                            "type": "string",
                            "description": "Short task title for create/update."
                        },
                        "owner": {
                            "type": "string",
                            "description": "Owner teammate name for create/update/claim. Claim does not change task status."
                        },
                        "clear_owner": {
                            "type": "boolean",
                            "description": "For action=update, clear the current owner."
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed task instructions for create/update. Empty string clears it on update."
                        },
                        "addBlockedBy": {
                            "type": "array",
                            "items": { "type": "integer", "minimum": 1 },
                            "description": "Additional task IDs that block this task, e.g. [1, 3, 4]. Use this for task dependencies."
                        },
                        "blockedBy": {
                            "type": "array",
                            "items": { "type": "integer", "minimum": 1 },
                            "description": "Replace dependencies with task IDs only, e.g. [1, 3, 4]. Unresolved dependencies keep the task blocked and clear automatically only after those tasks complete."
                        }
                    },
                    "required": ["action"],
                    "additionalProperties": false
                }),
            },
        ]
    }

    pub fn summary_for_tool_name(&self, name: &str) -> Option<String> {
        match name {
            TEAM_RUN_TOOL => Some("Agent Swarm · run".to_string()),
            TEAM_CREATE_TOOL => Some("Agent Swarm · disabled create".to_string()),
            AGENT_TOOL => Some("Agent Swarm · disabled agent spawn".to_string()),
            SEND_MESSAGE_TOOL => Some("Agent Swarm · message".to_string()),
            TASK_CREATE_TOOL => Some("Task · create".to_string()),
            TASK_UPDATE_TOOL => Some("Task · update".to_string()),
            TEAM_STATUS_TOOL => Some("Agent Swarm · status".to_string()),
            TEAM_STOP_TOOL => Some("Agent Swarm · stop".to_string()),
            _ => None,
        }
    }

    pub async fn current_agent_system_reminder(&self) -> Option<String> {
        let Some(identity) = self.current_agent.as_ref() else {
            let runtime = self.runtime.read().await;
            let scope = runtime.scopes.get(&self.scope_id)?;
            let team_name = scope.active_team.as_deref().or_else(|| {
                if scope.teams.len() == 1 {
                    scope.teams.keys().next().map(String::as_str)
                } else {
                    None
                }
            })?;
            let session = scope.teams.get(team_name)?;
            return Some(render_main_agent_team_system_reminder(session));
        };
        let mut runtime = self.runtime.write().await;
        let session = runtime
            .scopes
            .get_mut(&self.scope_id)
            .and_then(|scope| scope.teams.get_mut(&identity.team_name))?;
        let done_ids = completed_task_ids(session);
        refresh_unblocked_tasks(session, &done_ids);
        Some(render_agent_team_system_reminder(
            session,
            &identity.agent_name,
        ))
    }

    pub async fn drain_current_agent_messages_prompt(&self) -> Option<String> {
        let identity = self.current_agent.as_ref()?;
        let mut runtime = self.runtime.write().await;
        let session = runtime
            .scopes
            .get_mut(&self.scope_id)
            .and_then(|scope| scope.teams.get_mut(&identity.team_name))?;
        let key = agent_key(&identity.agent_name);
        let mut messages = Vec::new();
        let mut index = 0usize;
        while index < session.queued_messages.len() {
            if agent_key(&session.queued_messages[index].to) == key {
                messages.push(session.queued_messages.remove(index));
            } else {
                index += 1;
            }
        }
        if messages.is_empty() {
            return None;
        }
        session.updated_at_ms = now_ms();
        Some(queued_messages_prompt(&messages))
    }

    pub async fn record_current_agent_file_changes(
        &self,
        tool_name: &str,
        file_changes: &[FileChange],
    ) {
        if file_changes.is_empty() {
            return;
        }
        let Some(identity) = self.current_agent.as_ref() else {
            return;
        };
        let now = now_ms();
        let mut runtime = self.runtime.write().await;
        let Some(session) = runtime
            .scopes
            .get_mut(&self.scope_id)
            .and_then(|scope| scope.teams.get_mut(&identity.team_name))
        else {
            return;
        };
        for change in file_changes {
            let (added, removed) = file_change_line_counts(change);
            session.recent_file_changes.push(TeamRecentFileChange {
                agent: identity.agent_name.clone(),
                tool: tool_name.to_string(),
                relative_path: change.relative_path.clone(),
                kind: change.kind,
                added,
                removed,
                created_at_ms: now,
            });
        }
        if session.recent_file_changes.len() > TEAM_RECENT_FILE_CHANGE_LIMIT {
            let drain_count = session.recent_file_changes.len() - TEAM_RECENT_FILE_CHANGE_LIMIT;
            session.recent_file_changes.drain(0..drain_count);
        }
        session.updated_at_ms = now;
    }

    pub async fn run(
        &self,
        tool_call_id: &str,
        name: &str,
        input: Value,
        mode: AgentMode,
        parent_event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> Option<ToolRunResult> {
        let result = match name {
            TEAM_RUN_TOOL => {
                self.run_team_run(tool_call_id, input, mode, parent_event_tx)
                    .await
            }
            TEAM_CREATE_TOOL => ToolRunResult::err(
                "TeamCreate is disabled. Use TeamRun to start an agent team.",
                Vec::new(),
            ),
            AGENT_TOOL => ToolRunResult::err(
                "Agent is disabled for teams. Use TeamRun to start an agent team.",
                Vec::new(),
            ),
            SEND_MESSAGE_TOOL => {
                self.run_send_message(tool_call_id, input, mode, parent_event_tx)
                    .await
            }
            TASK_CREATE_TOOL => self.run_task_create(input, mode, parent_event_tx).await,
            TASK_LIST_TOOL => self.run_task_list(input, mode, parent_event_tx).await,
            TASK_UPDATE_TOOL => self.run_task_update(input, mode, parent_event_tx).await,
            TEAM_STATUS_TOOL => self.run_status(input).await,
            TEAM_STOP_TOOL => self.run_stop(input, parent_event_tx).await,
            _ => return None,
        };
        Some(result)
    }

    async fn run_team_run(
        &self,
        tool_call_id: &str,
        input: Value,
        mode: AgentMode,
        parent_event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> ToolRunResult {
        if self.current_agent.is_some() {
            return ToolRunResult::err(
                "TeamRun can only be started by the user-facing agent",
                Vec::new(),
            );
        }
        let parsed: TeamRunInput = match serde_json::from_value(input) {
            Ok(value) => value,
            Err(err) => {
                return ToolRunResult::err(format!("invalid TeamRun input: {err}"), Vec::new())
            }
        };
        if let Some(key) = parsed.extra.keys().next() {
            return ToolRunResult::err(format!("unknown TeamRun field `{key}`"), Vec::new());
        }
        let agent_profiles = match parsed.agent_profiles.as_ref() {
            Some(value) => match value.to_profile_map() {
                Ok(value) => Some(value),
                Err(err) => return ToolRunResult::err(err, Vec::new()),
            },
            None => None,
        };
        let agent_prompt_inputs = match parsed.agent_prompts.as_ref() {
            Some(value) => match value.to_prompt_map() {
                Ok(value) => Some(value),
                Err(err) => return ToolRunResult::err(err, Vec::new()),
            },
            None => None,
        };
        let has_start_only_fields = parsed.objective.is_some()
            || parsed.agent_names.is_some()
            || agent_profiles.is_some()
            || agent_prompt_inputs.is_some()
            || parsed.tasks.is_some();
        if let Some(agent_name) = parsed
            .agent
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            if has_start_only_fields {
                return ToolRunResult::err("TeamRun restart accepts only agent", Vec::new());
            }
            return self
                .run_team_agent_restart(tool_call_id, None, agent_name, mode, parent_event_tx)
                .await;
        }

        let objective = parsed.objective.as_deref().map(str::trim).unwrap_or("");
        if objective.is_empty() {
            return ToolRunResult::err(
                "objective is required when starting a new team",
                Vec::new(),
            );
        }
        let team_name = format!(
            "team-{}",
            agent_key(objective).chars().take(32).collect::<String>()
        );
        let agent_names = match prepare_team_agent_names(parsed.agent_names) {
            Ok(value) => value,
            Err(err) => return ToolRunResult::err(err, Vec::new()),
        };
        let initial_tasks = match prepare_team_run_tasks(parsed.tasks.as_deref(), &agent_names) {
            Ok(value) => value,
            Err(err) => return ToolRunResult::err(err, Vec::new()),
        };
        let agent_prompts =
            match prepare_team_agent_prompts(&agent_names, agent_prompt_inputs.as_ref()) {
                Ok(value) => value,
                Err(err) => return ToolRunResult::err(err, Vec::new()),
            };
        let agent_configs =
            match self.prepare_team_agent_configs(&agent_names, agent_profiles.as_ref()) {
                Ok(value) => value,
                Err(err) => return ToolRunResult::err(err, Vec::new()),
            };
        self.create_or_reset_team(&team_name, Some(objective.to_string()))
            .await;
        for config in &agent_configs {
            self.ensure_agent(
                &team_name,
                &config.name,
                config.description.clone(),
                config.prompt.clone(),
                config.model.clone(),
            )
            .await;
        }
        if let Err(err) = self.seed_team_run_tasks(&team_name, initial_tasks).await {
            return ToolRunResult::err(err, Vec::new());
        }

        let initial_turns = agent_names
            .iter()
            .map(|agent_name| {
                let agent_prompt = agent_prompts
                    .get(&agent_key(agent_name))
                    .map(String::as_str);
                TeamTurn {
                    agent_name: agent_name.clone(),
                    message: team_kickoff_message(objective, agent_name, agent_prompt),
                    task_id: None,
                    label: "initial team kickoff".to_string(),
                }
            })
            .collect::<Vec<_>>();
        let team_cancel = TurnCancel::empty();
        self.register_team_cancel(&team_name, team_cancel.clone())
            .await;
        self.with_cancel(team_cancel).spawn_agent_team_live(
            tool_call_id.to_string(),
            team_name.clone(),
            initial_turns,
            mode,
            parent_event_tx,
        );

        self.team_run_started_result(&team_name, "Agent Swarm started in background")
            .await
    }

    async fn run_team_agent_restart(
        &self,
        tool_call_id: &str,
        team_name: Option<&str>,
        agent_name: &str,
        mode: AgentMode,
        parent_event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> ToolRunResult {
        let team_name = match self.resolve_team_name(team_name).await {
            Ok(value) => value,
            Err(err) => return ToolRunResult::err(err, Vec::new()),
        };
        let agent_name = match self.prepare_agent_restart(&team_name, agent_name).await {
            Ok(value) => value,
            Err(err) => return ToolRunResult::err(err, Vec::new()),
        };
        let message = team_restart_message(&team_name, &agent_name);
        let team_cancel = TurnCancel::empty();
        self.register_team_cancel(&team_name, team_cancel.clone())
            .await;
        let restart_turn = TeamTurn {
            agent_name: agent_name.clone(),
            message,
            task_id: None,
            label: "restart".to_string(),
        };
        self.with_cancel(team_cancel).spawn_agent_team_live(
            tool_call_id.to_string(),
            team_name.clone(),
            vec![restart_turn],
            mode,
            parent_event_tx,
        );
        self.team_run_started_result(
            &team_name,
            &format!("restarted teammate @{agent_name} in background"),
        )
        .await
    }

    fn spawn_agent_team_live(
        &self,
        tool_call_id: String,
        team_name: String,
        initial_turns: Vec<TeamTurn>,
        mode: AgentMode,
        parent_event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) {
        let tool = self.clone();
        tokio::spawn(async move {
            let mut result = tool
                .run_agent_team_live(
                    &tool_call_id,
                    &team_name,
                    initial_turns,
                    mode,
                    parent_event_tx.clone(),
                )
                .await;
            tool.attach_team_run_status_meta(&team_name, &mut result, "completed")
                .await;
            let _ = parent_event_tx.send(AgentEvent::ToolFinished {
                id: tool_call_id,
                output: result.content,
                is_error: result.is_error,
                file_changes: result.file_changes,
                images: result.images,
                meta: result.meta,
            });
        });
    }

    async fn register_team_cancel(&self, team_name: &str, cancel: TurnCancel) {
        let mut runtime = self.runtime.write().await;
        runtime
            .scopes
            .entry(self.scope_id.clone())
            .or_default()
            .team_cancels
            .entry(team_name.to_string())
            .or_default()
            .push(cancel);
    }

    async fn workspace_write_lock(&self) -> Arc<Semaphore> {
        let key = workspace_write_lock_key(&self.workspace_root);
        let mut runtime = self.runtime.write().await;
        runtime
            .workspace_write_locks
            .entry(key)
            .or_insert_with(|| Arc::new(Semaphore::new(1)))
            .clone()
    }

    async fn agent_notify(&self, team_name: &str, agent_name: &str) -> Arc<Notify> {
        let key = agent_notify_key(&self.scope_id, team_name, agent_name);
        let mut runtime = self.runtime.write().await;
        runtime
            .agent_notifiers
            .entry(key)
            .or_insert_with(|| Arc::new(Notify::new()))
            .clone()
    }

    async fn notify_team_agents(&self, team_name: &str, agent_names: &[String]) {
        if agent_names.is_empty() {
            return;
        }
        let keys = agent_names
            .iter()
            .map(|agent_name| agent_notify_key(&self.scope_id, team_name, agent_name))
            .collect::<Vec<_>>();
        let notifiers = {
            let runtime = self.runtime.read().await;
            keys.iter()
                .filter_map(|key| runtime.agent_notifiers.get(key).cloned())
                .collect::<Vec<_>>()
        };
        for notifier in notifiers {
            wake_notifier(&notifier);
        }
    }

    async fn notify_all_team_agents(&self, team_name: &str) {
        let notifiers = {
            let runtime = self.runtime.read().await;
            let Some(session) = runtime
                .scopes
                .get(&self.scope_id)
                .and_then(|scope| scope.teams.get(team_name))
            else {
                return;
            };
            let mut notifiers = Vec::new();
            for agent in session.agents.values() {
                let key = agent_notify_key(&self.scope_id, team_name, &agent.name);
                if let Some(notifier) = runtime.agent_notifiers.get(&key) {
                    notifiers.push(notifier.clone());
                }
            }
            notifiers
        };
        for notifier in notifiers {
            wake_notifier(&notifier);
        }
    }

    async fn team_run_started_result(&self, team_name: &str, label: &str) -> ToolRunResult {
        let snapshot = self.team_snapshot(team_name).await;
        let subagents = self.team_subagents_meta(team_name).await;
        let content = match &snapshot {
            Some(snapshot) => format!(
                "{label}\n\n{}\n\nAgent Swarm is running asynchronously. Do not poll with shell commands or TeamStatus to check progress; end this turn after acknowledging launch and wait for a user/system wake.",
                render_team_snapshot(snapshot)
            ),
            None => format!(
                "{label}\n\nAgent Swarm is running asynchronously. Do not poll with shell commands or TeamStatus to check progress; end this turn after acknowledging launch and wait for a user/system wake."
            ),
        };
        let mut result = match snapshot {
            Some(snapshot) => ToolRunResult::ok_with_meta(
                content,
                Vec::new(),
                json!({ "team": snapshot, "subagents": subagents }),
            ),
            None => ToolRunResult::ok(content, Vec::new()),
        };
        self.attach_team_run_status_meta(team_name, &mut result, "running")
            .await;
        result
    }

    async fn run_agent_team_live(
        &self,
        tool_call_id: &str,
        team_name: &str,
        initial_turns: Vec<TeamTurn>,
        mode: AgentMode,
        parent_event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> ToolRunResult {
        let futures = initial_turns.into_iter().map(|initial_turn| {
            let agent_name = initial_turn.agent_name.clone();
            let tool = self.clone();
            let team_name = team_name.to_string();
            let tool_call_id = tool_call_id.to_string();
            let parent_event_tx = parent_event_tx.clone();
            async move {
                let mut report = LiveAgentReport::default();
                let mut next_turn = Some(initial_turn);

                loop {
                    let turn = match next_turn.take() {
                        Some(turn) => turn,
                        None => match tool.wait_for_next_live_turn(&team_name, &agent_name).await {
                            Some(turn) => turn,
                            None => break,
                        },
                    };

                    let child_tool_call_id = format!(
                        "{tool_call_id}-live-{}-{}",
                        agent_key(&agent_name),
                        Uuid::new_v4()
                    );
                    let result = tool
                        .run_agent_turn(
                            &child_tool_call_id,
                            &team_name,
                            &turn.agent_name,
                            turn.message.clone(),
                            mode,
                            parent_event_tx.clone(),
                        )
                        .await;
                    let line = first_line(&result.content)
                        .unwrap_or(if result.is_error {
                            "agent turn failed"
                        } else {
                            "agent turn finished"
                        })
                        .to_string();
                    if result.is_error {
                        if let Some(task_id) = turn.task_id {
                            tool.block_task_after_agent_error(
                                &team_name,
                                task_id,
                                &turn.agent_name,
                                &line,
                            )
                            .await;
                        }
                    }
                    report.file_changes.extend(result.file_changes);
                    report.images.extend(result.images);
                    report.last_meta = result.meta;
                    report
                        .reports
                        .push(format!("@{}: {} ({})", turn.agent_name, line, turn.label));
                    next_turn = tool.next_live_turn_for_agent(&team_name, &agent_name).await;
                }

                report
            }
        });

        let outputs = join_all(futures).await;
        let mut reports = Vec::new();
        let mut file_changes = Vec::new();
        let mut images = Vec::new();
        let mut last_meta = None;

        for mut output in outputs {
            reports.append(&mut output.reports);
            file_changes.append(&mut output.file_changes);
            images.append(&mut output.images);
            if output.last_meta.is_some() {
                last_meta = output.last_meta;
            }
        }

        let final_responses = self.team_agent_final_responses(team_name).await;
        let final_responses_text = render_team_agent_final_responses(&final_responses);

        let content = if reports.is_empty() {
            if final_responses_text.is_empty() {
                "Agent Swarm had no runnable teammates".to_string()
            } else {
                format!(
                    "Agent Swarm finished.\n\nFinal teammate responses:\n{final_responses_text}"
                )
            }
        } else {
            let mut content = format!(
                "Agent Swarm finished after {} teammate turn(s):\n{}",
                reports.len(),
                reports.join("\n")
            );
            if !final_responses_text.is_empty() {
                content.push_str("\n\nFinal teammate responses:\n");
                content.push_str(&final_responses_text);
            }
            content
        };
        let mut result = ToolRunResult::ok(content, file_changes);
        result.images = images;
        let mut meta = serde_json::Map::new();
        if let Some(last_meta) = last_meta {
            meta.insert("lastAgentTurnMeta".into(), last_meta);
        }
        if !final_responses.is_empty() {
            meta.insert("agentFinalResponses".into(), json!(final_responses));
        }
        result.meta = (!meta.is_empty()).then_some(Value::Object(meta));
        result
    }

    async fn wait_for_next_live_turn(&self, team_name: &str, agent_name: &str) -> Option<TeamTurn> {
        let notify = self.agent_notify(team_name, agent_name).await;
        loop {
            if let Some(turn) = self.next_live_turn_for_agent(team_name, agent_name).await {
                return Some(turn);
            }
            if self.team_is_settled_for_agent(team_name, agent_name).await {
                tokio::select! {
                    _ = notify.notified() => continue,
                    _ = tokio::time::sleep(Duration::from_millis(TEAM_SETTLE_GRACE_MS)) => {}
                }
                if let Some(turn) = self.next_live_turn_for_agent(team_name, agent_name).await {
                    return Some(turn);
                }
                if self.team_is_settled_for_agent(team_name, agent_name).await {
                    return None;
                }
                continue;
            }
            notify.notified().await;
        }
    }

    async fn next_live_turn_for_agent(
        &self,
        team_name: &str,
        agent_name: &str,
    ) -> Option<TeamTurn> {
        let mut runtime = self.runtime.write().await;
        let session = runtime
            .scopes
            .get_mut(&self.scope_id)
            .and_then(|scope| scope.teams.get_mut(team_name))?;

        let agent_key_value = agent_key(agent_name);
        let agent = session.agents.get(&agent_key_value)?;
        if matches!(
            agent.status,
            TeamAgentStatus::Stopped | TeamAgentStatus::Error
        ) {
            return None;
        }

        let has_wake_message = session.queued_messages.iter().any(|message| {
            agent_key(&message.to) == agent_key_value && queued_message_wakes_agent(message)
        });
        if has_wake_message {
            let mut messages = Vec::new();
            let mut index = 0usize;
            while index < session.queued_messages.len() {
                if agent_key(&session.queued_messages[index].to) == agent_key_value {
                    messages.push(session.queued_messages.remove(index));
                } else {
                    index += 1;
                }
            }
            let now = now_ms();
            if let Some(agent) = session.agents.get_mut(&agent_key_value) {
                agent.status = TeamAgentStatus::Running;
                agent.updated_at_ms = now;
            }
            session.updated_at_ms = now;
            return Some(TeamTurn {
                agent_name: agent_name.to_string(),
                message: queued_messages_prompt(&messages),
                task_id: None,
                label: format!("{} queued message(s)", messages.len()),
            });
        }

        let done_ids = completed_task_ids(session);
        refresh_unblocked_tasks(session, &done_ids);
        prune_stale_task_wakes(session, &done_ids);

        if let Some(task_id) = in_progress_task_id_for_agent(session, &agent_key_value) {
            let task = session
                .tasks
                .iter()
                .find(|task| task.id == task_id)
                .cloned()?;
            let now = now_ms();
            if let Some(agent) = session.agents.get_mut(&agent_key_value) {
                agent.status = TeamAgentStatus::Running;
                agent.updated_at_ms = now;
            }
            session.updated_at_ms = now;
            return Some(TeamTurn {
                agent_name: agent_name.to_string(),
                message: team_continue_task_message(&task),
                task_id: Some(task_id),
                label: format!("continue task #{}", task_id),
            });
        }

        let wake_task_ids = task_wake_ids_for_agent(session, &agent_key_value);
        let task_id =
            wake_task_ids.iter().next().copied().or_else(|| {
                ready_pending_task_id_for_agent(session, &agent_key_value, &done_ids)
            })?;
        let task = session
            .tasks
            .iter()
            .find(|task| task.id == task_id)
            .cloned()?;
        remove_task_wakes_for_task(session, task_id);
        let now = now_ms();
        if let Some(agent) = session.agents.get_mut(&agent_key_value) {
            agent.status = TeamAgentStatus::Running;
            agent.updated_at_ms = now;
        }
        session.updated_at_ms = now;

        Some(TeamTurn {
            agent_name: agent_name.to_string(),
            message: team_ready_task_message(&task),
            task_id: Some(task_id),
            label: format!("ready task #{}", task_id),
        })
    }

    async fn team_is_settled_for_agent(&self, team_name: &str, agent_name: &str) -> bool {
        let mut runtime = self.runtime.write().await;
        let Some(session) = runtime
            .scopes
            .get_mut(&self.scope_id)
            .and_then(|scope| scope.teams.get_mut(team_name))
        else {
            return true;
        };

        let done_ids = completed_task_ids(session);
        refresh_unblocked_tasks(session, &done_ids);
        prune_stale_task_wakes(session, &done_ids);

        let agent_key_value = agent_key(agent_name);
        let Some(agent) = session.agents.get(&agent_key_value) else {
            return true;
        };
        if matches!(
            agent.status,
            TeamAgentStatus::Stopped | TeamAgentStatus::Error
        ) {
            return true;
        }

        if session
            .agents
            .values()
            .any(|agent| agent.status == TeamAgentStatus::Running)
        {
            return false;
        }
        if session.queued_messages.iter().any(|message| {
            agent_key(&message.to) == agent_key_value && queued_message_wakes_agent(message)
        }) {
            return false;
        }
        if agent_has_runnable_task(session, &agent_key_value, &done_ids) {
            return false;
        }
        task_wake_ids_for_agent(session, &agent_key_value).is_empty()
    }

    async fn run_send_message(
        &self,
        _tool_call_id: &str,
        input: Value,
        _mode: AgentMode,
        _parent_event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> ToolRunResult {
        let parsed: SendMessageInput = match serde_json::from_value(input) {
            Ok(value) => value,
            Err(err) => {
                return ToolRunResult::err(format!("invalid SendMessage input: {err}"), Vec::new())
            }
        };
        let to = parsed.to.trim();
        let message = parsed.message.trim();
        if to.is_empty() {
            return ToolRunResult::err("to is required", Vec::new());
        }
        if message.is_empty() {
            return ToolRunResult::err("message is required", Vec::new());
        }
        let team_name = match self.resolve_team_name(parsed.team_name.as_deref()).await {
            Ok(value) => value,
            Err(err) => return ToolRunResult::err(err, Vec::new()),
        };
        let from = self
            .current_agent
            .as_ref()
            .filter(|identity| identity.team_name == team_name)
            .map(|identity| identity.agent_name.as_str())
            .unwrap_or("user");
        if self.current_agent.is_none() {
            return ToolRunResult::err(
                "SendMessage is peer-only. Start a team with TeamRun.",
                Vec::new(),
            );
        }
        if agent_key(to) == "team-lead" {
            return ToolRunResult::ok(
                "message not delivered: an Agent Swarm has no lead; use the shared task board or message teammates directly.",
                Vec::new(),
            );
        }

        let queued = self.queue_team_message(&team_name, from, to, message).await;
        match queued {
            Ok(messages) => {
                self.emit_peer_messages(&team_name, &messages, _parent_event_tx)
                    .await;
                let count = messages.len();
                ToolRunResult::ok(format!("queued {count} peer message(s)"), Vec::new())
            }
            Err(err) => ToolRunResult::err(err, Vec::new()),
        }
    }

    async fn run_task_create(
        &self,
        input: Value,
        _mode: AgentMode,
        _parent_event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> ToolRunResult {
        if self.current_agent.is_none() {
            return ToolRunResult::err(
                "TaskCreate is only available to team teammates. Use TeamRun to start the team.",
                Vec::new(),
            );
        }
        if input.get("blocker").is_some() {
            return ToolRunResult::err("blocker was removed; use blockedBy instead", Vec::new());
        }
        let parsed: TaskCreateInput = match serde_json::from_value(input) {
            Ok(value) => value,
            Err(err) => {
                return ToolRunResult::err(format!("invalid TaskCreate input: {err}"), Vec::new())
            }
        };
        let subject = parsed.subject.trim();
        if subject.is_empty() {
            return ToolRunResult::err("subject is required", Vec::new());
        }
        let description = parsed
            .description
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        let owner = normalized_owner(parsed.owner.as_deref());
        let blocked_by = match normalize_task_ids(merge_task_id_inputs(
            parsed.blocked_by,
            parsed.blocked_by_snake,
        )) {
            Ok(value) => value,
            Err(err) => return ToolRunResult::err(err, Vec::new()),
        };
        let team_name = match self.resolve_team_name(parsed.team_name.as_deref()).await {
            Ok(value) => value,
            Err(err) => return ToolRunResult::err(err, Vec::new()),
        };
        let actor = self.current_actor_name(&team_name);
        let task = {
            let mut runtime = self.runtime.write().await;
            let Some(scope) = runtime.scopes.get_mut(&self.scope_id) else {
                return ToolRunResult::err(
                    "no active team found; start one with TeamRun first",
                    Vec::new(),
                );
            };
            scope.active_team = Some(team_name.clone());
            let Some(session) = scope.teams.get_mut(&team_name) else {
                return ToolRunResult::err(format!("team `{team_name}` not found"), Vec::new());
            };
            if let Err(err) = validate_task_dependencies(session, None, &blocked_by) {
                return ToolRunResult::err(err, Vec::new());
            }
            let now = now_ms();
            let status = if blocked_by.is_empty() {
                TeamTaskStatus::Pending
            } else {
                TeamTaskStatus::Blocked
            };
            let task = TeamTask {
                id: session.next_task_id,
                subject: subject.to_string(),
                description,
                status,
                owner,
                blocked_by,
                created_by: actor,
                created_at_ms: now,
                updated_at_ms: now,
                completed_at_ms: None,
            };
            session.next_task_id += 1;
            session.updated_at_ms = now;
            session.tasks.push(task.clone());
            let done_ids = completed_task_ids(session);
            refresh_unblocked_tasks(session, &done_ids);
            if task.status == TeamTaskStatus::Pending
                && queue_task_wake_for_ready_task(session, task.id, &done_ids, now)
            {
                session.updated_at_ms = now;
            }
            task
        };

        self.notify_all_team_agents(&team_name).await;

        ToolRunResult::ok(
            format!("Task #{} created successfully: {}", task.id, task.subject),
            Vec::new(),
        )
    }

    async fn run_task_list(
        &self,
        input: Value,
        mode: AgentMode,
        parent_event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> ToolRunResult {
        if input.get("blocker").is_some() {
            return ToolRunResult::err("blocker was removed; use blockedBy instead", Vec::new());
        }
        let parsed: TaskListInput = match serde_json::from_value(input) {
            Ok(value) => value,
            Err(err) => {
                return ToolRunResult::err(format!("invalid TaskList input: {err}"), Vec::new())
            }
        };
        let Some(action) = parsed.action else {
            return ToolRunResult::err(
                "TaskList action is required: create, update, delete, or claim",
                Vec::new(),
            );
        };

        if self.current_agent.is_none() {
            return ToolRunResult::err(
                "TaskList mutations are only available to team teammates. Use TeamRun to start the team.",
                Vec::new(),
            );
        }

        match action {
            TaskListAction::List => self.run_task_list_snapshot(parsed).await,
            TaskListAction::Create => {
                let team_name = match self.resolve_team_name(parsed.team_name.as_deref()).await {
                    Ok(value) => value,
                    Err(err) => return ToolRunResult::err(err, Vec::new()),
                };
                let mut payload = serde_json::Map::new();
                payload.insert("team_name".into(), json!(team_name));
                if let Some(subject) = parsed.subject {
                    payload.insert("subject".into(), json!(subject));
                }
                if let Some(description) = parsed.description {
                    payload.insert("description".into(), json!(description));
                }
                if let Some(owner) = parsed.owner {
                    payload.insert("owner".into(), json!(owner));
                }
                if let Some(blocked_by) = parsed.blocked_by {
                    payload.insert(
                        "blockedBy".into(),
                        serde_json::to_value(blocked_by).unwrap_or(Value::Null),
                    );
                }
                if let Some(blocked_by) = parsed.blocked_by_snake {
                    payload.insert(
                        "blocked_by".into(),
                        serde_json::to_value(blocked_by).unwrap_or(Value::Null),
                    );
                }
                let mut result = self
                    .run_task_create(Value::Object(payload), mode, parent_event_tx)
                    .await;
                self.attach_team_snapshot_meta(&team_name, &mut result)
                    .await;
                result
            }
            TaskListAction::Update => {
                let mut payload = serde_json::Map::new();
                if let Some(team_name) = parsed.team_name {
                    payload.insert("team_name".into(), json!(team_name));
                }
                if let Some(task_id) = parsed.task_id {
                    payload.insert(
                        "taskId".into(),
                        serde_json::to_value(task_id).unwrap_or(Value::Null),
                    );
                }
                if let Some(status) = parsed.status {
                    payload.insert("status".into(), json!(status));
                }
                if let Some(owner) = parsed.owner {
                    payload.insert("owner".into(), json!(owner));
                }
                if let Some(clear_owner) = parsed.clear_owner {
                    payload.insert("clear_owner".into(), json!(clear_owner));
                }
                if let Some(subject) = parsed.subject {
                    payload.insert("subject".into(), json!(subject));
                }
                if let Some(description) = parsed.description {
                    payload.insert("description".into(), json!(description));
                }
                if let Some(blocked_by) = parsed.blocked_by {
                    payload.insert(
                        "blockedBy".into(),
                        serde_json::to_value(blocked_by).unwrap_or(Value::Null),
                    );
                }
                if let Some(blocked_by) = parsed.blocked_by_snake {
                    payload.insert(
                        "blocked_by".into(),
                        serde_json::to_value(blocked_by).unwrap_or(Value::Null),
                    );
                }
                if let Some(blocked_by) = parsed.add_blocked_by {
                    payload.insert(
                        "addBlockedBy".into(),
                        serde_json::to_value(blocked_by).unwrap_or(Value::Null),
                    );
                }
                if let Some(blocked_by) = parsed.add_blocked_by_snake {
                    payload.insert(
                        "add_blocked_by".into(),
                        serde_json::to_value(blocked_by).unwrap_or(Value::Null),
                    );
                }
                self.run_task_update(Value::Object(payload), mode, parent_event_tx)
                    .await
            }
            TaskListAction::Delete => self.run_task_delete(parsed).await,
            TaskListAction::Claim => self.run_task_claim(parsed).await,
        }
    }

    async fn run_task_list_snapshot(&self, parsed: TaskListInput) -> ToolRunResult {
        let team_name = match self.resolve_team_name(parsed.team_name.as_deref()).await {
            Ok(value) => value,
            Err(err) => return ToolRunResult::err(err, Vec::new()),
        };
        let (content, snapshot) = {
            let mut runtime = self.runtime.write().await;
            let Some(scope) = runtime.scopes.get_mut(&self.scope_id) else {
                return ToolRunResult::err(
                    "no active team found; start one with TeamRun first",
                    Vec::new(),
                );
            };
            scope.active_team = Some(team_name.clone());
            let Some(session) = scope.teams.get_mut(&team_name) else {
                return ToolRunResult::err(format!("team `{team_name}` not found"), Vec::new());
            };
            let done_ids = completed_task_ids(session);
            refresh_unblocked_tasks(session, &done_ids);
            prune_stale_task_wakes(session, &done_ids);
            let snapshot = TeamSnapshot::from_session(session);
            (
                format!("Task board:\n{}", render_team_snapshot(&snapshot)),
                snapshot,
            )
        };
        ToolRunResult::ok_with_meta(content, Vec::new(), json!({ "team": snapshot }))
    }

    async fn run_task_update(
        &self,
        input: Value,
        _mode: AgentMode,
        _parent_event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> ToolRunResult {
        if self.current_agent.is_none() {
            return ToolRunResult::err(
                "TaskUpdate is only available to team teammates. Use TeamRun to start the team.",
                Vec::new(),
            );
        }
        if input.get("blocker").is_some() {
            return ToolRunResult::err("blocker was removed; use blockedBy instead", Vec::new());
        }
        let parsed: TaskUpdateInput = match serde_json::from_value(input) {
            Ok(value) => value,
            Err(err) => {
                return ToolRunResult::err(format!("invalid TaskUpdate input: {err}"), Vec::new())
            }
        };
        let task_id = match parsed.task_id.to_u64() {
            Ok(value) => value,
            Err(err) => return ToolRunResult::err(err, Vec::new()),
        };
        let owner = normalized_owner(parsed.owner.as_deref());
        let subject = match parsed.subject.as_deref().map(str::trim) {
            Some("") => return ToolRunResult::err("subject cannot be empty", Vec::new()),
            Some(value) => Some(value.to_string()),
            None => None,
        };
        let description = parsed.description.as_ref().map(|value| {
            let value = value.trim();
            if value.is_empty() {
                None
            } else {
                Some(value.to_string())
            }
        });
        let replace_blocked_by = match normalize_optional_task_ids(merge_task_id_inputs(
            parsed.blocked_by,
            parsed.blocked_by_snake,
        )) {
            Ok(value) => value,
            Err(err) => return ToolRunResult::err(err, Vec::new()),
        };
        let add_blocked_by = match normalize_optional_task_ids(merge_task_id_inputs(
            parsed.add_blocked_by,
            parsed.add_blocked_by_snake,
        )) {
            Ok(value) => value,
            Err(err) => return ToolRunResult::err(err, Vec::new()),
        };
        let blocked_by_was_requested = replace_blocked_by.is_some() || add_blocked_by.is_some();
        let team_name = match self.resolve_team_name(parsed.team_name.as_deref()).await {
            Ok(value) => value,
            Err(err) => return ToolRunResult::err(err, Vec::new()),
        };
        let actor = self.current_actor_name(&team_name);
        let (content, snapshot) = {
            let mut runtime = self.runtime.write().await;
            let Some(scope) = runtime.scopes.get_mut(&self.scope_id) else {
                return ToolRunResult::err(
                    "no active team found; start one with TeamRun first",
                    Vec::new(),
                );
            };
            scope.active_team = Some(team_name.clone());
            let Some(session) = scope.teams.get_mut(&team_name) else {
                return ToolRunResult::err(format!("team `{team_name}` not found"), Vec::new());
            };
            let done_ids = completed_task_ids(session);
            refresh_unblocked_tasks(session, &done_ids);
            let Some(task_index) = session.tasks.iter().position(|task| task.id == task_id) else {
                return ToolRunResult::err(format!("task #{task_id} not found"), Vec::new());
            };

            let mut next_blocked_by = replace_blocked_by
                .clone()
                .unwrap_or_else(|| session.tasks[task_index].blocked_by.clone());
            if let Some(additional) = add_blocked_by {
                next_blocked_by.extend(additional);
                next_blocked_by = normalize_task_id_values(next_blocked_by);
            }
            if let Err(err) = validate_task_dependencies(session, Some(task_id), &next_blocked_by) {
                return ToolRunResult::err(err, Vec::new());
            }
            let done_ids = completed_task_ids(session);
            if let Err(err) = validate_task_dependency_lock(
                task_id,
                &session.tasks[task_index].blocked_by,
                &next_blocked_by,
                parsed.status,
                &done_ids,
            ) {
                return ToolRunResult::err(err, Vec::new());
            }
            if parsed.status == Some(TeamTaskStatus::Blocked) && next_blocked_by.is_empty() {
                return ToolRunResult::err(
                    "blocked tasks require blockedBy task IDs".to_string(),
                    Vec::new(),
                );
            }

            let now = now_ms();
            let task = &mut session.tasks[task_index];
            let mut updated_fields = Vec::new();

            if let Some(subject) = subject {
                if task.subject != subject {
                    task.subject = subject;
                    updated_fields.push("subject");
                }
            }
            if let Some(description) = description {
                if task.description != description {
                    task.description = description;
                    updated_fields.push("description");
                }
            }
            if parsed.clear_owner.unwrap_or(false) && task.owner.is_some() {
                task.owner = None;
                updated_fields.push("owner");
            }
            if let Some(owner) = owner {
                if task.owner.as_deref() != Some(owner.as_str()) {
                    task.owner = Some(owner);
                    updated_fields.push("owner");
                }
            }
            if task.blocked_by != next_blocked_by {
                task.blocked_by = next_blocked_by;
                updated_fields.push("blockedBy");
            }
            if parsed.status.is_none() && blocked_by_was_requested {
                if !task.blocked_by.is_empty() && task.status != TeamTaskStatus::Completed {
                    if task.status != TeamTaskStatus::Blocked {
                        task.status = TeamTaskStatus::Blocked;
                        task.completed_at_ms = None;
                        updated_fields.push("status");
                    }
                } else if task.blocked_by.is_empty() && task.status == TeamTaskStatus::Blocked {
                    task.status = TeamTaskStatus::Pending;
                    task.completed_at_ms = None;
                    updated_fields.push("status");
                }
            }
            if let Some(status) = parsed.status {
                if task.status != status {
                    task.status = status;
                    updated_fields.push("status");
                }
                match status {
                    TeamTaskStatus::Completed => {
                        task.completed_at_ms = Some(now);
                    }
                    TeamTaskStatus::InProgress => {
                        task.completed_at_ms = None;
                        if task.owner.is_none() && self.current_agent.is_some() {
                            task.owner = Some(actor.clone());
                            updated_fields.push("owner");
                        }
                    }
                    TeamTaskStatus::Blocked => {
                        task.completed_at_ms = None;
                    }
                    TeamTaskStatus::Pending => {
                        task.completed_at_ms = None;
                    }
                }
            }

            task.updated_at_ms = now;
            let updated_task_id = task.id;
            session.updated_at_ms = now;
            refresh_unblocked_tasks(session, &done_ids);
            let task_snapshot = session
                .tasks
                .iter()
                .find(|task| task.id == updated_task_id)
                .cloned()
                .expect("updated task should still exist");
            if task_snapshot.status == TeamTaskStatus::Pending
                && queue_task_wake_for_ready_task(session, task_snapshot.id, &done_ids, now)
            {
                session.updated_at_ms = now;
            }
            let snapshot = TeamSnapshot::from_session(session);
            let updated = if updated_fields.is_empty() {
                "no fields changed".to_string()
            } else {
                format!("updated {}", updated_fields.join(", "))
            };
            (
                format!(
                    "Task #{} {}:\n{}",
                    task_snapshot.id,
                    updated,
                    render_task_line(&task_snapshot)
                ),
                snapshot,
            )
        };

        self.notify_all_team_agents(&team_name).await;

        ToolRunResult::ok_with_meta(content, Vec::new(), json!({ "team": snapshot }))
    }

    async fn run_task_delete(&self, parsed: TaskListInput) -> ToolRunResult {
        let Some(task_id) = parsed.task_id else {
            return ToolRunResult::err("taskId is required for TaskList action=delete", Vec::new());
        };
        let task_id = match task_id.to_u64() {
            Ok(value) => value,
            Err(err) => return ToolRunResult::err(err, Vec::new()),
        };
        let team_name = match self.resolve_team_name(parsed.team_name.as_deref()).await {
            Ok(value) => value,
            Err(err) => return ToolRunResult::err(err, Vec::new()),
        };
        let (subject, snapshot) = {
            let mut runtime = self.runtime.write().await;
            let Some(scope) = runtime.scopes.get_mut(&self.scope_id) else {
                return ToolRunResult::err(
                    "no active team found; start one with TeamRun first",
                    Vec::new(),
                );
            };
            scope.active_team = Some(team_name.clone());
            let Some(session) = scope.teams.get_mut(&team_name) else {
                return ToolRunResult::err(format!("team `{team_name}` not found"), Vec::new());
            };
            let done_ids = completed_task_ids(session);
            refresh_unblocked_tasks(session, &done_ids);
            let Some(task_index) = session.tasks.iter().position(|task| task.id == task_id) else {
                return ToolRunResult::err(format!("task #{task_id} not found"), Vec::new());
            };
            let task = session.tasks.remove(task_index);
            for other in &mut session.tasks {
                other.blocked_by.retain(|id| *id != task_id);
            }
            session.updated_at_ms = now_ms();
            let done_ids = completed_task_ids(session);
            refresh_unblocked_tasks(session, &done_ids);
            (task.subject, TeamSnapshot::from_session(session))
        };
        self.notify_all_team_agents(&team_name).await;
        ToolRunResult::ok_with_meta(
            format!("Task #{task_id} deleted: {subject}"),
            Vec::new(),
            json!({ "team": snapshot }),
        )
    }

    async fn run_task_claim(&self, parsed: TaskListInput) -> ToolRunResult {
        let team_name = match self.resolve_team_name(parsed.team_name.as_deref()).await {
            Ok(value) => value,
            Err(err) => return ToolRunResult::err(err, Vec::new()),
        };
        let owner = normalized_owner(parsed.owner.as_deref())
            .unwrap_or_else(|| self.current_actor_name(&team_name));
        let requested_id = match parsed.task_id.as_ref() {
            Some(task_id) => match task_id.to_u64() {
                Ok(value) => Some(value),
                Err(err) => return ToolRunResult::err(err, Vec::new()),
            },
            None => None,
        };
        let (line, snapshot) = {
            let mut runtime = self.runtime.write().await;
            let Some(scope) = runtime.scopes.get_mut(&self.scope_id) else {
                return ToolRunResult::err(
                    "no active team found; start one with TeamRun first",
                    Vec::new(),
                );
            };
            scope.active_team = Some(team_name.clone());
            let Some(session) = scope.teams.get_mut(&team_name) else {
                return ToolRunResult::err(format!("team `{team_name}` not found"), Vec::new());
            };
            let done_ids = completed_task_ids(session);
            refresh_unblocked_tasks(session, &done_ids);
            prune_stale_task_wakes(session, &done_ids);
            let owner_key = agent_key(&owner);
            let task_index = if let Some(task_id) = requested_id {
                let Some(task_index) = session.tasks.iter().position(|task| task.id == task_id)
                else {
                    return ToolRunResult::err(format!("task #{task_id} not found"), Vec::new());
                };
                task_index
            } else {
                let Some(task_index) = session.tasks.iter().position(|task| {
                    task.status == TeamTaskStatus::Pending
                        && task_dependencies_satisfied(task, &done_ids)
                        && task
                            .owner
                            .as_deref()
                            .map(agent_key)
                            .map(|current_owner| current_owner == owner_key)
                            .unwrap_or(true)
                }) else {
                    return ToolRunResult::ok_with_meta(
                        "No unblocked pending task available to claim",
                        Vec::new(),
                        json!({ "team": TeamSnapshot::from_session(session) }),
                    );
                };
                task_index
            };
            let task = &mut session.tasks[task_index];
            if task.status == TeamTaskStatus::Completed {
                return ToolRunResult::err(
                    format!("task #{} is already completed", task.id),
                    Vec::new(),
                );
            }
            let now = now_ms();
            task.owner = Some(owner);
            task.updated_at_ms = now;
            session.updated_at_ms = now;
            let claimed_task_id = task.id;
            let line = render_task_line(task);
            remove_task_wakes_for_task(session, claimed_task_id);
            let snapshot = TeamSnapshot::from_session(session);
            (line, snapshot)
        };
        self.notify_all_team_agents(&team_name).await;
        ToolRunResult::ok_with_meta(
            format!("Task claimed:\n{line}"),
            Vec::new(),
            json!({ "team": snapshot }),
        )
    }

    async fn run_status(&self, input: Value) -> ToolRunResult {
        if self.current_agent.is_some() {
            return ToolRunResult::err(
                "TeamStatus is only available to the main agent.",
                Vec::new(),
            );
        }
        let input = normalize_optional_object_input(input);
        let parsed: TeamNameInput = match serde_json::from_value(input) {
            Ok(value) => value,
            Err(err) => {
                return ToolRunResult::err(format!("invalid TeamStatus input: {err}"), Vec::new())
            }
        };
        let team_name = match self.resolve_team_name(parsed.team_name.as_deref()).await {
            Ok(value) => value,
            Err(_) => return ToolRunResult::ok("no active Agent Swarm", Vec::new()),
        };
        let runtime = self.runtime.read().await;
        let Some(session) = runtime
            .scopes
            .get(&self.scope_id)
            .and_then(|scope| scope.teams.get(&team_name))
        else {
            return ToolRunResult::ok(format!("no Agent Swarm named `{team_name}`"), Vec::new());
        };
        let snapshot = TeamSnapshot::from_session(session);
        ToolRunResult::ok_with_meta(
            render_team_snapshot(&snapshot),
            Vec::new(),
            json!({ "team": snapshot }),
        )
    }

    async fn run_stop(
        &self,
        input: Value,
        parent_event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> ToolRunResult {
        if self.current_agent.is_some() {
            return ToolRunResult::err("TeamStop is only available to the main agent.", Vec::new());
        }
        let input = normalize_optional_object_input(input);
        let parsed: TeamStopInput = match serde_json::from_value(input) {
            Ok(value) => value,
            Err(err) => {
                return ToolRunResult::err(format!("invalid TeamStop input: {err}"), Vec::new())
            }
        };
        let team_name = match self.resolve_team_name(parsed.team_name.as_deref()).await {
            Ok(value) => value,
            Err(_) => return ToolRunResult::ok("no active Agent Swarm to stop", Vec::new()),
        };
        let mut runtime = self.runtime.write().await;
        let Some(session) = runtime
            .scopes
            .get_mut(&self.scope_id)
            .and_then(|scope| scope.teams.get_mut(&team_name))
        else {
            return ToolRunResult::ok(
                format!("no Agent Swarm named `{team_name}` to stop"),
                Vec::new(),
            );
        };
        let agent = parsed
            .agent
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        if let Some(agent_name) = agent {
            let agent_key_value = agent_key(agent_name);
            let Some(agent) = session.agents.get_mut(&agent_key_value) else {
                return ToolRunResult::err(
                    format!("teammate `{agent_name}` not found"),
                    Vec::new(),
                );
            };
            agent.status = TeamAgentStatus::Stopped;
            agent.updated_at_ms = now_ms();
            let stopped_agent_name = agent.name.clone();
            let now = agent.updated_at_ms;
            let mut reset_count = 0usize;
            for task in &mut session.tasks {
                if task.owner.as_deref().map(agent_key).as_deref() != Some(agent_key_value.as_str())
                {
                    continue;
                }
                if matches!(
                    task.status,
                    TeamTaskStatus::InProgress | TeamTaskStatus::Blocked
                ) {
                    task.status = TeamTaskStatus::Pending;
                    task.owner = None;
                    task.completed_at_ms = None;
                    task.updated_at_ms = now;
                    reset_count += 1;
                }
            }
            session.updated_at_ms = now;
            drop(runtime);
            self.notify_all_team_agents(&team_name).await;
            if let Ok(messages) = self
                .queue_team_message(
                    &team_name,
                    "system",
                    "*",
                    &format!(
                        "@{} left the team. Their open task(s) are back in pending.",
                        stopped_agent_name
                    ),
                )
                .await
            {
                self.emit_peer_messages(&team_name, &messages, parent_event_tx)
                    .await;
            }
            let mut result = ToolRunResult::ok(
                format!(
                    "stopped teammate: {} ({} open task(s) reset to pending)",
                    stopped_agent_name, reset_count
                ),
                Vec::new(),
            );
            self.attach_team_snapshot_meta(&team_name, &mut result)
                .await;
            return result;
        }

        for agent in session.agents.values_mut() {
            agent.status = TeamAgentStatus::Stopped;
            agent.updated_at_ms = now_ms();
        }
        session.updated_at_ms = now_ms();
        let stopped_count = session.agents.len();
        let agent_names = session
            .agents
            .values()
            .map(|agent| agent.name.clone())
            .collect::<Vec<_>>();
        let snapshot = TeamSnapshot::from_session(session);
        let notifiers = agent_names
            .iter()
            .filter_map(|agent_name| {
                runtime
                    .agent_notifiers
                    .get(&agent_notify_key(&self.scope_id, &team_name, agent_name))
                    .cloned()
            })
            .collect::<Vec<_>>();
        let cancels = runtime
            .scopes
            .get_mut(&self.scope_id)
            .map(|scope| {
                if scope.active_team.as_deref() == Some(team_name.as_str()) {
                    scope.active_team = None;
                }
                scope.teams.remove(&team_name);
                scope.team_cancels.remove(&team_name).unwrap_or_default()
            })
            .unwrap_or_default();
        let notify_prefix = team_notify_key_prefix(&self.scope_id, &team_name);
        runtime
            .agent_notifiers
            .retain(|key, _| !key.starts_with(&notify_prefix));
        drop(runtime);
        for notifier in notifiers {
            wake_notifier(&notifier);
        }
        for cancel in cancels {
            cancel.cancel_all();
        }
        ToolRunResult::ok_with_meta(
            format!("stopped Agent Swarm ({} teammate(s))", stopped_count),
            Vec::new(),
            json!({ "team": snapshot }),
        )
    }

    async fn create_or_reset_team(&self, team_name: &str, description: Option<String>) {
        let now = now_ms();
        let mut runtime = self.runtime.write().await;
        let notify_prefix = team_notify_key_prefix(&self.scope_id, team_name);
        runtime
            .agent_notifiers
            .retain(|key, _| !key.starts_with(&notify_prefix));
        let scope = runtime.scopes.entry(self.scope_id.clone()).or_default();
        scope.team_cancels.remove(team_name);
        scope.teams.insert(
            team_name.to_string(),
            TeamSession {
                name: team_name.to_string(),
                description,
                created_at_ms: now,
                updated_at_ms: now,
                agents: HashMap::new(),
                tasks: Vec::new(),
                next_task_id: 1,
                queued_messages: Vec::new(),
                next_message_id: 1,
                pending_task_wakes: Vec::new(),
                recent_file_changes: Vec::new(),
            },
        );
        scope.active_team = Some(team_name.to_string());
    }

    async fn seed_team_run_tasks(
        &self,
        team_name: &str,
        tasks: Vec<PreparedTeamRunTask>,
    ) -> std::result::Result<(), String> {
        if tasks.is_empty() {
            return Ok(());
        }
        let mut runtime = self.runtime.write().await;
        let scope = runtime
            .scopes
            .get_mut(&self.scope_id)
            .ok_or_else(|| "no active team found; start one with TeamRun first".to_string())?;
        let session = scope
            .teams
            .get_mut(team_name)
            .ok_or_else(|| format!("team `{team_name}` not found"))?;
        let now = now_ms();
        for task in tasks {
            let id = session.next_task_id;
            session.next_task_id += 1;
            let status = if task.blocked_by.is_empty() {
                TeamTaskStatus::Pending
            } else {
                TeamTaskStatus::Blocked
            };
            session.tasks.push(TeamTask {
                id,
                subject: task.subject,
                description: task.description,
                status,
                owner: task.owner,
                blocked_by: task.blocked_by,
                created_by: "main-agent".to_string(),
                created_at_ms: now,
                updated_at_ms: now,
                completed_at_ms: None,
            });
        }
        session.updated_at_ms = now;
        Ok(())
    }

    async fn team_snapshot(&self, team_name: &str) -> Option<TeamSnapshot> {
        let runtime = self.runtime.read().await;
        runtime
            .scopes
            .get(&self.scope_id)
            .and_then(|scope| scope.teams.get(team_name))
            .map(TeamSnapshot::from_session)
    }

    async fn attach_team_snapshot_meta(&self, team_name: &str, result: &mut ToolRunResult) {
        if result.is_error {
            return;
        }
        let Some(snapshot) = self.team_snapshot(team_name).await else {
            return;
        };
        let mut meta = match result.meta.take() {
            Some(Value::Object(map)) => map,
            Some(value) => {
                let mut map = serde_json::Map::new();
                map.insert("previousMeta".into(), value);
                map
            }
            None => serde_json::Map::new(),
        };
        meta.insert("team".into(), json!(snapshot));
        result.meta = Some(Value::Object(meta));
    }

    async fn attach_team_run_status_meta(
        &self,
        team_name: &str,
        result: &mut ToolRunResult,
        fallback_status: &str,
    ) {
        let snapshot = self.team_snapshot(team_name).await;
        let subagents = self.team_subagents_meta(team_name).await;
        let status = team_run_status_label(snapshot.as_ref(), result.is_error, fallback_status);
        let mut meta = match result.meta.take() {
            Some(Value::Object(map)) => map,
            Some(value) => {
                let mut map = serde_json::Map::new();
                map.insert("previousMeta".into(), value);
                map
            }
            None => serde_json::Map::new(),
        };
        if let Some(snapshot) = snapshot {
            meta.insert("team".into(), json!(snapshot));
        }
        meta.insert("subagents".into(), json!(subagents));
        meta.insert("teamRunStatus".into(), json!(status));
        result.meta = Some(Value::Object(meta));
    }

    async fn team_subagents_meta(&self, team_name: &str) -> Vec<Value> {
        let runtime = self.runtime.read().await;
        let mut agents = runtime
            .scopes
            .get(&self.scope_id)
            .and_then(|scope| scope.teams.get(team_name))
            .map(|session| {
                session
                    .agents
                    .values()
                    .map(|agent| {
                        let queued_messages = session
                            .queued_messages
                            .iter()
                            .filter(|message| agent_key(&message.to) == agent_key(&agent.name))
                            .map(|message| {
                                json!({
                                    "id": message.id.to_string(),
                                    "from": message.from.clone(),
                                    "to": message.target.clone().unwrap_or_else(|| message.to.clone()),
                                    "message": message.message.clone(),
                                })
                            })
                            .collect::<Vec<_>>();
                        json!({
                            "id": agent.id.clone(),
                            "name": agent.name.clone(),
                            "model": agent.model.clone(),
                            "history": agent.history.clone(),
                            "status": agent.status,
                            "queuedMessages": queued_messages,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        agents.sort_by(|left, right| {
            left.get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .cmp(
                    right
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                )
        });
        agents
    }

    async fn team_agent_final_responses(&self, team_name: &str) -> Vec<TeamAgentFinalResponse> {
        let runtime = self.runtime.read().await;
        runtime
            .scopes
            .get(&self.scope_id)
            .and_then(|scope| scope.teams.get(team_name))
            .map(team_agent_final_responses_from_session)
            .unwrap_or_default()
    }

    async fn block_task_after_agent_error(
        &self,
        team_name: &str,
        task_id: u64,
        _agent_name: &str,
        _error: &str,
    ) {
        let changed = {
            let mut runtime = self.runtime.write().await;
            let Some(session) = runtime
                .scopes
                .get_mut(&self.scope_id)
                .and_then(|scope| scope.teams.get_mut(team_name))
            else {
                return;
            };
            let Some(task) = session.tasks.iter_mut().find(|task| task.id == task_id) else {
                return;
            };
            let now = now_ms();
            task.status = TeamTaskStatus::Pending;
            task.owner = None;
            task.completed_at_ms = None;
            task.updated_at_ms = now;
            session.updated_at_ms = now;
            true
        };
        if changed {
            self.notify_all_team_agents(team_name).await;
        }
    }

    async fn ensure_agent(
        &self,
        team_name: &str,
        agent_name: &str,
        description: String,
        prompt: String,
        model: ModelRef,
    ) {
        let now = now_ms();
        let mut runtime = self.runtime.write().await;
        let scope = runtime.scopes.entry(self.scope_id.clone()).or_default();
        let session = scope
            .teams
            .entry(team_name.to_string())
            .or_insert_with(|| TeamSession {
                name: team_name.to_string(),
                description: None,
                created_at_ms: now,
                updated_at_ms: now,
                agents: HashMap::new(),
                tasks: Vec::new(),
                next_task_id: 1,
                queued_messages: Vec::new(),
                next_message_id: 1,
                pending_task_wakes: Vec::new(),
                recent_file_changes: Vec::new(),
            });
        let key = agent_key(agent_name);
        let agent = session.agents.entry(key).or_insert_with(|| TeamAgent {
            id: format!("{}@{}", agent_key(agent_name), team_name),
            name: agent_name.trim().to_string(),
            description: description.clone(),
            prompt: prompt.clone(),
            model: model.clone(),
            status: TeamAgentStatus::Idle,
            history: Vec::new(),
            created_at_ms: now,
            updated_at_ms: now,
            last_summary: None,
            last_error: None,
        });
        agent.description = description;
        agent.prompt = prompt;
        agent.model = model;
        if agent.status == TeamAgentStatus::Stopped {
            agent.status = TeamAgentStatus::Idle;
        }
        agent.updated_at_ms = now;
        session.updated_at_ms = now;
        scope.active_team = Some(team_name.to_string());
    }

    async fn prepare_agent_restart(
        &self,
        team_name: &str,
        agent_name: &str,
    ) -> std::result::Result<String, String> {
        let mut runtime = self.runtime.write().await;
        let scope = runtime
            .scopes
            .get_mut(&self.scope_id)
            .ok_or_else(|| "no active team found; start one with TeamRun first".to_string())?;
        scope.active_team = Some(team_name.to_string());
        let session = scope
            .teams
            .get_mut(team_name)
            .ok_or_else(|| format!("team `{team_name}` not found"))?;
        let key = agent_key(agent_name);
        let agent_name = {
            let agent = session
                .agents
                .get_mut(&key)
                .ok_or_else(|| format!("teammate `{agent_name}` not found"))?;
            if agent.status == TeamAgentStatus::Running {
                return Err(format!("teammate `{}` is already running", agent.name));
            }
            agent.status = TeamAgentStatus::Idle;
            agent.updated_at_ms = now_ms();
            agent.name.clone()
        };
        let done_ids = completed_task_ids(session);
        refresh_unblocked_tasks(session, &done_ids);
        session.updated_at_ms = now_ms();
        drop(runtime);
        self.notify_all_team_agents(team_name).await;
        Ok(agent_name)
    }

    async fn run_agent_turn(
        &self,
        tool_call_id: &str,
        team_name: &str,
        agent_name: &str,
        message: String,
        mode: AgentMode,
        parent_event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> ToolRunResult {
        let agent = match self.take_agent_for_run(team_name, agent_name).await {
            Ok(value) => value,
            Err(err) => return ToolRunResult::err(err, Vec::new()),
        };

        let Some(provider) = self.providers.get(&agent.model.provider).cloned() else {
            let error = format!(
                "provider `{}` is not configured or missing credentials",
                agent.model.provider
            );
            self.finish_agent_error(team_name, &agent.name, &error)
                .await;
            return ToolRunResult::err(error, Vec::new());
        };
        if provider.capabilities(&agent.model).is_none() {
            let error = format!("model `{}` is not supported", agent.model.name);
            self.finish_agent_error(team_name, &agent.name, &error)
                .await;
            return ToolRunResult::err(error, Vec::new());
        }

        let mut history = agent.history.clone();
        history.push(ChatMessage {
            role: Role::User,
            parts: vec![Part::Text {
                text: message.clone(),
                meta: None,
            }],
        });

        let (child_cmd_tx, child_cmd_rx) = mpsc::unbounded_channel();
        let (child_event_tx, mut child_event_rx) = mpsc::unbounded_channel();
        self.cancel.register(child_cmd_tx);
        let team_tool = Arc::new(self.for_agent(team_name.to_string(), agent.name.clone()));
        let workspace_write_lock = self.workspace_write_lock().await;
        let child_mode = if mode == AgentMode::Goal {
            AgentMode::Act
        } else {
            mode
        };
        let child_context = TurnContext {
            provider,
            model: agent.model.clone(),
            cache_key: Some(format!(
                "team:{}:{}:{}",
                self.scope_id,
                team_name,
                agent_key(&agent.name)
            )),
            cache_stable_message_count: agent.history.len(),
            auto_compact: true,
            mode: child_mode,
            stop_questions: false,
            system_prompt: team_agent_system_prompt(&self.system_prompt, team_name, &agent),
            history,
            todo_list: TodoListState::default(),
            goal_workflow: GoalWorkflowState::Idle,
            bash: Arc::new(BashTool::new(self.workspace_root.clone())),
            glob: Arc::new(GlobTool::new(self.workspace_root.clone())),
            grep: Arc::new(GrepTool::new(self.workspace_root.clone())),
            read: Arc::new(ReadTool::new(self.workspace_root.clone())),
            apply_patch: Arc::new(
                ApplyPatchTool::new(self.workspace_root.clone())
                    .with_workspace_write_lock(workspace_write_lock.clone()),
            ),
            create_image: Arc::new(
                CreateImageTool::with_settings(
                    self.workspace_root.clone(),
                    self.tool_settings.image_provider,
                    self.tool_settings.openai_image_api_key(),
                    self.tool_settings.nano_banana_api_key(),
                )
                .with_workspace_write_lock(workspace_write_lock),
            ),
            todo_list_tool: None,
            question: None,
            web_search: Arc::new(WebSearchTool::with_settings(
                self.tool_settings.web_search_provider,
                self.tool_settings.linkup_api_key(),
            )),
            web_fetch: Arc::new(WebFetchTool::new()),
            skill: Arc::new(SkillTool::with_settings(
                self.workspace_root.clone(),
                self.skill_settings.clone(),
            )),
            mcp: Arc::new(McpToolRegistry::new(self.mcp_settings.clone())),
            subagents: None,
            teams: Some(team_tool),
            tool_settings: self.tool_settings.clone(),
            event_scope: Some(AgentEventScope {
                id: tool_call_id.to_string(),
                agent_id: agent.id.clone(),
                agent_name: agent.name.clone(),
                team_name: Some(team_name.to_string()),
                model: agent.model.clone(),
                initial_message: message,
            }),
            max_tool_rounds: self.max_tool_rounds,
            event_tx: child_event_tx,
            cancel: self.cancel.clone(),
            cmd_rx: child_cmd_rx,
        };

        let engine = tokio::spawn(async move { run_turn(child_context).await });
        let mut child_error: Option<String> = None;
        while let Some(event) = child_event_rx.recv().await {
            if let AgentEvent::SubAgentEvent { event: inner, .. } = &event {
                if let AgentEvent::Error { message } = inner.as_ref() {
                    child_error.get_or_insert_with(|| message.clone());
                }
            }
            let _ = parent_event_tx.send(event);
        }
        let output = match engine.await {
            Ok(output) => output,
            Err(err) => {
                let error = format!("teammate task failed: {err}");
                self.finish_agent_error(team_name, &agent.name, &error)
                    .await;
                return ToolRunResult::err(error, Vec::new());
            }
        };
        let file_changes = file_changes_from_history(&output.history);
        if let Some(error) = child_error {
            let updated_agent = self
                .finish_agent_failure(team_name, &agent.name, output.history, error.clone())
                .await;
            let mut result = ToolRunResult::err(
                render_agent_result(team_name, &updated_agent, &error),
                file_changes,
            );
            result.meta = Some(json!({
                "subagent": {
                    "id": updated_agent.id,
                    "name": updated_agent.name,
                    "model": updated_agent.model,
                    "history": updated_agent.history,
                },
                "team": {
                    "name": team_name,
                    "agent": updated_agent,
                }
            }));
            return result;
        }
        let final_answer = final_assistant_text(&output.history)
            .unwrap_or_else(|| "Teammate finished without a final answer.".to_string());
        let updated_agent = self
            .finish_agent_success(team_name, &agent.name, output.history, final_answer.clone())
            .await;
        if self
            .agent_sleep_allowed(team_name, &updated_agent.name)
            .await
        {
            emit_agent_slept_event(tool_call_id, team_name, &updated_agent, &parent_event_tx);
        }

        ToolRunResult::ok_with_meta(
            render_agent_result(team_name, &updated_agent, &final_answer),
            file_changes,
            json!({
                "subagent": {
                    "id": updated_agent.id,
                    "name": updated_agent.name,
                    "model": updated_agent.model,
                    "history": updated_agent.history,
                },
                "team": {
                    "name": team_name,
                    "agent": updated_agent,
                }
            }),
        )
    }

    async fn take_agent_for_run(
        &self,
        team_name: &str,
        agent_name: &str,
    ) -> std::result::Result<TeamAgent, String> {
        let mut runtime = self.runtime.write().await;
        let scope = runtime
            .scopes
            .get_mut(&self.scope_id)
            .ok_or_else(|| "no active team found; start one with TeamRun first".to_string())?;
        let session = scope
            .teams
            .get_mut(team_name)
            .ok_or_else(|| format!("team `{team_name}` not found"))?;
        let key = agent_key(agent_name);
        let agent = session
            .agents
            .get_mut(&key)
            .ok_or_else(|| format!("teammate `{agent_name}` not found"))?;
        if agent.status == TeamAgentStatus::Running {
            return Ok(agent.clone());
        }
        if agent.status == TeamAgentStatus::Stopped {
            agent.status = TeamAgentStatus::Idle;
        }
        agent.status = TeamAgentStatus::Running;
        agent.updated_at_ms = now_ms();
        session.updated_at_ms = agent.updated_at_ms;
        Ok(agent.clone())
    }

    async fn finish_agent_success(
        &self,
        team_name: &str,
        agent_name: &str,
        history: Vec<ChatMessage>,
        summary: String,
    ) -> TeamAgent {
        let updated_agent = {
            let mut runtime = self.runtime.write().await;
            let session = runtime
                .scopes
                .entry(self.scope_id.clone())
                .or_default()
                .teams
                .entry(team_name.to_string())
                .or_insert_with(|| TeamSession {
                    name: team_name.to_string(),
                    description: None,
                    created_at_ms: now_ms(),
                    updated_at_ms: now_ms(),
                    agents: HashMap::new(),
                    tasks: Vec::new(),
                    next_task_id: 1,
                    queued_messages: Vec::new(),
                    next_message_id: 1,
                    pending_task_wakes: Vec::new(),
                    recent_file_changes: Vec::new(),
                });
            let key = agent_key(agent_name);
            let agent = session
                .agents
                .get_mut(&key)
                .expect("agent exists after run");
            if agent.status != TeamAgentStatus::Stopped {
                agent.status = TeamAgentStatus::Idle;
            }
            agent.history = history;
            agent.last_summary = Some(summary);
            agent.last_error = None;
            agent.updated_at_ms = now_ms();
            session.updated_at_ms = agent.updated_at_ms;
            agent.clone()
        };
        self.notify_all_team_agents(team_name).await;
        updated_agent
    }

    async fn finish_agent_failure(
        &self,
        team_name: &str,
        agent_name: &str,
        history: Vec<ChatMessage>,
        error: String,
    ) -> TeamAgent {
        let updated_agent = {
            let mut runtime = self.runtime.write().await;
            let session = runtime
                .scopes
                .entry(self.scope_id.clone())
                .or_default()
                .teams
                .entry(team_name.to_string())
                .or_insert_with(|| TeamSession {
                    name: team_name.to_string(),
                    description: None,
                    created_at_ms: now_ms(),
                    updated_at_ms: now_ms(),
                    agents: HashMap::new(),
                    tasks: Vec::new(),
                    next_task_id: 1,
                    queued_messages: Vec::new(),
                    next_message_id: 1,
                    pending_task_wakes: Vec::new(),
                    recent_file_changes: Vec::new(),
                });
            let key = agent_key(agent_name);
            let agent = session
                .agents
                .get_mut(&key)
                .expect("agent exists after run");
            let now = now_ms();
            if agent.status != TeamAgentStatus::Stopped {
                agent.status = TeamAgentStatus::Error;
            }
            agent.history = history;
            agent.last_error = Some(truncate_line(&error, 300));
            agent.last_summary = Some(format!("error: {}", truncate_line(&error, 180)));
            agent.updated_at_ms = now;
            session.updated_at_ms = now;
            agent.clone()
        };
        self.notify_all_team_agents(team_name).await;
        updated_agent
    }

    async fn finish_agent_error(&self, team_name: &str, agent_name: &str, error: &str) {
        let changed = {
            let mut runtime = self.runtime.write().await;
            if let Some(team) = runtime
                .scopes
                .get_mut(&self.scope_id)
                .and_then(|scope| scope.teams.get_mut(team_name))
            {
                if let Some(agent) = team.agents.get_mut(&agent_key(agent_name)) {
                    let now = now_ms();
                    agent.status = TeamAgentStatus::Error;
                    agent.last_error = Some(truncate_line(error, 300));
                    agent.last_summary = Some(format!("error: {}", truncate_line(error, 180)));
                    agent.updated_at_ms = now;
                    team.updated_at_ms = now;
                    true
                } else {
                    false
                }
            } else {
                false
            }
        };
        if changed {
            self.notify_all_team_agents(team_name).await;
        }
    }

    async fn resolve_team_name(
        &self,
        explicit: Option<&str>,
    ) -> std::result::Result<String, String> {
        if let Some(value) = explicit.map(str::trim).filter(|value| !value.is_empty()) {
            return Ok(value.to_string());
        }
        if let Some(identity) = &self.current_agent {
            return Ok(identity.team_name.clone());
        }
        let runtime = self.runtime.read().await;
        let Some(scope) = runtime.scopes.get(&self.scope_id) else {
            return Err("no active team found; start one with TeamRun first".to_string());
        };
        scope
            .active_team
            .clone()
            .or_else(|| {
                if scope.teams.len() == 1 {
                    scope.teams.keys().next().cloned()
                } else {
                    None
                }
            })
            .ok_or_else(|| "team_name is required when no active team exists".to_string())
    }

    async fn agent_sleep_allowed(&self, team_name: &str, agent_name: &str) -> bool {
        let mut runtime = self.runtime.write().await;
        let Some(session) = runtime
            .scopes
            .get_mut(&self.scope_id)
            .and_then(|scope| scope.teams.get_mut(team_name))
        else {
            return false;
        };
        let done_ids = completed_task_ids(session);
        refresh_unblocked_tasks(session, &done_ids);
        prune_stale_task_wakes(session, &done_ids);
        let agent_key_value = agent_key(agent_name);
        agent_has_blocked_task(session, &agent_key_value)
            && !agent_has_runnable_task(session, &agent_key_value, &done_ids)
            && !session.queued_messages.iter().any(|message| {
                agent_key(&message.to) == agent_key_value && queued_message_wakes_agent(message)
            })
    }

    async fn queue_team_message(
        &self,
        team_name: &str,
        from: &str,
        to: &str,
        message: &str,
    ) -> std::result::Result<Vec<TeamQueuedMessage>, String> {
        let (queued, recipients) = {
            let mut runtime = self.runtime.write().await;
            let session = runtime
                .scopes
                .get_mut(&self.scope_id)
                .and_then(|scope| scope.teams.get_mut(team_name))
                .ok_or_else(|| format!("team `{team_name}` not found"))?;
            let recipients = if to == "*" {
                session
                    .agents
                    .values()
                    .filter(|agent| {
                        agent.status != TeamAgentStatus::Stopped
                            && agent_key(&agent.name) != agent_key(from)
                    })
                    .map(|agent| agent.name.clone())
                    .collect::<Vec<_>>()
            } else {
                let key = agent_key(to);
                let Some(agent) = session
                    .agents
                    .values()
                    .find(|agent| agent_key(&agent.name) == key)
                else {
                    return Err(format!("teammate `{to}` not found"));
                };
                if agent.status == TeamAgentStatus::Stopped {
                    return Err(format!("teammate `{}` is stopped", agent.name));
                }
                vec![agent.name.clone()]
            };
            if recipients.is_empty() {
                return Err("no teammates to message".to_string());
            }
            let now = now_ms();
            let mut queued = Vec::new();
            for recipient in &recipients {
                let id = session.next_message_id;
                session.next_message_id += 1;
                let message = TeamQueuedMessage {
                    id,
                    from: from.to_string(),
                    to: recipient.clone(),
                    target: Some(to.to_string()),
                    message: message.to_string(),
                    created_at_ms: now,
                };
                session.queued_messages.push(message.clone());
                queued.push(message);
            }
            session.updated_at_ms = now;
            (queued, recipients)
        };
        self.notify_team_agents(team_name, &recipients).await;
        Ok(queued)
    }

    async fn emit_peer_messages(
        &self,
        team_name: &str,
        messages: &[TeamQueuedMessage],
        event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) {
        if messages.is_empty() {
            return;
        }
        let runtime = self.runtime.read().await;
        let Some(session) = runtime
            .scopes
            .get(&self.scope_id)
            .and_then(|scope| scope.teams.get(team_name))
        else {
            return;
        };
        for message in messages {
            let Some(agent) = session
                .agents
                .values()
                .find(|agent| agent_key(&agent.name) == agent_key(&message.to))
            else {
                continue;
            };
            let _ = event_tx.send(AgentEvent::SubAgentEvent {
                id: format!("agent:{}", agent.id),
                agent_id: agent.id.clone(),
                agent_name: agent.name.clone(),
                team_name: Some(team_name.to_string()),
                model: agent.model.clone(),
                initial_message: None,
                event: Box::new(AgentEvent::PeerMessageReceived {
                    id: message.id.to_string(),
                    from: message.from.clone(),
                    to: message.target.clone().unwrap_or_else(|| message.to.clone()),
                    message: message.message.clone(),
                }),
            });
        }
    }

    fn current_actor_name(&self, team_name: &str) -> String {
        self.current_agent
            .as_ref()
            .filter(|identity| identity.team_name == team_name)
            .map(|identity| identity.agent_name.clone())
            .unwrap_or_else(|| "user".to_string())
    }

    fn select_profile(&self, value: Option<&str>) -> Option<SubAgentConfig> {
        let needle = value?.trim();
        if needle.is_empty() {
            return None;
        }
        let wanted = agent_key(needle);
        self.sub_agent_settings
            .agents
            .iter()
            .find(|agent| {
                agent.enabled
                    && (agent_key(&agent.id) == wanted
                        || agent_key(&agent.name) == wanted
                        || agent_key(&format!("subagent_{}", agent.id)) == wanted)
            })
            .cloned()
    }

    fn prepare_team_agent_configs(
        &self,
        agent_names: &[String],
        agent_profiles: Option<&HashMap<String, String>>,
    ) -> std::result::Result<Vec<PreparedTeamAgentConfig>, String> {
        let mut profile_by_agent = HashMap::<String, SubAgentConfig>::new();
        if let Some(agent_profiles) = agent_profiles {
            for (agent_name, profile_name) in agent_profiles {
                let agent_name = agent_name.trim();
                let profile_name = profile_name.trim();
                if agent_name.is_empty() || profile_name.is_empty() {
                    return Err("agent_profiles keys and values cannot be empty".to_string());
                }
                let agent_key_value = agent_key(agent_name);
                let Some(canonical_name) = agent_names
                    .iter()
                    .find(|name| agent_key(name) == agent_key_value)
                else {
                    return Err(format!(
                        "agent_profiles references unknown teammate `{agent_name}`"
                    ));
                };
                let Some(profile) = self.select_profile(Some(profile_name)) else {
                    return Err(format!("sub-agent profile `{profile_name}` not found"));
                };
                profile_by_agent.insert(agent_key(canonical_name), profile);
            }
        }

        let mut configs = Vec::with_capacity(agent_names.len());
        for name in agent_names {
            let profile = profile_by_agent.get(&agent_key(name));
            let description = profile
                .map(|agent| agent.description.clone())
                .unwrap_or_else(|| "Team collaborator".to_string());
            let model = profile
                .map(|agent| agent.model.clone())
                .unwrap_or_else(|| self.default_model.clone());
            self.validate_model(&model)?;
            let prompt = profile
                .map(|agent| agent.prompt.clone())
                .unwrap_or_default();
            configs.push(PreparedTeamAgentConfig {
                name: name.clone(),
                description,
                prompt,
                model,
            });
        }
        Ok(configs)
    }

    fn validate_model(&self, model: &ModelRef) -> std::result::Result<(), String> {
        let provider = self.providers.get(&model.provider).ok_or_else(|| {
            format!(
                "provider `{}` is not configured or missing credentials",
                model.provider
            )
        })?;
        provider
            .capabilities(model)
            .map(|_| ())
            .ok_or_else(|| format!("model `{}` is not supported", model.name))
    }
}

fn emit_agent_slept_event(
    tool_call_id: &str,
    team_name: &str,
    agent: &TeamAgent,
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
) {
    let _ = event_tx.send(AgentEvent::SubAgentEvent {
        id: tool_call_id.to_string(),
        agent_id: agent.id.clone(),
        agent_name: agent.name.clone(),
        team_name: Some(team_name.to_string()),
        model: agent.model.clone(),
        initial_message: None,
        event: Box::new(AgentEvent::AgentSlept),
    });
}

#[derive(Debug, Deserialize)]
struct TeamRunInput {
    objective: Option<String>,
    agent: Option<String>,
    agent_names: Option<Vec<String>>,
    #[serde(default, alias = "agentProfiles")]
    agent_profiles: Option<AgentProfilesInput>,
    #[serde(default, alias = "agentPrompts")]
    agent_prompts: Option<AgentPromptsInput>,
    tasks: Option<Vec<TeamRunTaskInput>>,
    #[serde(flatten)]
    extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum AgentProfilesInput {
    Assignments(Vec<AgentProfileAssignmentInput>),
    Map(HashMap<String, String>),
}

impl AgentProfilesInput {
    fn to_profile_map(&self) -> std::result::Result<HashMap<String, String>, String> {
        match self {
            Self::Map(map) => Ok(map.clone()),
            Self::Assignments(assignments) => {
                let mut map = HashMap::new();
                for assignment in assignments {
                    let agent = assignment.agent.trim();
                    let profile = assignment.profile.trim();
                    if agent.is_empty() || profile.is_empty() {
                        return Err("agent_profiles entries require non-empty agent and profile"
                            .to_string());
                    }
                    if map.insert(agent.to_string(), profile.to_string()).is_some() {
                        return Err(format!(
                            "agent_profiles contains duplicate teammate `{agent}`"
                        ));
                    }
                }
                Ok(map)
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentProfileAssignmentInput {
    agent: String,
    profile: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum AgentPromptsInput {
    Assignments(Vec<AgentPromptAssignmentInput>),
    Map(HashMap<String, String>),
}

impl AgentPromptsInput {
    fn to_prompt_map(&self) -> std::result::Result<HashMap<String, String>, String> {
        match self {
            Self::Map(map) => Ok(map.clone()),
            Self::Assignments(assignments) => {
                let mut map = HashMap::new();
                let mut seen = BTreeSet::new();
                for assignment in assignments {
                    let agent = assignment.agent.trim();
                    let prompt = assignment.prompt.trim();
                    if agent.is_empty() || prompt.is_empty() {
                        return Err(
                            "agent_prompts entries require non-empty agent and prompt".to_string()
                        );
                    }
                    if !seen.insert(agent_key(agent)) {
                        return Err(format!(
                            "agent_prompts contains duplicate teammate `{agent}`"
                        ));
                    }
                    map.insert(agent.to_string(), prompt.to_string());
                }
                Ok(map)
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentPromptAssignmentInput {
    agent: String,
    prompt: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TeamRunTaskInput {
    subject: String,
    description: Option<String>,
    owner: Option<String>,
    #[serde(default, rename = "blockedBy")]
    blocked_by: Option<Vec<TaskIdInput>>,
    #[serde(default, rename = "blocked_by")]
    blocked_by_snake: Option<Vec<TaskIdInput>>,
}

#[derive(Debug, Clone)]
struct PreparedTeamRunTask {
    subject: String,
    description: Option<String>,
    owner: Option<String>,
    blocked_by: Vec<u64>,
}

#[derive(Debug, Clone)]
struct PreparedTeamAgentConfig {
    name: String,
    description: String,
    prompt: String,
    model: ModelRef,
}

#[derive(Debug, Deserialize)]
struct SendMessageInput {
    to: String,
    message: String,
    team_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TeamNameInput {
    team_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TeamStopInput {
    team_name: Option<String>,
    agent: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TaskCreateInput {
    subject: String,
    description: Option<String>,
    owner: Option<String>,
    #[serde(default, rename = "blockedBy")]
    blocked_by: Option<Vec<TaskIdInput>>,
    #[serde(default, rename = "blocked_by")]
    blocked_by_snake: Option<Vec<TaskIdInput>>,
    team_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TaskListInput {
    team_name: Option<String>,
    action: Option<TaskListAction>,
    status: Option<String>,
    #[serde(rename = "taskId", alias = "id", alias = "task_id")]
    task_id: Option<TaskIdInput>,
    subject: Option<String>,
    description: Option<String>,
    owner: Option<String>,
    #[serde(default, rename = "blockedBy")]
    blocked_by: Option<Vec<TaskIdInput>>,
    #[serde(default, rename = "blocked_by")]
    blocked_by_snake: Option<Vec<TaskIdInput>>,
    #[serde(default, rename = "addBlockedBy")]
    add_blocked_by: Option<Vec<TaskIdInput>>,
    #[serde(default, rename = "add_blocked_by")]
    add_blocked_by_snake: Option<Vec<TaskIdInput>>,
    clear_owner: Option<bool>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum TaskListAction {
    List,
    Create,
    Update,
    Delete,
    Claim,
}

#[derive(Debug, Deserialize)]
struct TaskUpdateInput {
    #[serde(rename = "taskId", alias = "id", alias = "task_id")]
    task_id: TaskIdInput,
    team_name: Option<String>,
    status: Option<TeamTaskStatus>,
    owner: Option<String>,
    subject: Option<String>,
    description: Option<String>,
    #[serde(default, rename = "blockedBy")]
    blocked_by: Option<Vec<TaskIdInput>>,
    #[serde(default, rename = "blocked_by")]
    blocked_by_snake: Option<Vec<TaskIdInput>>,
    #[serde(default, rename = "addBlockedBy")]
    add_blocked_by: Option<Vec<TaskIdInput>>,
    #[serde(default, rename = "add_blocked_by")]
    add_blocked_by_snake: Option<Vec<TaskIdInput>>,
    clear_owner: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum TaskIdInput {
    Number(u64),
    String(String),
}

impl TaskIdInput {
    fn to_u64(&self) -> std::result::Result<u64, String> {
        match self {
            Self::Number(value) if *value > 0 => Ok(*value),
            Self::Number(_) => Err("task id must be greater than zero".to_string()),
            Self::String(value) => {
                let trimmed = value.trim().trim_start_matches('#');
                trimmed.parse::<u64>().map_err(|_| {
                    format!(
                        "invalid task id `{}`; expected a positive integer",
                        value.trim()
                    )
                })
            }
        }
    }
}

#[derive(Debug, Clone)]
struct TeamTurn {
    agent_name: String,
    message: String,
    task_id: Option<u64>,
    label: String,
}

#[derive(Debug, Default)]
struct LiveAgentReport {
    reports: Vec<String>,
    file_changes: Vec<FileChange>,
    images: Vec<ToolRunImage>,
    last_meta: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct TeamSnapshot {
    name: String,
    description: Option<String>,
    agents: Vec<TeamAgentSnapshot>,
    tasks: Vec<TeamTaskSnapshot>,
    queued_messages: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct TeamAgentSnapshot {
    id: String,
    name: String,
    description: String,
    status: TeamAgentStatus,
    model: ModelRef,
    last_summary: Option<String>,
    last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct TeamTaskSnapshot {
    id: u64,
    subject: String,
    description: Option<String>,
    status: TeamTaskStatus,
    owner: Option<String>,
    blocked_by: Vec<u64>,
    updated_at_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct TeamAgentFinalResponse {
    agent: String,
    status: String,
    last_response: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
}

impl TeamSnapshot {
    fn from_session(session: &TeamSession) -> Self {
        let mut agents = session
            .agents
            .values()
            .map(TeamAgentSnapshot::from_agent)
            .collect::<Vec<_>>();
        agents.sort_by(|left, right| left.name.cmp(&right.name));
        let mut tasks = session
            .tasks
            .iter()
            .map(TeamTaskSnapshot::from_task)
            .collect::<Vec<_>>();
        tasks.sort_by_key(|task| task.id);
        Self {
            name: session.name.clone(),
            description: session.description.clone(),
            agents,
            tasks,
            queued_messages: session.queued_messages.len(),
        }
    }
}

impl TeamAgentSnapshot {
    fn from_agent(agent: &TeamAgent) -> Self {
        Self {
            id: agent.id.clone(),
            name: agent.name.clone(),
            description: agent.description.clone(),
            status: agent.status,
            model: agent.model.clone(),
            last_summary: agent.last_summary.clone(),
            last_error: agent.last_error.clone(),
        }
    }
}

impl TeamTaskSnapshot {
    fn from_task(task: &TeamTask) -> Self {
        Self {
            id: task.id,
            subject: task.subject.clone(),
            description: task.description.clone(),
            status: task.status,
            owner: task.owner.clone(),
            blocked_by: task.blocked_by.clone(),
            updated_at_ms: task.updated_at_ms,
        }
    }
}

fn team_agent_system_prompt(base: &str, team_name: &str, agent: &TeamAgent) -> String {
    let config_agent = SubAgentConfig {
        id: agent.id.clone(),
        name: agent.name.clone(),
        description: agent.description.clone(),
        prompt: agent.prompt.clone(),
        model: agent.model.clone(),
        enabled: true,
    };
    let base = subagent_system_prompt(base, &config_agent);
    format!(
        "{base}\n\n<agent_team_profile team=\"{}\" name=\"{}\">\nYou are part of an autonomous agent team.\nYour work is coordinated through the task system and teammate messaging, use SendMessage tool to talk with your team.\nYou may sleep only when your owned task is actually status=blocked in the task board with real blockedBy task IDs. If a task is pending or in_progress, keep working; if it is genuinely blocked, update the task to status=blocked with blockedBy before ending your turn. You will be woken automatically when your owned tasks unlock or when a teammate sends you a direct message.\n</agent_team_profile>",
        escape_attr(team_name),
        escape_attr(&agent.name)
    )
}

fn prepare_team_agent_names(
    names: Option<Vec<String>>,
) -> std::result::Result<Vec<String>, String> {
    let Some(names) = names else {
        return Err("agent_names is required when starting a new team".to_string());
    };
    let mut out: Vec<String> = Vec::new();
    let mut seen = BTreeSet::new();
    for (index, name) in names.into_iter().enumerate() {
        let name = name.trim().to_string();
        if name.is_empty() {
            return Err(format!("agent_names[{index}] cannot be empty"));
        }
        let key = agent_key(&name);
        if !seen.insert(key) {
            return Err(format!("duplicate teammate name `{name}`"));
        }
        out.push(name);
    }
    if out.len() < 2 {
        return Err("agent_names must include at least 2 teammates".to_string());
    }
    if out.len() > 8 {
        return Err("agent_names can include at most 8 teammates".to_string());
    }
    Ok(out)
}

fn team_kickoff_message(objective: &str, agent_name: &str, agent_prompt: Option<&str>) -> String {
    let mut sections = vec![format!("Objective:\n{}", objective.trim())];
    if let Some(prompt) = agent_prompt
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        sections.push(format!(
            "Message from the main agent for @{}:\n{}",
            agent_name.trim(),
            prompt
        ));
    }
    sections.join("\n\n")
}

fn team_restart_message(team_name: &str, agent_name: &str) -> String {
    format!(
        "You are being relaunched in Agent Swarm `{}` as @{}.\n\nReview the current team state in your system context, continue your owned work if it is still relevant, and coordinate with peers through TaskList and SendMessage.",
        team_name.trim(),
        agent_name.trim()
    )
}

fn queued_messages_prompt(messages: &[TeamQueuedMessage]) -> String {
    let mut lines = vec!["<queued_peer_messages>".to_string()];
    for message in messages {
        let to_attr = message
            .target
            .as_deref()
            .map(|target| format!(" to=\"{}\"", escape_attr(target)))
            .unwrap_or_default();
        lines.push(format!(
            "<teammate-message teammate_id=\"{}\"{}>\n{}\n</teammate-message>",
            escape_attr(&message.from),
            to_attr,
            escape_text(message.message.trim())
        ));
    }
    lines.push("</queued_peer_messages>".to_string());
    lines.join("\n")
}

fn render_agent_team_system_reminder(session: &TeamSession, agent_name: &str) -> String {
    let mut agents = session.agents.values().collect::<Vec<_>>();
    agents.sort_by(|left, right| left.name.cmp(&right.name));
    let mut tasks = session.tasks.iter().collect::<Vec<_>>();
    tasks.sort_by_key(|task| task.id);
    let mut lines = vec![
        "<agent_team_state>".to_string(),
        format!("team: {} | you: @{}", session.name, agent_name),
    ];
    if agents.is_empty() {
        lines.push("teammates: none".to_string());
    } else {
        lines.push("teammates:".to_string());
        for agent in agents {
            let you = if agent_key(&agent.name) == agent_key(agent_name) {
                " you"
            } else {
                ""
            };
            lines.push(format!(
                "- @{} [{}]{}",
                agent.name,
                status_label(agent.status),
                you
            ));
        }
    }
    if tasks.is_empty() {
        lines.push("tasks: none".to_string());
    } else {
        lines.push("tasks:".to_string());
        for task in tasks {
            lines.push(render_task_line(task));
        }
    }
    if !session.recent_file_changes.is_empty() {
        lines.push("recent file changes (newest -> oldest):".to_string());
        let total = session.recent_file_changes.len();
        for (index, change) in session.recent_file_changes.iter().rev().enumerate() {
            let marker = if index == 0 {
                "newest -> "
            } else if index + 1 == total {
                "oldest -> "
            } else {
                "          "
            };
            lines.push(format!("{marker}{}", render_recent_file_change(change)));
        }
    }
    lines.push("</agent_team_state>".to_string());
    lines.join("\n")
}

fn render_main_agent_team_system_reminder(session: &TeamSession) -> String {
    let mut agents = session.agents.values().collect::<Vec<_>>();
    agents.sort_by(|left, right| left.name.cmp(&right.name));
    let any_running = agents
        .iter()
        .any(|agent| agent.status == TeamAgentStatus::Running);
    let mut lines = vec![
        "<agent_swarm_state>".to_string(),
        format!("team: {}", session.name),
    ];
    if agents.is_empty() {
        lines.push("teammates: none".to_string());
    } else {
        lines.push("teammates:".to_string());
        for agent in &agents {
            let error = agent
                .last_error
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .map(|value| format!(" error: {}", truncate_line(value, 180)))
                .unwrap_or_default();
            lines.push(format!(
                "- @{} [{}]{}",
                agent.name,
                status_label(agent.status),
                error
            ));
        }
    }
    let errors = agents
        .into_iter()
        .filter(|agent| agent.status == TeamAgentStatus::Error)
        .filter_map(|agent| {
            agent
                .last_error
                .as_deref()
                .map(|error| format!("- @{}: {}", agent.name, truncate_line(error, 220)))
        })
        .collect::<Vec<_>>();
    if !errors.is_empty() {
        lines.push("errors:".to_string());
        lines.extend(errors);
        lines.push("main-agent guidance: handle only these failures. Relaunch the failed teammate with TeamRun agent=... when useful, or stop that teammate if it is looping. Do not take over normal team work.".to_string());
    } else if any_running {
        lines.push("main-agent guidance: the Agent Swarm runs asynchronously in the background. Do not poll with shell commands, file checks, or TeamStatus just to see whether it is done. End your turn after acknowledging launch/status and wait for a user or system wake.".to_string());
    } else {
        lines.push("main-agent guidance: the Agent Swarm has no running teammates right now. If the current turn was triggered by an agent_swarm_finished system reminder, tell the user the Agent Swarm finished and summarize the final teammate responses. Do not poll with shell commands, file checks, or TeamStatus just to check completion.".to_string());
    }
    lines.push("</agent_swarm_state>".to_string());
    lines.join("\n")
}

fn render_team_snapshot(snapshot: &TeamSnapshot) -> String {
    let mut lines = vec![format!("team: {}", snapshot.name)];
    if let Some(description) = snapshot.description.as_deref() {
        lines.push(format!("description: {description}"));
    }
    if snapshot.agents.is_empty() {
        lines.push("teammates: none".to_string());
    } else {
        lines.push(format!("teammates: {}", snapshot.agents.len()));
        for agent in &snapshot.agents {
            let summary = agent
                .last_summary
                .as_deref()
                .and_then(first_line)
                .unwrap_or("no report yet");
            lines.push(format!(
                "- @{} [{}] {} — {}",
                agent.name,
                status_label(agent.status),
                agent.description,
                summary
            ));
        }
    }
    if snapshot.tasks.is_empty() {
        lines.push("tasks: none".to_string());
    } else {
        lines.push(format!("tasks: {}", snapshot.tasks.len()));
        for task in &snapshot.tasks {
            lines.push(render_task_snapshot_line(task));
        }
    }
    if snapshot.queued_messages > 0 {
        lines.push(format!("queued messages: {}", snapshot.queued_messages));
    }
    lines.join("\n")
}

fn team_agent_final_responses_from_session(session: &TeamSession) -> Vec<TeamAgentFinalResponse> {
    let mut agents = session.agents.values().collect::<Vec<_>>();
    agents.sort_by(|left, right| left.name.cmp(&right.name));
    agents
        .into_iter()
        .map(|agent| TeamAgentFinalResponse {
            agent: agent.name.clone(),
            status: final_response_status_label(agent.status).to_string(),
            last_response: final_response_for_agent(agent),
            last_error: agent.last_error.clone(),
        })
        .collect()
}

fn final_response_for_agent(agent: &TeamAgent) -> String {
    final_assistant_text(&agent.history)
        .or_else(|| agent.last_summary.clone())
        .unwrap_or_else(|| "No final response recorded.".to_string())
}

fn render_team_agent_final_responses(responses: &[TeamAgentFinalResponse]) -> String {
    responses
        .iter()
        .map(|response| {
            let mut lines = vec![format!("- @{} [{}]", response.agent, response.status)];
            if let Some(error) = response
                .last_error
                .as_deref()
                .filter(|value| !value.trim().is_empty())
            {
                lines.push(format!("  error: {}", truncate_text(error, 500)));
            }
            lines.push(format!(
                "  lastResponse: {}",
                indent_multiline(&truncate_text(&response.last_response, 1200), "  ").trim_start()
            ));
            lines.join("\n")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn final_response_status_label(status: TeamAgentStatus) -> &'static str {
    match status {
        TeamAgentStatus::Idle => "finished",
        TeamAgentStatus::Running => "running",
        TeamAgentStatus::Stopped => "stopped",
        TeamAgentStatus::Error => "error",
    }
}

fn render_task_snapshot_line(task: &TeamTaskSnapshot) -> String {
    let owner = task
        .owner
        .as_deref()
        .map(|owner| format!(" @{}", owner))
        .unwrap_or_default();
    let mut detail = Vec::new();
    if !task.blocked_by.is_empty() {
        detail.push(format!("blocked by {}", render_task_ids(&task.blocked_by)));
    }
    let detail = if detail.is_empty() {
        String::new()
    } else {
        format!(" ({})", detail.join("; "))
    };
    format!(
        "- #{} [{}]{} {}{}",
        task.id,
        task_status_label(task.status),
        owner,
        task.subject,
        detail
    )
}

fn render_task_line(task: &TeamTask) -> String {
    render_task_snapshot_line(&TeamTaskSnapshot::from_task(task))
}

fn render_recent_file_change(change: &TeamRecentFileChange) -> String {
    format!(
        "@{} {} {} {} (+{} -{})",
        change.agent,
        change.tool,
        file_change_kind_label(change.kind),
        change.relative_path,
        change.added,
        change.removed
    )
}

fn render_agent_result(team_name: &str, agent: &TeamAgent, answer: &str) -> String {
    format!(
        "team: {team_name}\nagent: @{}\nstatus: {}\n\n{}",
        agent.name,
        status_label(agent.status),
        answer.trim()
    )
}

fn final_assistant_text(history: &[ChatMessage]) -> Option<String> {
    history.iter().rev().find_map(|message| {
        if !matches!(message.role, Role::Assistant) {
            return None;
        }
        let text = message
            .parts
            .iter()
            .filter_map(|part| match part {
                Part::Text { text, .. } if !text.trim().is_empty() => Some(text.trim()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        (!text.trim().is_empty()).then_some(text)
    })
}

fn file_changes_from_history(history: &[ChatMessage]) -> Vec<FileChange> {
    history
        .iter()
        .flat_map(|message| message.parts.iter())
        .filter_map(|part| match part {
            Part::ToolResult { meta, .. } => meta
                .as_ref()
                .and_then(|meta| meta.get("file_changes"))
                .and_then(|value| serde_json::from_value::<Vec<FileChange>>(value.clone()).ok()),
            _ => None,
        })
        .flatten()
        .collect()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis() as u64)
        .unwrap_or_default()
}

fn agent_notify_key(scope_id: &str, team_name: &str, agent_name: &str) -> String {
    format!(
        "{}\0{}\0{}",
        scope_id,
        agent_key(team_name),
        agent_key(agent_name)
    )
}

fn team_notify_key_prefix(scope_id: &str, team_name: &str) -> String {
    format!("{}\0{}\0", scope_id, agent_key(team_name))
}

fn workspace_write_lock_key(workspace_root: &Path) -> String {
    workspace_root.display().to_string()
}

fn wake_notifier(notifier: &Notify) {
    notifier.notify_waiters();
    notifier.notify_one();
}

fn agent_key(value: &str) -> String {
    let key = value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    if key.is_empty() {
        Uuid::new_v4().to_string()
    } else {
        key
    }
}

fn status_label(status: TeamAgentStatus) -> &'static str {
    match status {
        TeamAgentStatus::Idle => "idle",
        TeamAgentStatus::Running => "running",
        TeamAgentStatus::Stopped => "stopped",
        TeamAgentStatus::Error => "error",
    }
}

fn team_run_status_label(
    snapshot: Option<&TeamSnapshot>,
    is_error: bool,
    fallback: &str,
) -> String {
    if is_error {
        return "error".to_string();
    }
    let Some(snapshot) = snapshot else {
        return fallback.to_string();
    };
    if snapshot
        .agents
        .iter()
        .any(|agent| agent.status == TeamAgentStatus::Running)
    {
        return "running".to_string();
    }
    if snapshot
        .agents
        .iter()
        .any(|agent| agent.status == TeamAgentStatus::Error)
    {
        return "error".to_string();
    }
    if !snapshot.agents.is_empty()
        && snapshot
            .agents
            .iter()
            .all(|agent| agent.status == TeamAgentStatus::Stopped)
    {
        return "stopped".to_string();
    }
    fallback.to_string()
}

fn task_status_label(status: TeamTaskStatus) -> &'static str {
    match status {
        TeamTaskStatus::Pending => "pending",
        TeamTaskStatus::InProgress => "in_progress",
        TeamTaskStatus::Blocked => "blocked",
        TeamTaskStatus::Completed => "completed",
    }
}

fn file_change_kind_label(kind: FileChangeKind) -> &'static str {
    match kind {
        FileChangeKind::Added => "added",
        FileChangeKind::Modified => "modified",
        FileChangeKind::Deleted => "deleted",
    }
}

fn normalized_owner(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn prepare_team_agent_prompts(
    agent_names: &[String],
    agent_prompts: Option<&HashMap<String, String>>,
) -> std::result::Result<HashMap<String, String>, String> {
    let mut prompts = HashMap::new();
    let Some(agent_prompts) = agent_prompts else {
        return Ok(prompts);
    };
    for (agent_name, prompt) in agent_prompts {
        let agent_name = agent_name.trim();
        let prompt = prompt.trim();
        if agent_name.is_empty() || prompt.is_empty() {
            return Err("agent_prompts keys and values cannot be empty".to_string());
        }
        let agent_key_value = agent_key(agent_name);
        let Some(canonical_name) = agent_names
            .iter()
            .find(|name| agent_key(name) == agent_key_value)
        else {
            return Err(format!(
                "agent_prompts references unknown teammate `{agent_name}`"
            ));
        };
        let canonical_key = agent_key(canonical_name);
        if prompts.insert(canonical_key, prompt.to_string()).is_some() {
            return Err(format!(
                "agent_prompts contains duplicate teammate `{agent_name}`"
            ));
        }
    }
    Ok(prompts)
}

fn prepare_team_run_tasks(
    tasks: Option<&[TeamRunTaskInput]>,
    agent_names: &[String],
) -> std::result::Result<Vec<PreparedTeamRunTask>, String> {
    let Some(tasks) = tasks else {
        return Ok(Vec::new());
    };
    let task_count = tasks.len() as u64;
    let mut prepared = Vec::with_capacity(tasks.len());
    for (index, task) in tasks.iter().enumerate() {
        let task_id = index as u64 + 1;
        let subject = task.subject.trim();
        if subject.is_empty() {
            return Err(format!("tasks[{index}].subject is required"));
        }
        let description = task
            .description
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        let owner = match normalized_owner(task.owner.as_deref()) {
            Some(owner) => {
                let owner_key = agent_key(&owner);
                let Some(canonical) = agent_names
                    .iter()
                    .find(|agent_name| agent_key(agent_name) == owner_key)
                else {
                    return Err(format!(
                        "tasks[{index}].owner `{owner}` does not match a teammate"
                    ));
                };
                Some(canonical.clone())
            }
            None => None,
        };
        let blocked_by = normalize_task_ids(merge_task_id_inputs(
            task.blocked_by.clone(),
            task.blocked_by_snake.clone(),
        ))?;
        if blocked_by.contains(&task_id) {
            return Err(format!("task #{task_id} cannot block itself"));
        }
        let unknown = blocked_by
            .iter()
            .filter(|id| **id > task_count)
            .copied()
            .collect::<Vec<_>>();
        if !unknown.is_empty() {
            return Err(format!(
                "unknown initial blocking task(s): {}",
                render_task_ids(&unknown)
            ));
        }
        prepared.push(PreparedTeamRunTask {
            subject: subject.to_string(),
            description,
            owner,
            blocked_by,
        });
    }
    Ok(prepared)
}

fn queued_message_wakes_agent(message: &TeamQueuedMessage) -> bool {
    message
        .target
        .as_deref()
        .map(str::trim)
        .filter(|target| !target.is_empty())
        .map(|target| target != "*")
        .unwrap_or(true)
}

fn merge_task_id_inputs(
    first: Option<Vec<TaskIdInput>>,
    second: Option<Vec<TaskIdInput>>,
) -> Option<Vec<TaskIdInput>> {
    match (first, second) {
        (Some(mut first), Some(second)) => {
            first.extend(second);
            Some(first)
        }
        (Some(first), None) => Some(first),
        (None, Some(second)) => Some(second),
        (None, None) => None,
    }
}

fn normalize_task_ids(ids: Option<Vec<TaskIdInput>>) -> std::result::Result<Vec<u64>, String> {
    normalize_optional_task_ids(ids).map(Option::unwrap_or_default)
}

fn normalize_optional_task_ids(
    ids: Option<Vec<TaskIdInput>>,
) -> std::result::Result<Option<Vec<u64>>, String> {
    let Some(ids) = ids else {
        return Ok(None);
    };
    let mut values = Vec::new();
    for id in ids {
        values.push(id.to_u64()?);
    }
    Ok(Some(normalize_task_id_values(values)))
}

fn normalize_task_id_values(ids: Vec<u64>) -> Vec<u64> {
    ids.into_iter()
        .filter(|id| *id > 0)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn validate_task_dependencies(
    session: &TeamSession,
    task_id: Option<u64>,
    blocked_by: &[u64],
) -> std::result::Result<(), String> {
    if let Some(task_id) = task_id {
        if blocked_by.contains(&task_id) {
            return Err(format!("task #{task_id} cannot block itself"));
        }
    }
    let unknown = blocked_by
        .iter()
        .filter(|id| !session.tasks.iter().any(|task| task.id == **id))
        .copied()
        .collect::<Vec<_>>();
    if unknown.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "unknown blocking task(s): {}",
            render_task_ids(&unknown)
        ))
    }
}

fn validate_task_dependency_lock(
    task_id: u64,
    current_blocked_by: &[u64],
    next_blocked_by: &[u64],
    requested_status: Option<TeamTaskStatus>,
    done_ids: &BTreeSet<u64>,
) -> std::result::Result<(), String> {
    let removed_unresolved = current_blocked_by
        .iter()
        .filter(|id| !done_ids.contains(id) && !next_blocked_by.contains(id))
        .copied()
        .collect::<Vec<_>>();
    if !removed_unresolved.is_empty() {
        return Err(format!(
            "task #{task_id} is still blocked by {}; complete blocking tasks before clearing blockedBy",
            render_task_ids(&removed_unresolved)
        ));
    }

    let unresolved = next_blocked_by
        .iter()
        .filter(|id| !done_ids.contains(id))
        .copied()
        .collect::<Vec<_>>();
    if unresolved.is_empty() {
        return Ok(());
    }
    match requested_status {
        Some(status @ TeamTaskStatus::Pending)
        | Some(status @ TeamTaskStatus::InProgress)
        | Some(status @ TeamTaskStatus::Completed) => Err(format!(
            "task #{task_id} is blocked by {}; complete blocking tasks before setting status={}",
            render_task_ids(&unresolved),
            task_status_label(status)
        )),
        Some(TeamTaskStatus::Blocked) | None => Ok(()),
    }
}

fn completed_task_ids(session: &TeamSession) -> BTreeSet<u64> {
    session
        .tasks
        .iter()
        .filter(|task| task.status == TeamTaskStatus::Completed)
        .map(|task| task.id)
        .collect()
}

fn refresh_unblocked_tasks(session: &mut TeamSession, done_ids: &BTreeSet<u64>) -> Vec<u64> {
    let now = now_ms();
    let mut changed = false;
    let mut unblocked_task_ids = Vec::new();
    for task in &mut session.tasks {
        let dependencies_are_ready =
            !task.blocked_by.is_empty() && task_dependencies_satisfied(task, done_ids);
        let dependencies_are_unresolved = !task.blocked_by.is_empty() && !dependencies_are_ready;
        let invalid_empty_dependency = task.blocked_by.is_empty();
        if dependencies_are_ready {
            let was_blocked = task.status == TeamTaskStatus::Blocked;
            task.blocked_by.clear();
            if was_blocked {
                task.status = TeamTaskStatus::Pending;
                task.completed_at_ms = None;
                unblocked_task_ids.push(task.id);
            }
            task.updated_at_ms = now;
            changed = true;
        } else if task.status == TeamTaskStatus::Blocked && invalid_empty_dependency {
            task.status = TeamTaskStatus::Pending;
            task.completed_at_ms = None;
            task.updated_at_ms = now;
            changed = true;
            unblocked_task_ids.push(task.id);
        } else if dependencies_are_unresolved
            && matches!(
                task.status,
                TeamTaskStatus::Pending | TeamTaskStatus::InProgress
            )
        {
            task.status = TeamTaskStatus::Blocked;
            task.completed_at_ms = None;
            task.updated_at_ms = now;
            changed = true;
        }
    }
    if changed {
        session.updated_at_ms = now;
    }
    for task_id in &unblocked_task_ids {
        queue_task_wake_for_ready_task(session, *task_id, done_ids, now);
    }
    unblocked_task_ids
}

fn queue_task_wake_for_ready_task(
    session: &mut TeamSession,
    task_id: u64,
    done_ids: &BTreeSet<u64>,
    now: u64,
) -> bool {
    let Some(task) = session.tasks.iter().find(|task| task.id == task_id) else {
        return false;
    };
    if task.status != TeamTaskStatus::Pending || !task_dependencies_satisfied(task, done_ids) {
        return false;
    }
    let Some(owner) = task
        .owner
        .as_deref()
        .map(str::trim)
        .filter(|owner| !owner.is_empty())
    else {
        return false;
    };
    let owner_key = agent_key(owner);
    let Some(agent) = session.agents.get(&owner_key) else {
        return false;
    };
    if agent.status != TeamAgentStatus::Idle {
        return false;
    }
    if session
        .pending_task_wakes
        .iter()
        .any(|wake| wake.task_id == task_id && agent_key(&wake.owner) == owner_key)
    {
        return false;
    }
    session.pending_task_wakes.push(TeamTaskWake {
        task_id,
        owner: agent.name.clone(),
        created_at_ms: now,
    });
    true
}

fn prune_stale_task_wakes(session: &mut TeamSession, done_ids: &BTreeSet<u64>) {
    let wakes = std::mem::take(&mut session.pending_task_wakes);
    session.pending_task_wakes = wakes
        .into_iter()
        .filter(|wake| task_wake_is_still_valid(session, wake, done_ids))
        .collect();
}

fn task_wake_is_still_valid(
    session: &TeamSession,
    wake: &TeamTaskWake,
    done_ids: &BTreeSet<u64>,
) -> bool {
    let owner_key = agent_key(&wake.owner);
    let Some(agent) = session.agents.get(&owner_key) else {
        return false;
    };
    if agent.status != TeamAgentStatus::Idle {
        return false;
    }
    let Some(task) = session.tasks.iter().find(|task| task.id == wake.task_id) else {
        return false;
    };
    task.status == TeamTaskStatus::Pending
        && task_dependencies_satisfied(task, done_ids)
        && task.owner.as_deref().map(agent_key).as_deref() == Some(owner_key.as_str())
}

fn task_wake_ids_for_agent(session: &TeamSession, agent_key_value: &str) -> BTreeSet<u64> {
    session
        .pending_task_wakes
        .iter()
        .filter(|wake| agent_key(&wake.owner) == agent_key_value)
        .map(|wake| wake.task_id)
        .collect()
}

fn remove_task_wakes_for_task(session: &mut TeamSession, task_id: u64) {
    session
        .pending_task_wakes
        .retain(|wake| wake.task_id != task_id);
}

fn in_progress_task_id_for_agent(session: &TeamSession, agent_key_value: &str) -> Option<u64> {
    session
        .tasks
        .iter()
        .find(|task| {
            task.status == TeamTaskStatus::InProgress
                && task.owner.as_deref().map(agent_key).as_deref() == Some(agent_key_value)
        })
        .map(|task| task.id)
}

fn ready_pending_task_id_for_agent(
    session: &TeamSession,
    agent_key_value: &str,
    done_ids: &BTreeSet<u64>,
) -> Option<u64> {
    session
        .tasks
        .iter()
        .find(|task| {
            task.status == TeamTaskStatus::Pending
                && task_dependencies_satisfied(task, done_ids)
                && task.owner.as_deref().map(agent_key).as_deref() == Some(agent_key_value)
        })
        .map(|task| task.id)
}

fn agent_has_runnable_task(
    session: &TeamSession,
    agent_key_value: &str,
    done_ids: &BTreeSet<u64>,
) -> bool {
    in_progress_task_id_for_agent(session, agent_key_value).is_some()
        || ready_pending_task_id_for_agent(session, agent_key_value, done_ids).is_some()
}

fn agent_has_blocked_task(session: &TeamSession, agent_key_value: &str) -> bool {
    session.tasks.iter().any(|task| {
        task.status == TeamTaskStatus::Blocked
            && task.owner.as_deref().map(agent_key).as_deref() == Some(agent_key_value)
    })
}

fn task_dependencies_satisfied(task: &TeamTask, done_ids: &BTreeSet<u64>) -> bool {
    task.blocked_by.iter().all(|id| done_ids.contains(id))
}

fn file_change_line_counts(change: &FileChange) -> (usize, usize) {
    change
        .lines
        .iter()
        .fold((0usize, 0usize), |(added, removed), line| match line.kind {
            DiffLineKind::Added => (added + 1, removed),
            DiffLineKind::Removed => (added, removed + 1),
            DiffLineKind::Context => (added, removed),
        })
}

fn team_ready_task_message(task: &TeamTask) -> String {
    format!(
        "<task_ready>\n{}\n</task_ready>\n\nThis task is ready and assigned to you. The current board is already in your system context; do not call TaskList action=list just to inspect it. Start this task with TaskList action=update taskId={} status=in_progress, do the work, then mark it completed when finished. Only sleep if the task is actually status=blocked with real blockedBy task IDs.",
        render_task_line(task),
        task.id
    )
}

fn team_continue_task_message(task: &TeamTask) -> String {
    format!(
        "<task_continue>\n{}\n</task_continue>\n\nThis task is still in progress and assigned to you. Continue working now. If it is genuinely blocked, use TaskList action=update taskId={} status=blocked blockedBy=[...] with real blocking task IDs; otherwise keep working and mark it completed when finished. Do not sleep while this task is not truly blocked.",
        render_task_line(task),
        task.id
    )
}

#[cfg(test)]
fn team_unlocked_task_message(task: &TeamTask) -> String {
    team_ready_task_message(task)
}

fn normalize_optional_object_input(input: Value) -> Value {
    match input {
        Value::Null => json!({}),
        Value::String(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                json!({})
            } else {
                match serde_json::from_str::<Value>(trimmed) {
                    Ok(Value::Object(map)) => Value::Object(map),
                    _ => Value::String(raw),
                }
            }
        }
        other => other,
    }
}

fn render_task_ids(ids: &[u64]) -> String {
    ids.iter()
        .map(|id| format!("#{id}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn truncate_line(value: &str, limit: usize) -> String {
    let line = first_line(value).unwrap_or(value).trim();
    if line.chars().count() > limit {
        let keep = limit.saturating_sub(3);
        let mut truncated = line.chars().take(keep).collect::<String>();
        truncated.push_str("...");
        truncated
    } else {
        line.to_string()
    }
}

fn truncate_text(value: &str, limit: usize) -> String {
    let value = value.trim();
    if value.chars().count() > limit {
        let keep = limit.saturating_sub(3);
        let mut truncated = value.chars().take(keep).collect::<String>();
        truncated.push_str("...");
        truncated
    } else {
        value.to_string()
    }
}

fn indent_multiline(value: &str, indent: &str) -> String {
    value
        .lines()
        .map(|line| format!("{indent}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn first_line(value: &str) -> Option<&str> {
    value.lines().map(str::trim).find(|line| !line.is_empty())
}

fn escape_attr(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn escape_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

pub fn is_team_tool_name(name: &str) -> bool {
    matches!(
        name,
        TEAM_RUN_TOOL
            | TEAM_CREATE_TOOL
            | AGENT_TOOL
            | SEND_MESSAGE_TOOL
            | TASK_CREATE_TOOL
            | TASK_LIST_TOOL
            | TASK_UPDATE_TOOL
            | TEAM_STATUS_TOOL
            | TEAM_STOP_TOOL
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_object_tools_accept_empty_string_input() {
        let status: TeamNameInput =
            serde_json::from_value(normalize_optional_object_input(json!("")))
                .expect("empty status input should parse as an empty object");
        let stop: TeamStopInput =
            serde_json::from_value(normalize_optional_object_input(json!("  ")))
                .expect("empty stop input should parse as an empty object");

        assert_eq!(status.team_name, None);
        assert_eq!(stop.team_name, None);
        assert_eq!(stop.agent, None);
    }

    #[test]
    fn optional_object_tools_accept_json_string_input() {
        let stop: TeamStopInput =
            serde_json::from_value(normalize_optional_object_input(json!("{\"agent\":\"ui\"}")))
                .expect("json string stop input should parse as an object");

        assert_eq!(stop.agent.as_deref(), Some("ui"));
    }

    #[tokio::test]
    async fn team_stop_without_active_team_is_noop() {
        let tool = test_team_tool();
        let (event_tx, _event_rx) = mpsc::unbounded_channel();

        let result = tool.run_stop(json!(""), event_tx).await;

        assert!(!result.is_error);
        assert!(result.content.contains("no active Agent Swarm"));
    }

    #[tokio::test]
    async fn team_status_without_active_team_is_noop() {
        let tool = test_team_tool();

        let result = tool.run_status(json!("")).await;

        assert!(!result.is_error);
        assert!(result.content.contains("no active Agent Swarm"));
    }

    #[tokio::test]
    async fn team_stop_all_removes_active_runtime_team() {
        let tool = test_team_tool();
        {
            let mut runtime = tool.runtime.write().await;
            let scope = runtime.scopes.entry(tool.scope_id.clone()).or_default();
            scope.active_team = Some("test-team".to_string());
            scope.teams.insert(
                "test-team".to_string(),
                test_team_session(vec![test_agent("bravo", TeamAgentStatus::Idle)]),
            );
        }
        let (event_tx, _event_rx) = mpsc::unbounded_channel();

        let result = tool.run_stop(json!(""), event_tx).await;

        assert!(!result.is_error);
        let runtime = tool.runtime.read().await;
        let scope = runtime
            .scopes
            .get(&tool.scope_id)
            .expect("scope should remain");
        assert_eq!(scope.active_team, None);
        assert!(!scope.teams.contains_key("test-team"));
    }

    #[test]
    fn task_create_accepts_both_dependency_field_names() {
        let parsed: TaskCreateInput = serde_json::from_value(json!({
            "subject": "wire modules",
            "blockedBy": [1],
            "blocked_by": [2]
        }))
        .expect("task create input should parse");
        let blocked_by = normalize_task_ids(merge_task_id_inputs(
            parsed.blocked_by,
            parsed.blocked_by_snake,
        ))
        .expect("blocked ids should normalize");
        assert_eq!(blocked_by, vec![1, 2]);
    }

    #[test]
    fn task_update_accepts_both_dependency_field_names() {
        let parsed: TaskUpdateInput = serde_json::from_value(json!({
            "taskId": 3,
            "blockedBy": [1],
            "blocked_by": [2],
            "addBlockedBy": [4],
            "add_blocked_by": [5]
        }))
        .expect("task update input should parse");
        let replace = normalize_task_ids(merge_task_id_inputs(
            parsed.blocked_by,
            parsed.blocked_by_snake,
        ))
        .expect("replace ids should normalize");
        let add = normalize_task_ids(merge_task_id_inputs(
            parsed.add_blocked_by,
            parsed.add_blocked_by_snake,
        ))
        .expect("additional ids should normalize");
        assert_eq!(replace, vec![1, 2]);
        assert_eq!(add, vec![4, 5]);
    }

    #[test]
    fn team_run_tasks_accept_initial_dependencies_by_order() {
        let parsed: TeamRunInput = serde_json::from_value(json!({
            "objective": "ship app",
            "agent_names": ["player", "scene"],
            "tasks": [
                { "subject": "scaffold", "owner": "player" },
                { "subject": "polish", "blockedBy": [1] },
                { "subject": "review", "blocked_by": ["#1", "2"] }
            ]
        }))
        .expect("team run input should parse");
        let agent_names =
            prepare_team_agent_names(parsed.agent_names).expect("agent names should normalize");
        let tasks = prepare_team_run_tasks(parsed.tasks.as_deref(), &agent_names)
            .expect("initial tasks should normalize");
        assert_eq!(tasks.len(), 3);
        assert_eq!(tasks[0].owner.as_deref(), Some("player"));
        assert_eq!(tasks[1].blocked_by, vec![1]);
        assert_eq!(tasks[2].blocked_by, vec![1, 2]);
    }

    #[test]
    fn team_run_descriptor_exposes_agent_profiles_as_visible_array() {
        let descriptor = TeamTool::descriptors_static()
            .into_iter()
            .find(|tool| tool.name == TEAM_RUN_TOOL)
            .expect("TeamRun descriptor should exist");

        let schema_type = descriptor
            .input_schema
            .pointer("/properties/agent_profiles/type")
            .and_then(Value::as_str);

        assert_eq!(schema_type, Some("array"));
        assert!(descriptor
            .input_schema
            .pointer("/properties/agent_profiles/items/properties/profile")
            .is_some());
    }

    #[test]
    fn team_run_accepts_agent_profiles_assignments() {
        let parsed: TeamRunInput = serde_json::from_value(json!({
            "objective": "ship app",
            "agent_names": ["player", "scene"],
            "agent_profiles": [
                { "agent": "player", "profile": "gameplay_dev" },
                { "agent": "scene", "profile": "threejs_expert" }
            ]
        }))
        .expect("agent profile assignment list should parse");

        let profiles = parsed
            .agent_profiles
            .expect("profiles should exist")
            .to_profile_map()
            .expect("profiles should normalize");

        assert_eq!(
            profiles.get("player").map(String::as_str),
            Some("gameplay_dev")
        );
        assert_eq!(
            profiles.get("scene").map(String::as_str),
            Some("threejs_expert")
        );
    }

    #[test]
    fn team_run_still_accepts_agent_profiles_map() {
        let parsed: TeamRunInput = serde_json::from_value(json!({
            "objective": "ship app",
            "agent_names": ["player", "scene"],
            "agent_profiles": {
                "player": "gameplay_dev",
                "scene": "threejs_expert"
            }
        }))
        .expect("legacy agent profile map should parse");

        let profiles = parsed
            .agent_profiles
            .expect("profiles should exist")
            .to_profile_map()
            .expect("profiles should normalize");

        assert_eq!(
            profiles.get("player").map(String::as_str),
            Some("gameplay_dev")
        );
        assert_eq!(
            profiles.get("scene").map(String::as_str),
            Some("threejs_expert")
        );
    }

    #[test]
    fn final_responses_are_structured_by_agent_name() {
        let mut alpha = test_agent("alpha", TeamAgentStatus::Idle);
        alpha.history.push(ChatMessage {
            role: Role::Assistant,
            parts: vec![Part::Text {
                text: "Alpha final answer\nwith details".to_string(),
                meta: None,
            }],
        });
        alpha.last_summary = Some("Older summary".to_string());
        let mut bravo = test_agent("bravo", TeamAgentStatus::Error);
        bravo.last_summary = Some("error: failed".to_string());
        bravo.last_error = Some("failed".to_string());
        let session = test_team_session(vec![bravo, alpha]);

        let responses = team_agent_final_responses_from_session(&session);

        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0].agent, "alpha");
        assert_eq!(responses[0].status, "finished");
        assert_eq!(
            responses[0].last_response,
            "Alpha final answer\nwith details"
        );
        assert_eq!(responses[1].agent, "bravo");
        assert_eq!(responses[1].status, "error");
        assert_eq!(responses[1].last_response, "error: failed");
        assert_eq!(responses[1].last_error.as_deref(), Some("failed"));
    }

    #[test]
    fn team_run_tasks_reject_unknown_initial_dependencies() {
        let parsed: TeamRunInput = serde_json::from_value(json!({
            "objective": "ship app",
            "tasks": [
                { "subject": "review", "blockedBy": [2] }
            ]
        }))
        .expect("team run input should parse");
        let agent_names = vec!["reviewer".to_string(), "builder".to_string()];
        let err = prepare_team_run_tasks(parsed.tasks.as_deref(), &agent_names)
            .expect_err("unknown dependencies should fail");
        assert!(err.contains("unknown initial blocking task"));
    }

    #[test]
    fn team_run_tasks_reject_unknown_owner() {
        let parsed: TeamRunInput = serde_json::from_value(json!({
            "objective": "ship app",
            "agent_names": ["player", "scene"],
            "tasks": [
                { "subject": "scaffold", "owner": "audio" }
            ]
        }))
        .expect("team run input should parse");
        let agent_names =
            prepare_team_agent_names(parsed.agent_names).expect("agent names should normalize");
        let err = prepare_team_run_tasks(parsed.tasks.as_deref(), &agent_names)
            .expect_err("unknown owners should fail");
        assert!(err.contains("does not match a teammate"));
    }

    #[test]
    fn team_run_requires_explicit_agent_names() {
        let err = prepare_team_agent_names(None).expect_err("missing names should fail");
        assert!(err.contains("agent_names is required"));
    }

    #[test]
    fn team_run_accepts_up_to_eight_explicit_agent_names() {
        let names = prepare_team_agent_names(Some(vec![
            "player".to_string(),
            "scene".to_string(),
            "track".to_string(),
            "ui".to_string(),
            "audio".to_string(),
            "physics".to_string(),
            "qa".to_string(),
            "polish".to_string(),
        ]))
        .expect("eight agents should be accepted");
        assert_eq!(names.len(), 8);
    }

    #[test]
    fn unblocked_owned_task_queues_wake_for_idle_owner() {
        let mut session = test_team_session(vec![
            test_agent("alpha", TeamAgentStatus::Idle),
            test_agent("bravo", TeamAgentStatus::Idle),
        ]);
        session.tasks.push(test_task(
            1,
            TeamTaskStatus::Completed,
            Some("alpha"),
            Vec::new(),
        ));
        session.tasks.push(test_task(
            2,
            TeamTaskStatus::Blocked,
            Some("bravo"),
            vec![1],
        ));

        let done_ids = completed_task_ids(&session);
        let unblocked = refresh_unblocked_tasks(&mut session, &done_ids);

        assert_eq!(unblocked, vec![2]);
        assert_eq!(session.tasks[1].status, TeamTaskStatus::Pending);
        assert!(session.tasks[1].blocked_by.is_empty());
        assert_eq!(session.pending_task_wakes.len(), 1);
        assert_eq!(session.pending_task_wakes[0].task_id, 2);
        assert_eq!(session.pending_task_wakes[0].owner, "bravo");
    }

    #[test]
    fn pending_task_wake_targets_unblocked_owned_task() {
        let mut session = test_team_session(vec![test_agent("bravo", TeamAgentStatus::Idle)]);
        session
            .tasks
            .push(test_task(1, TeamTaskStatus::Pending, None, Vec::new()));
        session.tasks.push(test_task(
            2,
            TeamTaskStatus::Pending,
            Some("bravo"),
            Vec::new(),
        ));
        session.pending_task_wakes.push(TeamTaskWake {
            task_id: 2,
            owner: "bravo".to_string(),
            created_at_ms: 0,
        });
        let done_ids = completed_task_ids(&session);
        prune_stale_task_wakes(&mut session, &done_ids);
        let wake_task_ids = task_wake_ids_for_agent(&session, &agent_key("bravo"));

        assert_eq!(wake_task_ids.into_iter().collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn owned_pending_task_is_runnable_without_explicit_wake() {
        let mut session = test_team_session(vec![test_agent("bravo", TeamAgentStatus::Idle)]);
        session.tasks.push(test_task(
            1,
            TeamTaskStatus::Pending,
            Some("bravo"),
            Vec::new(),
        ));
        let done_ids = completed_task_ids(&session);

        assert_eq!(
            ready_pending_task_id_for_agent(&session, &agent_key("bravo"), &done_ids),
            Some(1)
        );
        assert!(agent_has_runnable_task(
            &session,
            &agent_key("bravo"),
            &done_ids
        ));
    }

    #[test]
    fn sleep_requires_blocked_task_and_no_runnable_owned_work() {
        let mut session = test_team_session(vec![test_agent("bravo", TeamAgentStatus::Idle)]);
        session.tasks.push(test_task(
            1,
            TeamTaskStatus::Blocked,
            Some("bravo"),
            vec![99],
        ));
        let done_ids = completed_task_ids(&session);
        let owner = agent_key("bravo");

        assert!(agent_has_blocked_task(&session, &owner));
        assert!(!agent_has_runnable_task(&session, &owner, &done_ids));

        session.tasks.push(test_task(
            2,
            TeamTaskStatus::Pending,
            Some("bravo"),
            Vec::new(),
        ));

        assert!(agent_has_blocked_task(&session, &owner));
        assert!(agent_has_runnable_task(&session, &owner, &done_ids));
    }

    #[test]
    fn unlocked_task_wake_message_names_task_and_start_command() {
        let task = test_task(2, TeamTaskStatus::Pending, Some("bravo"), Vec::new());

        let message = team_unlocked_task_message(&task);

        assert!(message.contains("#2"));
        assert!(message.contains("task 2"));
        assert!(message.contains("TaskList action=update taskId=2 status=in_progress"));
        assert!(message.contains("do not call TaskList action=list"));
    }

    #[test]
    fn dependency_lock_rejects_in_progress_with_unresolved_blocker() {
        let done_ids = BTreeSet::new();
        let err = validate_task_dependency_lock(
            2,
            &[1],
            &[1],
            Some(TeamTaskStatus::InProgress),
            &done_ids,
        )
        .expect_err("unresolved dependency should block in_progress");

        assert!(err.contains("blocked by #1"));
        assert!(err.contains("status=in_progress"));
    }

    #[test]
    fn dependency_lock_rejects_clearing_unresolved_blocker() {
        let done_ids = BTreeSet::new();
        let err = validate_task_dependency_lock(2, &[1], &[], None, &done_ids)
            .expect_err("unresolved dependency should not be manually cleared");

        assert!(err.contains("still blocked by #1"));
        assert!(err.contains("clearing blockedBy"));
    }

    #[test]
    fn dependency_lock_allows_non_status_update_while_blocked() {
        let done_ids = BTreeSet::new();
        validate_task_dependency_lock(2, &[1], &[1], None, &done_ids)
            .expect("blocked task metadata should remain editable");
    }

    #[test]
    fn refresh_reblocks_in_progress_task_with_unresolved_dependency() {
        let mut session = test_team_session(vec![test_agent("bravo", TeamAgentStatus::Idle)]);
        session.tasks.push(test_task(
            1,
            TeamTaskStatus::Pending,
            Some("alpha"),
            Vec::new(),
        ));
        session.tasks.push(test_task(
            2,
            TeamTaskStatus::InProgress,
            Some("bravo"),
            vec![1],
        ));

        let done_ids = completed_task_ids(&session);
        refresh_unblocked_tasks(&mut session, &done_ids);

        assert_eq!(session.tasks[1].status, TeamTaskStatus::Blocked);
        assert_eq!(session.tasks[1].blocked_by, vec![1]);
    }

    #[test]
    fn refresh_clears_satisfied_dependencies_without_forcing_status() {
        let mut session = test_team_session(vec![test_agent("bravo", TeamAgentStatus::Idle)]);
        session.tasks.push(test_task(
            1,
            TeamTaskStatus::Completed,
            Some("alpha"),
            Vec::new(),
        ));
        session.tasks.push(test_task(
            2,
            TeamTaskStatus::InProgress,
            Some("bravo"),
            vec![1],
        ));

        let done_ids = completed_task_ids(&session);
        refresh_unblocked_tasks(&mut session, &done_ids);

        assert_eq!(session.tasks[1].status, TeamTaskStatus::InProgress);
        assert!(session.tasks[1].blocked_by.is_empty());
    }

    #[test]
    fn broadcast_messages_do_not_wake_idle_agents() {
        let broadcast = TeamQueuedMessage {
            id: 1,
            from: "player".to_string(),
            to: "ui".to_string(),
            target: Some("*".to_string()),
            message: "FYI".to_string(),
            created_at_ms: 0,
        };
        assert!(!queued_message_wakes_agent(&broadcast));
    }

    #[test]
    fn direct_messages_wake_idle_agents() {
        let direct = TeamQueuedMessage {
            id: 1,
            from: "player".to_string(),
            to: "ui".to_string(),
            target: Some("ui".to_string()),
            message: "Need you".to_string(),
            created_at_ms: 0,
        };
        assert!(queued_message_wakes_agent(&direct));
    }

    fn test_team_session(agents: Vec<TeamAgent>) -> TeamSession {
        TeamSession {
            name: "test-team".to_string(),
            description: None,
            created_at_ms: 0,
            updated_at_ms: 0,
            agents: agents
                .into_iter()
                .map(|agent| (agent_key(&agent.name), agent))
                .collect(),
            tasks: Vec::new(),
            next_task_id: 1,
            queued_messages: Vec::new(),
            next_message_id: 1,
            pending_task_wakes: Vec::new(),
            recent_file_changes: Vec::new(),
        }
    }

    fn test_team_tool() -> TeamTool {
        TeamTool::new(
            "test-scope".to_string(),
            PathBuf::from("."),
            String::new(),
            HashMap::new(),
            SubAgentSettings::default(),
            McpSettings::default(),
            ToolSettings::default(),
            SkillSettings::default(),
            ModelRef::new("test", "model"),
            1,
            Arc::new(RwLock::new(TeamRuntime::default())),
            TurnCancel::empty(),
        )
    }

    fn test_agent(name: &str, status: TeamAgentStatus) -> TeamAgent {
        TeamAgent {
            id: format!("{}@test-team", agent_key(name)),
            name: name.to_string(),
            description: String::new(),
            prompt: String::new(),
            model: ModelRef::new("test", "model"),
            status,
            history: Vec::new(),
            created_at_ms: 0,
            updated_at_ms: 0,
            last_summary: None,
            last_error: None,
        }
    }

    fn test_task(
        id: u64,
        status: TeamTaskStatus,
        owner: Option<&str>,
        blocked_by: Vec<u64>,
    ) -> TeamTask {
        TeamTask {
            id,
            subject: format!("task {id}"),
            description: None,
            status,
            owner: owner.map(str::to_string),
            blocked_by,
            created_by: "test".to_string(),
            created_at_ms: 0,
            updated_at_ms: 0,
            completed_at_ms: (status == TeamTaskStatus::Completed).then_some(0),
        }
    }
}
