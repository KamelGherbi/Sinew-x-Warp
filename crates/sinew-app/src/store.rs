use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use directories::ProjectDirs;
use futures_util::StreamExt;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use sinew_core::{
    ChatMessage, Effort, ModelRef, Part, Provider, ProviderRequest, Role, ServiceTier, StreamEvent,
    ToolDescriptor,
};
use uuid::Uuid;

use crate::agent::AgentMode;
use crate::bash::active_shell_display_name;
use crate::mcp::{with_default_mcp_servers, McpSettings};
use crate::skill::SkillSettings;
use crate::subagent::{with_default_sub_agents, SubAgentSettings};
use crate::todo::TodoListState;
use crate::tool_names;
use crate::tool_run::TurnCheckpoint;
use crate::workspace::{workspace_info, WorkspaceInfo};

const DEFAULT_CONVERSATION_TITLE: &str = "New conversation";
const MODE_MODEL_SETTINGS_KEY: &str = "mode_model_settings";
const MCP_SETTINGS_KEY: &str = "mcp_settings";
const SUB_AGENT_SETTINGS_KEY: &str = "sub_agent_settings";
const TOOL_SETTINGS_KEY: &str = "tool_settings";
const SKILL_SETTINGS_KEY: &str = "skill_settings";
const OPENROUTER_MODELS_KEY: &str = "openrouter_models";
const HIDDEN_TOOL_SETTING_NAMES: &[&str] = &["skill"];
const TITLE_MAX_CHARS: usize = 48;
const TITLE_MAX_WORDS: usize = 3;
const TITLE_INPUT_MAX_CHARS: usize = 1_200;
const TITLE_MODEL_TIMEOUT_SECS: u64 = 12;

pub const DEFAULT_PLAN_MODE_PROMPT: &str = r#"You are in Plan mode.

Rules:
- Build understanding by reading/searching/running diagnostic shell commands as needed.
- Do not edit workspace files.
- You must keep the user in a Question loop until the user explicitly clicks "Send and stop questions".
- If the user message does not contain <plan_mode_control action="stop_questions">, your turn must end by calling the Question tool. Do not write the final plan yet.
- After each normal answer to a Question, inspect/explore more if needed, then ask the next Question.
- If you have no remaining substantive question, ask the user to confirm that you should create the plan now. Still use the Question tool.
- Only when the user message contains <plan_mode_control action="stop_questions">, stop asking questions and write the complete plan now.
- When the plan is ready, respond with only the Markdown plan. Do not implement it. The app will save this Markdown into `.sinew/plans/*.md` as the durable plan artifact.
- The Markdown plan must include a final section titled `## Suivi d’exécution` containing a granular checklist of the planned outcomes/steps. Leave every item unchecked (`- [ ] ...`) because implementation has not started yet. Each checklist item must represent one independently verifiable block of work so it can be checked off one by one during implementation. Keep each checklist item phrased as a user-visible outcome or validation point, not as code-level instructions.

STRICTLY FORBIDDEN:
- Low-level implementation details: code snippets, file paths/structures, function/variable names, or shell commands.

REQUIRED:
- All specific technologies, design choices, and parameters agreed upon during brainstorming. Do not invent extra components or options not discussed.

Focus on WHAT the system should do and how components behave, not HOW the code is written. Keep it clear and aligned with the discussed scope.

You may include Mermaid diagrams (in ```mermaid fenced blocks) when a flow, decision tree, sequence, or set of relationships would be clearer as a picture than as prose. Keep diagram labels at the same level of abstraction as the rest of the plan: describe intent and behavior, not files, functions, or implementation details."#;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationSummary {
    pub id: String,
    pub title: String,
    pub updated_at_ms: i64,
    pub archived_at_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSummary {
    pub id: String,
    pub workspace_id: String,
    pub workspace_name: String,
    pub title: String,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub message_count: i64,
    pub archived_at_ms: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct TurnCheckpointRecord {
    pub history_index: usize,
    pub checkpoint: TurnCheckpoint,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SavedConversation {
    pub id: String,
    pub workspace_id: String,
    pub title: String,
    pub model: ModelRef,
    pub mode_model_settings: ModeModelSettings,
    pub system_prompt: String,
    pub todo_list: TodoListState,
    pub plan_workflow: PlanWorkflowState,
    pub goal_workflow: GoalWorkflowState,
    pub history: Vec<ChatMessage>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModeModelSettings {
    pub act: ModelRef,
    pub plan: ModelRef,
    pub goal: ModelRef,
}

impl ModeModelSettings {
    pub fn new(default_model: &ModelRef) -> Self {
        Self {
            act: default_model.clone(),
            plan: default_model.clone(),
            goal: default_model.clone(),
        }
    }

    pub fn get(&self, mode: AgentMode) -> &ModelRef {
        match mode {
            AgentMode::Act => &self.act,
            AgentMode::Plan => &self.plan,
            AgentMode::Goal => &self.goal,
        }
    }

    pub fn set(&mut self, mode: AgentMode, model: ModelRef) {
        match mode {
            AgentMode::Act => self.act = model,
            AgentMode::Plan => self.plan = model,
            AgentMode::Goal => self.goal = model,
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawModeModelSettings {
    act: ModelRef,
    plan: ModelRef,
    #[serde(default)]
    goal: Option<ModelRef>,
}

impl<'de> Deserialize<'de> for ModeModelSettings {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawModeModelSettings::deserialize(deserializer)?;
        Ok(Self {
            goal: raw.goal.unwrap_or_else(|| raw.act.clone()),
            act: raw.act,
            plan: raw.plan,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PlanArtifactState {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub absolute_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "camelCase")]
#[derive(Default)]
pub enum PlanWorkflowState {
    #[default]
    Idle,
    PlanningQuestions,
    PlanReady {
        artifact: PlanArtifactState,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "camelCase")]
#[derive(Default)]
pub enum GoalWorkflowState {
    #[default]
    Idle,
    Active {
        objective: String,
        started_at_ms: i64,
        updated_at_ms: i64,
    },
    Paused {
        objective: String,
        started_at_ms: i64,
        updated_at_ms: i64,
    },
    Complete {
        objective: String,
        started_at_ms: i64,
        completed_at_ms: i64,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceBootstrap {
    pub workspace: WorkspaceInfo,
    pub conversations: Vec<ConversationSummary>,
    pub active_conversation: SavedConversation,
    pub mode_model_settings: ModeModelSettings,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolSettings {
    #[serde(default)]
    pub tools: Vec<ToolConfig>,
    #[serde(default)]
    pub plan_mode_prompt: String,
    #[serde(default)]
    pub image_provider: ImageProvider,
    #[serde(default)]
    pub openai_image_use_subscription: bool,
    #[serde(default)]
    pub openai_image_api_key: String,
    #[serde(default)]
    pub nano_banana_api_key: String,
    #[serde(default)]
    pub web_search_provider: WebSearchProvider,
    #[serde(default)]
    pub linkup_api_key: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageProvider {
    #[default]
    #[serde(rename = "gptImage2")]
    GptImage2,
    #[serde(rename = "nanoBanana2")]
    NanoBanana2,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebSearchProvider {
    #[serde(rename = "linkup")]
    LinkUp,
    #[default]
    #[serde(rename = "classic")]
    Classic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub description_override: bool,
    #[serde(default, skip_serializing)]
    pub default_description: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolSettingsView {
    pub tools: Vec<ToolConfigView>,
    pub plan_mode_prompt: String,
    pub default_plan_mode_prompt: String,
    pub image_provider: ImageProvider,
    pub openai_image_use_subscription: bool,
    pub openai_image_api_key: String,
    pub nano_banana_api_key: String,
    pub web_search_provider: WebSearchProvider,
    pub linkup_api_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct OpenRouterModelRecord {
    pub id: String,
    pub name: String,
    pub context_window: u32,
    pub max_output_tokens: u32,
    #[serde(default)]
    pub supports_images: bool,
    #[serde(default)]
    pub supports_thinking: bool,
    #[serde(default = "default_enabled")]
    pub supports_tools: bool,
    #[serde(default)]
    pub added_at_ms: i64,
}

impl OpenRouterModelRecord {
    pub fn normalized(mut self) -> Option<Self> {
        self.id = self.id.trim().to_string();
        self.name = self.name.trim().to_string();
        if self.id.is_empty() {
            return None;
        }
        if self.name.is_empty() {
            self.name = self.id.clone();
        }
        self.context_window = self.context_window.max(1);
        self.max_output_tokens = self.max_output_tokens.max(1).min(self.context_window);
        if self.added_at_ms <= 0 {
            self.added_at_ms = now_ms();
        }
        Some(self)
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfigView {
    pub name: String,
    pub display_name: String,
    pub description: String,
    pub default_description: String,
    pub enabled: bool,
}

impl ToolSettings {
    pub fn normalized(mut self) -> Self {
        let mut seen = HashSet::new();
        self.plan_mode_prompt = normalize_plan_mode_prompt(&self.plan_mode_prompt);
        self.openai_image_api_key = self.openai_image_api_key.trim().to_string();
        self.nano_banana_api_key = self.nano_banana_api_key.trim().to_string();
        self.linkup_api_key = self.linkup_api_key.trim().to_string();
        self.tools = self
            .tools
            .into_iter()
            .filter_map(|mut tool| {
                tool.name = tool_names::canonical_tool_name(tool.name.trim()).to_string();
                if tool.name.is_empty()
                    || HIDDEN_TOOL_SETTING_NAMES.contains(&tool.name.as_str())
                    || !seen.insert(tool.name.clone())
                {
                    return None;
                }
                tool.default_description.clear();
                if !tool.description_override {
                    tool.description.clear();
                }
                Some(tool)
            })
            .collect();
        self
    }

    pub fn normalized_for_catalog(mut self, catalog: &[ToolDescriptor]) -> Self {
        let defaults = catalog
            .iter()
            .map(|descriptor| (descriptor.name.as_str(), descriptor.description.as_str()))
            .collect::<HashMap<_, _>>();

        for tool in &mut self.tools {
            let canonical_name = tool_names::canonical_tool_name(tool.name.trim()).to_string();
            tool.name = canonical_name.clone();
            if let Some(default_description) =
                defaults.get(canonical_name.as_str()).copied().or_else(|| {
                    (!tool.default_description.is_empty())
                        .then_some(tool.default_description.as_str())
                })
            {
                tool.description_override = tool.description != default_description;
            }
        }

        self.normalized()
    }

    pub fn apply_to_descriptors(&self, descriptors: Vec<ToolDescriptor>) -> Vec<ToolDescriptor> {
        let by_name = self
            .tools
            .iter()
            .map(|tool| (tool.name.as_str(), tool))
            .collect::<HashMap<_, _>>();

        descriptors
            .into_iter()
            .filter_map(|mut descriptor| {
                let setting = by_name.get(descriptor.name.as_str());
                let enabled = setting
                    .map(|tool| tool.enabled)
                    .unwrap_or_else(|| default_tool_enabled(&descriptor.name));
                if !enabled {
                    return None;
                }
                if let Some(setting) = setting.filter(|tool| tool.description_override) {
                    descriptor.description = setting.description.clone();
                }
                Some(descriptor)
            })
            .collect()
    }

    pub fn plan_mode_prompt(&self) -> &str {
        let prompt = self.plan_mode_prompt.trim();
        if prompt.is_empty() {
            DEFAULT_PLAN_MODE_PROMPT
        } else {
            prompt
        }
    }

    pub fn is_enabled(&self, name: &str) -> bool {
        self.tools
            .iter()
            .find(|tool| tool.name == tool_names::canonical_tool_name(name))
            .map(|tool| tool.enabled)
            .unwrap_or_else(|| default_tool_enabled(tool_names::canonical_tool_name(name)))
    }

    pub fn openai_image_api_key(&self) -> Option<String> {
        let key = self.openai_image_api_key.trim();
        if key.is_empty() {
            None
        } else {
            Some(key.to_string())
        }
    }

    pub fn nano_banana_api_key(&self) -> Option<String> {
        let key = self.nano_banana_api_key.trim();
        if key.is_empty() {
            None
        } else {
            Some(key.to_string())
        }
    }

    pub fn linkup_api_key(&self) -> Option<String> {
        let key = self.linkup_api_key.trim();
        if key.is_empty() {
            None
        } else {
            Some(key.to_string())
        }
    }
}

fn normalize_plan_mode_prompt(value: &str) -> String {
    let prompt = value.trim();
    if prompt.is_empty() || prompt == DEFAULT_PLAN_MODE_PROMPT.trim() {
        String::new()
    } else {
        prompt.to_string()
    }
}

pub fn tool_settings_view(settings: &ToolSettings, catalog: &[ToolDescriptor]) -> ToolSettingsView {
    let by_name = settings
        .tools
        .iter()
        .map(|tool| (tool.name.as_str(), tool))
        .collect::<HashMap<_, _>>();
    let mut seen = HashSet::new();

    ToolSettingsView {
        plan_mode_prompt: settings.plan_mode_prompt().to_string(),
        default_plan_mode_prompt: DEFAULT_PLAN_MODE_PROMPT.to_string(),
        image_provider: settings.image_provider,
        openai_image_use_subscription: settings.openai_image_use_subscription,
        openai_image_api_key: settings.openai_image_api_key.clone(),
        nano_banana_api_key: settings.nano_banana_api_key.clone(),
        web_search_provider: settings.web_search_provider,
        linkup_api_key: settings.linkup_api_key.clone(),
        tools: catalog
            .iter()
            .filter_map(|descriptor| {
                if !seen.insert(descriptor.name.clone()) {
                    return None;
                }
                let setting = by_name.get(descriptor.name.as_str());
                Some(ToolConfigView {
                    name: descriptor.name.clone(),
                    display_name: tool_display_name(&descriptor.name),
                    description: setting
                        .filter(|tool| tool.description_override)
                        .map(|tool| tool.description.clone())
                        .unwrap_or_else(|| descriptor.description.clone()),
                    default_description: descriptor.description.clone(),
                    enabled: setting
                        .map(|tool| tool.enabled)
                        .unwrap_or_else(|| default_tool_enabled(&descriptor.name)),
                })
            })
            .collect(),
    }
}

fn tool_display_name(name: &str) -> String {
    match name {
        "bash" => active_shell_display_name().to_string(),
        "bash_input" => format!("{} input", active_shell_display_name()),
        _ => default_tool_display_name(name),
    }
}

fn default_tool_display_name(name: &str) -> String {
    match name {
        "read" => "Read".to_string(),
        "edit_file" => "Edit file".to_string(),
        "write_file" => "Write file".to_string(),
        "glob" => "Glob".to_string(),
        "grep" => "Grep".to_string(),
        "web_search" => "Web search".to_string(),
        "web_fetch" => "Web fetch".to_string(),
        "create_image" => "Create image".to_string(),
        "question" => "Question".to_string(),
        "todo_list" => "To-do list".to_string(),
        "load_mcp_tool" => "Load MCP tool".to_string(),
        "skill" => "Load skill".to_string(),
        "team_run" => "Team run".to_string(),
        "team_status" => "Team status".to_string(),
        "team_stop" => "Team stop".to_string(),
        "send_message" => "Send message".to_string(),
        "clean_context" => "Clean context".to_string(),
        "update_goal" => "Update goal".to_string(),
        "context_compaction" => "Compact context".to_string(),
        _ => humanize_tool_name(name),
    }
}

fn humanize_tool_name(name: &str) -> String {
    let mut out = String::new();
    let mut previous_was_separator = true;
    let mut previous_was_lowercase = false;

    for ch in name.chars() {
        if ch == '_' || ch == '-' || ch.is_whitespace() {
            if !out.ends_with(' ') && !out.is_empty() {
                out.push(' ');
            }
            previous_was_separator = true;
            previous_was_lowercase = false;
            continue;
        }
        if ch.is_uppercase() && previous_was_lowercase && !out.ends_with(' ') {
            out.push(' ');
        }
        if previous_was_separator {
            out.extend(ch.to_uppercase());
        } else {
            out.extend(ch.to_lowercase());
        }
        previous_was_separator = false;
        previous_was_lowercase = ch.is_lowercase();
    }

    let trimmed = out.trim();
    if trimmed.is_empty() {
        name.to_string()
    } else {
        trimmed.to_string()
    }
}

#[derive(Debug, Clone)]
pub struct AppStore {
    path: PathBuf,
}

impl AppStore {
    pub fn open_default() -> Result<Self> {
        let dirs = ProjectDirs::from("dev", "hyrak", "sinew")
            .context("unable to resolve local data directory")?;
        std::fs::create_dir_all(dirs.data_local_dir())
            .context("unable to create local data directory")?;

        let store = Self {
            path: dirs.data_local_dir().join("desktop-state.sqlite3"),
        };
        store.migrate()?;
        Ok(store)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn bootstrap_workspace(
        &self,
        workspace_root: &Path,
        default_model: &ModelRef,
        default_system: &str,
    ) -> Result<WorkspaceBootstrap> {
        let workspace_id = workspace_root.display().to_string();
        let mode_model_settings = self.load_mode_model_settings(default_model)?;
        let mut conversations = self.list_conversations(&workspace_id)?;
        let active_conversation = if let Some(first) = conversations.first() {
            self.load_conversation(&workspace_id, &first.id)?
                .context("conversation listed in index but missing from store")?
        } else {
            let created = self.create_conversation(&workspace_id, default_model, default_system)?;
            conversations = self.list_conversations(&workspace_id)?;
            created
        };

        Ok(WorkspaceBootstrap {
            workspace: workspace_info(workspace_root),
            conversations,
            active_conversation,
            mode_model_settings,
        })
    }

    pub fn create_conversation(
        &self,
        workspace_id: &str,
        default_model: &ModelRef,
        default_system: &str,
    ) -> Result<SavedConversation> {
        let id = Uuid::new_v4().to_string();
        let now = now_ms();
        let title = DEFAULT_CONVERSATION_TITLE.to_string();
        let todo_list = TodoListState::default();
        let todo_list_json = serde_json::to_string(&todo_list)?;
        let plan_workflow = PlanWorkflowState::default();
        let plan_workflow_json = serde_json::to_string(&plan_workflow)?;
        let goal_workflow = GoalWorkflowState::default();
        let goal_workflow_json = serde_json::to_string(&goal_workflow)?;
        let mode_model_settings = self.load_mode_model_settings(default_model)?;
        let conversation_model = mode_model_settings.act.clone();
        let mode_model_settings_json = serde_json::to_string(&mode_model_settings)?;
        let conn = self.connection()?;
        conn.execute(
            "insert into conversations (id, workspace_id, title, title_initialized, model_json, mode_model_settings_json, system_prompt, todo_list_json, plan_workflow_json, goal_workflow_json, created_at_ms, updated_at_ms, archived_at_ms)
             values (?1, ?2, ?3, 0, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, null)",
            params![
                &id,
                workspace_id,
                &title,
                serde_json::to_string(&conversation_model)?,
                mode_model_settings_json,
                default_system,
                todo_list_json,
                plan_workflow_json,
                goal_workflow_json,
                now,
                now,
            ],
        )
        .context("unable to insert conversation")?;

        Ok(SavedConversation {
            id,
            workspace_id: workspace_id.to_string(),
            title,
            model: conversation_model,
            mode_model_settings,
            system_prompt: default_system.to_string(),
            todo_list,
            plan_workflow,
            goal_workflow,
            history: Vec::new(),
        })
    }

    pub fn list_conversations(&self, workspace_id: &str) -> Result<Vec<ConversationSummary>> {
        let conn = self.connection()?;
        let mut statement = conn
            .prepare(
                "select id, title, updated_at_ms, archived_at_ms from conversations
                 where workspace_id = ?1
                   and archived_at_ms is null
                 order by updated_at_ms desc",
            )
            .context("unable to prepare conversation list query")?;

        let rows = statement
            .query_map(params![workspace_id], |row| {
                Ok(ConversationSummary {
                    id: row.get(0)?,
                    title: row.get(1)?,
                    updated_at_ms: row.get(2)?,
                    archived_at_ms: row.get(3)?,
                })
            })
            .context("unable to read conversation list")?;

        let mut conversations = Vec::new();
        for row in rows {
            conversations.push(row.context("bad conversation row")?);
        }
        Ok(conversations)
    }

    pub fn list_sessions(
        &self,
        query: Option<&str>,
        limit: usize,
        archived: bool,
    ) -> Result<Vec<SessionSummary>> {
        let conn = self.connection()?;
        let limit = limit.clamp(1, 500) as i64;
        let trimmed_query = query.unwrap_or_default().trim().to_lowercase();
        let like_query = if trimmed_query.is_empty() {
            None
        } else {
            Some(format!("%{}%", trimmed_query))
        };

        let mut statement = conn
            .prepare(
                "select c.id, c.workspace_id, c.title, c.created_at_ms, c.updated_at_ms, count(m.ordinal) as message_count, c.archived_at_ms
                 from conversations c
                 left join messages m on m.conversation_id = c.id
                 where (?1 is null or lower(c.title) like ?1 or lower(c.workspace_id) like ?1)
                   and ((?3 = 1 and c.archived_at_ms is not null) or (?3 = 0 and c.archived_at_ms is null))
                 group by c.id, c.workspace_id, c.title, c.created_at_ms, c.updated_at_ms, c.archived_at_ms
                 order by case when ?3 = 1 then c.archived_at_ms else c.updated_at_ms end desc
                 limit ?2",
            )
            .context("unable to prepare session list query")?;

        let rows = statement
            .query_map(
                params![like_query, limit, if archived { 1 } else { 0 }],
                |row| {
                    let workspace_id: String = row.get(1)?;
                    Ok(SessionSummary {
                        id: row.get(0)?,
                        workspace_name: workspace_name_from_id(&workspace_id),
                        workspace_id,
                        title: row.get(2)?,
                        created_at_ms: row.get(3)?,
                        updated_at_ms: row.get(4)?,
                        message_count: row.get(5)?,
                        archived_at_ms: row.get(6)?,
                    })
                },
            )
            .context("unable to read session list")?;

        let mut sessions = Vec::new();
        for row in rows {
            sessions.push(row.context("bad session row")?);
        }
        Ok(sessions)
    }

    pub fn load_conversation(
        &self,
        workspace_id: &str,
        id: &str,
    ) -> Result<Option<SavedConversation>> {
        let conn = self.connection()?;
        let conversation = conn
            .query_row(
                "select title, model_json, system_prompt, todo_list_json, plan_workflow_json, mode_model_settings_json, goal_workflow_json from conversations where workspace_id = ?1 and id = ?2",
                params![workspace_id, id],
                |row| {
                    let model_json: String = row.get(1)?;
                    let todo_list_json: String = row.get(3)?;
                    let plan_workflow_json: String = row.get(4)?;
                    let mode_model_settings_json: Option<String> = row.get(5)?;
                    let goal_workflow_json: String = row.get(6)?;
                    let model = serde_json::from_str::<ModelRef>(&model_json).map_err(|err| {
                        rusqlite::Error::FromSqlConversionFailure(
                            1,
                            rusqlite::types::Type::Text,
                            Box::new(err),
                        )
                    })?;
                    let mode_model_settings = mode_model_settings_json
                        .and_then(|json| serde_json::from_str::<ModeModelSettings>(&json).ok())
                        .unwrap_or_else(|| ModeModelSettings::new(&model));
                    let mut todo_list = serde_json::from_str::<TodoListState>(&todo_list_json)
                        .unwrap_or_default();
                    todo_list.normalize();
                    Ok((
                        row.get::<_, String>(0)?,
                        model,
                        mode_model_settings,
                        row.get::<_, String>(2)?,
                        todo_list,
                        serde_json::from_str::<PlanWorkflowState>(&plan_workflow_json)
                            .unwrap_or_default(),
                        serde_json::from_str::<GoalWorkflowState>(&goal_workflow_json)
                            .unwrap_or_default(),
                    ))
                },
            )
            .optional()
            .context("unable to load conversation metadata")?;

        let Some((
            title,
            model,
            mode_model_settings,
            system_prompt,
            todo_list,
            plan_workflow,
            goal_workflow,
        )) = conversation
        else {
            return Ok(None);
        };

        let mut statement = conn
            .prepare(
                "select message_json from messages
                 where conversation_id = ?1
                 order by ordinal asc",
            )
            .context("unable to prepare message query")?;
        let rows = statement
            .query_map(params![id], |row| {
                let message_json: String = row.get(0)?;
                serde_json::from_str::<ChatMessage>(&message_json).map_err(|err| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(err),
                    )
                })
            })
            .context("unable to read stored messages")?;

        let mut history = Vec::new();
        for row in rows {
            history.push(row.context("bad stored message")?);
        }

        Ok(Some(SavedConversation {
            id: id.to_string(),
            workspace_id: workspace_id.to_string(),
            title,
            model,
            mode_model_settings,
            system_prompt,
            todo_list,
            plan_workflow,
            goal_workflow,
            history,
        }))
    }

    pub fn save_conversation(&self, conversation: &SavedConversation) -> Result<()> {
        let now = now_ms();
        let mut todo_list = conversation.todo_list.clone();
        todo_list.normalize();
        let todo_list_json = serde_json::to_string(&todo_list)?;
        let plan_workflow_json = serde_json::to_string(&conversation.plan_workflow)?;
        let goal_workflow_json = serde_json::to_string(&conversation.goal_workflow)?;
        let mode_model_settings_json = serde_json::to_string(&conversation.mode_model_settings)?;
        let mut conn = self.connection()?;
        let tx = conn
            .transaction()
            .context("unable to open sqlite transaction")?;
        let current_title_state =
            load_conversation_title_state(&tx, &conversation.workspace_id, &conversation.id)?;
        let title_state = resolve_title_for_save(
            current_title_state.as_ref(),
            &conversation.title,
            &conversation.history,
        );

        tx.execute(
            "update conversations
             set title = ?2, model_json = ?3, system_prompt = ?4, updated_at_ms = ?5, todo_list_json = ?6, plan_workflow_json = ?7, mode_model_settings_json = ?8, goal_workflow_json = ?9, title_initialized = ?10
             where id = ?1 and workspace_id = ?11",
            params![
                &conversation.id,
                &title_state.title,
                serde_json::to_string(&conversation.model)?,
                &conversation.system_prompt,
                now,
                todo_list_json,
                plan_workflow_json,
                mode_model_settings_json,
                goal_workflow_json,
                title_state.initialized as i64,
                &conversation.workspace_id,
            ],
        )
        .context("unable to update conversation")?;

        tx.execute(
            "delete from messages where conversation_id = ?1",
            params![&conversation.id],
        )
        .context("unable to clear previous conversation messages")?;

        for (ordinal, message) in conversation.history.iter().enumerate() {
            tx.execute(
                "insert into messages (conversation_id, ordinal, message_json) values (?1, ?2, ?3)",
                params![
                    &conversation.id,
                    ordinal as i64,
                    serde_json::to_string(message)?
                ],
            )
            .context("unable to write conversation message")?;
        }

        tx.commit()
            .context("unable to commit conversation transaction")?;
        Ok(())
    }

    pub fn save_conversation_and_mode_model_settings(
        &self,
        conversation: &SavedConversation,
        settings: &ModeModelSettings,
    ) -> Result<()> {
        let now = now_ms();
        let mut todo_list = conversation.todo_list.clone();
        todo_list.normalize();
        let todo_list_json = serde_json::to_string(&todo_list)?;
        let plan_workflow_json = serde_json::to_string(&conversation.plan_workflow)?;
        let goal_workflow_json = serde_json::to_string(&conversation.goal_workflow)?;
        let mode_model_settings_json = serde_json::to_string(&conversation.mode_model_settings)?;
        let default_settings_json = serde_json::to_string(settings)?;
        let mut conn = self.connection()?;
        let tx = conn
            .transaction()
            .context("unable to open sqlite transaction")?;
        let current_title_state =
            load_conversation_title_state(&tx, &conversation.workspace_id, &conversation.id)?;
        let title_state = resolve_title_for_save(
            current_title_state.as_ref(),
            &conversation.title,
            &conversation.history,
        );

        tx.execute(
            "update conversations
             set title = ?2, model_json = ?3, system_prompt = ?4, updated_at_ms = ?5, todo_list_json = ?6, plan_workflow_json = ?7, mode_model_settings_json = ?8, goal_workflow_json = ?9, title_initialized = ?10
             where id = ?1 and workspace_id = ?11",
            params![
                &conversation.id,
                &title_state.title,
                serde_json::to_string(&conversation.model)?,
                &conversation.system_prompt,
                now,
                todo_list_json,
                plan_workflow_json,
                mode_model_settings_json,
                goal_workflow_json,
                title_state.initialized as i64,
                &conversation.workspace_id,
            ],
        )
        .context("unable to update conversation")?;

        tx.execute(
            "delete from messages where conversation_id = ?1",
            params![&conversation.id],
        )
        .context("unable to clear previous conversation messages")?;

        for (ordinal, message) in conversation.history.iter().enumerate() {
            tx.execute(
                "insert into messages (conversation_id, ordinal, message_json) values (?1, ?2, ?3)",
                params![
                    &conversation.id,
                    ordinal as i64,
                    serde_json::to_string(message)?
                ],
            )
            .context("unable to write conversation message")?;
        }

        tx.execute(
            "insert into app_settings (key, value_json, updated_at_ms)
             values (?1, ?2, ?3)
             on conflict(key) do update set
                value_json = excluded.value_json,
                updated_at_ms = excluded.updated_at_ms",
            params![MODE_MODEL_SETTINGS_KEY, default_settings_json, now],
        )
        .context("unable to save mode model settings")?;

        tx.commit()
            .context("unable to commit conversation/settings transaction")?;
        Ok(())
    }

    pub fn append_conversation_message(
        &self,
        workspace_id: &str,
        conversation_id: &str,
        message: &ChatMessage,
    ) -> Result<()> {
        let now = now_ms();
        let mut conn = self.connection()?;
        let tx = conn
            .transaction()
            .context("unable to open sqlite transaction")?;
        let next_ordinal: i64 = tx
            .query_row(
                "select coalesce(max(ordinal) + 1, 0) from messages where conversation_id = ?1",
                params![conversation_id],
                |row| row.get(0),
            )
            .context("unable to read next message ordinal")?;
        tx.execute(
            "insert into messages (conversation_id, ordinal, message_json) values (?1, ?2, ?3)",
            params![
                conversation_id,
                next_ordinal,
                serde_json::to_string(message)?
            ],
        )
        .context("unable to append conversation message")?;
        tx.execute(
            "update conversations set updated_at_ms = ?3 where workspace_id = ?1 and id = ?2",
            params![workspace_id, conversation_id, now],
        )
        .context("unable to update conversation timestamp")?;
        tx.commit()
            .context("unable to commit append message transaction")?;
        Ok(())
    }

    pub fn save_turn_checkpoint(
        &self,
        conversation_id: &str,
        history_index: usize,
        checkpoint: &TurnCheckpoint,
    ) -> Result<()> {
        let conn = self.connection()?;
        if checkpoint.files.is_empty() {
            conn.execute(
                "delete from turn_checkpoints where conversation_id = ?1 and history_index = ?2",
                params![conversation_id, history_index as i64],
            )
            .context("unable to clear empty turn checkpoint")?;
            return Ok(());
        }

        conn.execute(
            "insert into turn_checkpoints (conversation_id, history_index, checkpoint_json)
             values (?1, ?2, ?3)
             on conflict(conversation_id, history_index) do update set
                checkpoint_json = excluded.checkpoint_json",
            params![
                conversation_id,
                history_index as i64,
                serde_json::to_string(checkpoint)?,
            ],
        )
        .context("unable to save turn checkpoint")?;
        Ok(())
    }

    pub fn load_turn_checkpoints_from(
        &self,
        conversation_id: &str,
        history_index: usize,
    ) -> Result<Vec<TurnCheckpointRecord>> {
        let conn = self.connection()?;
        let mut statement = conn
            .prepare(
                "select history_index, checkpoint_json from turn_checkpoints
                 where conversation_id = ?1 and history_index >= ?2
                 order by history_index asc",
            )
            .context("unable to prepare turn checkpoint query")?;
        let rows = statement
            .query_map(params![conversation_id, history_index as i64], |row| {
                let checkpoint_json: String = row.get(1)?;
                let checkpoint =
                    serde_json::from_str::<TurnCheckpoint>(&checkpoint_json).map_err(|err| {
                        rusqlite::Error::FromSqlConversionFailure(
                            1,
                            rusqlite::types::Type::Text,
                            Box::new(err),
                        )
                    })?;
                let stored_index: i64 = row.get(0)?;
                Ok(TurnCheckpointRecord {
                    history_index: stored_index.max(0) as usize,
                    checkpoint,
                })
            })
            .context("unable to read turn checkpoints")?;

        let mut records = Vec::new();
        for row in rows {
            records.push(row.context("bad turn checkpoint row")?);
        }
        Ok(records)
    }

    pub fn delete_turn_checkpoints_from(
        &self,
        conversation_id: &str,
        history_index: usize,
    ) -> Result<()> {
        let conn = self.connection()?;
        conn.execute(
            "delete from turn_checkpoints where conversation_id = ?1 and history_index >= ?2",
            params![conversation_id, history_index as i64],
        )
        .context("unable to delete turn checkpoints")?;
        Ok(())
    }

    pub fn load_mode_model_settings(&self, default_model: &ModelRef) -> Result<ModeModelSettings> {
        let conn = self.connection()?;
        load_mode_model_settings_from_conn(&conn, default_model)
    }

    pub fn save_mode_model_settings(&self, settings: &ModeModelSettings) -> Result<()> {
        let conn = self.connection()?;
        conn.execute(
            "insert into app_settings (key, value_json, updated_at_ms)
             values (?1, ?2, ?3)
             on conflict(key) do update set
                value_json = excluded.value_json,
                updated_at_ms = excluded.updated_at_ms",
            params![
                MODE_MODEL_SETTINGS_KEY,
                serde_json::to_string(settings)?,
                now_ms(),
            ],
        )
        .context("unable to save mode model settings")?;
        Ok(())
    }

    pub fn load_mcp_settings(&self) -> Result<McpSettings> {
        let conn = self.connection()?;
        let stored = conn
            .query_row(
                "select value_json from app_settings where key = ?1",
                params![MCP_SETTINGS_KEY],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .context("unable to read MCP settings")?;

        if let Some(json) = stored {
            if let Ok(settings) = serde_json::from_str::<McpSettings>(&json) {
                return Ok(with_default_mcp_servers(settings));
            }
        }

        Ok(with_default_mcp_servers(McpSettings::default()))
    }

    pub fn save_mcp_settings(&self, settings: &McpSettings) -> Result<()> {
        let conn = self.connection()?;
        conn.execute(
            "insert into app_settings (key, value_json, updated_at_ms)
             values (?1, ?2, ?3)
             on conflict(key) do update set
                value_json = excluded.value_json,
                updated_at_ms = excluded.updated_at_ms",
            params![MCP_SETTINGS_KEY, serde_json::to_string(settings)?, now_ms()],
        )
        .context("unable to save MCP settings")?;
        Ok(())
    }

    pub fn load_tool_settings(&self) -> Result<ToolSettings> {
        let conn = self.connection()?;
        let stored = conn
            .query_row(
                "select value_json from app_settings where key = ?1",
                params![TOOL_SETTINGS_KEY],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .context("unable to read tool settings")?;

        if let Some(json) = stored {
            if let Ok(settings) = serde_json::from_str::<ToolSettings>(&json) {
                return Ok(settings.normalized());
            }
        }

        Ok(ToolSettings::default())
    }

    pub fn save_tool_settings(&self, settings: &ToolSettings) -> Result<ToolSettings> {
        self.save_tool_settings_for_catalog(settings, &[])
    }

    pub fn save_tool_settings_for_catalog(
        &self,
        settings: &ToolSettings,
        catalog: &[ToolDescriptor],
    ) -> Result<ToolSettings> {
        let normalized = settings.clone().normalized_for_catalog(catalog);
        let conn = self.connection()?;
        conn.execute(
            "insert into app_settings (key, value_json, updated_at_ms)
             values (?1, ?2, ?3)
             on conflict(key) do update set
                value_json = excluded.value_json,
                updated_at_ms = excluded.updated_at_ms",
            params![
                TOOL_SETTINGS_KEY,
                serde_json::to_string(&normalized)?,
                now_ms()
            ],
        )
        .context("unable to save tool settings")?;
        Ok(normalized)
    }

    pub fn load_skill_settings(&self) -> Result<SkillSettings> {
        let conn = self.connection()?;
        let stored = conn
            .query_row(
                "select value_json from app_settings where key = ?1",
                params![SKILL_SETTINGS_KEY],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .context("unable to read skill settings")?;

        if let Some(json) = stored {
            if let Ok(settings) = serde_json::from_str::<SkillSettings>(&json) {
                return Ok(settings.normalized());
            }
        }

        Ok(SkillSettings::default())
    }

    pub fn save_skill_settings(&self, settings: &SkillSettings) -> Result<SkillSettings> {
        let normalized = settings.clone().normalized();
        let conn = self.connection()?;
        conn.execute(
            "insert into app_settings (key, value_json, updated_at_ms)
             values (?1, ?2, ?3)
             on conflict(key) do update set
                value_json = excluded.value_json,
                updated_at_ms = excluded.updated_at_ms",
            params![
                SKILL_SETTINGS_KEY,
                serde_json::to_string(&normalized)?,
                now_ms()
            ],
        )
        .context("unable to save skill settings")?;
        Ok(normalized)
    }

    pub fn load_sub_agent_settings(&self) -> Result<SubAgentSettings> {
        let conn = self.connection()?;
        let stored = conn
            .query_row(
                "select value_json from app_settings where key = ?1",
                params![SUB_AGENT_SETTINGS_KEY],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .context("unable to read sub-agent settings")?;

        if let Some(json) = stored {
            if let Ok(settings) = serde_json::from_str::<SubAgentSettings>(&json) {
                return Ok(with_default_sub_agents(settings));
            }
        }

        Ok(with_default_sub_agents(SubAgentSettings::default()))
    }

    pub fn load_openrouter_models(&self) -> Result<Vec<OpenRouterModelRecord>> {
        let conn = self.connection()?;
        let stored = conn
            .query_row(
                "select value_json from app_settings where key = ?1",
                params![OPENROUTER_MODELS_KEY],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .context("unable to read OpenRouter model list")?;

        if let Some(json) = stored {
            if let Ok(models) = serde_json::from_str::<Vec<OpenRouterModelRecord>>(&json) {
                return Ok(normalize_openrouter_models(models));
            }
        }

        Ok(Vec::new())
    }

    pub fn save_openrouter_models(
        &self,
        models: &[OpenRouterModelRecord],
    ) -> Result<Vec<OpenRouterModelRecord>> {
        let normalized = normalize_openrouter_models(models.to_vec());
        let conn = self.connection()?;
        conn.execute(
            "insert into app_settings (key, value_json, updated_at_ms)
             values (?1, ?2, ?3)
             on conflict(key) do update set
                value_json = excluded.value_json,
                updated_at_ms = excluded.updated_at_ms",
            params![
                OPENROUTER_MODELS_KEY,
                serde_json::to_string(&normalized)?,
                now_ms(),
            ],
        )
        .context("unable to save OpenRouter model list")?;
        Ok(normalized)
    }

    pub fn add_openrouter_model(
        &self,
        model: OpenRouterModelRecord,
    ) -> Result<Vec<OpenRouterModelRecord>> {
        let Some(model) = model.normalized() else {
            anyhow::bail!("OpenRouter model id cannot be empty");
        };
        let mut models = self.load_openrouter_models()?;
        if !models.iter().any(|existing| existing.id == model.id) {
            models.push(model);
        }
        self.save_openrouter_models(&models)
    }

    pub fn remove_openrouter_model(&self, id: &str) -> Result<Vec<OpenRouterModelRecord>> {
        let id = id.trim();
        let models = self
            .load_openrouter_models()?
            .into_iter()
            .filter(|model| model.id != id)
            .collect::<Vec<_>>();
        self.save_openrouter_models(&models)
    }

    pub fn save_sub_agent_settings(&self, settings: &SubAgentSettings) -> Result<SubAgentSettings> {
        let normalized = settings.clone().normalized();
        let conn = self.connection()?;
        conn.execute(
            "insert into app_settings (key, value_json, updated_at_ms)
             values (?1, ?2, ?3)
             on conflict(key) do update set
                value_json = excluded.value_json,
                updated_at_ms = excluded.updated_at_ms",
            params![
                SUB_AGENT_SETTINGS_KEY,
                serde_json::to_string(&normalized)?,
                now_ms(),
            ],
        )
        .context("unable to save sub-agent settings")?;
        Ok(normalized)
    }

    pub fn update_generated_conversation_title(
        &self,
        workspace_id: &str,
        id: &str,
        expected_current_title: &str,
        generated_title: &str,
    ) -> Result<Option<i64>> {
        let generated_title = generated_title.trim();
        if generated_title.is_empty() {
            return Ok(None);
        }

        let conn = self.connection()?;
        let current_title = conn
            .query_row(
                "select title from conversations where workspace_id = ?1 and id = ?2",
                params![workspace_id, id],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .context("unable to read conversation title")?;
        let Some(current_title) = current_title else {
            return Ok(None);
        };
        if current_title.trim() != expected_current_title.trim()
            || current_title.trim() == generated_title
        {
            return Ok(None);
        }

        let updated_at_ms = now_ms();
        let changed = conn
            .execute(
                "update conversations set title = ?4, updated_at_ms = ?5 where workspace_id = ?1 and id = ?2 and title = ?3",
                params![workspace_id, id, current_title, generated_title, updated_at_ms],
            )
            .context("unable to update generated conversation title")?;
        if changed == 0 {
            return Ok(None);
        }
        Ok(Some(updated_at_ms))
    }

    pub fn rename_conversation(&self, workspace_id: &str, id: &str, title: &str) -> Result<()> {
        let conn = self.connection()?;
        conn.execute(
            "update conversations set title = ?3, title_initialized = 1, updated_at_ms = ?4 where workspace_id = ?1 and id = ?2",
            params![workspace_id, id, title.trim(), now_ms()],
        )
        .context("unable to rename conversation")?;
        Ok(())
    }

    pub fn delete_conversation(&self, workspace_id: &str, id: &str) -> Result<()> {
        let conn = self.connection()?;
        conn.execute(
            "delete from conversations where workspace_id = ?1 and id = ?2",
            params![workspace_id, id],
        )
        .context("unable to delete conversation")?;
        Ok(())
    }

    pub fn archive_conversation(&self, workspace_id: &str, id: &str) -> Result<()> {
        let now = now_ms();
        let conn = self.connection()?;
        conn.execute(
            "update conversations
             set archived_at_ms = coalesce(archived_at_ms, ?3), updated_at_ms = ?3
             where workspace_id = ?1 and id = ?2",
            params![workspace_id, id, now],
        )
        .context("unable to archive conversation")?;
        Ok(())
    }

    pub fn restore_conversation(&self, workspace_id: &str, id: &str) -> Result<()> {
        let conn = self.connection()?;
        conn.execute(
            "update conversations
             set archived_at_ms = null, updated_at_ms = ?3
             where workspace_id = ?1 and id = ?2",
            params![workspace_id, id, now_ms()],
        )
        .context("unable to restore conversation")?;
        Ok(())
    }

    pub fn load_conversation_model_by_id(&self, id: &str) -> Result<Option<ModelRef>> {
        let conn = self.connection()?;
        conn.query_row(
            "select model_json from conversations where id = ?1",
            params![id],
            |row| {
                let model_json: String = row.get(0)?;
                serde_json::from_str::<ModelRef>(&model_json).map_err(|err| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(err),
                    )
                })
            },
        )
        .optional()
        .context("unable to load conversation model")
    }

    fn migrate(&self) -> Result<()> {
        let conn = self.connection()?;
        let version: i64 = conn
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .unwrap_or(0);

        if version >= 9 {
            return Ok(());
        }

        if version < 2 {
            conn.execute_batch(
                "
            create table if not exists conversations (
                id text primary key,
                workspace_id text not null,
                title text not null,
                title_initialized integer not null default 0,
                model_json text not null,
                mode_model_settings_json text,
                system_prompt text not null,
                todo_list_json text not null default '{\"active\":false,\"tasks\":[],\"nextId\":1}',
                plan_workflow_json text not null default '{\"status\":\"idle\"}',
                goal_workflow_json text not null default '{\"status\":\"idle\"}',
                created_at_ms integer not null,
                updated_at_ms integer not null,
                archived_at_ms integer
            );

            create table if not exists messages (
                conversation_id text not null,
                ordinal integer not null,
                message_json text not null,
                primary key (conversation_id, ordinal),
                foreign key (conversation_id) references conversations(id) on delete cascade
            );

            create index if not exists idx_conversations_workspace_updated
                on conversations(workspace_id, updated_at_ms desc);

            create table if not exists app_settings (
                key text primary key,
                value_json text not null,
                updated_at_ms integer not null
            );
            ",
            )
            .context("unable to migrate sqlite schema")?;
        }
        ensure_conversations_todo_column(&conn)?;
        ensure_conversations_plan_workflow_column(&conn)?;
        ensure_conversations_goal_workflow_column(&conn)?;
        ensure_conversations_mode_model_settings_column(&conn)?;
        ensure_conversations_archived_column(&conn)?;
        ensure_conversations_title_initialized_column(&conn)?;
        ensure_app_settings_table(&conn)?;
        ensure_turn_checkpoints_table(&conn)?;
        if version < 8 {
            conn.execute("delete from turn_checkpoints", [])
                .context("unable to clear legacy turn checkpoints")?;
        }
        conn.pragma_update(None, "user_version", 9)
            .context("unable to set sqlite schema version")?;
        Ok(())
    }

    fn connection(&self) -> Result<Connection> {
        let conn = Connection::open(&self.path).context("unable to open sqlite database")?;
        conn.execute_batch("pragma foreign_keys = on;")
            .context("unable to enable foreign keys")?;
        Ok(conn)
    }
}

fn ensure_conversations_todo_column(conn: &Connection) -> Result<()> {
    if conversation_has_column(conn, "todo_list_json")? {
        return Ok(());
    }
    conn.execute_batch(
        r#"
        alter table conversations
            add column todo_list_json text not null
            default '{"active":false,"tasks":[],"nextId":1}';
        "#,
    )
    .context("unable to add todo list state column")?;
    Ok(())
}

fn ensure_conversations_plan_workflow_column(conn: &Connection) -> Result<()> {
    if conversation_has_column(conn, "plan_workflow_json")? {
        return Ok(());
    }
    conn.execute_batch(
        r#"
        alter table conversations
            add column plan_workflow_json text not null
            default '{"status":"idle"}';
        "#,
    )
    .context("unable to add plan workflow state column")?;
    Ok(())
}

fn ensure_conversations_goal_workflow_column(conn: &Connection) -> Result<()> {
    if conversation_has_column(conn, "goal_workflow_json")? {
        return Ok(());
    }
    conn.execute_batch(
        r#"
        alter table conversations
            add column goal_workflow_json text not null
            default '{"status":"idle"}';
        "#,
    )
    .context("unable to add goal workflow state column")?;
    Ok(())
}

fn ensure_conversations_mode_model_settings_column(conn: &Connection) -> Result<()> {
    if conversation_has_column(conn, "mode_model_settings_json")? {
        return Ok(());
    }
    conn.execute_batch(
        r#"
        alter table conversations
            add column mode_model_settings_json text;
        "#,
    )
    .context("unable to add mode model settings column")?;
    Ok(())
}

fn ensure_conversations_archived_column(conn: &Connection) -> Result<()> {
    if conversation_has_column(conn, "archived_at_ms")? {
        return Ok(());
    }
    conn.execute_batch(
        r#"
        alter table conversations
            add column archived_at_ms integer;
        "#,
    )
    .context("unable to add conversation archive column")?;
    Ok(())
}

fn ensure_conversations_title_initialized_column(conn: &Connection) -> Result<()> {
    if conversation_has_column(conn, "title_initialized")? {
        return Ok(());
    }
    conn.execute_batch(
        r#"
        alter table conversations
            add column title_initialized integer not null default 1;
        update conversations
            set title_initialized = case
                when trim(title) = 'New conversation' then 0
                else 1
            end;
        "#,
    )
    .context("unable to add conversation title initialization column")?;
    Ok(())
}

fn workspace_name_from_id(workspace_id: &str) -> String {
    Path::new(workspace_id)
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .map(|value| value.to_string())
        .unwrap_or_else(|| workspace_id.to_string())
}

fn ensure_app_settings_table(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        create table if not exists app_settings (
            key text primary key,
            value_json text not null,
            updated_at_ms integer not null
        );
        "#,
    )
    .context("unable to create app settings table")?;
    Ok(())
}

fn ensure_turn_checkpoints_table(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        create table if not exists turn_checkpoints (
            conversation_id text not null,
            history_index integer not null,
            checkpoint_json text not null,
            primary key (conversation_id, history_index),
            foreign key (conversation_id) references conversations(id) on delete cascade
        );
        "#,
    )
    .context("unable to create turn checkpoint table")?;
    Ok(())
}

fn default_enabled() -> bool {
    true
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn default_tool_enabled(name: &str) -> bool {
    !matches!(name, tool_names::CREATE_IMAGE | tool_names::WEB_SEARCH)
}

fn normalize_openrouter_models(models: Vec<OpenRouterModelRecord>) -> Vec<OpenRouterModelRecord> {
    let mut seen = HashSet::new();
    models
        .into_iter()
        .filter_map(OpenRouterModelRecord::normalized)
        .filter(|model| seen.insert(model.id.clone()))
        .collect()
}

fn load_mode_model_settings_from_conn(
    conn: &Connection,
    default_model: &ModelRef,
) -> Result<ModeModelSettings> {
    let stored = conn
        .query_row(
            "select value_json from app_settings where key = ?1",
            params![MODE_MODEL_SETTINGS_KEY],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .context("unable to read mode model settings")?;

    if let Some(json) = stored {
        if let Ok(settings) = serde_json::from_str::<ModeModelSettings>(&json) {
            return Ok(settings);
        }
    }

    let latest_conversation_settings = conn
        .query_row(
            "select mode_model_settings_json from conversations
             where mode_model_settings_json is not null
             order by updated_at_ms desc
             limit 1",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .context("unable to read latest conversation model settings")?;

    if let Some(json) = latest_conversation_settings {
        if let Ok(settings) = serde_json::from_str::<ModeModelSettings>(&json) {
            return Ok(settings);
        }
    }

    Ok(ModeModelSettings::new(default_model))
}

fn conversation_has_column(conn: &Connection, name: &str) -> Result<bool> {
    let mut statement = conn
        .prepare("pragma table_info(conversations)")
        .context("unable to inspect conversations table")?;
    let rows = statement
        .query_map([], |row| row.get::<_, String>(1))
        .context("unable to read conversations columns")?;

    for row in rows {
        if row.context("bad conversations column row")? == name {
            return Ok(true);
        }
    }
    Ok(false)
}

struct ConversationTitleState {
    title: String,
    initialized: bool,
}

fn load_conversation_title_state(
    conn: &Connection,
    workspace_id: &str,
    id: &str,
) -> Result<Option<ConversationTitleState>> {
    conn.query_row(
        "select title, title_initialized from conversations where workspace_id = ?1 and id = ?2",
        params![workspace_id, id],
        |row| {
            let initialized: i64 = row.get(1)?;
            Ok(ConversationTitleState {
                title: row.get(0)?,
                initialized: initialized != 0,
            })
        },
    )
    .optional()
    .context("unable to load conversation title state")
}

fn resolve_title_for_save(
    current_state: Option<&ConversationTitleState>,
    incoming_title: &str,
    history: &[ChatMessage],
) -> ConversationTitleState {
    if let Some(state) = current_state {
        if state.initialized && state.title.trim() != DEFAULT_CONVERSATION_TITLE {
            return ConversationTitleState {
                title: state.title.clone(),
                initialized: true,
            };
        }
    }

    let incoming_title = normalize_conversation_title(incoming_title);
    let fallback_title = current_state
        .map(|state| state.title.as_str())
        .unwrap_or(incoming_title.as_str());
    match legacy_title_from_history(history) {
        Some(title) => ConversationTitleState {
            title,
            initialized: true,
        },
        None => ConversationTitleState {
            title: normalize_conversation_title(fallback_title),
            initialized: false,
        },
    }
}

fn normalize_conversation_title(title: &str) -> String {
    let title = title.trim();
    if title.is_empty() {
        DEFAULT_CONVERSATION_TITLE.to_string()
    } else {
        title.to_string()
    }
}

fn conversation_needs_generated_title(current_title: &str, history: &[ChatMessage]) -> bool {
    if first_visible_user_text(history).is_none() {
        return false;
    }

    let current_title = current_title.trim();
    if current_title.is_empty() || current_title == DEFAULT_CONVERSATION_TITLE {
        return true;
    }

    legacy_title_from_history(history)
        .as_deref()
        .map(|legacy| legacy == current_title)
        .unwrap_or(false)
}

pub async fn summarized_conversation_title(
    current_title: &str,
    provider: Arc<dyn Provider>,
    model: ModelRef,
    history: &[ChatMessage],
) -> String {
    let current_title = normalize_conversation_title(current_title);
    if !conversation_needs_generated_title(&current_title, history) {
        return current_title;
    }

    let fallback = fallback_conversation_title(history).unwrap_or_else(|| current_title.clone());
    let Some(input) = title_generation_input(history) else {
        return fallback;
    };

    match tokio::time::timeout(
        Duration::from_secs(TITLE_MODEL_TIMEOUT_SECS),
        request_summarized_title(provider, model, input),
    )
    .await
    {
        Ok(Some(title)) => title,
        Ok(None) | Err(_) => fallback,
    }
}

async fn request_summarized_title(
    provider: Arc<dyn Provider>,
    model: ModelRef,
    input: String,
) -> Option<String> {
    let mut request = ProviderRequest::new(
        model,
        vec![ChatMessage::user_text(format!(
            "Generate a title for this conversation:\n{input}"
        ))],
    )
    .with_system(
        "You are a title generator. Output ONLY a very short conversation title. Nothing else. The title must be a single line of AT MOST 3 words, in the same language as the user message. Strongly prefer exactly 3 words; never exceed 3 words under any circumstance. Never include tool names, labels, quotes, markdown, or ending punctuation. Focus on the main topic or question the user needs to retrieve. Keep exact technical terms, numbers, filenames, and HTTP codes. Every word must be meaningful on its own; never output filler, partial, or cut-off words. Never say you cannot generate a title; always output something meaningful.",
    );
    request.max_output_tokens = Some(16);
    request.effort = Some(Effort::None);
    request.service_tier = Some(ServiceTier::Fast);

    let mut stream = provider.stream(request).await.ok()?;
    let mut title = String::new();
    while let Some(event) = stream.next().await {
        match event.ok()? {
            StreamEvent::TextDelta { delta, .. } => title.push_str(&delta),
            StreamEvent::MessageStop { .. } => break,
            _ => {}
        }
    }
    sanitize_generated_title(&title)
}

fn title_generation_input(history: &[ChatMessage]) -> Option<String> {
    first_visible_user_text(history).map(|text| {
        let text = compact_whitespace(text);
        if text.chars().count() <= TITLE_INPUT_MAX_CHARS {
            text
        } else {
            let mut shortened = text
                .chars()
                .take(TITLE_INPUT_MAX_CHARS.saturating_sub(1))
                .collect::<String>();
            shortened.push('…');
            shortened
        }
    })
}

fn fallback_conversation_title(history: &[ChatMessage]) -> Option<String> {
    first_visible_user_text(history).map(heuristic_title_from_text)
}

fn first_visible_user_text(history: &[ChatMessage]) -> Option<&str> {
    history
        .iter()
        .filter(|message| matches!(message.role, Role::User))
        .find_map(|message| {
            message.parts.iter().find_map(|part| match part {
                Part::Text { text, meta }
                    if !title_hidden_text(meta) && !text.trim().is_empty() =>
                {
                    Some(text.trim())
                }
                _ => None,
            })
        })
}

fn heuristic_title_from_text(text: &str) -> String {
    let mut title = compact_whitespace(text);
    title = strip_markdown_prefixes(&title);
    title = strip_request_prefixes(&title);
    title = title
        .split(['\n', '\r'])
        .next()
        .unwrap_or_default()
        .trim()
        .to_string();
    title = trim_title_edges(&title).to_string();
    truncate_title(&title)
}

fn legacy_title_from_history(history: &[ChatMessage]) -> Option<String> {
    first_visible_user_text(history).map(heuristic_title_from_text)
}

fn compact_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn strip_markdown_prefixes(value: &str) -> String {
    value
        .trim_start_matches(|ch: char| {
            ch.is_whitespace() || matches!(ch, '#' | '*' | '-' | '>' | '`')
        })
        .trim()
        .to_string()
}

fn strip_request_prefixes(value: &str) -> String {
    let prefixes = [
        "peux-tu ",
        "peux tu ",
        "est-ce que tu peux ",
        "tu peux ",
        "please ",
        "can you ",
        "could you ",
        "i want to ",
        "i want ",
        "je veux ",
        "j'aimerais ",
        "j aimerais ",
    ];
    let lower = value.to_lowercase();
    for prefix in prefixes {
        if lower.starts_with(prefix) {
            return value[prefix.len()..].trim().to_string();
        }
    }
    value.trim().to_string()
}

fn sanitize_generated_title(raw: &str) -> Option<String> {
    let mut title = raw
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())?
        .to_string();

    title = strip_markdown_prefixes(&title);
    title = strip_title_label(&title).to_string();
    title = trim_title_edges(&title).to_string();
    title = compact_whitespace(&title);
    title = trim_title_edges(&title).to_string();

    if title.is_empty() || title == DEFAULT_CONVERSATION_TITLE {
        return None;
    }

    Some(truncate_title(&title))
}

fn strip_title_label(value: &str) -> &str {
    let lower = value.to_lowercase();
    for label in ["title:", "titre:", "chat title:", "nom:"] {
        if lower.starts_with(label) {
            return value[label.len()..].trim();
        }
    }
    value.trim()
}

fn trim_title_edges(value: &str) -> &str {
    value.trim_matches(|ch: char| {
        ch.is_whitespace()
            || matches!(
                ch,
                '"' | '\''
                    | '`'
                    | '“'
                    | '”'
                    | '‘'
                    | '’'
                    | '«'
                    | '»'
                    | '*'
                    | '_'
                    | '-'
                    | '—'
                    | ':'
                    | ';'
                    | '.'
            )
    })
}

fn truncate_title(title: &str) -> String {
    let title = truncate_title_words(title);
    if title.chars().count() <= TITLE_MAX_CHARS {
        return title;
    }

    let mut shortened = title
        .chars()
        .take(TITLE_MAX_CHARS.saturating_sub(1))
        .collect::<String>();
    shortened.push('…');
    shortened
}

fn truncate_title_words(title: &str) -> String {
    let words = title
        .trim()
        .split_whitespace()
        .take(TITLE_MAX_WORDS)
        .collect::<Vec<_>>();
    trim_title_edges(&words.join(" ")).to_string()
}

fn title_hidden_text(meta: &Option<Value>) -> bool {
    let Some(Value::Object(meta)) = meta else {
        return false;
    };
    meta.get("ui_only").and_then(Value::as_bool) == Some(true)
        || meta
            .get("compaction_retained_user")
            .and_then(Value::as_bool)
            == Some(true)
        || meta.get("compaction_summary").and_then(Value::as_bool) == Some(true)
        || meta.get("system_reminder").and_then(Value::as_bool) == Some(true)
        || meta.get("attachment_context").and_then(Value::as_bool) == Some(true)
        || meta.get("agent_team_messages").and_then(Value::as_bool) == Some(true)
        || meta.get("plan_control").and_then(Value::as_str).is_some()
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;

    fn message(role: Role, text: &str, meta: Option<Value>) -> ChatMessage {
        ChatMessage {
            role,
            parts: vec![Part::Text {
                text: text.to_string(),
                meta,
            }],
        }
    }

    #[test]
    fn resolve_title_for_save_initializes_uninitialized_title_from_first_visible_user_text() {
        let history = vec![message(Role::User, "First request", None)];
        let current = ConversationTitleState {
            title: DEFAULT_CONVERSATION_TITLE.to_string(),
            initialized: false,
        };

        let resolved = resolve_title_for_save(Some(&current), DEFAULT_CONVERSATION_TITLE, &history);

        assert_eq!(resolved.title, "First request");
        assert!(resolved.initialized);
    }

    #[test]
    fn resolve_title_for_save_recovers_new_conversation_marked_initialized_by_migration_bug() {
        let history = vec![message(Role::User, "First request", None)];
        let current = ConversationTitleState {
            title: DEFAULT_CONVERSATION_TITLE.to_string(),
            initialized: true,
        };

        let resolved = resolve_title_for_save(Some(&current), DEFAULT_CONVERSATION_TITLE, &history);

        assert_eq!(resolved.title, "First request");
        assert!(resolved.initialized);
    }

    #[test]
    fn resolve_title_for_save_preserves_initialized_title() {
        let history = vec![message(Role::User, "A later request", None)];
        let current = ConversationTitleState {
            title: "Original request".to_string(),
            initialized: true,
        };

        let resolved = resolve_title_for_save(Some(&current), "Stale incoming title", &history);

        assert_eq!(resolved.title, "Original request");
        assert!(resolved.initialized);
    }

    #[test]
    fn resolve_title_for_save_preserves_initialized_title_after_compaction() {
        let history = vec![
            message(
                Role::User,
                "Retained compacted request",
                Some(json!({ "compaction_retained_user": true })),
            ),
            message(
                Role::User,
                "Compaction summary",
                Some(json!({ "compaction_summary": true })),
            ),
            message(Role::User, "New post-compaction request", None),
        ];
        let current = ConversationTitleState {
            title: "Original request".to_string(),
            initialized: true,
        };

        let resolved = resolve_title_for_save(Some(&current), "Original request", &history);

        assert_eq!(resolved.title, "Original request");
        assert!(resolved.initialized);
    }

    #[test]
    fn save_conversation_initializes_title_from_first_user_message_and_preserves_it() -> Result<()>
    {
        let path =
            std::env::temp_dir().join(format!("sinew-store-title-test-{}.sqlite3", Uuid::new_v4()));
        let store = AppStore { path: path.clone() };
        let result = (|| -> Result<()> {
            store.migrate()?;
            let model = ModelRef::new("test", "model");
            let mut conversation = store.create_conversation("workspace", &model, "system")?;
            conversation
                .history
                .push(message(Role::User, "First request", None));

            store.save_conversation(&conversation)?;
            let loaded = store
                .load_conversation("workspace", &conversation.id)?
                .expect("conversation should exist");
            assert_eq!(loaded.title, "First request");
            assert_eq!(
                store.list_conversations("workspace")?[0].title,
                "First request"
            );

            let mut compacted = loaded;
            compacted.history = vec![
                message(
                    Role::User,
                    "Retained compacted request",
                    Some(json!({ "compaction_retained_user": true })),
                ),
                message(
                    Role::User,
                    "Compaction summary",
                    Some(json!({ "compaction_summary": true })),
                ),
                message(Role::User, "New post-compaction request", None),
            ];
            store.save_conversation(&compacted)?;
            let reloaded = store
                .load_conversation("workspace", &conversation.id)?
                .expect("conversation should exist");
            assert_eq!(reloaded.title, "First request");
            Ok(())
        })();
        let _ = fs::remove_file(path);
        result
    }

    #[test]
    fn legacy_title_from_history_uses_first_visible_user_text() {
        let history = vec![
            message(Role::Assistant, "Assistant text", None),
            message(
                Role::User,
                "Hidden system reminder",
                Some(json!({ "system_reminder": true })),
            ),
            message(Role::User, "Real user request", None),
        ];

        assert_eq!(
            legacy_title_from_history(&history).as_deref(),
            Some("Real user request")
        );
    }

    #[test]
    fn legacy_title_from_history_ignores_assistant_when_no_visible_user_text() {
        let history = vec![
            message(
                Role::User,
                "Implement completely this plan",
                Some(json!({ "system_reminder": true })),
            ),
            message(Role::Assistant, "I'll start implementing the plan.", None),
        ];

        assert_eq!(legacy_title_from_history(&history), None);
    }

    #[test]
    fn conversation_needs_generated_title_for_default_and_legacy_titles() {
        let history = vec![message(
            Role::User,
            "Explain the new settings panel layout in detail",
            None,
        )];
        let legacy = legacy_title_from_history(&history).expect("legacy title");

        assert!(conversation_needs_generated_title(
            DEFAULT_CONVERSATION_TITLE,
            &history
        ));
        assert!(conversation_needs_generated_title(&legacy, &history));
        assert!(!conversation_needs_generated_title(
            "Settings layout",
            &history
        ));
    }

    #[test]
    fn sanitize_generated_title_removes_labels_and_quotes() {
        assert_eq!(
            sanitize_generated_title("Titre: \"Nommage résumé des chats.\"").as_deref(),
            Some("Nommage résumé des")
        );
    }

    #[test]
    fn sanitize_generated_title_limits_words() {
        assert_eq!(
            sanitize_generated_title(
                "Titre: Corriger les titres automatiques trop longs maintenant"
            )
            .as_deref(),
            Some("Corriger les titres")
        );
    }

    #[test]
    fn heuristic_title_from_text_strips_request_prefix_and_limits_words() {
        assert_eq!(
            heuristic_title_from_text(
                "Peux-tu corriger les titres automatiques trop longs dans Sinew ?",
            ),
            "corriger les titres"
        );
    }

    fn descriptor(name: &str, description: &str) -> ToolDescriptor {
        ToolDescriptor {
            name: name.to_string(),
            description: description.to_string(),
            input_schema: json!({ "type": "object" }),
        }
    }

    #[test]
    fn tool_settings_ignore_legacy_saved_descriptions_without_user_override() {
        let settings = ToolSettings {
            tools: vec![ToolConfig {
                name: "edit_file".to_string(),
                description: "old default from database".to_string(),
                enabled: true,
                description_override: false,
                default_description: String::new(),
            }],
            ..ToolSettings::default()
        }
        .normalized();

        let tools =
            settings.apply_to_descriptors(vec![descriptor("edit_file", "new code default")]);

        assert_eq!(tools[0].description, "new code default");
    }

    #[test]
    fn tool_settings_persist_only_descriptions_that_differ_from_catalog_default() {
        let settings = ToolSettings {
            tools: vec![
                ToolConfig {
                    name: "read".to_string(),
                    description: "read default".to_string(),
                    enabled: true,
                    description_override: false,
                    default_description: "read default".to_string(),
                },
                ToolConfig {
                    name: "edit_file".to_string(),
                    description: "custom edit instructions".to_string(),
                    enabled: true,
                    description_override: false,
                    default_description: "edit default".to_string(),
                },
            ],
            ..ToolSettings::default()
        }
        .normalized_for_catalog(&[
            descriptor("read", "read default"),
            descriptor("edit_file", "edit default"),
        ]);

        assert_eq!(settings.tools[0].description, "");
        assert!(!settings.tools[0].description_override);
        assert_eq!(settings.tools[1].description, "custom edit instructions");
        assert!(settings.tools[1].description_override);
    }

    #[test]
    fn tool_settings_apply_user_description_override() {
        let settings = ToolSettings {
            tools: vec![ToolConfig {
                name: "edit_file".to_string(),
                description: "custom edit instructions".to_string(),
                enabled: true,
                description_override: true,
                default_description: String::new(),
            }],
            ..ToolSettings::default()
        }
        .normalized();

        let tools =
            settings.apply_to_descriptors(vec![descriptor("edit_file", "new code default")]);

        assert_eq!(tools[0].description, "custom edit instructions");
    }
}
