use std::{collections::HashMap, path::PathBuf, sync::Arc};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_core::{
    ChatMessage, Effort, ModelRef, Part, Provider, Role, ServiceTier, ToolDescriptor,
};
use tokio::sync::mpsc;

use crate::tool_run::FileChange;
use crate::{
    run_turn, AgentEvent, AgentEventScope, AgentMode, BashTool, CreateImageTool, EditFileTool,
    GlobTool, GoalWorkflowState, GrepTool, McpSettings, McpToolRegistry, QuestionTool, ReadTool,
    SkillSettings, SkillTool, ToDoListTool, TodoListState, ToolRunResult, ToolSettings, TurnCancel,
    TurnContext, WebFetchTool, WebSearchTool, WriteFileTool,
};

const TOOL_PREFIX: &str = "subagent_";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubAgentConfig {
    pub id: String,
    pub name: String,
    pub description: String,
    pub prompt: String,
    pub model: ModelRef,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubAgentSettings {
    #[serde(default)]
    pub agents: Vec<SubAgentConfig>,
}

impl SubAgentSettings {
    pub fn normalized(mut self) -> Self {
        let mut seen = HashMap::<String, usize>::new();
        for (index, agent) in self.agents.iter_mut().enumerate() {
            agent.id = clean_id(&agent.id).unwrap_or_else(|| format!("agent-{}", index + 1));
            let count = seen.entry(agent.id.clone()).or_insert(0);
            if *count > 0 {
                agent.id = format!("{}-{}", agent.id, *count + 1);
            }
            *count += 1;

            agent.name = agent.name.trim().to_string();
            if agent.name.is_empty() {
                agent.name = format!("Sub-agent {}", index + 1);
            }
            agent.description = agent.description.trim().to_string();
            agent.prompt = agent.prompt.trim().to_string();
        }
        self
    }
}

pub fn with_default_sub_agents(mut settings: SubAgentSettings) -> SubAgentSettings {
    settings = settings.normalized();
    let mut defaults = aiblueprint_default_sub_agents();
    defaults.retain(|default_agent| {
        !settings
            .agents
            .iter()
            .any(|agent| agent.id == default_agent.id)
    });
    defaults.extend(settings.agents);
    SubAgentSettings { agents: defaults }.normalized()
}

pub fn aiblueprint_default_sub_agents() -> Vec<SubAgentConfig> {
    fn model() -> ModelRef {
        ModelRef::new("anthropic", "claude-sonnet-4-6").with_effort(Effort::Medium)
    }

    fn agent(id: &str, name: &str, description: &str, prompt: &str) -> SubAgentConfig {
        SubAgentConfig {
            id: id.to_string(),
            name: name.to_string(),
            description: description.to_string(),
            prompt: prompt.trim().to_string(),
            model: model(),
            enabled: true,
        }
    }

    vec![
        agent(
            "apex-coordinator",
            "APEX Coordinator",
            "Coordinates Sinew-native APEX workflows: analyze, plan, execute, validate, examine, resolve, and verify.",
            r#"You coordinate structured implementation work using Sinew tools. Break work into phases, keep a concise ToDoList, prefer parallel TeamRun only for independent workstreams, and always preserve user changes. Translate Claude-style APEX concepts to Sinew equivalents: TeamRun for teams, ToDoList for task tracking, apply_patch for edits, and Skill for loading skills."#,
        ),
        agent(
            "code-architect",
            "Code Architect",
            "Designs implementation blueprints by extracting existing codebase patterns and making concrete architecture decisions.",
            r#"Act as a senior software architect. Inspect the codebase before deciding. Find existing patterns, adjacent features, data flows, and constraints. Produce a decisive architecture blueprint with exact files, responsibilities, integration points, risks, and validation steps. Do not implement unless explicitly asked."#,
        ),
        agent(
            "code-explorer",
            "Code Explorer",
            "Maps relevant code paths, conventions, dependencies, and risks before implementation.",
            r#"Explore the repository efficiently. Use Glob/Grep/Read to identify relevant files, ownership boundaries, architecture, conventions, and unknowns. Return a compact report with file paths and findings. Do not edit files."#,
        ),
        agent(
            "code-reviewer",
            "Code Reviewer",
            "Performs adversarial review for correctness, regressions, security, performance, and maintainability.",
            r#"Review changes skeptically. Look for logic bugs, missing edge cases, broken UX, race conditions, security issues, style drift, and test gaps. Provide prioritized findings with file references and concrete fixes. Do not modify files unless explicitly asked."#,
        ),
        agent(
            "implementer",
            "Implementer",
            "Focused task implementer for APEX/team workflows with strict scope boundaries.",
            r#"Implement only the assigned task. Read assigned files before editing. Stay inside the requested scope, follow existing patterns, use apply_patch for changes, run targeted checks when appropriate, and report exactly what changed. If blocked or scope expands, stop and report instead of improvising."#,
        ),
        agent(
            "verifier",
            "Verifier",
            "Validates that implementation satisfies acceptance criteria and does not regress existing behavior.",
            r#"Verify completed work. Run or recommend the smallest meaningful checks. Inspect relevant code paths and acceptance criteria. Report pass/fail, evidence, remaining risks, and exact commands run. Do not edit unless explicitly asked."#,
        ),
        agent(
            "worker",
            "Worker",
            "General-purpose autonomous worker for bounded research or implementation tasks.",
            r#"Execute a bounded assignment independently. Build context first, keep progress concise, avoid scope creep, and return a final report with files touched, decisions made, and validation status."#,
        ),
        agent(
            "clean-code-runner",
            "Clean Code Runner",
            "Applies clean-code improvements while preserving behavior and minimizing churn.",
            r#"Improve readability and maintainability without changing behavior. Prefer small, local improvements, clear names, simpler control flow, and removal of duplication. Avoid broad refactors unless requested. Validate after edits."#,
        ),
        agent(
            "code-simplifier",
            "Code Simplifier",
            "Simplifies complex code paths and removes unnecessary abstraction.",
            r#"Find the simplest correct design. Reduce unnecessary layers, branching, cleverness, and indirection. Preserve public behavior. Explain tradeoffs and keep changes minimal."#,
        ),
        agent(
            "explore-codebase",
            "Explore Codebase",
            "Performs broad codebase orientation and system mapping.",
            r#"Create a practical map of the codebase for the requested area: modules, entry points, data flow, key types, config, tests, and conventions. Use file references. Do not modify files."#,
        ),
        agent(
            "explore-docs",
            "Explore Docs",
            "Finds and summarizes relevant documentation, specs, changelogs, and external references.",
            r#"Search local docs first, then web only if useful. Summarize relevant guidance, API constraints, version-specific details, and cite sources or file paths. Do not implement."#,
        ),
        agent(
            "explore-fast",
            "Explore Fast",
            "Fast, shallow reconnaissance for obvious files and likely implementation path.",
            r#"Do a quick targeted scan. Identify the likely files, commands, and risks in a compact answer. Optimize for speed over completeness. Do not edit."#,
        ),
        agent(
            "websearch",
            "Web Search",
            "Researches external technical documentation and current best practices.",
            r#"Use web search/fetch when local context is insufficient. Prefer official docs and primary sources. Return concise findings with links, version caveats, and implementation implications."#,
        ),
        agent(
            "fast-websearch",
            "Fast Web Search",
            "Quick external lookup for a specific API, error, or version question.",
            r#"Answer a narrow external research question quickly. Use minimal sources, prefer official docs, and return only actionable facts with links."#,
        ),
        agent(
            "snipper",
            "Snipper",
            "Extracts minimal code snippets, examples, and reusable patterns from existing code.",
            r#"Find small, relevant code snippets or patterns. Return file paths and concise excerpts or descriptions. Do not edit files."#,
        ),
        agent(
            "action",
            "Action",
            "Direct action agent for small, well-defined changes.",
            r#"For small explicit tasks, act directly after reading relevant files. Keep changes minimal, use apply_patch, and validate if practical. Ask/report if the task is ambiguous or broader than expected."#,
        ),
    ]
}

#[derive(Clone)]
pub struct SubAgentTool {
    workspace_root: PathBuf,
    system_prompt: String,
    providers: HashMap<String, Arc<dyn Provider>>,
    settings: SubAgentSettings,
    mcp_settings: McpSettings,
    tool_settings: ToolSettings,
    skill_settings: SkillSettings,
    max_tool_rounds: usize,
    service_tier: Option<ServiceTier>,
    cancel: TurnCancel,
}

impl SubAgentTool {
    pub fn new(
        workspace_root: PathBuf,
        system_prompt: String,
        providers: HashMap<String, Arc<dyn Provider>>,
        settings: SubAgentSettings,
        mcp_settings: McpSettings,
        tool_settings: ToolSettings,
        skill_settings: SkillSettings,
        max_tool_rounds: usize,
        service_tier: Option<ServiceTier>,
        cancel: TurnCancel,
    ) -> Self {
        Self {
            workspace_root,
            system_prompt,
            providers,
            settings: settings.normalized(),
            mcp_settings,
            tool_settings,
            skill_settings,
            max_tool_rounds,
            service_tier,
            cancel,
        }
    }

    pub fn descriptors(&self) -> Vec<ToolDescriptor> {
        self.settings
            .agents
            .iter()
            .filter(|agent| agent.enabled)
            .map(|agent| ToolDescriptor {
                name: tool_name_for_agent(agent),
                description: descriptor_description(agent),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The full free-form message to send to the sub-agent."
                        }
                    },
                    "required": ["prompt"],
                    "additionalProperties": false
                }),
            })
            .collect()
    }

    pub fn summary_for_tool_name(&self, name: &str) -> Option<String> {
        self.settings
            .agents
            .iter()
            .find(|agent| agent.enabled && tool_name_for_agent(agent) == name)
            .map(|agent| format!("Sub-agent · {}", agent.name))
    }

    pub async fn run(
        &self,
        tool_call_id: &str,
        name: &str,
        input: Value,
        mode: AgentMode,
        parent_event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> Option<ToolRunResult> {
        let agent = self
            .settings
            .agents
            .iter()
            .find(|agent| agent.enabled && tool_name_for_agent(agent) == name)?
            .clone();

        Some(
            self.run_agent(tool_call_id, agent, input, mode, parent_event_tx)
                .await,
        )
    }

    async fn run_agent(
        &self,
        tool_call_id: &str,
        agent: SubAgentConfig,
        input: Value,
        mode: AgentMode,
        parent_event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> ToolRunResult {
        let parsed: SubAgentInput = match serde_json::from_value(input) {
            Ok(value) => value,
            Err(err) => {
                return ToolRunResult::err(format!("invalid sub-agent input: {err}"), Vec::new())
            }
        };
        let prompt = parsed.prompt.trim();
        if prompt.is_empty() {
            return ToolRunResult::err("prompt is required", Vec::new());
        }

        let Some(provider) = self.providers.get(&agent.model.provider).cloned() else {
            return ToolRunResult::err(
                format!(
                    "provider `{}` is not configured or missing credentials",
                    agent.model.provider
                ),
                Vec::new(),
            );
        };
        if provider.capabilities(&agent.model).is_none() {
            return ToolRunResult::err(
                format!("model `{}` is not supported", agent.model.name),
                Vec::new(),
            );
        }

        let initial_message = prompt.to_string();
        let (child_cmd_tx, child_cmd_rx) = mpsc::unbounded_channel();
        self.cancel.register(child_cmd_tx);
        let child_mode = if mode == AgentMode::Goal {
            AgentMode::Act
        } else {
            mode
        };
        let child_context = TurnContext {
            provider,
            model: agent.model.clone(),
            cache_key: Some(format!("subagent:{}:{}", agent.id, tool_call_id)),
            cache_stable_message_count: 0,
            service_tier: self.service_tier,
            auto_compact: true,
            mode: child_mode,
            stop_questions: false,
            system_prompt: subagent_system_prompt(&self.system_prompt, &agent),
            history: vec![ChatMessage {
                role: Role::User,
                parts: vec![Part::Text {
                    text: initial_message.clone(),
                    meta: None,
                }],
            }],
            todo_list: TodoListState::default(),
            goal_workflow: GoalWorkflowState::Idle,
            bash: Arc::new(BashTool::new(self.workspace_root.clone())),
            glob: Arc::new(GlobTool::new(self.workspace_root.clone())),
            grep: Arc::new(GrepTool::new(self.workspace_root.clone())),
            read: Arc::new(ReadTool::new(self.workspace_root.clone())),
            edit_file: Arc::new(EditFileTool::new(self.workspace_root.clone())),
            write_file: Arc::new(WriteFileTool::new(self.workspace_root.clone())),
            create_image: Arc::new(CreateImageTool::with_settings(
                self.workspace_root.clone(),
                self.tool_settings.image_provider,
                self.tool_settings.openai_image_use_subscription,
                self.tool_settings.openai_image_api_key(),
                self.tool_settings.nano_banana_api_key(),
            )),
            todo_list_tool: Some(Arc::new(ToDoListTool::new())),
            question: Some(Arc::new(QuestionTool::new())),
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
            teams: None,
            tool_settings: self.tool_settings.clone(),
            event_scope: Some(AgentEventScope {
                id: tool_call_id.to_string(),
                agent_id: agent.id.clone(),
                agent_name: agent.name.clone(),
                team_name: None,
                model: agent.model.clone(),
                initial_message,
            }),
            max_tool_rounds: self.max_tool_rounds,
            event_tx: parent_event_tx,
            cancel: self.cancel.clone(),
            cmd_rx: child_cmd_rx,
        };

        let output = Box::pin(run_turn(child_context)).await;
        let file_changes = file_changes_from_history(&output.history);

        let final_answer = final_assistant_text(&output.history)
            .unwrap_or_else(|| "Sub-agent finished without a final answer.".to_string());
        ToolRunResult::ok_with_meta(
            final_answer,
            file_changes,
            json!({
                "subagent": {
                    "id": agent.id,
                    "name": agent.name,
                    "model": agent.model,
                    "history": output.history,
                }
            }),
        )
    }
}

#[derive(Debug, Deserialize)]
struct SubAgentInput {
    prompt: String,
}

pub fn subagent_system_prompt(base: &str, agent: &SubAgentConfig) -> String {
    let prompt = agent.prompt.trim();
    let profile = if prompt.is_empty() {
        "No extra profile prompt was provided.".to_string()
    } else {
        prompt.to_string()
    };
    format!(
        "{base}\n\n<sub_agent_profile name=\"{}\">\nYou are a delegated sub-agent. Work independently in your own context window. Use the normal workspace tools when useful. Do not ask the user questions; if you are blocked, explain the blocker in your final answer. When finished, return a concise final report for the main agent.\n\n{profile}\n</sub_agent_profile>",
        escape_attr(&agent.name)
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

fn descriptor_description(agent: &SubAgentConfig) -> String {
    let desc = agent.description.trim();
    if desc.is_empty() {
        format!("Delegate a focused task to the {} sub-agent.", agent.name)
    } else {
        desc.to_string()
    }
}

pub fn is_subagent_tool_name(name: &str) -> bool {
    name.starts_with(TOOL_PREFIX)
}

pub fn subagent_summary(name: &str, settings: &SubAgentSettings) -> Option<String> {
    settings
        .agents
        .iter()
        .find(|agent| tool_name_for_agent(agent) == name)
        .map(|agent| format!("Sub-agent · {}", agent.name))
}

fn tool_name_for_agent(agent: &SubAgentConfig) -> String {
    format!("{TOOL_PREFIX}{}", slug(&agent.id))
}

fn clean_id(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn slug(value: &str) -> String {
    let slug = value
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>()
        .trim_matches('_')
        .to_string();
    if slug.is_empty() {
        "agent".to_string()
    } else {
        slug
    }
}

fn escape_attr(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn default_enabled() -> bool {
    true
}
