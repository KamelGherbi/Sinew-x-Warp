use std::{
    collections::{BTreeSet, HashMap},
    sync::{Arc, Mutex as StdMutex},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use futures_util::StreamExt;
use serde::Serialize;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;
use uuid::Uuid;

use sinew_core::{
    AppError, ChatMessage, ModelRef, Part, PartKind, Provider, ProviderRequest, Role, StopReason,
    StreamEvent, ToolDescriptor, ToolResultImage, Usage,
};

use crate::{
    compact_conversation_history, system_prompt_with_todo,
    tool_run::{FileChange, ToolRunImage, ToolRunResult},
    ApplyPatchTool, BashTool, CreateImageTool, GlobTool, GoalWorkflowState, GrepTool,
    McpToolRegistry, QuestionTool, ReadTool, SkillTool, SubAgentTool, TeamTool, ToDoListTool,
    TodoListState, ToolSettings, WebFetchTool, WebSearchTool,
};

const PLAN_MODE_PROMPT: &str = r#"You are in Plan mode.

Rules:
- Build understanding by reading/searching/running diagnostic shell commands as needed.
- Do not edit workspace files and do not use apply_patch.
- You must keep the user in a Question loop until the user explicitly clicks "Send and stop questions".
- If the user message does not contain <plan_mode_control action="stop_questions">, your turn must end by calling the Question tool. Do not write the final plan yet.
- After each normal answer to a Question, inspect/explore more if needed, then ask the next Question.
- If you have no remaining substantive question, ask the user to confirm that you should create the plan now. Still use the Question tool.
- Only when the user message contains <plan_mode_control action="stop_questions">, stop asking questions and write the complete plan now.
- When the plan is ready, respond with only the Markdown plan. Do not implement it.

STRICTLY FORBIDDEN in the plan (unless the user explicitly requests it):
- Code snippets, pseudo-code, or inline code
- File paths, directory structures, or tree views
- Function, class, variable, or module names
- Shell commands or CLI instructions
- Technical configuration details
- Any implementation-specific notation

The plan should read as a clear description of intent and expected behavior that anyone could understand without technical background. Bullet points and paragraphs are both acceptable. The focus is on WHAT the system should do, not HOW the code should be written.

If technical specifics become necessary to avoid ambiguity, the AI may include them at its discretion, integrated naturally into the plan - but this should remain the exception, not the default."#;

const GOAL_MODE_PROMPT: &str = r#"You are in Goal mode.

Rules:
- Work autonomously toward the objective across as many turns as needed.
- Do not treat one answer as the end of the goal unless the objective is genuinely complete.
- Do not repeat completed work. First orient from existing context, then continue from the next useful step.
- Use tools and make edits normally, like Act mode.
- Before deciding the goal is complete, audit the objective against the current workspace state.
- When the objective is truly complete, you MUST call update_goal with status "complete" before your final response.
- If the objective is not complete by the end of this turn, briefly report progress and the next step. The app will continue automatically."#;

const CLEAN_CONTEXT_RESULT_PLACEHOLDER: &str =
    "[Tool result cleaned by you: irrelevant to future context.]";
const AUTO_COMPACT_OUTPUT_TOKEN_MAX: u32 = 32_000;
const MAX_AUTO_COMPACTIONS_PER_TURN: usize = 3;
const AUTO_COMPACTION_TOOL_NAME: &str = "context_compaction";

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    TurnStarted,
    TextStarted,
    TextChunk {
        delta: String,
    },
    TextFinished,
    ThinkingStarted,
    ThinkingChunk {
        delta: String,
    },
    ThinkingFinished,
    ToolStarted {
        id: String,
        name: String,
    },
    ToolArgsDelta {
        id: String,
        delta: String,
    },
    ToolOutputDelta {
        id: String,
        delta: String,
    },
    ToolReady {
        id: String,
        summary: String,
        args_pretty: String,
    },
    ToolFinished {
        id: String,
        output: String,
        is_error: bool,
        file_changes: Vec<FileChange>,
        images: Vec<ToolRunImage>,
        #[serde(skip_serializing_if = "Option::is_none")]
        meta: Option<Value>,
    },
    TokenUsage {
        provider: String,
        model: String,
        context_window: u32,
        preferred_window: u32,
        max_output_tokens: u32,
        usage: Usage,
    },
    Interrupted,
    Error {
        message: String,
    },
    PeerMessageReceived {
        id: String,
        from: String,
        to: String,
        message: String,
    },
    SubAgentEvent {
        id: String,
        agent_id: String,
        agent_name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        team_name: Option<String>,
        model: ModelRef,
        #[serde(skip_serializing_if = "Option::is_none")]
        initial_message: Option<String>,
        event: Box<AgentEvent>,
    },
    AgentSlept,
    TurnFinished,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationEvent {
    pub workspace_id: String,
    pub conversation_id: String,
    pub event: AgentEvent,
}

#[derive(Debug)]
pub enum EngineCommand {
    Cancel,
}

#[derive(Debug, Clone, Default)]
pub struct TurnCancel {
    state: Arc<StdMutex<TurnCancelState>>,
}

#[derive(Debug, Default)]
struct TurnCancelState {
    senders: Vec<mpsc::UnboundedSender<EngineCommand>>,
    cancelled: bool,
}

impl TurnCancel {
    pub fn new(root: mpsc::UnboundedSender<EngineCommand>) -> Self {
        let group = Self::default();
        group.register(root);
        group
    }

    pub fn empty() -> Self {
        Self::default()
    }

    pub fn register(&self, sender: mpsc::UnboundedSender<EngineCommand>) {
        if let Ok(mut state) = self.state.lock() {
            if state.cancelled {
                let _ = sender.send(EngineCommand::Cancel);
            }
            state.senders.push(sender);
        }
    }

    pub fn cancel_all(&self) -> bool {
        let senders = self
            .state
            .lock()
            .map(|mut state| {
                state.cancelled = true;
                state.senders.clone()
            })
            .unwrap_or_default();
        let mut sent = false;
        for sender in senders {
            sent |= sender.send(EngineCommand::Cancel).is_ok();
        }
        sent
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentMode {
    Act,
    Plan,
    Goal,
}

impl Default for AgentMode {
    fn default() -> Self {
        Self::Act
    }
}

pub struct TurnContext {
    pub provider: Arc<dyn Provider>,
    pub model: sinew_core::ModelRef,
    pub cache_key: Option<String>,
    pub cache_stable_message_count: usize,
    pub auto_compact: bool,
    pub mode: AgentMode,
    pub stop_questions: bool,
    pub system_prompt: String,
    pub history: Vec<ChatMessage>,
    pub todo_list: TodoListState,
    pub goal_workflow: GoalWorkflowState,
    pub bash: Arc<BashTool>,
    pub glob: Arc<GlobTool>,
    pub grep: Arc<GrepTool>,
    pub read: Arc<ReadTool>,
    pub apply_patch: Arc<ApplyPatchTool>,
    pub create_image: Arc<CreateImageTool>,
    pub todo_list_tool: Option<Arc<ToDoListTool>>,
    pub question: Option<Arc<QuestionTool>>,
    pub web_search: Arc<WebSearchTool>,
    pub web_fetch: Arc<WebFetchTool>,
    pub skill: Arc<SkillTool>,
    pub mcp: Arc<McpToolRegistry>,
    pub subagents: Option<Arc<SubAgentTool>>,
    pub teams: Option<Arc<TeamTool>>,
    pub tool_settings: ToolSettings,
    pub event_scope: Option<AgentEventScope>,
    pub max_tool_rounds: usize,
    pub event_tx: mpsc::UnboundedSender<AgentEvent>,
    pub cancel: TurnCancel,
    pub cmd_rx: mpsc::UnboundedReceiver<EngineCommand>,
}

#[derive(Debug, Clone)]
pub struct AgentEventScope {
    pub id: String,
    pub agent_id: String,
    pub agent_name: String,
    pub team_name: Option<String>,
    pub model: ModelRef,
    pub initial_message: String,
}

pub struct TurnOutput {
    pub history: Vec<ChatMessage>,
    pub todo_list: TodoListState,
    pub goal_workflow: GoalWorkflowState,
    pub interrupted: bool,
}

pub async fn run_turn(ctx: TurnContext) -> TurnOutput {
    let TurnContext {
        provider,
        model,
        cache_key,
        mut cache_stable_message_count,
        auto_compact,
        mode,
        stop_questions,
        system_prompt,
        mut history,
        mut todo_list,
        mut goal_workflow,
        bash,
        glob,
        grep,
        read,
        apply_patch,
        create_image,
        todo_list_tool,
        question,
        web_search,
        web_fetch,
        skill,
        mcp,
        subagents,
        teams,
        tool_settings,
        event_scope,
        max_tool_rounds,
        event_tx,
        cancel: _cancel,
        mut cmd_rx,
    } = ctx;

    send_event(&event_tx, event_scope.as_ref(), AgentEvent::TurnStarted);
    strip_all_visible_tool_result_ids(&mut history);
    normalize_tool_call_inputs(&mut history);
    repair_missing_tool_results(&mut history);
    mcp.refresh_catalog(&history).await;

    let mut cancelled = false;
    let mut loops = 0usize;
    let mut auto_compaction_attempts = 0usize;
    let mut current_turn_tool_result_ids = BTreeSet::new();
    let mut read_paths = successful_read_paths(&history, &read);
    todo_list.normalize();

    'conversation: loop {
        if let Some(teams) = &teams {
            if let Some(messages_prompt) = teams.drain_current_agent_messages_prompt().await {
                history.push(ChatMessage {
                    role: Role::User,
                    parts: vec![Part::Text {
                        text: messages_prompt,
                        meta: Some(json!({ "agent_team_messages": true })),
                    }],
                });
            }
        }

        let mut tool_descriptors = vec![
            bash.descriptor(),
            bash.input_descriptor(),
            glob.descriptor(),
            grep.descriptor(),
            read.descriptor(),
            clean_context_descriptor(),
            web_search.descriptor(),
            web_fetch.descriptor(),
        ];
        if let Some(question) = &question {
            tool_descriptors.insert(6, question.descriptor());
        }
        if let Some(todo_list_tool) = &todo_list_tool {
            tool_descriptors.insert(6, todo_list_tool.descriptor());
        }
        if let Some(descriptor) = skill.descriptor() {
            tool_descriptors.push(descriptor);
        }
        if mode != AgentMode::Plan {
            tool_descriptors.insert(4, apply_patch.descriptor());
            tool_descriptors.push(create_image.descriptor());
        }
        if mode == AgentMode::Goal {
            tool_descriptors.push(update_goal_descriptor());
        }
        tool_descriptors.extend(mcp.descriptors().await);
        if let Some(subagents) = &subagents {
            tool_descriptors.extend(subagents.descriptors());
        }
        if let Some(teams) = &teams {
            tool_descriptors.extend(teams.descriptors());
        }
        let tool_descriptors = tool_settings.apply_to_descriptors(tool_descriptors);
        let question_enabled = question.is_some() && tool_settings.is_enabled("Question");

        let mut current_system_prompt = system_prompt_with_todo(&system_prompt, &todo_list);
        if let Some(teams) = &teams {
            if let Some(team_reminder) = teams.current_agent_system_reminder().await {
                current_system_prompt.push_str("\n\n");
                current_system_prompt.push_str(&team_reminder);
            }
        }
        let current_system_prompt =
            system_prompt_for_turn(&current_system_prompt, mode, &goal_workflow);

        if auto_compact {
            match maybe_auto_compact_history(
                &provider,
                &model,
                cache_key.as_ref(),
                &mut cache_stable_message_count,
                &mut history,
                &mut current_turn_tool_result_ids,
                &current_system_prompt,
                &tool_descriptors,
                &event_tx,
                event_scope.as_ref(),
                &mut cmd_rx,
                &mut auto_compaction_attempts,
            )
            .await
            {
                Ok(true) => continue,
                Ok(false) => {}
                Err(err) => {
                    send_event(
                        &event_tx,
                        event_scope.as_ref(),
                        AgentEvent::Error { message: err },
                    );
                    break;
                }
            }
        }

        let request_history =
            history_with_current_tool_result_ids(&history, &current_turn_tool_result_ids);
        let request = ProviderRequest::new(model.clone(), request_history)
            .with_system(current_system_prompt.clone())
            .with_tools(tool_descriptors.clone())
            .with_cache_stable_message_count(cache_stable_message_count);
        let request = match &cache_key {
            Some(cache_key) => request.with_cache_key(cache_key.clone()),
            None => request,
        };

        let mut stream = match provider.stream(request).await {
            Ok(stream) => stream,
            Err(err) => {
                if auto_compact
                    && is_context_length_error(&err)
                    && can_auto_compact_history(&history, auto_compaction_attempts)
                {
                    match run_auto_compaction(
                        &provider,
                        &model,
                        cache_key.as_ref(),
                        &mut cache_stable_message_count,
                        &mut history,
                        &mut current_turn_tool_result_ids,
                        &current_system_prompt,
                        &event_tx,
                        event_scope.as_ref(),
                        &mut cmd_rx,
                        &mut auto_compaction_attempts,
                    )
                    .await
                    {
                        Ok(()) => continue,
                        Err(compaction_err) => {
                            send_event(
                                &event_tx,
                                event_scope.as_ref(),
                                AgentEvent::Error {
                                    message: format!(
                                        "provider error: {err}; context compaction failed: {compaction_err}"
                                    ),
                                },
                            );
                            break;
                        }
                    }
                }
                send_event(
                    &event_tx,
                    event_scope.as_ref(),
                    AgentEvent::Error {
                        message: format!("provider error: {err}"),
                    },
                );
                break;
            }
        };

        let mut message_builder = AssistantMessageBuilder::default();
        let mut stop_reason = StopReason::EndTurn;
        let mut response_usage = None;
        let mut stream_error = None;

        loop {
            tokio::select! {
                biased;

                command = cmd_rx.recv() => {
                    if matches!(command, Some(EngineCommand::Cancel)) {
                        cancelled = true;
                        break;
                    }
                }
                event = stream.next() => {
                    let Some(event) = event else { break; };
                    let event = match event {
                        Ok(event) => event,
                        Err(err) => {
                            stream_error = Some(err);
                            break;
                        }
                    };

                    match event {
                        StreamEvent::MessageStart { .. } => {}
                        StreamEvent::PartStart { index, kind, tool } => {
                            message_builder.open(index, kind);
                            match kind {
                                PartKind::Text => { send_event(&event_tx, event_scope.as_ref(), AgentEvent::TextStarted); }
                                PartKind::Thinking => { send_event(&event_tx, event_scope.as_ref(), AgentEvent::ThinkingStarted); }
                                PartKind::ToolCall => {
                                    if let Some(tool) = tool {
                                        message_builder.register_tool(index, tool.id.clone(), tool.name.clone());
                                        send_event(&event_tx, event_scope.as_ref(), AgentEvent::ToolStarted { id: tool.id, name: tool.name });
                                    }
                                }
                            }
                        }
                        StreamEvent::TextDelta { index, delta } => {
                            message_builder.push_text(index, &delta);
                            send_event(&event_tx, event_scope.as_ref(), AgentEvent::TextChunk { delta });
                        }
                        StreamEvent::ThinkingDelta { index, delta } => {
                            message_builder.push_text(index, &delta);
                            send_event(&event_tx, event_scope.as_ref(), AgentEvent::ThinkingChunk { delta });
                        }
                        StreamEvent::ToolJsonDelta { index, chunk } => {
                            message_builder.push_tool_json(index, &chunk);
                            if let Some((id, name)) = message_builder.tool_head(index) {
                                if should_stream_tool_args(&name) {
                                    send_event(&event_tx, event_scope.as_ref(), AgentEvent::ToolArgsDelta { id, delta: chunk });
                                }
                            }
                        }
                        StreamEvent::PartMeta { index, meta } => {
                            message_builder.push_meta(index, meta);
                        }
                        StreamEvent::PartStop { index } => {
                            match message_builder.kind(index) {
                                Some(PartKind::Text) => { send_event(&event_tx, event_scope.as_ref(), AgentEvent::TextFinished); }
                                Some(PartKind::Thinking) => {
                                    if let Some(ms) = message_builder.thinking_duration_ms(index) {
                                        message_builder.insert_meta_field(index, "duration_ms", json!(ms));
                                    }
                                    send_event(&event_tx, event_scope.as_ref(), AgentEvent::ThinkingFinished);
                                }
                                Some(PartKind::ToolCall) => {
                                    let (id, name, args) = message_builder.finalize_tool(index);
                                    let mcp_label = mcp.tool_label(&name).await;
                                    let summary = mcp_label
                                        .as_ref()
                                        .map(|label| {
                                            format!(
                                                "{} · {}",
                                                display_mcp_server_name(&label.server_name),
                                                label.tool_name
                                            )
                                        })
                                        .or_else(|| {
                                            subagents
                                                .as_ref()
                                                .and_then(|tool| tool.summary_for_tool_name(&name))
                                        })
                                        .or_else(|| {
                                            teams
                                                .as_ref()
                                                .and_then(|tool| tool.summary_for_tool_name(&name))
                                        })
                                        .unwrap_or_else(|| summarize_tool(&name, &args));
                                    if let Some(label) = mcp_label {
                                        message_builder.insert_meta_field(index, "mcp", json!(label));
                                    }
                                    send_event(&event_tx, event_scope.as_ref(), AgentEvent::ToolReady {
                                        id,
                                        summary,
                                        args_pretty: pretty_json(&args),
                                    });
                                }
                                None => {}
                            }
                        }
                        StreamEvent::Usage { usage } => {
                            response_usage = Some(usage);
                            send_token_usage_event(&event_tx, event_scope.as_ref(), &provider, &model, usage);
                        }
                        StreamEvent::MessageStop { stop_reason: reason, usage } => {
                            stop_reason = reason;
                            response_usage = Some(usage);
                            send_token_usage_event(&event_tx, event_scope.as_ref(), &provider, &model, usage);
                            break;
                        }
                    }
                }
            }
        }

        if let Some(err) = stream_error {
            if auto_compact
                && message_builder.is_empty()
                && is_context_length_error(&err)
                && can_auto_compact_history(&history, auto_compaction_attempts)
            {
                match run_auto_compaction(
                    &provider,
                    &model,
                    cache_key.as_ref(),
                    &mut cache_stable_message_count,
                    &mut history,
                    &mut current_turn_tool_result_ids,
                    &current_system_prompt,
                    &event_tx,
                    event_scope.as_ref(),
                    &mut cmd_rx,
                    &mut auto_compaction_attempts,
                )
                .await
                {
                    Ok(()) => continue,
                    Err(compaction_err) => {
                        send_event(
                            &event_tx,
                            event_scope.as_ref(),
                            AgentEvent::Error {
                                message: format!(
                                    "stream error: {err}; context compaction failed: {compaction_err}"
                                ),
                            },
                        );
                        break 'conversation;
                    }
                }
            }

            send_event(
                &event_tx,
                event_scope.as_ref(),
                AgentEvent::Error {
                    message: format!("stream error: {err}"),
                },
            );
            break 'conversation;
        }

        let mut assistant = message_builder.finish();
        if cancelled {
            retain_cancelled_visible_parts(&mut assistant);
            if !assistant.parts.is_empty() {
                history.push(assistant);
            }
            break 'conversation;
        }
        if mode == AgentMode::Plan && !stop_questions && question_enabled {
            if !assistant_has_question_tool(&assistant)
                && !matches!(stop_reason, StopReason::ToolUse)
            {
                append_plan_fallback_question(&mut assistant, &event_tx, event_scope.as_ref());
                stop_reason = StopReason::ToolUse;
            } else if assistant_has_question_tool(&assistant) {
                stop_reason = StopReason::ToolUse;
            }
        }
        if let Some(usage) = response_usage {
            attach_token_usage(&mut assistant, provider.name(), &model.name, usage);
        }
        if !assistant.parts.is_empty() {
            history.push(assistant.clone());
        }

        if !matches!(stop_reason, StopReason::ToolUse) {
            break;
        }

        if loops >= max_tool_rounds {
            send_event(
                &event_tx,
                event_scope.as_ref(),
                AgentEvent::Error {
                    message: format!("tool loop limit reached ({max_tool_rounds})"),
                },
            );
            break;
        }
        loops += 1;

        let mut tool_results = Vec::new();
        for part in &assistant.parts {
            if let Part::ToolCall {
                id, name, input, ..
            } = part
            {
                let result = if name == "clean_context" {
                    run_clean_context(&mut history, input.clone(), &current_turn_tool_result_ids)
                } else if name == "update_goal" {
                    run_update_goal(&mut goal_workflow, input.clone())
                } else if should_wait_for_cooperative_cancel(
                    name,
                    subagents.as_ref(),
                    teams.as_ref(),
                ) {
                    let result = run_tool(
                        &bash,
                        &glob,
                        &grep,
                        &read,
                        &apply_patch,
                        &create_image,
                        todo_list_tool.as_deref(),
                        question.as_deref(),
                        &web_search,
                        &web_fetch,
                        &skill,
                        &mcp,
                        subagents.as_deref(),
                        teams.as_deref(),
                        &tool_settings,
                        &read_paths,
                        &mut todo_list,
                        mode,
                        &event_tx,
                        id,
                        name,
                        input.clone(),
                    )
                    .await;
                    if matches!(cmd_rx.try_recv(), Ok(EngineCommand::Cancel)) {
                        cancelled = true;
                    }
                    result
                } else {
                    tokio::select! {
                    biased;
                        command = cmd_rx.recv() => {
                            if matches!(command, Some(EngineCommand::Cancel)) {
                                cancelled = true;
                                ToolRunResult::err("tool call interrupted by user", Vec::new())
                            } else {
                                continue;
                            }
                        }
                        result = run_tool(
                            &bash,
                            &glob,
                            &grep,
                            &read,
                            &apply_patch,
                            &create_image,
                            todo_list_tool.as_deref(),
                            question.as_deref(),
                            &web_search,
                            &web_fetch,
                            &skill,
                            &mcp,
                            subagents.as_deref(),
                            teams.as_deref(),
                            &tool_settings,
                            &read_paths,
                            &mut todo_list,
                            mode,
                            &event_tx,
                            id,
                            name,
                            input.clone(),
                        ) => result,
                    }
                };
                if name == "read" && !result.is_error {
                    if let Some(path) = input.get("path").and_then(|value| value.as_str()) {
                        if let Ok(normalized) = read.normalize_path(path) {
                            read_paths.insert(normalized);
                        }
                    }
                }
                let result_images = result.images.clone();
                let result_content = result.content.clone();
                if let Some(teams) = &teams {
                    teams
                        .record_current_agent_file_changes(name, &result.file_changes)
                        .await;
                }
                let mut meta = Map::new();
                if !result.file_changes.is_empty() {
                    meta.insert("file_changes".into(), json!(result.file_changes.clone()));
                }
                if name == "ToDoList" && !result.is_error {
                    meta.insert("todo_list".into(), json!(&todo_list));
                }
                if let Some(Value::Object(result_meta)) = result.meta.clone() {
                    for (key, value) in result_meta {
                        meta.insert(key, value);
                    }
                }
                let result_meta = (!meta.is_empty()).then(|| Value::Object(meta));
                send_event(
                    &event_tx,
                    event_scope.as_ref(),
                    AgentEvent::ToolFinished {
                        id: id.clone(),
                        output: result_content.clone(),
                        is_error: result.is_error,
                        file_changes: result.file_changes.clone(),
                        images: result_images.clone(),
                        meta: result_meta.clone(),
                    },
                );
                if name != "clean_context" {
                    current_turn_tool_result_ids.insert(id.clone());
                }
                tool_results.push(Part::ToolResult {
                    tool_call_id: id.clone(),
                    content: result_content,
                    images: result_images
                        .into_iter()
                        .map(|image| ToolResultImage {
                            media_type: image.media_type,
                            data: if name == "CreateImage" {
                                String::new()
                            } else {
                                image.data
                            },
                            path: image.path,
                        })
                        .collect(),
                    is_error: result.is_error,
                    meta: result_meta,
                });
                if cancelled {
                    break;
                }
            }
        }

        if cancelled {
            append_interrupted_tool_results(&assistant, &mut tool_results);
        }

        if tool_results.is_empty() {
            break;
        }

        history.push(ChatMessage {
            role: Role::User,
            parts: tool_results,
        });
        if cancelled {
            break 'conversation;
        }
        if mode == AgentMode::Plan && !stop_questions && assistant_has_question_tool(&assistant) {
            break;
        }
    }

    if cancelled {
        send_event(&event_tx, event_scope.as_ref(), AgentEvent::Interrupted);
    }
    send_event(&event_tx, event_scope.as_ref(), AgentEvent::TurnFinished);
    todo_list.normalize();
    TurnOutput {
        history,
        todo_list,
        goal_workflow,
        interrupted: cancelled,
    }
}

fn send_event(
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
    scope: Option<&AgentEventScope>,
    event: AgentEvent,
) {
    let event = match scope {
        Some(scope) => {
            let initial_message = if matches!(&event, AgentEvent::TurnStarted) {
                Some(scope.initial_message.clone())
            } else {
                None
            };
            AgentEvent::SubAgentEvent {
                id: scope.id.clone(),
                agent_id: scope.agent_id.clone(),
                agent_name: scope.agent_name.clone(),
                team_name: scope.team_name.clone(),
                model: scope.model.clone(),
                initial_message,
                event: Box::new(event),
            }
        }
        None => event,
    };
    let _ = event_tx.send(event);
}

fn send_token_usage_event(
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
    scope: Option<&AgentEventScope>,
    provider: &Arc<dyn Provider>,
    model: &ModelRef,
    usage: Usage,
) {
    if usage.total_tokens == 0 && usage.input_tokens == 0 && usage.output_tokens == 0 {
        return;
    }

    let Some(caps) = provider.capabilities(model) else {
        return;
    };

    send_event(
        event_tx,
        scope,
        AgentEvent::TokenUsage {
            provider: model.provider.clone(),
            model: model.name.clone(),
            context_window: caps.context_window,
            preferred_window: caps.preferred_window,
            max_output_tokens: caps.max_output_tokens,
            usage,
        },
    );
}

async fn maybe_auto_compact_history(
    provider: &Arc<dyn Provider>,
    model: &ModelRef,
    cache_key: Option<&String>,
    cache_stable_message_count: &mut usize,
    history: &mut Vec<ChatMessage>,
    current_turn_tool_result_ids: &mut BTreeSet<String>,
    system_prompt: &str,
    tool_descriptors: &[ToolDescriptor],
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
    event_scope: Option<&AgentEventScope>,
    cmd_rx: &mut mpsc::UnboundedReceiver<EngineCommand>,
    auto_compaction_attempts: &mut usize,
) -> std::result::Result<bool, String> {
    if !can_auto_compact_history(history, *auto_compaction_attempts) {
        return Ok(false);
    }

    let Some(caps) = provider.capabilities(model) else {
        return Ok(false);
    };
    if caps.context_window == 0 {
        return Ok(false);
    }

    let request_history =
        history_with_current_tool_result_ids(history, current_turn_tool_result_ids);
    let mut request = ProviderRequest::new(model.clone(), request_history)
        .with_system(system_prompt.to_string())
        .with_tools(tool_descriptors.to_vec())
        .with_cache_stable_message_count(*cache_stable_message_count);
    if let Some(cache_key) = cache_key {
        request = request.with_cache_key(cache_key.clone());
    }

    let should_compact = match provider.estimate_tokens(request).await {
        Ok(estimate) => {
            let threshold = auto_compact_threshold(caps.context_window, caps.max_output_tokens);
            estimate.input_tokens >= threshold
        }
        Err(err) if is_context_length_error(&err) => true,
        Err(_) => false,
    };

    if !should_compact {
        return Ok(false);
    }

    run_auto_compaction(
        provider,
        model,
        cache_key,
        cache_stable_message_count,
        history,
        current_turn_tool_result_ids,
        system_prompt,
        event_tx,
        event_scope,
        cmd_rx,
        auto_compaction_attempts,
    )
    .await?;
    Ok(true)
}

async fn run_auto_compaction(
    provider: &Arc<dyn Provider>,
    model: &ModelRef,
    cache_key: Option<&String>,
    cache_stable_message_count: &mut usize,
    history: &mut Vec<ChatMessage>,
    current_turn_tool_result_ids: &mut BTreeSet<String>,
    system_prompt: &str,
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
    event_scope: Option<&AgentEventScope>,
    cmd_rx: &mut mpsc::UnboundedReceiver<EngineCommand>,
    auto_compaction_attempts: &mut usize,
) -> std::result::Result<(), String> {
    if !can_auto_compact_history(history, *auto_compaction_attempts) {
        return Err("context is still too large, but there is no new content to compact".into());
    }

    let compaction_id = format!("auto-context-compaction-{}", Uuid::new_v4());
    send_event(
        event_tx,
        event_scope,
        AgentEvent::ToolStarted {
            id: compaction_id.clone(),
            name: AUTO_COMPACTION_TOOL_NAME.to_string(),
        },
    );
    send_event(
        event_tx,
        event_scope,
        AgentEvent::ToolReady {
            id: compaction_id.clone(),
            summary: "Compact context".to_string(),
            args_pretty: "{}".to_string(),
        },
    );

    let before_len = history.len();
    let (summary_delta_tx, mut summary_delta_rx) = mpsc::unbounded_channel();
    let delta_event_tx = event_tx.clone();
    let delta_event_scope = event_scope.cloned();
    let delta_compaction_id = compaction_id.clone();
    let delta_forwarder = tokio::spawn(async move {
        while let Some(delta) = summary_delta_rx.recv().await {
            send_event(
                &delta_event_tx,
                delta_event_scope.as_ref(),
                AgentEvent::ToolOutputDelta {
                    id: delta_compaction_id.clone(),
                    delta,
                },
            );
        }
    });
    let result = compact_conversation_history(
        provider.clone(),
        model.clone(),
        system_prompt.to_string(),
        history.clone(),
        cache_key.cloned(),
        *cache_stable_message_count,
        cmd_rx,
        Some(summary_delta_tx),
    )
    .await;
    let _ = delta_forwarder.await;

    match result {
        Ok(output) => {
            let retained = output.retained_user_messages;
            let summary = output.summary;
            *history = output.history;
            current_turn_tool_result_ids.clear();
            *cache_stable_message_count = 0;
            *auto_compaction_attempts += 1;
            let label = match retained {
                0 => "Context compacted. No raw user messages retained".to_string(),
                1 => format!(
                    "Context compacted from {before_len} messages. Retained 1 recent user message"
                ),
                count => format!(
                    "Context compacted from {before_len} messages. Retained {count} recent user messages"
                ),
            };
            send_event(
                event_tx,
                event_scope,
                AgentEvent::ToolFinished {
                    id: compaction_id,
                    output: label,
                    is_error: false,
                    file_changes: Vec::new(),
                    images: Vec::new(),
                    meta: Some(json!({
                        "retainedUserMessages": retained,
                        "compactionSummary": summary,
                    })),
                },
            );
            Ok(())
        }
        Err(err) => {
            let message = err.to_string();
            send_event(
                event_tx,
                event_scope,
                AgentEvent::ToolFinished {
                    id: compaction_id,
                    output: message.clone(),
                    is_error: true,
                    file_changes: Vec::new(),
                    images: Vec::new(),
                    meta: None,
                },
            );
            Err(message)
        }
    }
}

fn can_auto_compact_history(history: &[ChatMessage], attempts: usize) -> bool {
    attempts < MAX_AUTO_COMPACTIONS_PER_TURN
        && (attempts > 0 || has_content_after_latest_compaction(history))
}

fn has_content_after_latest_compaction(history: &[ChatMessage]) -> bool {
    let latest_boundary = history
        .iter()
        .rposition(|message| message.parts.iter().any(is_auto_compaction_boundary_part));
    history
        .iter()
        .skip(latest_boundary.map(|index| index + 1).unwrap_or(0))
        .any(|message| message.parts.iter().any(is_auto_compaction_meaningful_part))
}

fn is_auto_compaction_boundary_part(part: &Part) -> bool {
    let Some(meta) = part_meta(part) else {
        return false;
    };
    meta.get("compaction_summary").and_then(Value::as_bool) == Some(true)
        || meta.get("compaction_marker").and_then(Value::as_bool) == Some(true)
}

fn is_auto_compaction_meaningful_part(part: &Part) -> bool {
    match part {
        Part::Text { text, meta } => {
            !text.trim().is_empty() && !is_auto_compaction_hidden_text(meta)
        }
        _ => true,
    }
}

fn is_auto_compaction_hidden_text(meta: &Option<Value>) -> bool {
    let Some(Value::Object(meta)) = meta else {
        return false;
    };
    meta.get("attachment_context").and_then(Value::as_bool) == Some(true)
        || meta.get("ui_only").and_then(Value::as_bool) == Some(true)
        || meta.get("system_reminder").and_then(Value::as_bool) == Some(true)
        || meta
            .get("compaction_retained_user")
            .and_then(Value::as_bool)
            == Some(true)
        || meta.get("compaction_summary").and_then(Value::as_bool) == Some(true)
        || meta.get("plan_control").and_then(Value::as_str).is_some()
}

fn is_context_length_error(err: &AppError) -> bool {
    matches!(err, AppError::ContextLength(_))
}

fn auto_compact_threshold(context_window: u32, max_output_tokens: u32) -> u32 {
    if context_window == 0 {
        return 0;
    }
    let reserved_output = if max_output_tokens == 0 {
        AUTO_COMPACT_OUTPUT_TOKEN_MAX
    } else {
        max_output_tokens.min(AUTO_COMPACT_OUTPUT_TOKEN_MAX)
    };
    context_window.saturating_sub(reserved_output)
}

fn part_meta(part: &Part) -> Option<&Value> {
    match part {
        Part::Text { meta, .. }
        | Part::Image { meta, .. }
        | Part::Thinking { meta, .. }
        | Part::ToolCall { meta, .. }
        | Part::ToolResult { meta, .. } => meta.as_ref(),
    }
}

fn attach_token_usage(message: &mut ChatMessage, provider: &str, model: &str, usage: Usage) {
    if usage.total_tokens == 0 && usage.input_tokens == 0 && usage.output_tokens == 0 {
        return;
    }

    let Some(first_part) = message.parts.first_mut() else {
        return;
    };

    let slot = part_meta_mut(first_part);
    let mut meta = match slot.take() {
        Some(Value::Object(map)) => map,
        Some(value) => {
            let mut map = Map::new();
            map.insert("previous_meta".into(), value);
            map
        }
        None => Map::new(),
    };

    meta.insert(
        "token_usage".into(),
        json!({
            "source": "stream",
            "provider": provider,
            "model": model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "cache_creation_tokens": usage.cache_creation_tokens,
        }),
    );
    *slot = Some(Value::Object(meta));
}

fn part_meta_mut(part: &mut Part) -> &mut Option<Value> {
    match part {
        Part::Text { meta, .. }
        | Part::Image { meta, .. }
        | Part::Thinking { meta, .. }
        | Part::ToolCall { meta, .. }
        | Part::ToolResult { meta, .. } => meta,
    }
}

pub fn clean_context_descriptor() -> ToolDescriptor {
    ToolDescriptor {
        name: "clean_context".into(),
        description: "Prune useless tool results from your own context. MANDATORY whenever your turn included tool calls AND at least one produced noise (example : irrelevant Glob/Grep paths you never opened, a Read of an unrelated file, a failed exploration you retried elsewhere, etc.) — in that case you MUST call this before finishing. Keep anything you quoted, referenced, edited from, or based a decision on. If unsure, keep it. Current-turn tool results start with a tool_call_id line; use those exact ids.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "tool_call_ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Exact tool_call_id values."
                }
            },
            "required": ["tool_call_ids"],
            "additionalProperties": false
        }),
    }
}

fn update_goal_descriptor() -> ToolDescriptor {
    ToolDescriptor {
        name: "update_goal".into(),
        description: "Mark the active Goal mode objective complete. Use this only after auditing that the full objective is genuinely finished.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["complete"],
                    "description": "Use complete only when the goal is truly done."
                },
                "summary": {
                    "type": "string",
                    "description": "A concise summary of what was completed."
                }
            },
            "required": ["status"],
            "additionalProperties": false
        }),
    }
}

fn run_update_goal(goal_workflow: &mut GoalWorkflowState, input: Value) -> ToolRunResult {
    let status = input
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if status != "complete" {
        return ToolRunResult::err("status must be complete", Vec::new());
    }

    let Some((objective, started_at_ms)) = goal_objective_and_started(goal_workflow) else {
        return ToolRunResult::err("no active goal to update", Vec::new());
    };
    *goal_workflow = GoalWorkflowState::Complete {
        objective,
        started_at_ms,
        completed_at_ms: now_ms(),
    };

    let summary = input
        .get("summary")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("Goal marked complete");
    ToolRunResult::ok(summary.to_string(), Vec::new())
}

fn goal_objective_and_started(goal_workflow: &GoalWorkflowState) -> Option<(String, i64)> {
    match goal_workflow {
        GoalWorkflowState::Active {
            objective,
            started_at_ms,
            ..
        }
        | GoalWorkflowState::Paused {
            objective,
            started_at_ms,
            ..
        }
        | GoalWorkflowState::Complete {
            objective,
            started_at_ms,
            ..
        } => Some((objective.clone(), *started_at_ms)),
        GoalWorkflowState::Idle => None,
    }
}

fn goal_objective(goal_workflow: &GoalWorkflowState) -> Option<&str> {
    match goal_workflow {
        GoalWorkflowState::Active { objective, .. }
        | GoalWorkflowState::Paused { objective, .. }
        | GoalWorkflowState::Complete { objective, .. } => Some(objective.as_str()),
        GoalWorkflowState::Idle => None,
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

fn tool_result_content_with_id(tool_call_id: &str, content: &str) -> String {
    format!("tool_call_id: {tool_call_id}\n{content}")
}

fn history_with_current_tool_result_ids(
    history: &[ChatMessage],
    current_turn_tool_result_ids: &BTreeSet<String>,
) -> Vec<ChatMessage> {
    let mut history = history.to_vec();
    if current_turn_tool_result_ids.is_empty() {
        return history;
    }

    for message in &mut history {
        for part in &mut message.parts {
            let Part::ToolResult {
                tool_call_id,
                content,
                meta,
                ..
            } = part
            else {
                continue;
            };
            if !current_turn_tool_result_ids.contains(tool_call_id) || tool_result_cleaned(meta) {
                continue;
            }
            let stripped = strip_visible_tool_result_id(content);
            *content = tool_result_content_with_id(tool_call_id, &stripped);
        }
    }

    history
}

fn strip_all_visible_tool_result_ids(history: &mut [ChatMessage]) {
    for message in history {
        for part in &mut message.parts {
            let Part::ToolResult { content, .. } = part else {
                continue;
            };
            *content = strip_visible_tool_result_id(content);
        }
    }
}

fn normalize_tool_call_inputs(history: &mut [ChatMessage]) {
    for message in history {
        for part in &mut message.parts {
            let Part::ToolCall { input, .. } = part else {
                continue;
            };
            let normalized = normalize_tool_call_input(std::mem::take(input));
            *input = normalized;
        }
    }
}

fn normalize_tool_call_input(input: Value) -> Value {
    match input {
        Value::Object(_) => input,
        Value::Null => json!({}),
        Value::String(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                json!({})
            } else {
                match serde_json::from_str::<Value>(trimmed) {
                    Ok(Value::Object(map)) => Value::Object(map),
                    Ok(value) => json!({ "value": value }),
                    Err(_) => json!({ "value": raw }),
                }
            }
        }
        value => json!({ "value": value }),
    }
}

fn repair_missing_tool_results(history: &mut Vec<ChatMessage>) {
    let mut index = 0usize;
    while index < history.len() {
        if !matches!(history[index].role, Role::Assistant) {
            index += 1;
            continue;
        }
        let tool_call_ids = tool_call_ids(&history[index]);
        if tool_call_ids.is_empty() {
            index += 1;
            continue;
        }

        let next_user_tool_results = history
            .get(index + 1)
            .filter(|message| matches!(message.role, Role::User))
            .map(tool_result_ids)
            .unwrap_or_default();
        let missing = tool_call_ids
            .into_iter()
            .filter(|id| !next_user_tool_results.contains(id))
            .collect::<Vec<_>>();
        if missing.is_empty() {
            index += 1;
            continue;
        }

        let missing_parts = missing
            .into_iter()
            .map(|id| {
                interrupted_tool_result(id, "tool call was interrupted before a result was saved")
            })
            .collect::<Vec<_>>();
        let next_is_tool_result_message = history
            .get(index + 1)
            .filter(|message| matches!(message.role, Role::User))
            .map(|message| {
                !message.parts.is_empty()
                    && message
                        .parts
                        .iter()
                        .all(|part| matches!(part, Part::ToolResult { .. }))
            })
            .unwrap_or(false);
        if next_is_tool_result_message {
            history[index + 1].parts.extend(missing_parts);
        } else {
            history.insert(
                index + 1,
                ChatMessage {
                    role: Role::User,
                    parts: missing_parts,
                },
            );
        }
        index += 2;
    }
}

fn append_interrupted_tool_results(assistant: &ChatMessage, tool_results: &mut Vec<Part>) {
    let completed = tool_results
        .iter()
        .filter_map(|part| match part {
            Part::ToolResult { tool_call_id, .. } => Some(tool_call_id.clone()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    for id in tool_call_ids(assistant) {
        if completed.contains(&id) {
            continue;
        }
        tool_results.push(interrupted_tool_result(id, "tool call interrupted by user"));
    }
}

fn tool_call_ids(message: &ChatMessage) -> Vec<String> {
    message
        .parts
        .iter()
        .filter_map(|part| match part {
            Part::ToolCall { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect()
}

fn tool_result_ids(message: &ChatMessage) -> BTreeSet<String> {
    message
        .parts
        .iter()
        .filter_map(|part| match part {
            Part::ToolResult { tool_call_id, .. } => Some(tool_call_id.clone()),
            _ => None,
        })
        .collect()
}

fn interrupted_tool_result(id: String, content: &'static str) -> Part {
    Part::ToolResult {
        tool_call_id: id,
        content: content.to_string(),
        images: Vec::new(),
        is_error: true,
        meta: Some(json!({ "interrupted": true })),
    }
}

fn should_wait_for_cooperative_cancel(
    name: &str,
    subagents: Option<&Arc<SubAgentTool>>,
    teams: Option<&Arc<TeamTool>>,
) -> bool {
    name.starts_with("subagent_")
        || teams
            .and_then(|tool| tool.summary_for_tool_name(name))
            .is_some()
        || subagents
            .and_then(|tool| tool.summary_for_tool_name(name))
            .is_some()
}

fn strip_visible_tool_result_id(content: &str) -> String {
    let Some(rest) = content.strip_prefix("tool_call_id:") else {
        return content.to_string();
    };
    let Some(newline_index) = rest.find('\n') else {
        return String::new();
    };
    rest[newline_index + 1..].to_string()
}

#[cfg(test)]
fn tool_result_exposes_id(content: &str) -> bool {
    content.starts_with("tool_call_id:")
}

fn run_clean_context(
    history: &mut [ChatMessage],
    input: Value,
    current_turn_tool_result_ids: &BTreeSet<String>,
) -> ToolRunResult {
    let Some(values) = input
        .get("tool_call_ids")
        .or_else(|| input.get("ids"))
        .and_then(Value::as_array)
    else {
        return ToolRunResult::err("tool_call_ids must be an array", Vec::new());
    };
    let requested_ids = values
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
    let ids = requested_ids
        .intersection(current_turn_tool_result_ids)
        .cloned()
        .collect::<BTreeSet<_>>();
    let cleaned = clean_tool_results_by_ids(history, &ids);
    ToolRunResult::ok(
        format!(
            "cleaned: {}\nrequested: {}",
            cleaned.len(),
            requested_ids.len()
        ),
        Vec::new(),
    )
}

fn clean_tool_results_by_ids(history: &mut [ChatMessage], ids: &BTreeSet<String>) -> Vec<String> {
    let mut cleaned = Vec::new();
    if ids.is_empty() {
        return cleaned;
    }

    for message in history {
        for part in &mut message.parts {
            let Part::ToolResult {
                tool_call_id,
                content,
                images,
                meta,
                ..
            } = part
            else {
                continue;
            };
            if !ids.contains(tool_call_id) {
                continue;
            }
            *content = CLEAN_CONTEXT_RESULT_PLACEHOLDER.to_string();
            images.clear();
            mark_tool_result_cleaned(meta);
            cleaned.push(tool_call_id.clone());
        }
    }
    cleaned
}

fn mark_tool_result_cleaned(meta: &mut Option<Value>) {
    let mut map = match meta.take() {
        Some(Value::Object(map)) => map,
        Some(value) => {
            let mut map = Map::new();
            map.insert("previous_meta".into(), value);
            map
        }
        None => Map::new(),
    };
    map.insert("tool_result_cleaned".into(), json!(true));
    *meta = Some(Value::Object(map));
}

fn tool_result_cleaned(meta: &Option<Value>) -> bool {
    meta.as_ref()
        .and_then(|meta| meta.get("tool_result_cleaned"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

async fn run_tool(
    bash: &BashTool,
    glob: &GlobTool,
    grep: &GrepTool,
    read: &ReadTool,
    apply_patch: &ApplyPatchTool,
    create_image: &CreateImageTool,
    todo_list_tool: Option<&ToDoListTool>,
    question: Option<&QuestionTool>,
    web_search: &WebSearchTool,
    web_fetch: &WebFetchTool,
    skill: &SkillTool,
    mcp: &McpToolRegistry,
    subagents: Option<&SubAgentTool>,
    teams: Option<&TeamTool>,
    tool_settings: &ToolSettings,
    _read_paths: &BTreeSet<String>,
    todo_list: &mut TodoListState,
    mode: AgentMode,
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
    tool_call_id: &str,
    name: &str,
    input: Value,
) -> ToolRunResult {
    if !tool_settings.is_enabled(name) {
        return ToolRunResult::err(format!("{name} is disabled in Settings"), Vec::new());
    }
    if name == "bash" {
        bash.run(input).await
    } else if name == "bash_input" {
        bash.run_input(input).await
    } else if name == "Glob" {
        glob.run(input).await
    } else if name == "Grep" {
        grep.run(input).await
    } else if name == "read" {
        read.run(input).await
    } else if name == "apply_patch" {
        if mode == AgentMode::Plan {
            return ToolRunResult::err("apply_patch is unavailable in Plan mode", Vec::new());
        }
        apply_patch.run_with_read_paths(input).await
    } else if name == "CreateImage" {
        if mode == AgentMode::Plan {
            return ToolRunResult::err("CreateImage is unavailable in Plan mode", Vec::new());
        }
        create_image.run(input).await
    } else if name == "ToDoList" {
        let Some(todo_list_tool) = todo_list_tool else {
            return ToolRunResult::err("ToDoList is unavailable in this context", Vec::new());
        };
        todo_list_tool.run(input, todo_list).await
    } else if name == "Question" {
        let Some(question) = question else {
            return ToolRunResult::err("Question is unavailable in this context", Vec::new());
        };
        question.run(input).await
    } else if name == "WebSearch" {
        web_search.run(input).await
    } else if name == "WebFetch" {
        web_fetch.run(input).await
    } else if name == "skill" {
        skill.run(input).await
    } else if name.starts_with("subagent_") {
        let Some(subagents) = subagents else {
            return ToolRunResult::err(format!("unknown tool: {name}"), Vec::new());
        };
        subagents
            .run(tool_call_id, name, input, mode, event_tx.clone())
            .await
            .unwrap_or_else(|| ToolRunResult::err(format!("unknown tool: {name}"), Vec::new()))
    } else if let Some(teams) = teams {
        if let Some(result) = teams
            .run(tool_call_id, name, input.clone(), mode, event_tx.clone())
            .await
        {
            result
        } else if let Some(result) = mcp.run_tool(name, input).await {
            result
        } else {
            ToolRunResult::err(format!("unknown tool: {name}"), Vec::new())
        }
    } else if let Some(result) = mcp.run_tool(name, input).await {
        result
    } else {
        ToolRunResult::err(format!("unknown tool: {name}"), Vec::new())
    }
}

pub fn system_prompt_for_mode(base: &str, mode: AgentMode) -> String {
    match mode {
        AgentMode::Act => base.to_string(),
        AgentMode::Plan => format!("{base}\n\n<plan_mode>\n{PLAN_MODE_PROMPT}\n</plan_mode>"),
        AgentMode::Goal => format!("{base}\n\n<goal_mode>\n{GOAL_MODE_PROMPT}\n</goal_mode>"),
    }
}

fn system_prompt_for_turn(
    base: &str,
    mode: AgentMode,
    goal_workflow: &GoalWorkflowState,
) -> String {
    let prompt = system_prompt_for_mode(base, mode);
    if mode != AgentMode::Goal {
        return prompt;
    }
    let Some(objective) = goal_objective(goal_workflow) else {
        return prompt;
    };
    format!("{prompt}\n\n<goal_objective>\n{objective}\n</goal_objective>")
}

fn retain_cancelled_visible_parts(message: &mut ChatMessage) {
    message.parts.retain(|part| match part {
        Part::Text { text, .. } | Part::Thinking { text, .. } => !text.is_empty(),
        _ => false,
    });
}

fn append_plan_fallback_question(
    message: &mut ChatMessage,
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
    event_scope: Option<&AgentEventScope>,
) {
    let id = format!("plan-question-{}", Uuid::new_v4());
    let name = "Question".to_string();
    let input = json!({
        "question": "Je peux continuer a preparer le plan. Tu veux ajouter une contrainte avant que je le cree ?",
        "type": "single_choice",
        "options": [
            {
                "label": "Ajouter une contrainte",
                "description": "Je precise le scope, le gameplay, le style ou les priorites."
            },
            {
                "label": "Creer le plan maintenant",
                "description": "Je suis pret a generer le plan."
            }
        ]
    });

    send_event(
        event_tx,
        event_scope,
        AgentEvent::ToolStarted {
            id: id.clone(),
            name: name.clone(),
        },
    );
    send_event(
        event_tx,
        event_scope,
        AgentEvent::ToolReady {
            id: id.clone(),
            summary: summarize_tool(&name, &input),
            args_pretty: pretty_json(&input),
        },
    );

    message.parts.push(Part::ToolCall {
        id,
        name,
        input,
        meta: None,
    });
}

fn assistant_has_question_tool(message: &ChatMessage) -> bool {
    message.parts.iter().any(|part| {
        matches!(
            part,
            Part::ToolCall { name, .. } if name == "Question"
        )
    })
}

fn successful_read_paths(history: &[ChatMessage], read: &ReadTool) -> BTreeSet<String> {
    let mut pending_reads = HashMap::new();
    let mut successful = BTreeSet::new();

    for message in history {
        match message.role {
            Role::Assistant => {
                for part in &message.parts {
                    let Part::ToolCall {
                        id, name, input, ..
                    } = part
                    else {
                        continue;
                    };
                    if name != "read" {
                        continue;
                    }
                    let Some(path) = input.get("path").and_then(|value| value.as_str()) else {
                        continue;
                    };
                    if let Ok(normalized) = read.normalize_path(path) {
                        pending_reads.insert(id.clone(), normalized);
                    }
                }
            }
            Role::User => {
                for part in &message.parts {
                    let Part::ToolResult {
                        tool_call_id,
                        is_error,
                        meta,
                        ..
                    } = part
                    else {
                        continue;
                    };
                    if *is_error || tool_result_cleaned(meta) {
                        pending_reads.remove(tool_call_id);
                        continue;
                    }
                    if let Some(path) = pending_reads.remove(tool_call_id) {
                        successful.insert(path);
                    }
                }
            }
        }
    }

    successful
}

#[derive(Default)]
struct AssistantMessageBuilder {
    order: Vec<(usize, PartKind)>,
    text_parts: std::collections::HashMap<usize, String>,
    tool_json_parts: std::collections::HashMap<usize, String>,
    tool_heads: std::collections::HashMap<usize, (String, String)>,
    meta: std::collections::HashMap<usize, Value>,
    thinking_started: std::collections::HashMap<usize, Instant>,
}

impl AssistantMessageBuilder {
    fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    fn open(&mut self, index: usize, kind: PartKind) {
        self.order.push((index, kind));
        if matches!(kind, PartKind::Thinking) {
            self.thinking_started.insert(index, Instant::now());
        }
    }

    fn thinking_duration_ms(&self, index: usize) -> Option<u64> {
        self.thinking_started
            .get(&index)
            .map(|start| start.elapsed().as_millis() as u64)
    }

    fn kind(&self, index: usize) -> Option<PartKind> {
        self.order
            .iter()
            .find(|(candidate, _)| *candidate == index)
            .map(|(_, kind)| *kind)
    }

    fn register_tool(&mut self, index: usize, id: String, name: String) {
        self.tool_heads.insert(index, (id, name));
    }

    fn tool_head(&self, index: usize) -> Option<(String, String)> {
        self.tool_heads.get(&index).cloned()
    }

    fn push_text(&mut self, index: usize, chunk: &str) {
        self.text_parts.entry(index).or_default().push_str(chunk);
    }

    fn push_tool_json(&mut self, index: usize, chunk: &str) {
        self.tool_json_parts
            .entry(index)
            .or_default()
            .push_str(chunk);
    }

    fn push_meta(&mut self, index: usize, meta: Value) {
        self.meta.insert(index, meta);
    }

    fn insert_meta_field(&mut self, index: usize, key: &str, value: Value) {
        let current = self.meta.remove(&index);
        let mut meta = match current {
            Some(Value::Object(map)) => map,
            Some(value) => {
                let mut map = Map::new();
                map.insert("previous_meta".into(), value);
                map
            }
            None => Map::new(),
        };
        meta.insert(key.to_string(), value);
        self.meta.insert(index, Value::Object(meta));
    }

    fn finalize_tool(&self, index: usize) -> (String, String, Value) {
        let (id, name) = self.tool_heads.get(&index).cloned().unwrap_or_default();
        let raw = self
            .tool_json_parts
            .get(&index)
            .cloned()
            .unwrap_or_default();
        let value =
            normalize_tool_call_input(serde_json::from_str(&raw).unwrap_or(Value::String(raw)));
        (id, name, value)
    }

    fn finish(mut self) -> ChatMessage {
        let pending_thinking: Vec<(usize, u64)> = self
            .order
            .iter()
            .filter_map(|(index, kind)| {
                if !matches!(kind, PartKind::Thinking) {
                    return None;
                }
                let has_duration = matches!(
                    self.meta.get(index),
                    Some(Value::Object(map)) if map.contains_key("duration_ms")
                );
                if has_duration {
                    return None;
                }
                self.thinking_duration_ms(*index).map(|ms| (*index, ms))
            })
            .collect();
        for (index, ms) in pending_thinking {
            self.insert_meta_field(index, "duration_ms", json!(ms));
        }

        let mut parts = Vec::with_capacity(self.order.len());
        let order = self.order.clone();
        for (index, kind) in order {
            let meta = self.meta.get(&index).cloned();
            match kind {
                PartKind::Text => parts.push(Part::Text {
                    text: self.text_parts.get(&index).cloned().unwrap_or_default(),
                    meta,
                }),
                PartKind::Thinking => parts.push(Part::Thinking {
                    text: self.text_parts.get(&index).cloned().unwrap_or_default(),
                    meta,
                }),
                PartKind::ToolCall => {
                    let (id, name, input) = self.finalize_tool(index);
                    parts.push(Part::ToolCall {
                        id,
                        name,
                        input,
                        meta,
                    });
                }
            }
        }

        ChatMessage {
            role: Role::Assistant,
            parts,
        }
    }
}

fn summarize_tool(name: &str, input: &Value) -> String {
    if name == "bash" {
        if let Some(desc) = input
            .get("description")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            return desc.to_string();
        }
        if let Some(command) = input.get("command").and_then(|value| value.as_str()) {
            return command.to_string();
        }
    }
    if name == "bash_input" {
        if let Some(session_id) = input.get("session_id").and_then(|value| value.as_u64()) {
            if input
                .get("kill")
                .and_then(|value| value.as_bool())
                .unwrap_or(false)
            {
                return format!("Stop bash session {session_id}");
            }
            if input
                .get("input")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .is_some()
            {
                return format!("Send input to bash session {session_id}");
            }
            return format!("Poll bash session {session_id}");
        }
    }
    if name == "read" {
        if let Some(path) = input.get("path").and_then(|value| value.as_str()) {
            return format!("Read {path}");
        }
    }
    if name == "Grep" {
        let scope = input
            .get("path")
            .or_else(|| input.get("include"))
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty() && *value != ".")
            .unwrap_or("workspace");
        return format!("Grep in {scope}");
    }
    if name == "Glob" {
        let pattern = input
            .get("pattern")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("*");
        let scope = input
            .get("path")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty() && *value != ".")
            .unwrap_or("workspace");
        return format!("Glob {pattern} in {scope}");
    }
    if name == "apply_patch" {
        return "Apply patch".to_string();
    }
    if name == "clean_context" {
        let count = input
            .get("tool_call_ids")
            .or_else(|| input.get("ids"))
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or_default();
        return if count == 0 {
            "Clean context".to_string()
        } else {
            format!("Clean context · {count} results")
        };
    }
    if name == "update_goal" {
        return "Goal finished".to_string();
    }
    if name == "CreateImage" {
        if let Some(prompt) = input
            .get("prompt")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            let mut clipped = prompt.chars().take(64).collect::<String>();
            if prompt.chars().count() > 64 {
                clipped.push_str("...");
            }
            return format!("Create image: {clipped}");
        }
        return "Create image".to_string();
    }
    if name == "ToDoList" {
        if let Some(changes) = input
            .get("changes")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            if changes.eq_ignore_ascii_case("close") || changes.eq_ignore_ascii_case("clear") {
                return "Close ToDoList".to_string();
            }
        }
        return "Update ToDoList".to_string();
    }
    if name == "Question" {
        if let Some(count) = input
            .get("questions")
            .and_then(|value| value.as_array())
            .map(Vec::len)
        {
            return if count == 1 {
                "Question".to_string()
            } else {
                format!("{count} questions")
            };
        }
        if let Some(question) = input
            .get("question")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return question.to_string();
        }
        return "Question".to_string();
    }
    if name == "LoadMcpTool" {
        let server = input
            .get("server")
            .or_else(|| input.get("serverName"))
            .or_else(|| input.get("server_name"))
            .and_then(|value| value.as_str())
            .map(display_mcp_server_name)
            .filter(|value| !value.is_empty());
        let tool = input
            .get("tool")
            .or_else(|| input.get("toolName"))
            .or_else(|| input.get("tool_name"))
            .or_else(|| input.get("name"))
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty());
        if let (Some(server), Some(tool)) = (server, tool) {
            return format!("Load {server} · {tool}");
        }
        return "Load MCP tool".to_string();
    }
    if name == "skill" {
        if let Some(skill) = input
            .get("name")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return format!("Load skill · {skill}");
        }
        return "Load skill".to_string();
    }
    if name.starts_with("subagent_") {
        if let Some(task) = input
            .get("task")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return format!("Sub-agent · {task}");
        }
        return "Sub-agent".to_string();
    }
    if name == "TeamRun" {
        let team = input
            .get("team_name")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let agent = input
            .get("agent")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let objective = input
            .get("objective")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty());
        return match (team, agent, objective) {
            (Some(team), Some(agent), _) => format!("Agent Swarm · restart @{agent} · {team}"),
            (None, Some(agent), _) => format!("Agent Swarm · restart @{agent}"),
            (Some(team), None, Some(objective)) => format!("Agent Swarm · {team} · {objective}"),
            (Some(team), None, None) => format!("Agent Swarm · {team}"),
            (None, None, Some(objective)) => format!("Agent Swarm · {objective}"),
            _ => "Agent Swarm".to_string(),
        };
    }
    if name == "TeamCreate" {
        if let Some(team) = input
            .get("team_name")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return format!("Agent Swarm · {team}");
        }
        return "Create Agent Swarm".to_string();
    }
    if name == "Agent" {
        let teammate = input
            .get("name")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let task = input
            .get("description")
            .or_else(|| input.get("prompt"))
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty());
        return match (teammate, task) {
            (Some(teammate), Some(task)) => format!("Agent · @{teammate} · {task}"),
            (Some(teammate), None) => format!("Agent · @{teammate}"),
            _ => "Agent teammate".to_string(),
        };
    }
    if name == "SendMessage" {
        if let Some(to) = input
            .get("to")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return format!("Message · {to}");
        }
        return "Send Agent Swarm message".to_string();
    }
    if name == "TaskCreate" {
        if let Some(subject) = input
            .get("subject")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return format!("Task · create · {subject}");
        }
        return "Create task".to_string();
    }
    if name == "TaskList" {
        let action = input
            .get("action")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let task_id = input
            .get("taskId")
            .or_else(|| input.get("id"))
            .and_then(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .or_else(|| value.as_u64().map(|value| value.to_string()))
            });
        let subject = input
            .get("subject")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty());
        return match (action, task_id, subject) {
            (Some("create"), _, Some(subject)) => format!("Task · create · {subject}"),
            (Some(action @ ("update" | "claim" | "delete")), Some(task_id), _) => {
                format!("Task · {action} · #{task_id}")
            }
            (Some(action), _, _) => format!("Task · {action}"),
            _ => "Task list".to_string(),
        };
    }
    if name == "TaskUpdate" {
        let task_id = input
            .get("taskId")
            .or_else(|| input.get("id"))
            .and_then(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .or_else(|| value.as_u64().map(|value| value.to_string()))
            });
        let status = input
            .get("status")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty());
        return match (task_id, status) {
            (Some(task_id), Some(status)) => format!("Task · #{task_id} · {status}"),
            (Some(task_id), None) => format!("Task · #{task_id}"),
            _ => "Update task".to_string(),
        };
    }
    if name == "TeamStatus" {
        return "Agent Swarm status".to_string();
    }
    if name == "TeamStop" {
        return "Stop Agent Swarm".to_string();
    }
    if name == "WebSearch" {
        if let Some(q) = input
            .get("q")
            .or_else(|| input.get("query"))
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return format!("Search web: {q}");
        }
        return "Search web".to_string();
    }
    if name == "WebFetch" {
        if let Some(url) = input
            .get("url")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return format!("Fetch {url}");
        }
        return "Fetch URL".to_string();
    }

    if let Ok(pretty) = serde_json::to_string(input) {
        if pretty.len() <= 72 {
            return pretty;
        }
        let mut clipped = pretty.chars().take(69).collect::<String>();
        clipped.push_str("...");
        return clipped;
    }

    name.to_string()
}

fn should_stream_tool_args(name: &str) -> bool {
    matches!(name, "apply_patch" | "read")
}

fn display_mcp_server_name(value: &str) -> String {
    let trimmed = value.trim();
    let Some(rest) = trimmed.get(3..) else {
        return trimmed.to_string();
    };
    if !trimmed[..3].eq_ignore_ascii_case("mcp") {
        return trimmed.to_string();
    }

    let stripped = rest
        .trim_start_matches(|ch: char| ch == '-' || ch == '_' || ch == '.' || ch.is_whitespace())
        .trim();
    if stripped.is_empty() {
        trimmed.to_string()
    } else {
        stripped.to_string()
    }
}

fn pretty_json(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancelled_visible_parts_keep_partial_text_only() {
        let mut message = ChatMessage {
            role: Role::Assistant,
            parts: vec![
                Part::Text {
                    text: "partial answer".to_string(),
                    meta: None,
                },
                Part::Thinking {
                    text: "partial thought".to_string(),
                    meta: None,
                },
                Part::Text {
                    text: String::new(),
                    meta: None,
                },
                Part::ToolCall {
                    id: "call-1".to_string(),
                    name: "read".to_string(),
                    input: json!({ "path": "Cargo.toml" }),
                    meta: None,
                },
            ],
        };

        retain_cancelled_visible_parts(&mut message);

        assert_eq!(message.parts.len(), 2);
        assert!(matches!(&message.parts[0], Part::Text { text, .. } if text == "partial answer"));
        assert!(
            matches!(&message.parts[1], Part::Thinking { text, .. } if text == "partial thought")
        );
    }

    #[test]
    fn clean_context_replaces_matching_tool_results() {
        let mut history = vec![ChatMessage {
            role: Role::User,
            parts: vec![
                Part::ToolResult {
                    tool_call_id: "call-1".to_string(),
                    content: "noisy grep output".to_string(),
                    images: Vec::new(),
                    is_error: false,
                    meta: None,
                },
                Part::ToolResult {
                    tool_call_id: "call-2".to_string(),
                    content: "useful read output".to_string(),
                    images: Vec::new(),
                    is_error: false,
                    meta: None,
                },
            ],
        }];

        let result = run_clean_context(
            &mut history,
            json!({ "tool_call_ids": ["call-1", "missing"] }),
            &BTreeSet::from(["call-1".to_string()]),
        );

        assert!(!result.is_error);
        assert!(result.content.contains("cleaned: 1"));
        let Part::ToolResult { content, meta, .. } = &history[0].parts[0] else {
            panic!("expected tool result");
        };
        assert_eq!(content, CLEAN_CONTEXT_RESULT_PLACEHOLDER);
        assert!(tool_result_cleaned(meta));

        let Part::ToolResult { content, .. } = &history[0].parts[1] else {
            panic!("expected tool result");
        };
        assert_eq!(content, "useful read output");
    }

    #[test]
    fn clean_context_ignores_ids_outside_current_turn() {
        let mut history = vec![ChatMessage {
            role: Role::User,
            parts: vec![Part::ToolResult {
                tool_call_id: "old-call".to_string(),
                content: "old useful output".to_string(),
                images: Vec::new(),
                is_error: false,
                meta: None,
            }],
        }];

        let result = run_clean_context(
            &mut history,
            json!({ "tool_call_ids": ["old-call"] }),
            &BTreeSet::new(),
        );

        assert!(!result.is_error);
        assert!(result.content.contains("cleaned: 0"));
        assert!(result.content.contains("requested: 1"));
        let Part::ToolResult { content, meta, .. } = &history[0].parts[0] else {
            panic!("expected tool result");
        };
        assert_eq!(content, "old useful output");
        assert!(!tool_result_cleaned(meta));
    }

    #[test]
    fn cleaned_read_results_do_not_count_as_successful_reads() {
        let read = ReadTool::new(".");
        let history = vec![
            ChatMessage {
                role: Role::Assistant,
                parts: vec![Part::ToolCall {
                    id: "read-1".to_string(),
                    name: "read".to_string(),
                    input: json!({ "path": "src/lib.rs", "limit": 10 }),
                    meta: None,
                }],
            },
            ChatMessage {
                role: Role::User,
                parts: vec![Part::ToolResult {
                    tool_call_id: "read-1".to_string(),
                    content: CLEAN_CONTEXT_RESULT_PLACEHOLDER.to_string(),
                    images: Vec::new(),
                    is_error: false,
                    meta: Some(json!({ "tool_result_cleaned": true })),
                }],
            },
        ];

        assert!(successful_read_paths(&history, &read).is_empty());
    }

    #[test]
    fn tool_result_content_exposes_tool_call_id() {
        assert_eq!(
            tool_result_content_with_id("call-1", "hello"),
            "tool_call_id: call-1\nhello"
        );
    }

    #[test]
    fn request_history_exposes_only_current_turn_tool_result_ids() {
        let history = vec![ChatMessage {
            role: Role::User,
            parts: vec![
                Part::ToolResult {
                    tool_call_id: "call-1".to_string(),
                    content: "current result".to_string(),
                    images: Vec::new(),
                    is_error: false,
                    meta: None,
                },
                Part::ToolResult {
                    tool_call_id: "call-2".to_string(),
                    content: "old result".to_string(),
                    images: Vec::new(),
                    is_error: false,
                    meta: None,
                },
            ],
        }];
        let ids = BTreeSet::from(["call-1".to_string()]);

        let request_history = history_with_current_tool_result_ids(&history, &ids);
        let Part::ToolResult {
            content: current_content,
            ..
        } = &request_history[0].parts[0]
        else {
            panic!("expected tool result");
        };
        let Part::ToolResult {
            content: old_content,
            ..
        } = &request_history[0].parts[1]
        else {
            panic!("expected tool result");
        };

        assert!(tool_result_exposes_id(current_content));
        assert!(!tool_result_exposes_id(old_content));
        let Part::ToolResult { content, .. } = &history[0].parts[0] else {
            panic!("expected tool result");
        };
        assert!(!tool_result_exposes_id(content));
    }

    #[test]
    fn legacy_visible_tool_result_ids_are_stripped_from_history() {
        let mut history = vec![ChatMessage {
            role: Role::User,
            parts: vec![Part::ToolResult {
                tool_call_id: "call-1".to_string(),
                content: "tool_call_id: call-1\nhello".to_string(),
                images: Vec::new(),
                is_error: false,
                meta: None,
            }],
        }];

        strip_all_visible_tool_result_ids(&mut history);

        let Part::ToolResult { content, .. } = &history[0].parts[0] else {
            panic!("expected tool result");
        };
        assert_eq!(content, "hello");
    }

    #[test]
    fn tool_call_inputs_are_normalized_for_provider_replay() {
        let mut history = vec![ChatMessage {
            role: Role::Assistant,
            parts: vec![
                Part::ToolCall {
                    id: "call-empty".to_string(),
                    name: "TeamStop".to_string(),
                    input: json!(""),
                    meta: None,
                },
                Part::ToolCall {
                    id: "call-json".to_string(),
                    name: "TeamStop".to_string(),
                    input: json!("{\"agent\":\"ui\"}"),
                    meta: None,
                },
                Part::ToolCall {
                    id: "call-string".to_string(),
                    name: "bash".to_string(),
                    input: json!("ls"),
                    meta: None,
                },
            ],
        }];

        normalize_tool_call_inputs(&mut history);

        let Part::ToolCall {
            input: empty_input, ..
        } = &history[0].parts[0]
        else {
            panic!("expected tool call");
        };
        let Part::ToolCall {
            input: json_input, ..
        } = &history[0].parts[1]
        else {
            panic!("expected tool call");
        };
        let Part::ToolCall {
            input: string_input,
            ..
        } = &history[0].parts[2]
        else {
            panic!("expected tool call");
        };

        assert_eq!(empty_input, &json!({}));
        assert_eq!(json_input, &json!({ "agent": "ui" }));
        assert_eq!(string_input, &json!({ "value": "ls" }));
    }
}
