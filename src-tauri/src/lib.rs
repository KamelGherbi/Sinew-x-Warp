use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex as StdMutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
#[cfg(target_os = "macos")]
use objc2::{
    ffi::class_addMethod,
    rc::Retained,
    runtime::{AnyClass, AnyObject, Imp, Sel},
    MainThreadMarker,
};
#[cfg(target_os = "macos")]
use objc2_app_kit::{NSApplication, NSMenu, NSMenuItem};
#[cfg(target_os = "macos")]
use objc2_foundation::NSString;
use portable_pty::{native_pty_system, Child, ChildKiller, CommandBuilder, MasterPty, PtySize};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_anthropic::{
    delete_default_auth as delete_default_anthropic_auth,
    exchange_oauth_code as exchange_anthropic_oauth_code, generate_pkce as generate_anthropic_pkce,
    generate_state as generate_anthropic_state,
    load_default_auth_status as load_default_anthropic_auth_status,
    oauth_authorize_url as anthropic_oauth_authorize_url, AnthropicAuthStatus, AnthropicProvider,
    PkceCodes as AnthropicPkceCodes, MODEL_ID as ANTHROPIC_MODEL_ID,
};
use sinew_app::{
    checkpoint_from_snapshots, clean_context_descriptor, compact_conversation_history,
    copy_workspace_entries, create_workspace_directory, create_workspace_file,
    delete_workspace_entry, import_workspace_paths, list_installed_skills, list_workspace_entries,
    list_workspace_files, normalize_workspace_root, probe_mcp_servers, read_external_file,
    read_workspace_file, rename_workspace_entry, resolve_terminal_path, restore_turn_checkpoints,
    restore_workspace_deleted_entries, run_turn, search_workspace_files,
    snapshot_workspace_for_checkpoint, subagent_system_prompt, system_prompt_for_mode,
    system_prompt_with_todo, todo_list_from_history, tool_settings_view, trash_workspace_entry,
    write_workspace_file, AgentEvent, AgentMode, AppStore, ApplyPatchTool, BashTool,
    ConversationEvent, ConversationSummary, CreateImageTool, GlobTool, GoalWorkflowState, GrepTool,
    ImportedEntry, InstalledSkill, McpSettings, McpToolRegistry, ModeModelSettings,
    PlanArtifactState, PlanWorkflowState, QuestionTool, ReadTool, SavedConversation, SkillSettings,
    SkillTool, SubAgentConfig, SubAgentSettings, SubAgentTool, TeamRuntime, TeamTool,
    TerminalPathResolution, ToDoListTool, TodoListState, ToolSettings, ToolSettingsView,
    TurnCancel, TurnContext, WebFetchTool, WebSearchTool, WorkspaceBootstrap,
    WorkspaceCopyOperation, WorkspaceDeletedEntry, WorkspaceFileChangeEvent, WorkspaceSearchResult,
};
use sinew_core::{
    ChatMessage, Effort, ModelRef, Part, Provider, ProviderRequest, Role, ToolDescriptor,
};
use sinew_google::{
    delete_default_auth as delete_default_google_auth,
    exchange_oauth_code as exchange_google_oauth_code, generate_state as generate_google_state,
    load_default_auth_status as load_default_google_auth_status,
    oauth_authorize_url as google_oauth_authorize_url, GoogleAuthStatus, GoogleProvider,
    MODEL_ID as GOOGLE_MODEL_ID,
};
use sinew_kimi::{
    delete_default_auth as delete_default_kimi_auth, generate_state as generate_kimi_state,
    load_default_auth_status as load_default_kimi_auth_status,
    request_device_authorization as request_kimi_device_authorization,
    wait_for_device_token as wait_for_kimi_device_token,
    DeviceAuthorization as KimiDeviceAuthorization, KimiAuthStatus, KimiProvider,
    MODEL_ID as KIMI_MODEL_ID,
};
use sinew_openai::{
    delete_default_auth, exchange_oauth_code, generate_pkce, generate_state,
    load_default_auth_status, oauth_authorize_url, OpenAiAuthStatus, OpenAiProvider, PkceCodes,
    MODEL_ID as OPENAI_MODEL_ID,
};
use tauri::{AppHandle, Emitter, Manager, State, WebviewUrl, WebviewWindowBuilder};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    sync::{mpsc, Mutex, Notify, RwLock},
};

const DEFAULT_SYSTEM_PROMPT: &str = "You are Sinew, a coding assistant. You build context by examining the codebase first without making assumptions or jumping to conclusions. When exploring, you provide user updates frequently, every 30s. ALWAYS check for a dedicated tool that fits the task before falling back to bash.";
const WORKSPACE_INSTRUCTIONS_FILE: &str = "AGENTS.md";
const WORKSPACE_DESIGN_FILE: &str = "DESIGN.md";
const AGENT_EVENT_NAME: &str = "agent-event";
const FILE_CHANGE_EVENT_NAME: &str = "workspace-file-changed";
const TERMINAL_DATA_EVENT_NAME: &str = "terminal-data";
const TERMINAL_EXIT_EVENT_NAME: &str = "terminal-exit";
const TERMINAL_OPEN_EVENT_NAME: &str = "terminal-open-requested";
const TERMINAL_OPEN_MENU_ID: &str = "terminal-open";
const NEW_WINDOW_MENU_ID: &str = "new-window";
const NEW_WINDOW_LABEL_PREFIX: &str = "sinew-window";
const NEW_WINDOW_URL: &str = "index.html?newWindow=1";
const MAX_ATTACHMENT_BYTES: usize = 128 * 1024;
const MAX_IMAGE_BYTES: usize = 10 * 1024 * 1024;
const TURN_SLOT_WAIT_ATTEMPTS: usize = 30;
const TURN_SLOT_WAIT_INTERVAL_MS: u64 = 50;
const SWARM_WAKE_TURN_SLOT_WAIT_ATTEMPTS: usize = 600;

#[cfg(target_os = "macos")]
static MACOS_APP_HANDLE: std::sync::OnceLock<AppHandle> = std::sync::OnceLock::new();

struct TerminalProcess {
    token: String,
    master: Box<dyn MasterPty + Send>,
    writer: Arc<StdMutex<Box<dyn Write + Send>>>,
    killer: Arc<StdMutex<Box<dyn ChildKiller + Send + Sync>>>,
}

#[derive(Clone)]
struct DesktopState {
    providers: Arc<StdMutex<HashMap<String, Arc<dyn Provider>>>>,
    store: AppStore,
    default_model: ModelRef,
    system_prompt: String,
    max_tool_rounds: usize,
    active_turns: Arc<Mutex<HashMap<String, TurnCancel>>>,
    team_runtime: Arc<RwLock<TeamRuntime>>,
    file_watchers: Arc<Mutex<HashMap<String, RecommendedWatcher>>>,
    terminal_sessions: Arc<Mutex<HashMap<String, TerminalProcess>>>,
    openai_login: Arc<Mutex<Option<OpenAiLoginAttempt>>>,
    anthropic_login: Arc<Mutex<Option<AnthropicLoginAttempt>>>,
    google_login: Arc<Mutex<Option<GoogleLoginAttempt>>>,
    kimi_login: Arc<Mutex<Option<KimiLoginAttempt>>>,
}

#[derive(Clone)]
struct OpenAiLoginAttempt {
    id: String,
    cancel: Arc<Notify>,
    outcome: Arc<StdMutex<Option<OpenAiLoginOutcome>>>,
}

#[derive(Clone)]
struct OpenAiLoginOutcome {
    success: bool,
    error: Option<String>,
}

#[derive(Clone)]
struct AnthropicLoginAttempt {
    id: String,
    cancel: Arc<Notify>,
    outcome: Arc<StdMutex<Option<AnthropicLoginOutcome>>>,
}

#[derive(Clone)]
struct AnthropicLoginOutcome {
    success: bool,
    error: Option<String>,
}

#[derive(Clone)]
struct GoogleLoginAttempt {
    id: String,
    cancel: Arc<Notify>,
    outcome: Arc<StdMutex<Option<GoogleLoginOutcome>>>,
}

#[derive(Clone)]
struct GoogleLoginOutcome {
    success: bool,
    error: Option<String>,
}

#[derive(Clone)]
struct KimiLoginAttempt {
    id: String,
    cancel: Arc<Notify>,
    outcome: Arc<StdMutex<Option<KimiLoginOutcome>>>,
}

#[derive(Clone)]
struct KimiLoginOutcome {
    success: bool,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct OpenAiProviderStatus {
    connected: bool,
    connection_state: String,
    email: Option<String>,
    account_id: Option<String>,
    plan_type: Option<String>,
    expires_at_ms: Option<i64>,
    last_refresh_ms: Option<i64>,
    login_id: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StartOpenAiLoginOutput {
    login_id: String,
    auth_url: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AnthropicProviderStatus {
    connected: bool,
    connection_state: String,
    expires_at_ms: Option<i64>,
    last_refresh_ms: Option<i64>,
    login_id: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StartAnthropicLoginOutput {
    login_id: String,
    auth_url: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleProviderStatus {
    connected: bool,
    connection_state: String,
    email: Option<String>,
    project_id: Option<String>,
    user_tier: Option<String>,
    expires_at_ms: Option<i64>,
    last_refresh_ms: Option<i64>,
    login_id: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StartGoogleLoginOutput {
    login_id: String,
    auth_url: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct KimiProviderStatus {
    connected: bool,
    connection_state: String,
    expires_at_ms: Option<i64>,
    last_refresh_ms: Option<i64>,
    login_id: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StartKimiLoginOutput {
    login_id: String,
    auth_url: String,
    user_code: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WorkspaceInput {
    workspace_path: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WorkspaceEntriesInput {
    workspace_path: String,
    relative_path: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WorkspaceFileInput {
    workspace_path: String,
    relative_path: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RestoreWorkspaceDeletedEntriesInput {
    workspace_path: String,
    entries: Vec<WorkspaceDeletedEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WorkspaceSearchInput {
    workspace_path: String,
    query: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WriteWorkspaceFileInput {
    workspace_path: String,
    relative_path: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateWorkspaceEntryInput {
    workspace_path: String,
    target_relative_path: Option<String>,
    name: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RenameWorkspaceEntryInput {
    workspace_path: String,
    relative_path: String,
    new_name: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CopyWorkspaceEntriesInput {
    workspace_path: String,
    target_relative_path: Option<String>,
    sources: Vec<String>,
    cut: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConversationInput {
    workspace_path: String,
    conversation_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StopAgentSwarmInput {
    workspace_path: String,
    conversation_id: String,
    team_name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RenameConversationInput {
    workspace_path: String,
    conversation_id: String,
    title: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct AttachmentInput {
    path: String,
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClipboardImageInput {
    workspace_path: String,
    name: Option<String>,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ClipboardImageAttachment {
    path: String,
    name: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SendMessageInput {
    workspace_path: String,
    conversation_id: String,
    text: String,
    #[serde(default)]
    attachments: Vec<AttachmentInput>,
    model: Option<ModelInput>,
    thinking: Option<ThinkingLevelInput>,
    mode: Option<AgentModeInput>,
    plan_control: Option<PlanControlInput>,
    message_visibility: Option<MessageVisibilityInput>,
    #[serde(default)]
    rewrite_from_history_index: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CompactConversationInput {
    workspace_path: String,
    conversation_id: String,
    model: Option<ModelInput>,
    thinking: Option<ThinkingLevelInput>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContextEstimateInput {
    workspace_path: String,
    conversation_id: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    attachments: Vec<AttachmentInput>,
    model: Option<ModelInput>,
    thinking: Option<ThinkingLevelInput>,
    mode: Option<AgentModeInput>,
    #[serde(default)]
    rewrite_from_history_index: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SubAgentContextEstimateInput {
    workspace_path: String,
    agent_id: String,
    #[serde(default)]
    agent_name: Option<String>,
    history: Vec<ChatMessage>,
    model: ModelRef,
    mode: Option<AgentModeInput>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConversationModeInput {
    workspace_path: String,
    conversation_id: String,
    mode: AgentModeInput,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConversationModelPreferenceInput {
    workspace_path: String,
    conversation_id: String,
    mode: AgentModeInput,
    model: Option<ModelInput>,
    thinking: Option<ThinkingLevelInput>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ContextEstimateOutput {
    used_tokens: u32,
    context_window: u32,
    preferred_window: u32,
    max_output_tokens: u32,
    input_tokens: u32,
    output_tokens: u32,
    reasoning_tokens: u32,
    cache_read_tokens: u32,
    cache_creation_tokens: u32,
    exact: bool,
    error: Option<String>,
    breakdown: Vec<ContextBreakdownItem>,
}

#[derive(Debug, Clone, Copy, Default)]
struct ContextTokenUsage {
    input_tokens: u32,
    output_tokens: u32,
    reasoning_tokens: u32,
    cache_read_tokens: u32,
    cache_creation_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ContextBreakdownItem {
    key: String,
    label: String,
    tokens: u32,
}

#[derive(Debug, Clone)]
struct ContextBreakdownWeight {
    key: &'static str,
    label: &'static str,
    weight: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SaveMcpSettingsInput {
    settings: McpSettings,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SaveToolSettingsInput {
    workspace_path: String,
    settings: ToolSettings,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SaveSkillSettingsInput {
    workspace_path: String,
    settings: SkillSettings,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SaveSubAgentSettingsInput {
    settings: SubAgentSettings,
}

#[derive(Debug, Deserialize)]
struct ModelInput {
    provider: String,
    name: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TerminalCommandInput {
    workspace_path: String,
    command: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TerminalCommandOutput {
    content: String,
    is_error: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TerminalSpawnInput {
    workspace_path: String,
    session_id: String,
    token: String,
    cols: u16,
    rows: u16,
    #[serde(default)]
    pixel_width: u16,
    #[serde(default)]
    pixel_height: u16,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TerminalSpawnOutput {
    session_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TerminalWriteInput {
    session_id: String,
    token: String,
    data: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TerminalResizeInput {
    session_id: String,
    token: String,
    cols: u16,
    rows: u16,
    #[serde(default)]
    pixel_width: u16,
    #[serde(default)]
    pixel_height: u16,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TerminalControlInput {
    session_id: String,
    token: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OpenExternalUrlInput {
    url: String,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct TerminalDataEvent {
    session_id: String,
    token: String,
    data: String,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct TerminalExitEvent {
    session_id: String,
    token: String,
    exit_code: Option<u32>,
    signal: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum AgentModeInput {
    Act,
    Plan,
    Goal,
}

impl From<AgentModeInput> for AgentMode {
    fn from(value: AgentModeInput) -> Self {
        match value {
            AgentModeInput::Act => AgentMode::Act,
            AgentModeInput::Plan => AgentMode::Plan,
            AgentModeInput::Goal => AgentMode::Goal,
        }
    }
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum PlanControlInput {
    StopQuestions,
    UpdatePlan,
    ImplementPlan,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum MessageVisibilityInput {
    Normal,
    SystemReminder,
}

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum ThinkingLevelInput {
    Off,
    Low,
    Medium,
    High,
    Max,
    Xhigh,
}

impl ThinkingLevelInput {
    fn into_effort(self) -> Effort {
        match self {
            Self::Off => Effort::None,
            Self::Low => Effort::Low,
            Self::Medium => Effort::Medium,
            Self::High => Effort::High,
            Self::Xhigh => Effort::Xhigh,
            Self::Max => Effort::Max,
        }
    }
}

fn model_with_optional_selection(
    current: &ModelRef,
    model: Option<ModelInput>,
    thinking: Option<ThinkingLevelInput>,
) -> ModelRef {
    let mut selected = match model {
        Some(model) => ModelRef::new(model.provider, model.name),
        None => current.clone(),
    };
    if let Some(thinking) = thinking {
        selected.effort = Some(thinking.into_effort());
    }
    selected
}

fn provider_registry_snapshot(
    state: &DesktopState,
) -> std::result::Result<HashMap<String, Arc<dyn Provider>>, String> {
    state
        .providers
        .lock()
        .map_err(|_| "provider registry is unavailable".to_string())
        .map(|providers| providers.clone())
}

fn provider_from_registry(
    state: &DesktopState,
    provider_id: &str,
) -> std::result::Result<Arc<dyn Provider>, String> {
    state
        .providers
        .lock()
        .map_err(|_| "provider registry is unavailable".to_string())?
        .get(provider_id)
        .cloned()
        .ok_or_else(|| format!("provider `{provider_id}` is not configured or missing credentials"))
}

#[tauri::command]
fn list_configured_model_providers(
    state: State<'_, DesktopState>,
) -> std::result::Result<Vec<String>, String> {
    let mut providers = state
        .providers
        .lock()
        .map_err(|_| "provider registry is unavailable".to_string())?
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    providers.sort();
    Ok(providers)
}

fn install_openai_provider(
    providers: &Arc<StdMutex<HashMap<String, Arc<dyn Provider>>>>,
) -> std::result::Result<(), String> {
    let provider = OpenAiProvider::from_default_sources().map_err(error_to_string)?;
    providers
        .lock()
        .map_err(|_| "provider registry is unavailable".to_string())?
        .insert("openai".into(), Arc::new(provider) as Arc<dyn Provider>);
    Ok(())
}

fn install_anthropic_provider(
    providers: &Arc<StdMutex<HashMap<String, Arc<dyn Provider>>>>,
) -> std::result::Result<(), String> {
    let provider = AnthropicProvider::from_default_sources().map_err(error_to_string)?;
    providers
        .lock()
        .map_err(|_| "provider registry is unavailable".to_string())?
        .insert("anthropic".into(), Arc::new(provider) as Arc<dyn Provider>);
    Ok(())
}

fn install_google_provider(
    providers: &Arc<StdMutex<HashMap<String, Arc<dyn Provider>>>>,
) -> std::result::Result<(), String> {
    let provider = GoogleProvider::from_default_sources().map_err(error_to_string)?;
    providers
        .lock()
        .map_err(|_| "provider registry is unavailable".to_string())?
        .insert("google".into(), Arc::new(provider) as Arc<dyn Provider>);
    Ok(())
}

fn install_kimi_provider(
    providers: &Arc<StdMutex<HashMap<String, Arc<dyn Provider>>>>,
) -> std::result::Result<(), String> {
    let provider = KimiProvider::from_default_sources().map_err(error_to_string)?;
    providers
        .lock()
        .map_err(|_| "provider registry is unavailable".to_string())?
        .insert("kimi".into(), Arc::new(provider) as Arc<dyn Provider>);
    Ok(())
}

fn remove_openai_provider(
    providers: &Arc<StdMutex<HashMap<String, Arc<dyn Provider>>>>,
) -> std::result::Result<(), String> {
    providers
        .lock()
        .map_err(|_| "provider registry is unavailable".to_string())?
        .remove("openai");
    Ok(())
}

fn remove_anthropic_provider(
    providers: &Arc<StdMutex<HashMap<String, Arc<dyn Provider>>>>,
) -> std::result::Result<(), String> {
    providers
        .lock()
        .map_err(|_| "provider registry is unavailable".to_string())?
        .remove("anthropic");
    Ok(())
}

fn remove_google_provider(
    providers: &Arc<StdMutex<HashMap<String, Arc<dyn Provider>>>>,
) -> std::result::Result<(), String> {
    providers
        .lock()
        .map_err(|_| "provider registry is unavailable".to_string())?
        .remove("google");
    Ok(())
}

fn remove_kimi_provider(
    providers: &Arc<StdMutex<HashMap<String, Arc<dyn Provider>>>>,
) -> std::result::Result<(), String> {
    providers
        .lock()
        .map_err(|_| "provider registry is unavailable".to_string())?
        .remove("kimi");
    Ok(())
}

fn openai_provider_status_from_auth(
    auth: OpenAiAuthStatus,
    connection_state: &str,
    login_id: Option<String>,
    error: Option<String>,
) -> OpenAiProviderStatus {
    OpenAiProviderStatus {
        connected: auth.connected,
        connection_state: connection_state.to_string(),
        email: auth.email,
        account_id: auth.account_id,
        plan_type: auth.plan_type,
        expires_at_ms: auth.expires_at_ms,
        last_refresh_ms: auth.last_refresh_ms,
        login_id,
        error,
    }
}

fn anthropic_provider_status_from_auth(
    auth: AnthropicAuthStatus,
    connection_state: &str,
    login_id: Option<String>,
    error: Option<String>,
) -> AnthropicProviderStatus {
    AnthropicProviderStatus {
        connected: auth.connected,
        connection_state: connection_state.to_string(),
        expires_at_ms: auth.expires_at_ms,
        last_refresh_ms: auth.last_refresh_ms,
        login_id,
        error,
    }
}

fn google_provider_status_from_auth(
    auth: GoogleAuthStatus,
    connection_state: &str,
    login_id: Option<String>,
    error: Option<String>,
) -> GoogleProviderStatus {
    GoogleProviderStatus {
        connected: auth.connected,
        connection_state: connection_state.to_string(),
        email: auth.email,
        project_id: auth.project_id,
        user_tier: auth.user_tier,
        expires_at_ms: auth.expires_at_ms,
        last_refresh_ms: auth.last_refresh_ms,
        login_id,
        error,
    }
}

fn kimi_provider_status_from_auth(
    auth: KimiAuthStatus,
    connection_state: &str,
    login_id: Option<String>,
    error: Option<String>,
) -> KimiProviderStatus {
    KimiProviderStatus {
        connected: auth.connected,
        connection_state: connection_state.to_string(),
        expires_at_ms: auth.expires_at_ms,
        last_refresh_ms: auth.last_refresh_ms,
        login_id,
        error,
    }
}

async fn bind_openai_oauth_listener() -> Result<tokio::net::TcpListener> {
    const DEFAULT_PORT: u16 = 1455;
    const FALLBACK_PORT: u16 = 1457;

    match tokio::net::TcpListener::bind(("127.0.0.1", DEFAULT_PORT)).await {
        Ok(listener) => Ok(listener),
        Err(default_err) => {
            tokio::net::TcpListener::bind(("127.0.0.1", FALLBACK_PORT))
                .await
                .with_context(|| {
                    format!(
                        "unable to bind OAuth callback ports {DEFAULT_PORT} or {FALLBACK_PORT}: {default_err}"
                    )
                })
        }
    }
}

async fn run_openai_oauth_server(
    listener: tokio::net::TcpListener,
    redirect_uri: String,
    expected_state: String,
    pkce: PkceCodes,
    cancel: Arc<Notify>,
) -> Result<()> {
    let http = reqwest::Client::builder()
        .user_agent("sinew/0.1")
        .build()
        .context("unable to build OAuth client")?;

    loop {
        tokio::select! {
            _ = cancel.notified() => {
                anyhow::bail!("Login canceled");
            }
            accepted = listener.accept() => {
                let (mut stream, _) = accepted.context("OAuth callback accept failed")?;
                if let Some(result) = handle_openai_oauth_request(
                    &http,
                    &mut stream,
                    &redirect_uri,
                    &expected_state,
                    &pkce,
                ).await? {
                    return result;
                }
            }
        }
    }
}

async fn handle_openai_oauth_request(
    http: &reqwest::Client,
    stream: &mut tokio::net::TcpStream,
    redirect_uri: &str,
    expected_state: &str,
    pkce: &PkceCodes,
) -> Result<Option<Result<()>>> {
    let mut buffer = [0u8; 8192];
    let read = stream
        .read(&mut buffer)
        .await
        .context("OAuth callback read failed")?;
    if read == 0 {
        return Ok(None);
    }

    let request = String::from_utf8_lossy(&buffer[..read]);
    let Some(first_line) = request.lines().next() else {
        write_http_response(stream, 400, "Bad Request", "Bad Request").await?;
        return Ok(None);
    };
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let target = parts.next().unwrap_or_default();
    if method != "GET" {
        write_http_response(stream, 405, "Method Not Allowed", "Method Not Allowed").await?;
        return Ok(None);
    }

    let parsed = parse_local_oauth_url(target)?;
    match parsed.path() {
        "/auth/callback" => {
            let params = parsed
                .query_pairs()
                .into_owned()
                .collect::<HashMap<String, String>>();
            if params.get("state").map(String::as_str) != Some(expected_state) {
                write_html_response(stream, 400, openai_login_error_html("State mismatch")).await?;
                return Ok(Some(Err(anyhow::anyhow!("State mismatch"))));
            }
            if let Some(error) = params.get("error") {
                let message = params
                    .get("error_description")
                    .filter(|value| !value.trim().is_empty())
                    .cloned()
                    .unwrap_or_else(|| error.clone());
                write_html_response(stream, 400, openai_login_error_html(&message)).await?;
                return Ok(Some(Err(anyhow::anyhow!(message))));
            }
            let Some(code) = params.get("code").filter(|value| !value.is_empty()) else {
                write_html_response(
                    stream,
                    400,
                    openai_login_error_html("Missing authorization code"),
                )
                .await?;
                return Ok(Some(Err(anyhow::anyhow!("Missing authorization code"))));
            };

            match exchange_oauth_code(http, code, redirect_uri, pkce).await {
                Ok(_) => {
                    write_html_response(stream, 200, openai_login_success_html()).await?;
                    Ok(Some(Ok(())))
                }
                Err(err) => {
                    let message = err.to_string();
                    write_html_response(stream, 500, openai_login_error_html(&message)).await?;
                    Ok(Some(Err(anyhow::anyhow!(message))))
                }
            }
        }
        "/cancel" => {
            write_http_response(stream, 200, "OK", "Login canceled").await?;
            Ok(Some(Err(anyhow::anyhow!("Login canceled"))))
        }
        _ => {
            write_http_response(stream, 404, "Not Found", "Not Found").await?;
            Ok(None)
        }
    }
}

async fn bind_anthropic_oauth_listener() -> Result<tokio::net::TcpListener> {
    const CALLBACK_PORT: u16 = 53692;
    tokio::net::TcpListener::bind(("127.0.0.1", CALLBACK_PORT))
        .await
        .context("unable to bind Anthropic OAuth callback port 53692")
}

async fn run_anthropic_oauth_server(
    listener: tokio::net::TcpListener,
    redirect_uri: String,
    expected_state: String,
    pkce: AnthropicPkceCodes,
    cancel: Arc<Notify>,
) -> Result<()> {
    let http = reqwest::Client::builder()
        .user_agent("sinew/0.1")
        .build()
        .context("unable to build OAuth client")?;

    loop {
        tokio::select! {
            _ = cancel.notified() => {
                anyhow::bail!("Login canceled");
            }
            accepted = listener.accept() => {
                let (mut stream, _) = accepted.context("OAuth callback accept failed")?;
                if let Some(result) = handle_anthropic_oauth_request(
                    &http,
                    &mut stream,
                    &redirect_uri,
                    &expected_state,
                    &pkce,
                ).await? {
                    return result;
                }
            }
        }
    }
}

async fn handle_anthropic_oauth_request(
    http: &reqwest::Client,
    stream: &mut tokio::net::TcpStream,
    redirect_uri: &str,
    expected_state: &str,
    pkce: &AnthropicPkceCodes,
) -> Result<Option<Result<()>>> {
    let mut buffer = [0u8; 8192];
    let read = stream
        .read(&mut buffer)
        .await
        .context("OAuth callback read failed")?;
    if read == 0 {
        return Ok(None);
    }

    let request = String::from_utf8_lossy(&buffer[..read]);
    let Some(first_line) = request.lines().next() else {
        write_http_response(stream, 400, "Bad Request", "Bad Request").await?;
        return Ok(None);
    };
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let target = parts.next().unwrap_or_default();
    if method != "GET" {
        write_http_response(stream, 405, "Method Not Allowed", "Method Not Allowed").await?;
        return Ok(None);
    }

    let parsed = parse_local_oauth_url(target)?;
    match parsed.path() {
        "/callback" => {
            let params = parsed
                .query_pairs()
                .into_owned()
                .collect::<HashMap<String, String>>();
            if let Some(error) = params.get("error") {
                let message = params
                    .get("error_description")
                    .filter(|value| !value.trim().is_empty())
                    .cloned()
                    .unwrap_or_else(|| error.clone());
                write_html_response(stream, 400, openai_login_error_html(&message)).await?;
                return Ok(Some(Err(anyhow::anyhow!(message))));
            }
            if params.get("state").map(String::as_str) != Some(expected_state) {
                write_html_response(stream, 400, openai_login_error_html("State mismatch")).await?;
                return Ok(Some(Err(anyhow::anyhow!("State mismatch"))));
            }
            let Some(code) = params.get("code").filter(|value| !value.is_empty()) else {
                write_html_response(
                    stream,
                    400,
                    openai_login_error_html("Missing authorization code"),
                )
                .await?;
                return Ok(Some(Err(anyhow::anyhow!("Missing authorization code"))));
            };

            match exchange_anthropic_oauth_code(http, code, expected_state, redirect_uri, pkce)
                .await
            {
                Ok(_) => {
                    write_html_response(stream, 200, anthropic_login_success_html()).await?;
                    Ok(Some(Ok(())))
                }
                Err(err) => {
                    let message = err.to_string();
                    write_html_response(stream, 500, openai_login_error_html(&message)).await?;
                    Ok(Some(Err(anyhow::anyhow!(message))))
                }
            }
        }
        "/cancel" => {
            write_http_response(stream, 200, "OK", "Login canceled").await?;
            Ok(Some(Err(anyhow::anyhow!("Login canceled"))))
        }
        _ => {
            write_http_response(stream, 404, "Not Found", "Not Found").await?;
            Ok(None)
        }
    }
}

async fn bind_google_oauth_listener() -> Result<tokio::net::TcpListener> {
    tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .context("unable to bind Google OAuth callback port")
}

async fn run_google_oauth_server(
    listener: tokio::net::TcpListener,
    redirect_uri: String,
    expected_state: String,
    cancel: Arc<Notify>,
) -> Result<()> {
    let http = reqwest::Client::builder()
        .user_agent("sinew/0.1")
        .build()
        .context("unable to build OAuth client")?;

    loop {
        tokio::select! {
            _ = cancel.notified() => {
                anyhow::bail!("Login canceled");
            }
            accepted = listener.accept() => {
                let (mut stream, _) = accepted.context("OAuth callback accept failed")?;
                if let Some(result) = handle_google_oauth_request(
                    &http,
                    &mut stream,
                    &redirect_uri,
                    &expected_state,
                ).await? {
                    return result;
                }
            }
        }
    }
}

async fn handle_google_oauth_request(
    http: &reqwest::Client,
    stream: &mut tokio::net::TcpStream,
    redirect_uri: &str,
    expected_state: &str,
) -> Result<Option<Result<()>>> {
    let mut buffer = [0u8; 8192];
    let read = stream
        .read(&mut buffer)
        .await
        .context("OAuth callback read failed")?;
    if read == 0 {
        return Ok(None);
    }

    let request = String::from_utf8_lossy(&buffer[..read]);
    let Some(first_line) = request.lines().next() else {
        write_http_response(stream, 400, "Bad Request", "Bad Request").await?;
        return Ok(None);
    };
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let target = parts.next().unwrap_or_default();
    if method != "GET" {
        write_http_response(stream, 405, "Method Not Allowed", "Method Not Allowed").await?;
        return Ok(None);
    }

    let parsed = parse_local_oauth_url(target)?;
    match parsed.path() {
        "/oauth2callback" => {
            let params = parsed
                .query_pairs()
                .into_owned()
                .collect::<HashMap<String, String>>();
            if let Some(error) = params.get("error") {
                let message = params
                    .get("error_description")
                    .filter(|value| !value.trim().is_empty())
                    .cloned()
                    .unwrap_or_else(|| error.clone());
                write_html_response(stream, 400, openai_login_error_html(&message)).await?;
                return Ok(Some(Err(anyhow::anyhow!(message))));
            }
            if params.get("state").map(String::as_str) != Some(expected_state) {
                write_html_response(stream, 400, openai_login_error_html("State mismatch")).await?;
                return Ok(Some(Err(anyhow::anyhow!("State mismatch"))));
            }
            let Some(code) = params.get("code").filter(|value| !value.is_empty()) else {
                write_html_response(
                    stream,
                    400,
                    openai_login_error_html("Missing authorization code"),
                )
                .await?;
                return Ok(Some(Err(anyhow::anyhow!("Missing authorization code"))));
            };

            match exchange_google_oauth_code(http, code, redirect_uri).await {
                Ok(_) => {
                    write_html_response(stream, 200, google_login_success_html()).await?;
                    Ok(Some(Ok(())))
                }
                Err(err) => {
                    let message = err.to_string();
                    write_html_response(stream, 500, openai_login_error_html(&message)).await?;
                    Ok(Some(Err(anyhow::anyhow!(message))))
                }
            }
        }
        "/cancel" => {
            write_http_response(stream, 200, "OK", "Login canceled").await?;
            Ok(Some(Err(anyhow::anyhow!("Login canceled"))))
        }
        _ => {
            write_http_response(stream, 404, "Not Found", "Not Found").await?;
            Ok(None)
        }
    }
}

fn parse_local_oauth_url(target: &str) -> Result<url::Url> {
    if target.starts_with('/') {
        url::Url::parse(&format!("http://localhost{target}")).context("invalid OAuth callback URL")
    } else {
        url::Url::parse(target).context("invalid OAuth callback URL")
    }
}

async fn write_http_response(
    stream: &mut tokio::net::TcpStream,
    status: u16,
    reason: &str,
    body: &str,
) -> Result<()> {
    write_response(stream, status, reason, "text/plain; charset=utf-8", body).await
}

async fn write_html_response(
    stream: &mut tokio::net::TcpStream,
    status: u16,
    body: String,
) -> Result<()> {
    let reason = if status < 400 { "OK" } else { "Error" };
    write_response(stream, status, reason, "text/html; charset=utf-8", &body).await
}

async fn write_response(
    stream: &mut tokio::net::TcpStream,
    status: u16,
    reason: &str,
    content_type: &str,
    body: &str,
) -> Result<()> {
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.as_bytes().len()
    );
    stream
        .write_all(response.as_bytes())
        .await
        .context("OAuth callback write failed")
}

fn openai_login_success_html() -> String {
    r#"<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Sinew connected</title>
    <style>
      body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0a0b0d;color:#f4f4f5;font:15px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
      main{max-width:420px;padding:32px;text-align:center}
      h1{font-size:22px;margin:0 0 10px}
      p{margin:0;color:#a1a1aa;line-height:1.5}
    </style>
  </head>
  <body><main><h1>OpenAI is connected</h1><p>You can close this tab and return to Sinew.</p></main></body>
</html>"#
        .to_string()
}

fn anthropic_login_success_html() -> String {
    r#"<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Sinew connected</title>
    <style>
      body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0a0b0d;color:#f4f4f5;font:15px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
      main{max-width:420px;padding:32px;text-align:center}
      h1{font-size:22px;margin:0 0 10px}
      p{margin:0;color:#a1a1aa;line-height:1.5}
    </style>
  </head>
  <body><main><h1>Anthropic is connected</h1><p>You can close this tab and return to Sinew.</p></main></body>
</html>"#
        .to_string()
}

fn google_login_success_html() -> String {
    r#"<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Sinew connected</title>
    <style>
      body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0a0b0d;color:#f4f4f5;font:15px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
      main{max-width:420px;padding:32px;text-align:center}
      h1{font-size:22px;margin:0 0 10px}
      p{margin:0;color:#a1a1aa;line-height:1.5}
    </style>
  </head>
  <body><main><h1>Google is connected</h1><p>You can close this tab and return to Sinew.</p></main></body>
</html>"#
        .to_string()
}

fn openai_login_error_html(message: &str) -> String {
    let escaped = html_escape(message);
    format!(
        r#"<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Sinew connection failed</title>
    <style>
      body{{margin:0;min-height:100vh;display:grid;place-items:center;background:#0a0b0d;color:#f4f4f5;font:15px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
      main{{max-width:480px;padding:32px;text-align:center}}
      h1{{font-size:22px;margin:0 0 10px}}
      p{{margin:0;color:#a1a1aa;line-height:1.5;overflow-wrap:anywhere}}
    </style>
  </head>
  <body><main><h1>Connection failed</h1><p>{escaped}</p></main></body>
</html>"#
    )
}

fn html_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn conversation_active_mode(conversation: &SavedConversation) -> AgentMode {
    let mode = match &conversation.plan_workflow {
        PlanWorkflowState::Idle => AgentMode::Act,
        PlanWorkflowState::PlanningQuestions | PlanWorkflowState::PlanReady { .. } => {
            AgentMode::Plan
        }
    };
    let mode = if mode == AgentMode::Act
        && matches!(conversation.goal_workflow, GoalWorkflowState::Active { .. })
    {
        AgentMode::Goal
    } else {
        mode
    };
    mode
}

#[derive(Debug, Clone)]
struct PlanTurnPolicy {
    mode: AgentMode,
    stop_questions: bool,
    next_workflow: PlanWorkflowState,
    attach_plan: bool,
}

fn plan_turn_policy(
    current: &PlanWorkflowState,
    requested_mode: AgentMode,
    control: Option<PlanControlInput>,
) -> std::result::Result<PlanTurnPolicy, String> {
    match current {
        PlanWorkflowState::Idle => match control {
            Some(PlanControlInput::StopQuestions) => {
                Err("no active plan workflow for this action".into())
            }
            Some(PlanControlInput::UpdatePlan) => Ok(PlanTurnPolicy {
                mode: AgentMode::Plan,
                stop_questions: false,
                next_workflow: PlanWorkflowState::PlanningQuestions,
                attach_plan: false,
            }),
            Some(PlanControlInput::ImplementPlan) => Ok(PlanTurnPolicy {
                mode: AgentMode::Act,
                stop_questions: false,
                next_workflow: PlanWorkflowState::Idle,
                attach_plan: false,
            }),
            None if requested_mode == AgentMode::Plan => Ok(PlanTurnPolicy {
                mode: AgentMode::Plan,
                stop_questions: false,
                next_workflow: PlanWorkflowState::PlanningQuestions,
                attach_plan: false,
            }),
            None if requested_mode == AgentMode::Goal => Ok(PlanTurnPolicy {
                mode: AgentMode::Goal,
                stop_questions: false,
                next_workflow: PlanWorkflowState::Idle,
                attach_plan: false,
            }),
            None => Ok(PlanTurnPolicy {
                mode: AgentMode::Act,
                stop_questions: false,
                next_workflow: PlanWorkflowState::Idle,
                attach_plan: false,
            }),
        },
        PlanWorkflowState::PlanningQuestions => match control {
            Some(PlanControlInput::ImplementPlan) => {
                Err("create the plan before implementing it".into())
            }
            Some(PlanControlInput::UpdatePlan) => Ok(PlanTurnPolicy {
                mode: AgentMode::Plan,
                stop_questions: false,
                next_workflow: PlanWorkflowState::PlanningQuestions,
                attach_plan: false,
            }),
            Some(PlanControlInput::StopQuestions) => Ok(PlanTurnPolicy {
                mode: AgentMode::Plan,
                stop_questions: true,
                next_workflow: PlanWorkflowState::PlanningQuestions,
                attach_plan: true,
            }),
            None => Ok(PlanTurnPolicy {
                mode: AgentMode::Plan,
                stop_questions: false,
                next_workflow: PlanWorkflowState::PlanningQuestions,
                attach_plan: false,
            }),
        },
        PlanWorkflowState::PlanReady { .. } => match control {
            Some(PlanControlInput::ImplementPlan) => Ok(PlanTurnPolicy {
                mode: AgentMode::Act,
                stop_questions: false,
                next_workflow: PlanWorkflowState::Idle,
                attach_plan: false,
            }),
            Some(PlanControlInput::UpdatePlan) => Ok(PlanTurnPolicy {
                mode: AgentMode::Plan,
                stop_questions: false,
                next_workflow: PlanWorkflowState::PlanningQuestions,
                attach_plan: false,
            }),
            Some(PlanControlInput::StopQuestions) => Ok(PlanTurnPolicy {
                mode: AgentMode::Plan,
                stop_questions: true,
                next_workflow: PlanWorkflowState::PlanningQuestions,
                attach_plan: true,
            }),
            None if requested_mode == AgentMode::Plan => Ok(PlanTurnPolicy {
                mode: AgentMode::Plan,
                stop_questions: false,
                next_workflow: PlanWorkflowState::PlanningQuestions,
                attach_plan: false,
            }),
            None => Err("plan is ready; choose update plan or implement plan".into()),
        },
    }
}

fn plan_estimate_mode(current: &PlanWorkflowState, requested_mode: AgentMode) -> AgentMode {
    match current {
        PlanWorkflowState::Idle => requested_mode,
        PlanWorkflowState::PlanningQuestions | PlanWorkflowState::PlanReady { .. } => {
            AgentMode::Plan
        }
    }
}

fn start_goal_workflow(objective: &str) -> GoalWorkflowState {
    let now = now_ms();
    GoalWorkflowState::Active {
        objective: objective.trim().to_string(),
        started_at_ms: now,
        updated_at_ms: now,
    }
}

fn resume_goal_workflow(workflow: GoalWorkflowState) -> GoalWorkflowState {
    match workflow {
        GoalWorkflowState::Paused {
            objective,
            started_at_ms,
            ..
        } => GoalWorkflowState::Active {
            objective,
            started_at_ms,
            updated_at_ms: now_ms(),
        },
        current => current,
    }
}

fn pause_goal_workflow(workflow: GoalWorkflowState) -> GoalWorkflowState {
    match workflow {
        GoalWorkflowState::Active {
            objective,
            started_at_ms,
            ..
        } => GoalWorkflowState::Paused {
            objective,
            started_at_ms,
            updated_at_ms: now_ms(),
        },
        current => current,
    }
}

#[tauri::command]
async fn open_workspace(
    state: State<'_, DesktopState>,
    input: WorkspaceInput,
) -> std::result::Result<WorkspaceBootstrap, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    state
        .store
        .bootstrap_workspace(&workspace_root, &state.default_model, &state.system_prompt)
        .map_err(error_to_string)
}

#[tauri::command]
async fn open_new_window(app: AppHandle) -> std::result::Result<(), String> {
    create_new_window(&app).map_err(error_to_string)
}

#[tauri::command]
async fn watch_workspace_command(
    app: AppHandle,
    state: State<'_, DesktopState>,
    input: WorkspaceInput,
) -> std::result::Result<(), String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let workspace_id = workspace_root.display().to_string();
    let mut watchers = state.file_watchers.lock().await;
    if watchers.contains_key(&workspace_id) {
        return Ok(());
    }

    let watcher_root = workspace_root.clone();
    let app_for_watcher = app.clone();
    let workspace_id_for_watcher = workspace_id.clone();
    let mut watcher =
        notify::recommended_watcher(move |event: notify::Result<notify::Event>| match event {
            Ok(event) => {
                if !is_workspace_file_event(&event.kind) {
                    return;
                }
                if event.paths.is_empty() {
                    let _ = app_for_watcher.emit(
                        FILE_CHANGE_EVENT_NAME,
                        WorkspaceFileChangeEvent {
                            workspace_path: workspace_id_for_watcher.clone(),
                            relative_path: String::new(),
                        },
                    );
                    return;
                }
                for path in event.paths {
                    if should_ignore_workspace_event_path(&watcher_root, &path) {
                        continue;
                    }
                    if let Some(relative_path) = event_relative_path(&watcher_root, &path) {
                        let _ = app_for_watcher.emit(
                            FILE_CHANGE_EVENT_NAME,
                            WorkspaceFileChangeEvent {
                                workspace_path: workspace_id_for_watcher.clone(),
                                relative_path,
                            },
                        );
                    }
                }
            }
            Err(err) => tracing::warn!(%err, "workspace watcher error"),
        })
        .map_err(error_to_string)?;
    watcher
        .watch(&workspace_root, RecursiveMode::Recursive)
        .map_err(error_to_string)?;
    watchers.insert(workspace_id, watcher);
    Ok(())
}

#[tauri::command]
async fn unwatch_workspace_command(
    state: State<'_, DesktopState>,
    input: WorkspaceInput,
) -> std::result::Result<bool, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let workspace_id = workspace_root.display().to_string();
    Ok(state
        .file_watchers
        .lock()
        .await
        .remove(&workspace_id)
        .is_some())
}

#[tauri::command]
async fn list_workspace_entries_command(
    input: WorkspaceEntriesInput,
) -> std::result::Result<Vec<sinew_app::WorkspaceEntry>, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    list_workspace_entries(&workspace_root, input.relative_path.as_deref()).map_err(error_to_string)
}

#[tauri::command]
async fn list_workspace_files_command(
    input: WorkspaceInput,
) -> std::result::Result<Vec<sinew_app::WorkspaceEntry>, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    list_workspace_files(&workspace_root).map_err(error_to_string)
}

#[tauri::command]
async fn search_workspace_files_command(
    input: WorkspaceSearchInput,
) -> std::result::Result<WorkspaceSearchResult, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    search_workspace_files(&workspace_root, &input.query).map_err(error_to_string)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ImportPathsInput {
    workspace_path: String,
    target_relative_path: Option<String>,
    sources: Vec<String>,
}

#[tauri::command]
async fn import_workspace_paths_command(
    app: AppHandle,
    input: ImportPathsInput,
) -> std::result::Result<Vec<ImportedEntry>, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let imported = import_workspace_paths(
        &workspace_root,
        input.target_relative_path.as_deref(),
        &input.sources,
    )
    .map_err(error_to_string)?;
    for entry in &imported {
        emit_workspace_file_change(&app, &workspace_root, &entry.relative_path);
    }
    Ok(imported)
}

#[tauri::command]
async fn read_workspace_file_command(
    input: WorkspaceFileInput,
) -> std::result::Result<sinew_app::FileDocument, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    read_workspace_file(&workspace_root, &input.relative_path).map_err(error_to_string)
}

#[tauri::command]
async fn write_workspace_file_command(
    app: AppHandle,
    input: WriteWorkspaceFileInput,
) -> std::result::Result<sinew_app::FileDocument, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let doc = write_workspace_file(&workspace_root, &input.relative_path, &input.content)
        .map_err(error_to_string)?;
    emit_workspace_file_change(&app, &workspace_root, &doc.relative_path);
    Ok(doc)
}

#[tauri::command]
async fn create_workspace_file_command(
    app: AppHandle,
    input: CreateWorkspaceEntryInput,
) -> std::result::Result<sinew_app::WorkspaceEntry, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let entry = create_workspace_file(
        &workspace_root,
        input.target_relative_path.as_deref(),
        &input.name,
    )
    .map_err(error_to_string)?;
    emit_workspace_file_change(&app, &workspace_root, &entry.relative_path);
    Ok(entry)
}

#[tauri::command]
async fn create_workspace_directory_command(
    app: AppHandle,
    input: CreateWorkspaceEntryInput,
) -> std::result::Result<sinew_app::WorkspaceEntry, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let entry = create_workspace_directory(
        &workspace_root,
        input.target_relative_path.as_deref(),
        &input.name,
    )
    .map_err(error_to_string)?;
    emit_workspace_file_change(&app, &workspace_root, &entry.relative_path);
    Ok(entry)
}

#[tauri::command]
async fn save_clipboard_image_attachment_command(
    input: ClipboardImageInput,
) -> std::result::Result<ClipboardImageAttachment, String> {
    normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let (_, extension) = clipboard_image_type(&input.media_type, input.name.as_deref())
        .ok_or_else(|| "unsupported pasted image type".to_string())?;
    let raw_data = input
        .data
        .split_once(',')
        .map(|(_, data)| data)
        .unwrap_or(input.data.as_str())
        .trim();
    let bytes = BASE64_STANDARD.decode(raw_data).map_err(error_to_string)?;
    if bytes.is_empty() {
        return Err("pasted image is empty".into());
    }
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err("pasted image is too large".into());
    }

    let display_name = clipboard_image_display_name(input.name.as_deref(), extension);
    let stem = Path::new(&display_name)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("pasted-image");
    let safe_stem = safe_temp_file_stem(stem);
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let file_name = format!("{safe_stem}-{}-{now_ms}.{extension}", std::process::id());
    let dir = std::env::temp_dir().join("sinew-clipboard-attachments");
    fs::create_dir_all(&dir).map_err(error_to_string)?;
    let path = dir.join(file_name);
    fs::write(&path, bytes).map_err(error_to_string)?;

    Ok(ClipboardImageAttachment {
        path: path.display().to_string(),
        name: display_name,
    })
}

#[tauri::command]
async fn rename_workspace_entry_command(
    app: AppHandle,
    input: RenameWorkspaceEntryInput,
) -> std::result::Result<sinew_app::WorkspaceEntry, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let entry = rename_workspace_entry(&workspace_root, &input.relative_path, &input.new_name)
        .map_err(error_to_string)?;
    emit_workspace_file_change(&app, &workspace_root, &input.relative_path);
    emit_workspace_file_change(&app, &workspace_root, &entry.relative_path);
    Ok(entry)
}

#[tauri::command]
async fn delete_workspace_entry_command(
    app: AppHandle,
    input: WorkspaceFileInput,
) -> std::result::Result<(), String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    delete_workspace_entry(&workspace_root, &input.relative_path).map_err(error_to_string)?;
    emit_workspace_file_change(&app, &workspace_root, &input.relative_path);
    Ok(())
}

#[tauri::command]
async fn trash_workspace_entry_command(
    app: AppHandle,
    input: WorkspaceFileInput,
) -> std::result::Result<WorkspaceDeletedEntry, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let deleted =
        trash_workspace_entry(&workspace_root, &input.relative_path).map_err(error_to_string)?;
    emit_workspace_file_change(&app, &workspace_root, &deleted.relative_path);
    Ok(deleted)
}

#[tauri::command]
async fn restore_workspace_deleted_entries_command(
    app: AppHandle,
    input: RestoreWorkspaceDeletedEntriesInput,
) -> std::result::Result<Vec<sinew_app::WorkspaceEntry>, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let entries = restore_workspace_deleted_entries(&workspace_root, &input.entries)
        .map_err(error_to_string)?;
    for entry in &entries {
        emit_workspace_file_change(&app, &workspace_root, &entry.relative_path);
    }
    Ok(entries)
}

#[tauri::command]
async fn reveal_workspace_entry_command(
    input: WorkspaceFileInput,
) -> std::result::Result<(), String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let path = sinew_app::workspace::resolve_workspace_path(&workspace_root, &input.relative_path)
        .map_err(error_to_string)?;
    reveal_path(&path).map_err(error_to_string)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AbsolutePathInput {
    path: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SkillPathInput {
    workspace_path: String,
    path: String,
}

#[tauri::command]
async fn reveal_absolute_path_command(input: AbsolutePathInput) -> std::result::Result<(), String> {
    let path = std::path::PathBuf::from(&input.path);
    reveal_path(&path).map_err(error_to_string)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ResolveTerminalPathInput {
    workspace_path: String,
    raw_path: String,
}

#[tauri::command]
async fn resolve_terminal_path_command(
    input: ResolveTerminalPathInput,
) -> std::result::Result<TerminalPathResolution, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    resolve_terminal_path(&workspace_root, &input.raw_path).map_err(error_to_string)
}

#[tauri::command]
async fn read_external_file_command(
    input: AbsolutePathInput,
) -> std::result::Result<sinew_app::FileDocument, String> {
    let path = std::path::PathBuf::from(&input.path);
    read_external_file(&path).map_err(error_to_string)
}

#[tauri::command]
async fn delete_skill_command(
    app: AppHandle,
    input: SkillPathInput,
) -> std::result::Result<(), String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let skill_md = PathBuf::from(&input.path);
    let folder = delete_installed_skill(&workspace_root, &skill_md).map_err(error_to_string)?;
    if let Ok(relative) = folder.strip_prefix(&workspace_root) {
        let relative_path = relative.to_string_lossy().to_string();
        emit_workspace_file_change(&app, &workspace_root, &relative_path);
    }
    Ok(())
}

#[tauri::command]
async fn open_external_url_command(input: OpenExternalUrlInput) -> std::result::Result<(), String> {
    open_external_url(&input.url).map_err(error_to_string)
}

#[tauri::command]
async fn copy_workspace_entries_command(
    app: AppHandle,
    input: CopyWorkspaceEntriesInput,
) -> std::result::Result<Vec<sinew_app::WorkspaceEntry>, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let operation = if input.cut {
        WorkspaceCopyOperation::Move
    } else {
        WorkspaceCopyOperation::Copy
    };
    let entries = copy_workspace_entries(
        &workspace_root,
        input.target_relative_path.as_deref(),
        &input.sources,
        operation,
    )
    .map_err(error_to_string)?;
    for source in &input.sources {
        emit_workspace_file_change(&app, &workspace_root, source);
    }
    for entry in &entries {
        emit_workspace_file_change(&app, &workspace_root, &entry.relative_path);
    }
    Ok(entries)
}

#[tauri::command]
async fn read_clipboard_file_paths_command() -> std::result::Result<Vec<String>, String> {
    tauri::async_runtime::spawn_blocking(read_clipboard_file_paths)
        .await
        .map_err(error_to_string)?
        .map_err(error_to_string)
}

#[tauri::command]
async fn list_conversations(
    state: State<'_, DesktopState>,
    input: WorkspaceInput,
) -> std::result::Result<Vec<ConversationSummary>, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    state
        .store
        .list_conversations(&workspace_root.display().to_string())
        .map_err(error_to_string)
}

#[tauri::command]
async fn create_conversation(
    state: State<'_, DesktopState>,
    input: WorkspaceInput,
) -> std::result::Result<WorkspaceBootstrap, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    state
        .store
        .create_conversation(
            &workspace_root.display().to_string(),
            &state.default_model,
            &state.system_prompt,
        )
        .map_err(error_to_string)?;
    state
        .store
        .bootstrap_workspace(&workspace_root, &state.default_model, &state.system_prompt)
        .map_err(error_to_string)
}

#[tauri::command]
async fn load_conversation(
    state: State<'_, DesktopState>,
    input: ConversationInput,
) -> std::result::Result<SavedConversation, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    state
        .store
        .load_conversation(
            &workspace_root.display().to_string(),
            &input.conversation_id,
        )
        .map_err(error_to_string)?
        .ok_or_else(|| "conversation not found".to_string())
}

#[tauri::command]
async fn rename_conversation(
    state: State<'_, DesktopState>,
    input: RenameConversationInput,
) -> std::result::Result<Vec<ConversationSummary>, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let title = input.title.trim();
    if title.is_empty() {
        return Err("title cannot be empty".into());
    }
    let workspace_id = workspace_root.display().to_string();
    state
        .store
        .rename_conversation(&workspace_id, &input.conversation_id, title)
        .map_err(error_to_string)?;
    state
        .store
        .list_conversations(&workspace_id)
        .map_err(error_to_string)
}

#[tauri::command]
async fn delete_conversation(
    state: State<'_, DesktopState>,
    input: ConversationInput,
) -> std::result::Result<WorkspaceBootstrap, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let workspace_id = workspace_root.display().to_string();
    {
        let active_turns = state.active_turns.lock().await;
        if active_turns.contains_key(&input.conversation_id) {
            return Err("a turn is already running for this conversation".into());
        }
    }
    state
        .store
        .delete_conversation(&workspace_id, &input.conversation_id)
        .map_err(error_to_string)?;
    state
        .store
        .bootstrap_workspace(&workspace_root, &state.default_model, &state.system_prompt)
        .map_err(error_to_string)
}

#[tauri::command]
async fn set_conversation_mode(
    state: State<'_, DesktopState>,
    input: ConversationModeInput,
) -> std::result::Result<SavedConversation, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let workspace_id = workspace_root.display().to_string();
    {
        let active_turns = state.active_turns.lock().await;
        if active_turns.contains_key(&input.conversation_id) {
            return Err("a turn is already running for this conversation".into());
        }
    }

    let mut conversation = state
        .store
        .load_conversation(&workspace_id, &input.conversation_id)
        .map_err(error_to_string)?
        .ok_or_else(|| "conversation not found".to_string())?;

    let mode = AgentMode::from(input.mode);
    let current_plan_workflow = std::mem::take(&mut conversation.plan_workflow);
    conversation.plan_workflow = match mode {
        AgentMode::Act => PlanWorkflowState::Idle,
        AgentMode::Plan => match current_plan_workflow {
            PlanWorkflowState::Idle => PlanWorkflowState::PlanningQuestions,
            current => current,
        },
        AgentMode::Goal => PlanWorkflowState::Idle,
    };
    conversation.goal_workflow = match mode {
        AgentMode::Goal => resume_goal_workflow(std::mem::take(&mut conversation.goal_workflow)),
        AgentMode::Act | AgentMode::Plan => {
            pause_goal_workflow(std::mem::take(&mut conversation.goal_workflow))
        }
    };
    conversation.model = conversation.mode_model_settings.get(mode).clone();

    state
        .store
        .save_conversation(&conversation)
        .map_err(error_to_string)?;
    Ok(conversation)
}

#[tauri::command]
async fn set_conversation_model_preference(
    state: State<'_, DesktopState>,
    input: ConversationModelPreferenceInput,
) -> std::result::Result<ModeModelSettings, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let workspace_id = workspace_root.display().to_string();
    let conversation_id = input.conversation_id;
    let mode = AgentMode::from(input.mode);

    {
        let active_turns = state.active_turns.lock().await;
        if active_turns.contains_key(&conversation_id) {
            return Err("a turn is already running for this conversation".into());
        }
    }

    let mut conversation = state
        .store
        .load_conversation(&workspace_id, &conversation_id)
        .map_err(error_to_string)?
        .ok_or_else(|| "conversation not found".to_string())?;
    let selected = model_with_optional_selection(
        conversation.mode_model_settings.get(mode),
        input.model,
        input.thinking,
    );
    let provider = provider_from_registry(&state, &selected.provider)?;
    provider
        .capabilities(&selected)
        .ok_or_else(|| format!("model `{}` is not supported", selected.name))?;

    conversation.mode_model_settings.set(mode, selected.clone());
    if conversation_active_mode(&conversation) == mode {
        conversation.model = selected.clone();
    }

    let mut default_settings = state
        .store
        .load_mode_model_settings(&state.default_model)
        .map_err(error_to_string)?;
    default_settings.set(mode, selected);

    state
        .store
        .save_conversation_and_mode_model_settings(&conversation, &default_settings)
        .map_err(error_to_string)?;
    Ok(conversation.mode_model_settings)
}

#[tauri::command]
async fn list_mcp_settings(
    state: State<'_, DesktopState>,
) -> std::result::Result<McpSettings, String> {
    state.store.load_mcp_settings().map_err(error_to_string)
}

#[tauri::command]
async fn save_mcp_settings(
    state: State<'_, DesktopState>,
    input: SaveMcpSettingsInput,
) -> std::result::Result<McpSettings, String> {
    state
        .store
        .save_mcp_settings(&input.settings)
        .map_err(error_to_string)?;
    Ok(input.settings)
}

#[tauri::command]
async fn list_tool_settings(
    state: State<'_, DesktopState>,
    input: WorkspaceInput,
) -> std::result::Result<ToolSettingsView, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let settings = state.store.load_tool_settings().map_err(error_to_string)?;
    Ok(tool_settings_view(
        &settings,
        &configurable_tool_catalog(&workspace_root),
    ))
}

#[tauri::command]
async fn save_tool_settings(
    state: State<'_, DesktopState>,
    input: SaveToolSettingsInput,
) -> std::result::Result<ToolSettingsView, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let saved = state
        .store
        .save_tool_settings(&input.settings)
        .map_err(error_to_string)?;
    Ok(tool_settings_view(
        &saved,
        &configurable_tool_catalog(&workspace_root),
    ))
}

#[tauri::command]
async fn list_sub_agent_settings(
    state: State<'_, DesktopState>,
) -> std::result::Result<SubAgentSettings, String> {
    state
        .store
        .load_sub_agent_settings()
        .map_err(error_to_string)
}

#[tauri::command]
async fn save_sub_agent_settings(
    state: State<'_, DesktopState>,
    input: SaveSubAgentSettingsInput,
) -> std::result::Result<SubAgentSettings, String> {
    for agent in input.settings.agents.iter().filter(|agent| agent.enabled) {
        let provider = provider_from_registry(&state, &agent.model.provider)?;
        provider
            .capabilities(&agent.model)
            .ok_or_else(|| format!("model `{}` is not supported", agent.model.name))?;
    }
    state
        .store
        .save_sub_agent_settings(&input.settings)
        .map_err(error_to_string)
}

#[tauri::command]
async fn get_openai_provider_status(
    state: State<'_, DesktopState>,
) -> std::result::Result<OpenAiProviderStatus, String> {
    let mut active_login = state.openai_login.lock().await;
    let attempt = active_login.clone();
    if let Some(attempt) = attempt {
        let outcome = attempt
            .outcome
            .lock()
            .map_err(|_| "login state is unavailable".to_string())?
            .clone();

        if let Some(outcome) = outcome {
            *active_login = None;
            let auth = load_default_auth_status().map_err(error_to_string)?;
            if outcome.success {
                return Ok(openai_provider_status_from_auth(
                    auth,
                    "connected",
                    None,
                    None,
                ));
            }
            return Ok(openai_provider_status_from_auth(
                auth,
                "error",
                None,
                outcome.error,
            ));
        }

        let auth = load_default_auth_status().map_err(error_to_string)?;
        return Ok(openai_provider_status_from_auth(
            auth,
            "connecting",
            Some(attempt.id),
            None,
        ));
    }

    let auth = load_default_auth_status().map_err(error_to_string)?;
    let state = if auth.connected {
        "connected"
    } else {
        "disconnected"
    };
    Ok(openai_provider_status_from_auth(auth, state, None, None))
}

#[tauri::command]
async fn start_openai_oauth_login(
    state: State<'_, DesktopState>,
) -> std::result::Result<StartOpenAiLoginOutput, String> {
    if let Some(existing) = state.openai_login.lock().await.take() {
        existing.cancel.notify_one();
    }

    let listener = bind_openai_oauth_listener()
        .await
        .map_err(error_to_string)?;
    let port = listener.local_addr().map_err(error_to_string)?.port();
    let redirect_uri = format!("http://localhost:{port}/auth/callback");
    let pkce = generate_pkce();
    let oauth_state = generate_state();
    let auth_url = oauth_authorize_url(&redirect_uri, &pkce, &oauth_state);
    let login_id = generate_state();
    let cancel = Arc::new(Notify::new());
    let outcome = Arc::new(StdMutex::new(None));

    {
        let mut active_login = state.openai_login.lock().await;
        *active_login = Some(OpenAiLoginAttempt {
            id: login_id.clone(),
            cancel: cancel.clone(),
            outcome: outcome.clone(),
        });
    }

    let providers = state.providers.clone();
    tauri::async_runtime::spawn(async move {
        let result =
            run_openai_oauth_server(listener, redirect_uri, oauth_state, pkce, cancel).await;
        let login_outcome = match result {
            Ok(()) => match install_openai_provider(&providers) {
                Ok(()) => OpenAiLoginOutcome {
                    success: true,
                    error: None,
                },
                Err(err) => OpenAiLoginOutcome {
                    success: false,
                    error: Some(err),
                },
            },
            Err(err) => OpenAiLoginOutcome {
                success: false,
                error: Some(err.to_string()),
            },
        };
        if let Ok(mut slot) = outcome.lock() {
            *slot = Some(login_outcome);
        }
    });

    Ok(StartOpenAiLoginOutput { login_id, auth_url })
}

#[tauri::command]
async fn cancel_openai_oauth_login(
    state: State<'_, DesktopState>,
) -> std::result::Result<OpenAiProviderStatus, String> {
    if let Some(attempt) = state.openai_login.lock().await.take() {
        attempt.cancel.notify_one();
    }
    let auth = load_default_auth_status().map_err(error_to_string)?;
    let state = if auth.connected {
        "connected"
    } else {
        "disconnected"
    };
    Ok(openai_provider_status_from_auth(auth, state, None, None))
}

#[tauri::command]
async fn disconnect_openai_provider(
    state: State<'_, DesktopState>,
) -> std::result::Result<OpenAiProviderStatus, String> {
    if let Some(attempt) = state.openai_login.lock().await.take() {
        attempt.cancel.notify_one();
    }
    delete_default_auth().map_err(error_to_string)?;
    remove_openai_provider(&state.providers)?;
    Ok(openai_provider_status_from_auth(
        OpenAiAuthStatus::disconnected(),
        "disconnected",
        None,
        None,
    ))
}

#[tauri::command]
async fn get_anthropic_provider_status(
    state: State<'_, DesktopState>,
) -> std::result::Result<AnthropicProviderStatus, String> {
    let mut active_login = state.anthropic_login.lock().await;
    let attempt = active_login.clone();
    if let Some(attempt) = attempt {
        let outcome = attempt
            .outcome
            .lock()
            .map_err(|_| "login state is unavailable".to_string())?
            .clone();

        if let Some(outcome) = outcome {
            *active_login = None;
            let auth = load_default_anthropic_auth_status().map_err(error_to_string)?;
            if outcome.success {
                return Ok(anthropic_provider_status_from_auth(
                    auth,
                    "connected",
                    None,
                    None,
                ));
            }
            return Ok(anthropic_provider_status_from_auth(
                auth,
                "error",
                None,
                outcome.error,
            ));
        }

        let auth = load_default_anthropic_auth_status().map_err(error_to_string)?;
        return Ok(anthropic_provider_status_from_auth(
            auth,
            "connecting",
            Some(attempt.id),
            None,
        ));
    }

    let auth = load_default_anthropic_auth_status().map_err(error_to_string)?;
    let state = if auth.connected {
        "connected"
    } else {
        "disconnected"
    };
    Ok(anthropic_provider_status_from_auth(auth, state, None, None))
}

#[tauri::command]
async fn start_anthropic_oauth_login(
    state: State<'_, DesktopState>,
) -> std::result::Result<StartAnthropicLoginOutput, String> {
    if let Some(existing) = state.anthropic_login.lock().await.take() {
        existing.cancel.notify_one();
    }

    let listener = bind_anthropic_oauth_listener()
        .await
        .map_err(error_to_string)?;
    let port = listener.local_addr().map_err(error_to_string)?.port();
    let redirect_uri = format!("http://localhost:{port}/callback");
    let pkce = generate_anthropic_pkce();
    let oauth_state = pkce.code_verifier.clone();
    let auth_url = anthropic_oauth_authorize_url(&redirect_uri, &pkce, &oauth_state);
    let login_id = generate_anthropic_state();
    let cancel = Arc::new(Notify::new());
    let outcome = Arc::new(StdMutex::new(None));

    {
        let mut active_login = state.anthropic_login.lock().await;
        *active_login = Some(AnthropicLoginAttempt {
            id: login_id.clone(),
            cancel: cancel.clone(),
            outcome: outcome.clone(),
        });
    }

    let providers = state.providers.clone();
    tauri::async_runtime::spawn(async move {
        let result =
            run_anthropic_oauth_server(listener, redirect_uri, oauth_state, pkce, cancel).await;
        let login_outcome = match result {
            Ok(()) => match install_anthropic_provider(&providers) {
                Ok(()) => AnthropicLoginOutcome {
                    success: true,
                    error: None,
                },
                Err(err) => AnthropicLoginOutcome {
                    success: false,
                    error: Some(err),
                },
            },
            Err(err) => AnthropicLoginOutcome {
                success: false,
                error: Some(err.to_string()),
            },
        };
        if let Ok(mut slot) = outcome.lock() {
            *slot = Some(login_outcome);
        }
    });

    Ok(StartAnthropicLoginOutput { login_id, auth_url })
}

#[tauri::command]
async fn cancel_anthropic_oauth_login(
    state: State<'_, DesktopState>,
) -> std::result::Result<AnthropicProviderStatus, String> {
    if let Some(attempt) = state.anthropic_login.lock().await.take() {
        attempt.cancel.notify_one();
    }
    let auth = load_default_anthropic_auth_status().map_err(error_to_string)?;
    let state = if auth.connected {
        "connected"
    } else {
        "disconnected"
    };
    Ok(anthropic_provider_status_from_auth(auth, state, None, None))
}

#[tauri::command]
async fn disconnect_anthropic_provider(
    state: State<'_, DesktopState>,
) -> std::result::Result<AnthropicProviderStatus, String> {
    if let Some(attempt) = state.anthropic_login.lock().await.take() {
        attempt.cancel.notify_one();
    }
    delete_default_anthropic_auth().map_err(error_to_string)?;
    remove_anthropic_provider(&state.providers)?;
    Ok(anthropic_provider_status_from_auth(
        AnthropicAuthStatus::disconnected(),
        "disconnected",
        None,
        None,
    ))
}

#[tauri::command]
async fn get_google_provider_status(
    state: State<'_, DesktopState>,
) -> std::result::Result<GoogleProviderStatus, String> {
    let mut active_login = state.google_login.lock().await;
    let attempt = active_login.clone();
    if let Some(attempt) = attempt {
        let outcome = attempt
            .outcome
            .lock()
            .map_err(|_| "login state is unavailable".to_string())?
            .clone();

        if let Some(outcome) = outcome {
            *active_login = None;
            let auth = load_default_google_auth_status().map_err(error_to_string)?;
            if outcome.success {
                return Ok(google_provider_status_from_auth(
                    auth,
                    "connected",
                    None,
                    None,
                ));
            }
            return Ok(google_provider_status_from_auth(
                auth,
                "error",
                None,
                outcome.error,
            ));
        }

        let auth = load_default_google_auth_status().map_err(error_to_string)?;
        return Ok(google_provider_status_from_auth(
            auth,
            "connecting",
            Some(attempt.id),
            None,
        ));
    }

    let auth = load_default_google_auth_status().map_err(error_to_string)?;
    let state = if auth.connected {
        "connected"
    } else {
        "disconnected"
    };
    Ok(google_provider_status_from_auth(auth, state, None, None))
}

#[tauri::command]
async fn start_google_oauth_login(
    state: State<'_, DesktopState>,
) -> std::result::Result<StartGoogleLoginOutput, String> {
    if let Some(existing) = state.google_login.lock().await.take() {
        existing.cancel.notify_one();
    }

    let listener = bind_google_oauth_listener()
        .await
        .map_err(error_to_string)?;
    let port = listener.local_addr().map_err(error_to_string)?.port();
    let redirect_uri = format!("http://127.0.0.1:{port}/oauth2callback");
    let oauth_state = generate_google_state();
    let auth_url = google_oauth_authorize_url(&redirect_uri, &oauth_state);
    let login_id = generate_google_state();
    let cancel = Arc::new(Notify::new());
    let outcome = Arc::new(StdMutex::new(None));

    {
        let mut active_login = state.google_login.lock().await;
        *active_login = Some(GoogleLoginAttempt {
            id: login_id.clone(),
            cancel: cancel.clone(),
            outcome: outcome.clone(),
        });
    }

    let providers = state.providers.clone();
    tauri::async_runtime::spawn(async move {
        let result = run_google_oauth_server(listener, redirect_uri, oauth_state, cancel).await;
        let login_outcome = match result {
            Ok(()) => match install_google_provider(&providers) {
                Ok(()) => GoogleLoginOutcome {
                    success: true,
                    error: None,
                },
                Err(err) => GoogleLoginOutcome {
                    success: false,
                    error: Some(err),
                },
            },
            Err(err) => GoogleLoginOutcome {
                success: false,
                error: Some(err.to_string()),
            },
        };
        if let Ok(mut slot) = outcome.lock() {
            *slot = Some(login_outcome);
        }
    });

    Ok(StartGoogleLoginOutput { login_id, auth_url })
}

#[tauri::command]
async fn cancel_google_oauth_login(
    state: State<'_, DesktopState>,
) -> std::result::Result<GoogleProviderStatus, String> {
    if let Some(attempt) = state.google_login.lock().await.take() {
        attempt.cancel.notify_one();
    }
    let auth = load_default_google_auth_status().map_err(error_to_string)?;
    let state = if auth.connected {
        "connected"
    } else {
        "disconnected"
    };
    Ok(google_provider_status_from_auth(auth, state, None, None))
}

#[tauri::command]
async fn disconnect_google_provider(
    state: State<'_, DesktopState>,
) -> std::result::Result<GoogleProviderStatus, String> {
    if let Some(attempt) = state.google_login.lock().await.take() {
        attempt.cancel.notify_one();
    }
    delete_default_google_auth().map_err(error_to_string)?;
    remove_google_provider(&state.providers)?;
    Ok(google_provider_status_from_auth(
        GoogleAuthStatus::disconnected(),
        "disconnected",
        None,
        None,
    ))
}

#[tauri::command]
async fn get_kimi_provider_status(
    state: State<'_, DesktopState>,
) -> std::result::Result<KimiProviderStatus, String> {
    let mut active_login = state.kimi_login.lock().await;
    let attempt = active_login.clone();
    if let Some(attempt) = attempt {
        let outcome = attempt
            .outcome
            .lock()
            .map_err(|_| "login state is unavailable".to_string())?
            .clone();

        if let Some(outcome) = outcome {
            *active_login = None;
            let auth = load_default_kimi_auth_status().map_err(error_to_string)?;
            if outcome.success {
                return Ok(kimi_provider_status_from_auth(
                    auth,
                    "connected",
                    None,
                    None,
                ));
            }
            return Ok(kimi_provider_status_from_auth(
                auth,
                "error",
                None,
                outcome.error,
            ));
        }

        let auth = load_default_kimi_auth_status().map_err(error_to_string)?;
        return Ok(kimi_provider_status_from_auth(
            auth,
            "connecting",
            Some(attempt.id),
            None,
        ));
    }

    let auth = load_default_kimi_auth_status().map_err(error_to_string)?;
    let state = if auth.connected {
        "connected"
    } else {
        "disconnected"
    };
    Ok(kimi_provider_status_from_auth(auth, state, None, None))
}

#[tauri::command]
async fn start_kimi_oauth_login(
    state: State<'_, DesktopState>,
) -> std::result::Result<StartKimiLoginOutput, String> {
    if let Some(existing) = state.kimi_login.lock().await.take() {
        existing.cancel.notify_one();
    }

    let http = reqwest::Client::builder()
        .user_agent("sinew/0.1")
        .build()
        .map_err(error_to_string)?;
    let auth = request_kimi_device_authorization(&http)
        .await
        .map_err(error_to_string)?;
    let login_id = generate_kimi_state();
    let auth_url = auth.verification_uri_complete.clone();
    let user_code = auth.user_code.clone();
    let cancel = Arc::new(Notify::new());
    let outcome = Arc::new(StdMutex::new(None));

    {
        let mut active_login = state.kimi_login.lock().await;
        *active_login = Some(KimiLoginAttempt {
            id: login_id.clone(),
            cancel: cancel.clone(),
            outcome: outcome.clone(),
        });
    }

    let providers = state.providers.clone();
    tauri::async_runtime::spawn(async move {
        let result = run_kimi_device_login(http, auth, cancel).await;
        let login_outcome = match result {
            Ok(()) => match install_kimi_provider(&providers) {
                Ok(()) => KimiLoginOutcome {
                    success: true,
                    error: None,
                },
                Err(err) => KimiLoginOutcome {
                    success: false,
                    error: Some(err),
                },
            },
            Err(err) => KimiLoginOutcome {
                success: false,
                error: Some(err.to_string()),
            },
        };
        if let Ok(mut slot) = outcome.lock() {
            *slot = Some(login_outcome);
        }
    });

    Ok(StartKimiLoginOutput {
        login_id,
        auth_url,
        user_code,
    })
}

async fn run_kimi_device_login(
    http: reqwest::Client,
    auth: KimiDeviceAuthorization,
    cancel: Arc<Notify>,
) -> Result<()> {
    tokio::select! {
        _ = cancel.notified() => {
            anyhow::bail!("Login canceled");
        }
        result = wait_for_kimi_device_token(&http, &auth) => {
            result.map(|_| ()).map_err(|err| anyhow::anyhow!(err.to_string()))
        }
    }
}

#[tauri::command]
async fn cancel_kimi_oauth_login(
    state: State<'_, DesktopState>,
) -> std::result::Result<KimiProviderStatus, String> {
    if let Some(attempt) = state.kimi_login.lock().await.take() {
        attempt.cancel.notify_one();
    }
    let auth = load_default_kimi_auth_status().map_err(error_to_string)?;
    let state = if auth.connected {
        "connected"
    } else {
        "disconnected"
    };
    Ok(kimi_provider_status_from_auth(auth, state, None, None))
}

#[tauri::command]
async fn disconnect_kimi_provider(
    state: State<'_, DesktopState>,
) -> std::result::Result<KimiProviderStatus, String> {
    if let Some(attempt) = state.kimi_login.lock().await.take() {
        attempt.cancel.notify_one();
    }
    delete_default_kimi_auth().map_err(error_to_string)?;
    remove_kimi_provider(&state.providers)?;
    Ok(kimi_provider_status_from_auth(
        KimiAuthStatus::disconnected(),
        "disconnected",
        None,
        None,
    ))
}

#[tauri::command]
async fn probe_mcp_tools(
    state: State<'_, DesktopState>,
) -> std::result::Result<Vec<sinew_app::McpServerProbe>, String> {
    let settings = state.store.load_mcp_settings().map_err(error_to_string)?;
    Ok(probe_mcp_servers(&settings).await)
}

#[tauri::command]
async fn list_installed_skills_command(
    state: State<'_, DesktopState>,
    input: WorkspaceInput,
) -> std::result::Result<Vec<InstalledSkill>, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let settings = state.store.load_skill_settings().map_err(error_to_string)?;
    Ok(list_installed_skills(workspace_root, &settings))
}

#[tauri::command]
async fn save_skill_settings(
    state: State<'_, DesktopState>,
    input: SaveSkillSettingsInput,
) -> std::result::Result<Vec<InstalledSkill>, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let saved = state
        .store
        .save_skill_settings(&input.settings)
        .map_err(error_to_string)?;
    Ok(list_installed_skills(workspace_root, &saved))
}

#[tauri::command]
async fn send_message(
    app: AppHandle,
    state: State<'_, DesktopState>,
    input: SendMessageInput,
) -> std::result::Result<(), String> {
    let text = input.text.trim();
    if text.is_empty() {
        return Err("message cannot be empty".into());
    }
    let requested_mode = input.mode.map(AgentMode::from).unwrap_or_default();
    let plan_control = input.plan_control;
    let message_visibility = input
        .message_visibility
        .unwrap_or(MessageVisibilityInput::Normal);

    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let workspace_id = workspace_root.display().to_string();
    let effective_system_prompt =
        system_prompt_for_workspace(&workspace_root, &state.system_prompt)
            .map_err(error_to_string)?;
    if !wait_for_conversation_turn_slot(&state.active_turns, &input.conversation_id).await {
        return Err("a turn is already running for this conversation".into());
    }

    let mut conversation = state
        .store
        .load_conversation(&workspace_id, &input.conversation_id)
        .map_err(error_to_string)?
        .ok_or_else(|| "conversation not found".to_string())?;

    if let Some(index) = input.rewrite_from_history_index {
        if index > conversation.history.len() {
            return Err("rewrite index out of bounds".into());
        }
        if let Some(message) = conversation.history.get(index) {
            if !matches!(message.role, Role::User) {
                return Err("rewrite index must point to a user message".into());
            }
        }
        restore_workspace_for_rewrite(
            &app,
            &state.store,
            &workspace_root,
            &input.conversation_id,
            index,
        )
        .map_err(error_to_string)?;
        conversation.history.truncate(index);
        conversation.todo_list = todo_list_from_history(&conversation.history);
        conversation.plan_workflow = PlanWorkflowState::Idle;
    }

    let policy = plan_turn_policy(&conversation.plan_workflow, requested_mode, plan_control)?;
    let turn_plan_reminder = plan_implementation_turn_reminder(
        &workspace_root,
        &conversation.plan_workflow,
        &input.attachments,
        plan_control,
    )?;
    let turn_system_prompt = with_turn_plan_reminder(&effective_system_prompt, turn_plan_reminder);
    let mut mode_model_settings = conversation.mode_model_settings.clone();
    let selected_model = model_with_optional_selection(
        mode_model_settings.get(policy.mode),
        input.model,
        input.thinking,
    );
    mode_model_settings.set(policy.mode, selected_model.clone());
    conversation.mode_model_settings = mode_model_settings.clone();
    conversation.model = selected_model;
    let provider = provider_from_registry(&state, &conversation.model.provider)?;
    provider
        .capabilities(&conversation.model)
        .ok_or_else(|| format!("model `{}` is not supported", conversation.model.name))?;
    let mcp_settings = state.store.load_mcp_settings().map_err(error_to_string)?;
    let sub_agent_settings = state
        .store
        .load_sub_agent_settings()
        .map_err(error_to_string)?;
    let tool_settings = state.store.load_tool_settings().map_err(error_to_string)?;
    let skill_settings = state.store.load_skill_settings().map_err(error_to_string)?;
    let next_plan_workflow = policy.next_workflow.clone();
    conversation.plan_workflow = next_plan_workflow.clone();
    conversation.goal_workflow = if policy.mode == AgentMode::Goal {
        match message_visibility {
            MessageVisibilityInput::Normal => start_goal_workflow(text),
            MessageVisibilityInput::SystemReminder => {
                resume_goal_workflow(std::mem::take(&mut conversation.goal_workflow))
            }
        }
    } else {
        pause_goal_workflow(std::mem::take(&mut conversation.goal_workflow))
    };

    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let cancel = TurnCancel::new(cmd_tx);
    {
        let mut active_turns = state.active_turns.lock().await;
        if active_turns.contains_key(&input.conversation_id) {
            return Err("a turn is already running for this conversation".into());
        }
        active_turns.insert(input.conversation_id.clone(), cancel.clone());
    }

    let turn_user_history_index = conversation.history.len();
    let before_turn_snapshot = snapshot_workspace_for_checkpoint(&workspace_root);
    conversation.history.push(build_user_message(
        text,
        &input.attachments,
        &workspace_root,
        plan_control,
        message_visibility,
    ));
    state
        .store
        .save_conversation(&conversation)
        .map_err(|err| {
            let active_turns = state.active_turns.clone();
            let conversation_id = input.conversation_id.clone();
            tauri::async_runtime::spawn(async move {
                active_turns.lock().await.remove(&conversation_id);
            });
            error_to_string(err)
        })?;

    let providers = provider_registry_snapshot(&state)?;
    let context = TurnContext {
        provider,
        model: conversation.model.clone(),
        cache_key: Some(conversation.id.clone()),
        cache_stable_message_count: turn_user_history_index,
        auto_compact: true,
        mode: policy.mode,
        stop_questions: policy.stop_questions,
        system_prompt: turn_system_prompt.clone(),
        history: conversation.history.clone(),
        todo_list: conversation.todo_list.clone(),
        goal_workflow: conversation.goal_workflow.clone(),
        bash: Arc::new(BashTool::new(workspace_root.clone())),
        glob: Arc::new(GlobTool::new(workspace_root.clone())),
        grep: Arc::new(GrepTool::new(workspace_root.clone())),
        read: Arc::new(ReadTool::new(workspace_root.clone())),
        apply_patch: Arc::new(ApplyPatchTool::new(workspace_root.clone())),
        create_image: Arc::new(CreateImageTool::with_settings(
            workspace_root.clone(),
            tool_settings.image_provider,
            tool_settings.openai_image_api_key(),
            tool_settings.nano_banana_api_key(),
        )),
        todo_list_tool: Some(Arc::new(ToDoListTool::new())),
        question: Some(Arc::new(QuestionTool::new())),
        web_search: Arc::new(WebSearchTool::with_settings(
            tool_settings.web_search_provider,
            tool_settings.linkup_api_key(),
        )),
        web_fetch: Arc::new(WebFetchTool::new()),
        skill: Arc::new(SkillTool::with_settings(
            workspace_root.clone(),
            skill_settings.clone(),
        )),
        mcp: Arc::new(McpToolRegistry::new(mcp_settings.clone())),
        subagents: Some(Arc::new(SubAgentTool::new(
            workspace_root.clone(),
            turn_system_prompt.clone(),
            providers.clone(),
            sub_agent_settings.clone(),
            mcp_settings.clone(),
            tool_settings.clone(),
            skill_settings.clone(),
            state.max_tool_rounds,
            cancel.clone(),
        ))),
        teams: Some(Arc::new(TeamTool::new(
            conversation.id.clone(),
            workspace_root.clone(),
            turn_system_prompt.clone(),
            providers,
            sub_agent_settings,
            mcp_settings,
            tool_settings.clone(),
            skill_settings,
            conversation.model.clone(),
            state.max_tool_rounds,
            state.team_runtime.clone(),
            cancel.clone(),
        ))),
        tool_settings,
        event_scope: None,
        max_tool_rounds: state.max_tool_rounds,
        event_tx,
        cancel,
        cmd_rx,
    };

    let store = state.store.clone();
    let active_turns = state.active_turns.clone();
    let state_for_wake = state.inner().clone();
    let conversation_id = conversation.id.clone();
    let conversation_title = conversation.title.clone();
    let conversation_model = conversation.model.clone();
    let conversation_mode_model_settings = conversation.mode_model_settings.clone();
    let conversation_system_prompt = conversation.system_prompt.clone();
    let workspace_root_for_output = workspace_root.clone();
    let workspace_root_for_wake = workspace_root.clone();
    let plan_requested = policy.attach_plan;
    let before_turn_snapshot_for_checkpoint = before_turn_snapshot;

    tauri::async_runtime::spawn(async move {
        let mut engine = Box::pin(tauri::async_runtime::spawn(async move {
            run_turn(context).await
        }));
        let mut engine_done = false;
        let mut events_done = false;

        loop {
            tokio::select! {
                event = event_rx.recv(), if !events_done => {
                    match event {
                        Some(event) => {
                            if matches!(event, AgentEvent::TurnFinished) {
                                continue;
                            }
                            schedule_main_wake_for_swarm_event(
                                &app,
                                &state_for_wake,
                                &workspace_root_for_wake,
                                &conversation_id,
                                &event,
                            );
                            let _ = emit_agent_event(&app, &workspace_id, &conversation_id, &event);
                            emit_agent_file_changes(&app, &workspace_id, &event);
                        }
                        None => {
                            events_done = true;
                        }
                    }
                }
                engine_result = &mut engine, if !engine_done => {
                    engine_done = true;
                    match engine_result {
                        Ok(output) => {
                            let mut history = output.history;
                            let mut plan_workflow = next_plan_workflow.clone();
                            let mut goal_workflow = output.goal_workflow;
                            if output.interrupted {
                                goal_workflow = pause_goal_workflow(goal_workflow);
                            }
                            if plan_requested {
                                match attach_latest_plan_artifact(
                                    &workspace_root_for_output,
                                    &conversation_id,
                                    &mut history,
                                    turn_user_history_index,
                                ) {
                                    Ok(Some(artifact)) => {
                                        emit_workspace_file_change(
                                            &app,
                                            &workspace_root_for_output,
                                            &artifact.path,
                                        );
                                        plan_workflow = PlanWorkflowState::PlanReady { artifact };
                                    }
                                    Ok(None) => {}
                                    Err(err) => {
                                        let _ = emit_agent_event(
                                            &app,
                                            &workspace_id,
                                            &conversation_id,
                                            &AgentEvent::Error {
                                                message: format!("plan save failed: {err}"),
                                            },
                                        );
                                    }
                                }
                            }
                            let saved = SavedConversation {
                                id: conversation_id.clone(),
                                workspace_id: workspace_id.clone(),
                                title: conversation_title.clone(),
                                model: conversation_model.clone(),
                                mode_model_settings: conversation_mode_model_settings.clone(),
                                system_prompt: conversation_system_prompt.clone(),
                                todo_list: output.todo_list,
                                plan_workflow,
                                goal_workflow,
                                history,
                            };
                            let saved_ok = match store.save_conversation(&saved) {
                                Ok(()) => true,
                                Err(err) => {
                                    let _ = emit_agent_event(
                                        &app,
                                        &workspace_id,
                                        &conversation_id,
                                        &AgentEvent::Error {
                                            message: format!("save failed: {err}"),
                                        },
                                    );
                                    false
                                }
                            };
                            if saved_ok {
                                let after_turn_snapshot =
                                    snapshot_workspace_for_checkpoint(&workspace_root_for_output);
                                let checkpoint = checkpoint_from_snapshots(
                                    &before_turn_snapshot_for_checkpoint,
                                    &after_turn_snapshot,
                                );
                                if let Err(err) = store.save_turn_checkpoint(
                                    &conversation_id,
                                    turn_user_history_index,
                                    &checkpoint,
                                ) {
                                    let _ = emit_agent_event(
                                        &app,
                                        &workspace_id,
                                        &conversation_id,
                                        &AgentEvent::Error {
                                            message: format!("checkpoint save failed: {err}"),
                                        },
                                    );
                                }
                            }
                            active_turns.lock().await.remove(&conversation_id);
                            let _ = emit_agent_event(
                                &app,
                                &workspace_id,
                                &conversation_id,
                                &AgentEvent::TurnFinished,
                            );
                        }
                        Err(err) => {
                            let _ = emit_agent_event(
                                &app,
                                &workspace_id,
                                &conversation_id,
                                &AgentEvent::Error {
                                    message: format!("turn task failed: {err}"),
                                },
                            );
                            active_turns.lock().await.remove(&conversation_id);
                            let _ = emit_agent_event(
                                &app,
                                &workspace_id,
                                &conversation_id,
                                &AgentEvent::TurnFinished,
                            );
                        }
                    }
                }
            }

            if engine_done && events_done {
                break;
            }
        }
    });

    Ok(())
}

#[tauri::command]
async fn compact_conversation(
    app: AppHandle,
    state: State<'_, DesktopState>,
    input: CompactConversationInput,
) -> std::result::Result<(), String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let workspace_id = workspace_root.display().to_string();
    let effective_system_prompt =
        system_prompt_for_workspace(&workspace_root, &state.system_prompt)
            .map_err(error_to_string)?;
    if !wait_for_conversation_turn_slot(&state.active_turns, &input.conversation_id).await {
        return Err("a turn is already running for this conversation".into());
    }

    let mut conversation = state
        .store
        .load_conversation(&workspace_id, &input.conversation_id)
        .map_err(error_to_string)?
        .ok_or_else(|| "conversation not found".to_string())?;
    if conversation.history.is_empty() {
        return Err("conversation has no history to compact".into());
    }

    let selected_model =
        model_with_optional_selection(&conversation.model, input.model, input.thinking);
    let provider = provider_from_registry(&state, &selected_model.provider)?;
    provider
        .capabilities(&selected_model)
        .ok_or_else(|| format!("model `{}` is not supported", selected_model.name))?;
    let compact_mode = conversation_active_mode(&conversation);

    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();
    let cancel = TurnCancel::new(cmd_tx);
    {
        let mut active_turns = state.active_turns.lock().await;
        if active_turns.contains_key(&input.conversation_id) {
            return Err("a turn is already running for this conversation".into());
        }
        active_turns.insert(input.conversation_id.clone(), cancel);
    }

    let conversation_id = conversation.id.clone();
    let source_history = conversation.history.clone();
    let compaction_id = format!("context-compaction-{}", now_ms());

    let _ = emit_agent_event(
        &app,
        &workspace_id,
        &conversation_id,
        &AgentEvent::TurnStarted,
    );
    let _ = emit_agent_event(
        &app,
        &workspace_id,
        &conversation_id,
        &AgentEvent::ToolStarted {
            id: compaction_id.clone(),
            name: "context_compaction".to_string(),
        },
    );
    let _ = emit_agent_event(
        &app,
        &workspace_id,
        &conversation_id,
        &AgentEvent::ToolReady {
            id: compaction_id.clone(),
            summary: "Compact context".to_string(),
            args_pretty: "{}".to_string(),
        },
    );

    let (summary_delta_tx, mut summary_delta_rx) = mpsc::unbounded_channel();
    let app_for_deltas = app.clone();
    let workspace_id_for_deltas = workspace_id.clone();
    let conversation_id_for_deltas = conversation_id.clone();
    let compaction_id_for_deltas = compaction_id.clone();
    let delta_forwarder = tauri::async_runtime::spawn(async move {
        while let Some(delta) = summary_delta_rx.recv().await {
            let _ = emit_agent_event(
                &app_for_deltas,
                &workspace_id_for_deltas,
                &conversation_id_for_deltas,
                &AgentEvent::ToolOutputDelta {
                    id: compaction_id_for_deltas.clone(),
                    delta,
                },
            );
        }
    });

    let result = compact_conversation_history(
        provider,
        selected_model.clone(),
        effective_system_prompt,
        source_history.clone(),
        Some(conversation_id.clone()),
        source_history.len(),
        &mut cmd_rx,
        Some(summary_delta_tx),
    )
    .await;
    let _ = delta_forwarder.await;

    let command_result = match result {
        Ok(output) => {
            let retained = output.retained_user_messages;
            let summary = output.summary;
            conversation.model = selected_model.clone();
            conversation
                .mode_model_settings
                .set(compact_mode, selected_model);
            conversation.history = output.history;
            conversation.todo_list = todo_list_from_history(&conversation.history);
            match state.store.save_conversation(&conversation) {
                Ok(()) => {
                    let label = match retained {
                        0 => "No raw user messages retained".to_string(),
                        1 => "Retained 1 recent user message".to_string(),
                        count => format!("Retained {count} recent user messages"),
                    };
                    let _ = emit_agent_event(
                        &app,
                        &workspace_id,
                        &conversation_id,
                        &AgentEvent::ToolFinished {
                            id: compaction_id.clone(),
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
                    let message = format!("save failed: {err}");
                    let _ = emit_agent_event(
                        &app,
                        &workspace_id,
                        &conversation_id,
                        &AgentEvent::ToolFinished {
                            id: compaction_id.clone(),
                            output: message.clone(),
                            is_error: true,
                            file_changes: Vec::new(),
                            images: Vec::new(),
                            meta: None,
                        },
                    );
                    let _ = emit_agent_event(
                        &app,
                        &workspace_id,
                        &conversation_id,
                        &AgentEvent::Error {
                            message: message.clone(),
                        },
                    );
                    Err(message)
                }
            }
        }
        Err(err) => {
            let message = err.to_string();
            let _ = emit_agent_event(
                &app,
                &workspace_id,
                &conversation_id,
                &AgentEvent::ToolFinished {
                    id: compaction_id.clone(),
                    output: message.clone(),
                    is_error: true,
                    file_changes: Vec::new(),
                    images: Vec::new(),
                    meta: None,
                },
            );
            let _ = emit_agent_event(
                &app,
                &workspace_id,
                &conversation_id,
                &AgentEvent::Error {
                    message: message.clone(),
                },
            );
            Err(message)
        }
    };

    let _ = emit_agent_event(
        &app,
        &workspace_id,
        &conversation_id,
        &AgentEvent::TurnFinished,
    );
    state.active_turns.lock().await.remove(&conversation_id);

    command_result
}

async fn wait_for_conversation_turn_slot(
    active_turns: &Arc<Mutex<HashMap<String, TurnCancel>>>,
    conversation_id: &str,
) -> bool {
    wait_for_conversation_turn_slot_with_attempts(
        active_turns,
        conversation_id,
        TURN_SLOT_WAIT_ATTEMPTS,
    )
    .await
}

async fn wait_for_conversation_turn_slot_with_attempts(
    active_turns: &Arc<Mutex<HashMap<String, TurnCancel>>>,
    conversation_id: &str,
    attempts: usize,
) -> bool {
    for attempt in 0..attempts {
        let is_busy = active_turns.lock().await.contains_key(conversation_id);
        if !is_busy {
            return true;
        }
        if attempt + 1 < attempts {
            tokio::time::sleep(Duration::from_millis(TURN_SLOT_WAIT_INTERVAL_MS)).await;
        }
    }
    false
}

fn agent_swarm_error_from_event(event: &AgentEvent) -> Option<(String, String)> {
    let AgentEvent::SubAgentEvent {
        agent_name,
        team_name,
        event,
        ..
    } = event
    else {
        return None;
    };
    team_name.as_ref()?;
    let AgentEvent::Error { message } = event.as_ref() else {
        return None;
    };
    Some((agent_name.clone(), message.clone()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AgentSwarmCompletion {
    team_name: String,
    responses: Vec<AgentSwarmFinalResponse>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AgentSwarmFinalResponse {
    agent: String,
    status: String,
    last_response: String,
    last_error: Option<String>,
}

fn schedule_main_wake_for_swarm_event(
    app: &AppHandle,
    state: &DesktopState,
    workspace_root: &Path,
    conversation_id: &str,
    event: &AgentEvent,
) {
    schedule_main_wake_for_swarm_error(app, state, workspace_root, conversation_id, event);
    schedule_main_wake_for_swarm_completion(app, state, workspace_root, conversation_id, event);
}

fn schedule_main_wake_for_swarm_error(
    app: &AppHandle,
    state: &DesktopState,
    workspace_root: &Path,
    conversation_id: &str,
    event: &AgentEvent,
) {
    let Some((agent_name, error)) = agent_swarm_error_from_event(event) else {
        return;
    };
    let app_for_wake = app.clone();
    let state_for_wake = state.clone();
    let workspace_root_for_wake = workspace_root.to_path_buf();
    let conversation_id_for_wake = conversation_id.to_string();
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        if let Err(err) = wake_main_agent_for_swarm_error(
            app_for_wake,
            state_for_wake,
            workspace_root_for_wake,
            conversation_id_for_wake,
            agent_name,
            error,
        )
        .await
        {
            tracing::warn!(%err, "failed to wake main agent for swarm error");
        }
    });
}

fn schedule_main_wake_for_swarm_completion(
    app: &AppHandle,
    state: &DesktopState,
    workspace_root: &Path,
    conversation_id: &str,
    event: &AgentEvent,
) {
    let Some(completion) = agent_swarm_completion_from_event(event) else {
        return;
    };
    let app_for_wake = app.clone();
    let state_for_wake = state.clone();
    let workspace_root_for_wake = workspace_root.to_path_buf();
    let conversation_id_for_wake = conversation_id.to_string();
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        if let Err(err) = wake_main_agent_for_swarm_completion(
            app_for_wake,
            state_for_wake,
            workspace_root_for_wake,
            conversation_id_for_wake,
            completion,
        )
        .await
        {
            tracing::warn!(%err, "failed to wake main agent for swarm completion");
        }
    });
}

fn agent_swarm_completion_from_event(event: &AgentEvent) -> Option<AgentSwarmCompletion> {
    let AgentEvent::ToolFinished {
        is_error,
        meta: Some(meta),
        ..
    } = event
    else {
        return None;
    };
    if *is_error {
        return None;
    }
    let meta = meta.as_object()?;
    let status = meta
        .get("teamRunStatus")
        .and_then(Value::as_str)
        .map(str::trim)?;
    if status != "completed" {
        return None;
    }
    let team = meta.get("team")?.as_object()?;
    let team_name = team.get("name")?.as_str()?.trim();
    if team_name.is_empty() {
        return None;
    }
    let mut responses = agent_swarm_final_responses_from_value(meta.get("agentFinalResponses"));
    if responses.is_empty() {
        responses = agent_swarm_final_responses_from_team(meta.get("team"));
    }
    Some(AgentSwarmCompletion {
        team_name: team_name.to_string(),
        responses,
    })
}

fn agent_swarm_final_responses_from_value(value: Option<&Value>) -> Vec<AgentSwarmFinalResponse> {
    value
        .and_then(Value::as_array)
        .map(|responses| {
            responses
                .iter()
                .filter_map(|value| agent_swarm_final_response_from_record(value, false))
                .collect()
        })
        .unwrap_or_default()
}

fn agent_swarm_final_responses_from_team(value: Option<&Value>) -> Vec<AgentSwarmFinalResponse> {
    value
        .and_then(Value::as_object)
        .and_then(|team| team.get("agents"))
        .and_then(Value::as_array)
        .map(|agents| {
            agents
                .iter()
                .filter_map(|value| agent_swarm_final_response_from_record(value, true))
                .collect()
        })
        .unwrap_or_default()
}

fn agent_swarm_final_response_from_record(
    value: &Value,
    team_agent_snapshot: bool,
) -> Option<AgentSwarmFinalResponse> {
    let record = value.as_object()?;
    let agent = record
        .get("agent")
        .or_else(|| record.get("name"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    let status = final_response_status_from_meta(record.get("status"));
    let last_response_key = if team_agent_snapshot {
        "lastSummary"
    } else {
        "lastResponse"
    };
    let last_response = record
        .get(last_response_key)
        .or_else(|| record.get("lastResponse"))
        .or_else(|| record.get("lastSummary"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("No final response recorded.");
    let last_error = record
        .get("lastError")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    Some(AgentSwarmFinalResponse {
        agent: agent.to_string(),
        status,
        last_response: last_response.to_string(),
        last_error,
    })
}

fn final_response_status_from_meta(value: Option<&Value>) -> String {
    let status = value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("finished");
    if status == "idle" {
        "finished".to_string()
    } else {
        status.to_string()
    }
}

async fn wake_main_agent_for_swarm_error(
    app: AppHandle,
    state: DesktopState,
    workspace_root: PathBuf,
    conversation_id: String,
    agent_name: String,
    error: String,
) -> std::result::Result<(), String> {
    let wake_text = agent_swarm_error_wake_text(&agent_name, &error);
    wake_main_agent_for_swarm_notice(app, state, workspace_root, conversation_id, wake_text).await
}

async fn wake_main_agent_for_swarm_completion(
    app: AppHandle,
    state: DesktopState,
    workspace_root: PathBuf,
    conversation_id: String,
    completion: AgentSwarmCompletion,
) -> std::result::Result<(), String> {
    let wake_text = agent_swarm_completion_wake_text(&completion);
    wake_main_agent_for_swarm_notice(app, state, workspace_root, conversation_id, wake_text).await
}

async fn wake_main_agent_for_swarm_notice(
    app: AppHandle,
    state: DesktopState,
    workspace_root: PathBuf,
    conversation_id: String,
    wake_text: String,
) -> std::result::Result<(), String> {
    if !wait_for_conversation_turn_slot_with_attempts(
        &state.active_turns,
        &conversation_id,
        SWARM_WAKE_TURN_SLOT_WAIT_ATTEMPTS,
    )
    .await
    {
        return Ok(());
    }

    let workspace_id = workspace_root.display().to_string();
    let effective_system_prompt =
        system_prompt_for_workspace(&workspace_root, &state.system_prompt)
            .map_err(error_to_string)?;
    let mut conversation = state
        .store
        .load_conversation(&workspace_id, &conversation_id)
        .map_err(error_to_string)?
        .ok_or_else(|| "conversation not found".to_string())?;

    let selected_model = conversation.mode_model_settings.get(AgentMode::Act).clone();
    conversation.model = selected_model;
    let provider = provider_from_registry(&state, &conversation.model.provider)?;
    provider
        .capabilities(&conversation.model)
        .ok_or_else(|| format!("model `{}` is not supported", conversation.model.name))?;

    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let cancel = TurnCancel::new(cmd_tx);
    {
        let mut active_turns = state.active_turns.lock().await;
        if active_turns.contains_key(&conversation_id) {
            return Ok(());
        }
        active_turns.insert(conversation_id.clone(), cancel.clone());
    }

    let turn_user_history_index = conversation.history.len();
    let before_turn_snapshot = snapshot_workspace_for_checkpoint(&workspace_root);
    conversation.history.push(build_user_message(
        &wake_text,
        &[],
        &workspace_root,
        None,
        MessageVisibilityInput::SystemReminder,
    ));
    state
        .store
        .save_conversation(&conversation)
        .map_err(|err| {
            let active_turns = state.active_turns.clone();
            let conversation_id = conversation_id.clone();
            tauri::async_runtime::spawn(async move {
                active_turns.lock().await.remove(&conversation_id);
            });
            error_to_string(err)
        })?;

    let mcp_settings = state.store.load_mcp_settings().map_err(error_to_string)?;
    let sub_agent_settings = state
        .store
        .load_sub_agent_settings()
        .map_err(error_to_string)?;
    let tool_settings = state.store.load_tool_settings().map_err(error_to_string)?;
    let skill_settings = state.store.load_skill_settings().map_err(error_to_string)?;
    let turn_system_prompt = with_turn_plan_reminder(&effective_system_prompt, None);
    let providers = provider_registry_snapshot(&state)?;
    let context = TurnContext {
        provider,
        model: conversation.model.clone(),
        cache_key: Some(conversation.id.clone()),
        cache_stable_message_count: turn_user_history_index,
        auto_compact: true,
        mode: AgentMode::Act,
        stop_questions: false,
        system_prompt: turn_system_prompt.clone(),
        history: conversation.history.clone(),
        todo_list: conversation.todo_list.clone(),
        goal_workflow: conversation.goal_workflow.clone(),
        bash: Arc::new(BashTool::new(workspace_root.clone())),
        glob: Arc::new(GlobTool::new(workspace_root.clone())),
        grep: Arc::new(GrepTool::new(workspace_root.clone())),
        read: Arc::new(ReadTool::new(workspace_root.clone())),
        apply_patch: Arc::new(ApplyPatchTool::new(workspace_root.clone())),
        create_image: Arc::new(CreateImageTool::with_settings(
            workspace_root.clone(),
            tool_settings.image_provider,
            tool_settings.openai_image_api_key(),
            tool_settings.nano_banana_api_key(),
        )),
        todo_list_tool: Some(Arc::new(ToDoListTool::new())),
        question: Some(Arc::new(QuestionTool::new())),
        web_search: Arc::new(WebSearchTool::with_settings(
            tool_settings.web_search_provider,
            tool_settings.linkup_api_key(),
        )),
        web_fetch: Arc::new(WebFetchTool::new()),
        skill: Arc::new(SkillTool::with_settings(
            workspace_root.clone(),
            skill_settings.clone(),
        )),
        mcp: Arc::new(McpToolRegistry::new(mcp_settings.clone())),
        subagents: Some(Arc::new(SubAgentTool::new(
            workspace_root.clone(),
            turn_system_prompt.clone(),
            providers.clone(),
            sub_agent_settings.clone(),
            mcp_settings.clone(),
            tool_settings.clone(),
            skill_settings.clone(),
            state.max_tool_rounds,
            cancel.clone(),
        ))),
        teams: Some(Arc::new(TeamTool::new(
            conversation.id.clone(),
            workspace_root.clone(),
            turn_system_prompt.clone(),
            providers,
            sub_agent_settings,
            mcp_settings,
            tool_settings.clone(),
            skill_settings,
            conversation.model.clone(),
            state.max_tool_rounds,
            state.team_runtime.clone(),
            cancel.clone(),
        ))),
        tool_settings,
        event_scope: None,
        max_tool_rounds: state.max_tool_rounds,
        event_tx,
        cancel,
        cmd_rx,
    };

    let store = state.store.clone();
    let active_turns = state.active_turns.clone();
    let conversation_title = conversation.title.clone();
    let conversation_model = conversation.model.clone();
    let conversation_mode_model_settings = conversation.mode_model_settings.clone();
    let conversation_system_prompt = conversation.system_prompt.clone();
    let plan_workflow = conversation.plan_workflow.clone();
    let conversation_id_for_events = conversation_id.clone();
    let workspace_root_for_checkpoint = workspace_root.clone();
    let before_turn_snapshot_for_checkpoint = before_turn_snapshot;

    tauri::async_runtime::spawn(async move {
        let mut engine = Box::pin(tauri::async_runtime::spawn(async move {
            run_turn(context).await
        }));
        let mut engine_done = false;
        let mut events_done = false;

        loop {
            tokio::select! {
                event = event_rx.recv(), if !events_done => {
                    match event {
                        Some(event) => {
                            if matches!(event, AgentEvent::TurnFinished) {
                                continue;
                            }
                            schedule_main_wake_for_swarm_event(
                                &app,
                                &state,
                                &workspace_root,
                                &conversation_id_for_events,
                                &event,
                            );
                            let _ = emit_agent_event(
                                &app,
                                &workspace_id,
                                &conversation_id_for_events,
                                &event,
                            );
                            emit_agent_file_changes(&app, &workspace_id, &event);
                        }
                        None => {
                            events_done = true;
                        }
                    }
                }
                engine_result = &mut engine, if !engine_done => {
                    engine_done = true;
                    match engine_result {
                        Ok(output) => {
                            let saved = SavedConversation {
                                id: conversation_id_for_events.clone(),
                                workspace_id: workspace_id.clone(),
                                title: conversation_title.clone(),
                                model: conversation_model.clone(),
                                mode_model_settings: conversation_mode_model_settings.clone(),
                                system_prompt: conversation_system_prompt.clone(),
                                todo_list: output.todo_list,
                                plan_workflow: plan_workflow.clone(),
                                goal_workflow: output.goal_workflow,
                                history: output.history,
                            };
                            let saved_ok = match store.save_conversation(&saved) {
                                Ok(()) => true,
                                Err(err) => {
                                    let _ = emit_agent_event(
                                        &app,
                                        &workspace_id,
                                        &conversation_id_for_events,
                                        &AgentEvent::Error {
                                            message: format!("save failed: {err}"),
                                        },
                                    );
                                    false
                                }
                            };
                            if saved_ok {
                                let after_turn_snapshot = snapshot_workspace_for_checkpoint(
                                    &workspace_root_for_checkpoint,
                                );
                                let checkpoint = checkpoint_from_snapshots(
                                    &before_turn_snapshot_for_checkpoint,
                                    &after_turn_snapshot,
                                );
                                if let Err(err) = store.save_turn_checkpoint(
                                    &conversation_id_for_events,
                                    turn_user_history_index,
                                    &checkpoint,
                                ) {
                                    let _ = emit_agent_event(
                                        &app,
                                        &workspace_id,
                                        &conversation_id_for_events,
                                        &AgentEvent::Error {
                                            message: format!("checkpoint save failed: {err}"),
                                        },
                                    );
                                }
                            }
                            active_turns.lock().await.remove(&conversation_id_for_events);
                            let _ = emit_agent_event(
                                &app,
                                &workspace_id,
                                &conversation_id_for_events,
                                &AgentEvent::TurnFinished,
                            );
                        }
                        Err(err) => {
                            let _ = emit_agent_event(
                                &app,
                                &workspace_id,
                                &conversation_id_for_events,
                                &AgentEvent::Error {
                                    message: format!("turn task failed: {err}"),
                                },
                            );
                            active_turns.lock().await.remove(&conversation_id_for_events);
                            let _ = emit_agent_event(
                                &app,
                                &workspace_id,
                                &conversation_id_for_events,
                                &AgentEvent::TurnFinished,
                            );
                        }
                    }
                }
            }

            if engine_done && events_done {
                break;
            }
        }
    });

    Ok(())
}

fn agent_swarm_error_wake_text(agent_name: &str, error: &str) -> String {
    format!(
        "<agent_swarm_error>\nagent: @{agent_name}\nerror: {}\n</agent_swarm_error>\n\nHandle this Agent Swarm failure now. Relaunch only the failed teammate when that is the right recovery. If it keeps failing, stop that teammate so their open work returns to pending.",
        truncate_hidden_turn_line(error, 1200)
    )
}

fn agent_swarm_completion_wake_text(completion: &AgentSwarmCompletion) -> String {
    let mut lines = vec![
        "<agent_swarm_finished>".to_string(),
        format!("team: {}", completion.team_name),
        "agentResponses:".to_string(),
    ];
    if completion.responses.is_empty() {
        lines.push("- none".to_string());
    } else {
        for response in &completion.responses {
            lines.push(format!("- agent: @{}", response.agent));
            lines.push(format!("  status: {}", response.status));
            if let Some(error) = response
                .last_error
                .as_deref()
                .filter(|value| !value.trim().is_empty())
            {
                lines.push(format!(
                    "  error: {}",
                    truncate_hidden_turn_line(error.trim(), 1200)
                ));
            }
            lines.push("  lastResponse: |".to_string());
            lines.extend(indent_hidden_lines(
                &truncate_hidden_turn_line(&response.last_response, 4000),
                "    ",
            ));
        }
    }
    lines.push("</agent_swarm_finished>".to_string());
    lines.push(String::new());
    lines.push("L'Agent Swarm a terminé. Réponds maintenant à l'utilisateur pour lui dire que l'Agent Swarm a terminé, puis résume les dernières réponses structurées ci-dessus agent par agent. N'utilise pas TeamStatus, le shell, ni les fichiers juste pour vérifier que le swarm est terminé.".to_string());
    lines.join("\n")
}

fn indent_hidden_lines(value: &str, indent: &str) -> Vec<String> {
    let lines = value.lines().collect::<Vec<_>>();
    if lines.is_empty() {
        return vec![indent.to_string()];
    }
    lines
        .into_iter()
        .map(|line| format!("{indent}{line}"))
        .collect()
}

fn truncate_hidden_turn_line(value: &str, max_chars: usize) -> String {
    let mut out = String::new();
    let mut count = 0usize;
    for ch in value.chars() {
        if count >= max_chars {
            out.push_str("...");
            break;
        }
        out.push(ch);
        count += 1;
    }
    out
}

#[tauri::command]
async fn estimate_context(
    state: State<'_, DesktopState>,
    input: ContextEstimateInput,
) -> std::result::Result<ContextEstimateOutput, String> {
    let requested_mode = input.mode.map(AgentMode::from).unwrap_or_default();
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let workspace_id = workspace_root.display().to_string();
    let effective_system_prompt =
        system_prompt_for_workspace(&workspace_root, &state.system_prompt)
            .map_err(error_to_string)?;

    let mut conversation = state
        .store
        .load_conversation(&workspace_id, &input.conversation_id)
        .map_err(error_to_string)?
        .ok_or_else(|| "conversation not found".to_string())?;

    if let Some(index) = input.rewrite_from_history_index {
        if index > conversation.history.len() {
            return Err("rewrite index out of bounds".into());
        }
        if let Some(message) = conversation.history.get(index) {
            if !matches!(message.role, Role::User) {
                return Err("rewrite index must point to a user message".into());
            }
        }
        conversation.history.truncate(index);
        conversation.todo_list = todo_list_from_history(&conversation.history);
        conversation.plan_workflow = PlanWorkflowState::Idle;
    }

    let mode = plan_estimate_mode(&conversation.plan_workflow, requested_mode);
    let mode_model_settings = conversation.mode_model_settings.clone();
    let selected_model =
        model_with_optional_selection(mode_model_settings.get(mode), input.model, input.thinking);
    conversation.mode_model_settings = mode_model_settings;
    conversation.model = selected_model;
    let provider = provider_from_registry(&state, &conversation.model.provider)?;

    let draft = input.text.trim();
    let has_pending_user_input = !draft.is_empty() || !input.attachments.is_empty();
    let cache_stable_message_count = conversation.history.len();
    if has_pending_user_input {
        conversation.history.push(build_user_message(
            draft,
            &input.attachments,
            &workspace_root,
            None,
            MessageVisibilityInput::Normal,
        ));
    }

    let tool_settings = state.store.load_tool_settings().map_err(error_to_string)?;
    let skill_settings = state.store.load_skill_settings().map_err(error_to_string)?;
    let mut tools = tool_descriptors_for_workspace(&workspace_root, mode, &skill_settings);
    let mcp_settings = state.store.load_mcp_settings().map_err(error_to_string)?;
    let mcp = McpToolRegistry::new(mcp_settings.clone());
    let mcp_tools = mcp.refresh_catalog(&conversation.history).await;
    let mcp_tool_names = tool_name_set(&mcp_tools);
    tools.extend(mcp_tools);
    let sub_agent_settings = state
        .store
        .load_sub_agent_settings()
        .map_err(error_to_string)?;
    let sub_agent_tools = SubAgentTool::new(
        workspace_root.clone(),
        effective_system_prompt.clone(),
        provider_registry_snapshot(&state)?,
        sub_agent_settings,
        mcp_settings,
        tool_settings.clone(),
        skill_settings,
        state.max_tool_rounds,
        TurnCancel::empty(),
    )
    .descriptors();
    let team_tools = TeamTool::descriptors_static();
    let mut sub_agent_tool_names = tool_name_set(&sub_agent_tools);
    sub_agent_tool_names.extend(tool_name_set(&team_tools));
    tools.extend(sub_agent_tools);
    tools.extend(team_tools);
    let tools = tool_settings.apply_to_descriptors(tools);
    let system = system_prompt_with_todo(&effective_system_prompt, &conversation.todo_list);
    let system_prompt = system_prompt_for_mode(&system, mode);
    let workspace_rules_weight =
        workspace_rules_weight(&workspace_root).map_err(error_to_string)?;
    let breakdown_weights = context_breakdown_weights(
        &system_prompt,
        workspace_rules_weight,
        &conversation.history,
        &tools,
        &mcp_tool_names,
        &sub_agent_tool_names,
    );
    estimate_model_context(
        provider,
        conversation.model.clone(),
        conversation.history.clone(),
        system_prompt,
        tools,
        Some(conversation.id.clone()),
        cache_stable_message_count,
        breakdown_weights,
        !has_pending_user_input,
    )
    .await
}

#[tauri::command]
async fn estimate_sub_agent_context(
    state: State<'_, DesktopState>,
    input: SubAgentContextEstimateInput,
) -> std::result::Result<ContextEstimateOutput, String> {
    let mode = input.mode.map(AgentMode::from).unwrap_or_default();
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let effective_system_prompt =
        system_prompt_for_workspace(&workspace_root, &state.system_prompt)
            .map_err(error_to_string)?;
    let settings = state
        .store
        .load_sub_agent_settings()
        .map_err(error_to_string)?
        .normalized();
    let configured_agent = settings
        .agents
        .iter()
        .find(|agent| agent.id == input.agent_id);
    let team_agent = configured_agent
        .is_none()
        .then(|| team_agent_estimate_identity(&input.agent_id, input.agent_name.as_deref()))
        .flatten();
    if configured_agent.is_none() && team_agent.is_none() {
        return Err("sub-agent not found".to_string());
    }
    let provider = provider_from_registry(&state, &input.model.provider)?;

    let tool_settings = state.store.load_tool_settings().map_err(error_to_string)?;
    let skill_settings = state.store.load_skill_settings().map_err(error_to_string)?;
    let mut tools = tool_descriptors_for_workspace(&workspace_root, mode, &skill_settings);
    let mcp_settings = state.store.load_mcp_settings().map_err(error_to_string)?;
    let mcp = McpToolRegistry::new(mcp_settings);
    let mcp_tools = mcp.refresh_catalog(&input.history).await;
    let mcp_tool_names = tool_name_set(&mcp_tools);
    tools.extend(mcp_tools);
    if team_agent.is_some() {
        tools.retain(|tool| tool.name != "ToDoList" && tool.name != "Question");
        tools.extend(TeamTool::agent_descriptors_static());
    }
    let tools = tool_settings.apply_to_descriptors(tools);
    let agent_system_prompt = if let Some(agent) = configured_agent {
        subagent_system_prompt(&effective_system_prompt, agent)
    } else if let Some((team_name, agent_name)) = team_agent.as_ref() {
        team_agent_system_prompt_for_estimate(
            &effective_system_prompt,
            team_name,
            &input.agent_id,
            agent_name,
            &input.model,
        )
    } else {
        return Err("sub-agent not found".to_string());
    };
    let system = system_prompt_with_todo(&agent_system_prompt, &TodoListState::default());
    let system_prompt = system_prompt_for_mode(&system, mode);
    let workspace_rules_weight =
        workspace_rules_weight(&workspace_root).map_err(error_to_string)?;
    let breakdown_weights = context_breakdown_weights(
        &system_prompt,
        workspace_rules_weight,
        &input.history,
        &tools,
        &mcp_tool_names,
        &HashSet::new(),
    );
    estimate_model_context(
        provider,
        input.model.clone(),
        input.history.clone(),
        system_prompt,
        tools,
        Some(format!(
            "subagent:{}:{}",
            workspace_root.display(),
            input.agent_id
        )),
        input.history.len(),
        breakdown_weights,
        true,
    )
    .await
}

fn team_agent_estimate_identity(
    agent_id: &str,
    agent_name: Option<&str>,
) -> Option<(String, String)> {
    let (raw_agent, raw_team) = agent_id.rsplit_once('@')?;
    let team_name = raw_team.trim();
    if team_name.is_empty() {
        return None;
    }
    let name = agent_name
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .or_else(|| {
            let fallback = raw_agent.trim();
            (!fallback.is_empty()).then_some(fallback)
        })?;
    Some((team_name.to_string(), name.to_string()))
}

fn team_agent_system_prompt_for_estimate(
    base: &str,
    team_name: &str,
    agent_id: &str,
    agent_name: &str,
    model: &ModelRef,
) -> String {
    let config_agent = SubAgentConfig {
        id: agent_id.to_string(),
        name: agent_name.to_string(),
        description: String::new(),
        prompt: String::new(),
        model: model.clone(),
        enabled: true,
    };
    let base = subagent_system_prompt(base, &config_agent);
    format!(
        "{base}\n\n<agent_team_profile team=\"{}\" name=\"{}\">\nYou are part of an autonomous agent team.\nYour work is coordinated through the task system and teammate messaging, use SendMessage tool to talk with your team.\nIf your owned work is blocked by incomplete tasks, end your turn and sleep; you will be woken automatically when your owned tasks unlock or when a teammate sends you a direct message.\n</agent_team_profile>",
        html_escape(team_name),
        html_escape(agent_name)
    )
}

async fn estimate_model_context(
    provider: Arc<dyn Provider>,
    model: ModelRef,
    history: Vec<ChatMessage>,
    system_prompt: String,
    tools: Vec<ToolDescriptor>,
    cache_key: Option<String>,
    cache_stable_message_count: usize,
    breakdown_weights: Vec<ContextBreakdownWeight>,
    prefer_latest_stream_usage: bool,
) -> std::result::Result<ContextEstimateOutput, String> {
    let caps = provider
        .capabilities(&model)
        .ok_or_else(|| format!("model `{}` is not supported", model.name))?;
    let latest_stream_usage = prefer_latest_stream_usage
        .then(|| latest_stream_context_usage(&history, &model.provider, &model.name))
        .flatten();
    let (usage, exact, error) = match latest_stream_usage {
        Some(usage) => (usage, true, None),
        None => {
            let mut request = ProviderRequest::new(model.clone(), history)
                .with_system(system_prompt)
                .with_tools(tools)
                .with_cache_stable_message_count(cache_stable_message_count);
            if let Some(cache_key) = cache_key {
                request = request.with_cache_key(cache_key);
            }

            match provider.estimate_tokens(request).await {
                Ok(estimate) => (
                    ContextTokenUsage::from_input_tokens(estimate.input_tokens),
                    estimate.exact,
                    None,
                ),
                Err(err) => {
                    let local_estimate = local_context_token_estimate(&breakdown_weights);
                    (
                        ContextTokenUsage::from_input_tokens(local_estimate),
                        false,
                        Some(err.to_string()),
                    )
                }
            }
        }
    };
    let used_tokens = usage.total();

    Ok(ContextEstimateOutput {
        used_tokens,
        context_window: caps.context_window,
        preferred_window: caps.preferred_window,
        max_output_tokens: caps.max_output_tokens,
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        reasoning_tokens: usage.reasoning_tokens,
        cache_read_tokens: usage.cache_read_tokens,
        cache_creation_tokens: usage.cache_creation_tokens,
        exact,
        error,
        breakdown: context_usage_breakdown(usage)
            .unwrap_or_else(|| scale_context_breakdown(used_tokens, breakdown_weights)),
    })
}

#[tauri::command]
async fn cancel_turn(
    state: State<'_, DesktopState>,
    input: ConversationInput,
) -> std::result::Result<bool, String> {
    let sender = state
        .active_turns
        .lock()
        .await
        .get(&input.conversation_id)
        .cloned();

    Ok(match sender {
        Some(sender) => sender.cancel_all(),
        None => false,
    })
}

#[tauri::command]
async fn stop_agent_swarm_command(
    app: AppHandle,
    state: State<'_, DesktopState>,
    input: StopAgentSwarmInput,
) -> std::result::Result<String, String> {
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let workspace_id = workspace_root.display().to_string();
    let conversation = state
        .store
        .load_conversation(&workspace_id, &input.conversation_id)
        .map_err(error_to_string)?
        .ok_or_else(|| "conversation not found".to_string())?;
    let effective_system_prompt =
        system_prompt_for_workspace(&workspace_root, &state.system_prompt)
            .map_err(error_to_string)?;
    let mcp_settings = state.store.load_mcp_settings().map_err(error_to_string)?;
    let sub_agent_settings = state
        .store
        .load_sub_agent_settings()
        .map_err(error_to_string)?;
    let tool_settings = state.store.load_tool_settings().map_err(error_to_string)?;
    let skill_settings = state.store.load_skill_settings().map_err(error_to_string)?;
    let team_tool = TeamTool::new(
        conversation.id.clone(),
        workspace_root.clone(),
        effective_system_prompt,
        provider_registry_snapshot(&state)?,
        sub_agent_settings,
        mcp_settings,
        tool_settings,
        skill_settings,
        conversation.model.clone(),
        state.max_tool_rounds,
        state.team_runtime.clone(),
        TurnCancel::empty(),
    );
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let mut payload = serde_json::Map::new();
    if let Some(team_name) = input
        .team_name
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        payload.insert("team_name".into(), json!(team_name));
    }
    let result = team_tool
        .run(
            "ui-agent-swarm-stop",
            "TeamStop",
            Value::Object(payload),
            AgentMode::Act,
            event_tx,
        )
        .await
        .ok_or_else(|| "TeamStop is unavailable".to_string())?;
    while let Ok(event) = event_rx.try_recv() {
        let _ = emit_agent_event(&app, &workspace_id, &conversation.id, &event);
        emit_agent_file_changes(&app, &workspace_id, &event);
    }
    if result.is_error {
        Err(result.content)
    } else {
        Ok(result.content)
    }
}

#[tauri::command]
async fn run_terminal_command(
    app: AppHandle,
    input: TerminalCommandInput,
) -> std::result::Result<TerminalCommandOutput, String> {
    let command = input.command.trim();
    if command.is_empty() {
        return Err("command cannot be empty".into());
    }

    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;
    let result = BashTool::new(workspace_root.clone())
        .run(json!({
            "command": command,
            "timeout_secs": 120,
        }))
        .await;

    for change in &result.file_changes {
        emit_workspace_file_change(&app, &workspace_root, &change.relative_path);
    }

    Ok(TerminalCommandOutput {
        content: result.content,
        is_error: result.is_error,
    })
}

#[tauri::command]
async fn spawn_terminal(
    app: AppHandle,
    state: State<'_, DesktopState>,
    input: TerminalSpawnInput,
) -> std::result::Result<TerminalSpawnOutput, String> {
    let session_id = validate_terminal_value(&input.session_id, "session id")?.to_string();
    let token = validate_terminal_value(&input.token, "session token")?.to_string();
    let workspace_root =
        normalize_workspace_root(&input.workspace_path).map_err(error_to_string)?;

    let pty_system = native_pty_system();
    let pair = pty_system
        .openpty(terminal_size(
            input.cols,
            input.rows,
            input.pixel_width,
            input.pixel_height,
        ))
        .map_err(error_to_string)?;

    let mut command = CommandBuilder::new_default_prog();
    command.cwd(workspace_root.as_os_str());
    command.env("TERM", "xterm-256color");
    command.env("COLORTERM", "truecolor");
    command.env("SINEW_WORKSPACE", workspace_root.as_os_str());

    let child = pair.slave.spawn_command(command).map_err(error_to_string)?;
    drop(pair.slave);

    let reader = pair.master.try_clone_reader().map_err(error_to_string)?;
    let writer = Arc::new(StdMutex::new(
        pair.master.take_writer().map_err(error_to_string)?,
    ));
    let killer = Arc::new(StdMutex::new(child.clone_killer()));

    if let Some(previous) = state.terminal_sessions.lock().await.remove(&session_id) {
        terminate_terminal_process(previous);
    }

    state.terminal_sessions.lock().await.insert(
        session_id.clone(),
        TerminalProcess {
            token: token.clone(),
            master: pair.master,
            writer,
            killer,
        },
    );

    spawn_terminal_reader(app.clone(), session_id.clone(), token.clone(), reader);
    spawn_terminal_waiter(
        app,
        state.terminal_sessions.clone(),
        session_id.clone(),
        token,
        child,
    );

    Ok(TerminalSpawnOutput { session_id })
}

#[tauri::command]
async fn write_terminal(
    state: State<'_, DesktopState>,
    input: TerminalWriteInput,
) -> std::result::Result<(), String> {
    let writer = {
        let sessions = state.terminal_sessions.lock().await;
        let Some(process) = sessions.get(&input.session_id) else {
            return Ok(());
        };
        if process.token != input.token {
            return Ok(());
        }
        process.writer.clone()
    };

    let mut writer = writer
        .lock()
        .map_err(|_| "terminal writer unavailable".to_string())?;
    writer
        .write_all(input.data.as_bytes())
        .map_err(error_to_string)?;
    writer.flush().map_err(error_to_string)?;
    Ok(())
}

#[tauri::command]
async fn resize_terminal(
    state: State<'_, DesktopState>,
    input: TerminalResizeInput,
) -> std::result::Result<(), String> {
    let sessions = state.terminal_sessions.lock().await;
    let Some(process) = sessions.get(&input.session_id) else {
        return Ok(());
    };
    if process.token != input.token {
        return Ok(());
    }
    process
        .master
        .resize(terminal_size(
            input.cols,
            input.rows,
            input.pixel_width,
            input.pixel_height,
        ))
        .map_err(error_to_string)
}

#[tauri::command]
async fn kill_terminal(
    state: State<'_, DesktopState>,
    input: TerminalControlInput,
) -> std::result::Result<bool, String> {
    let process = {
        let mut sessions = state.terminal_sessions.lock().await;
        match sessions.get(&input.session_id) {
            Some(process) if process.token == input.token => sessions.remove(&input.session_id),
            _ => None,
        }
    };

    if let Some(process) = process {
        terminate_terminal_process(process);
        Ok(true)
    } else {
        Ok(false)
    }
}

fn validate_terminal_value<'a>(
    value: &'a str,
    label: &str,
) -> std::result::Result<&'a str, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(format!("{label} cannot be empty"));
    }
    if trimmed.len() > 256 {
        return Err(format!("{label} is too long"));
    }
    Ok(trimmed)
}

fn terminal_size(cols: u16, rows: u16, pixel_width: u16, pixel_height: u16) -> PtySize {
    PtySize {
        rows: rows.clamp(4, 200),
        cols: cols.clamp(20, 500),
        pixel_width,
        pixel_height,
    }
}

fn spawn_terminal_reader(
    app: AppHandle,
    session_id: String,
    token: String,
    mut reader: Box<dyn Read + Send>,
) {
    std::thread::spawn(move || {
        let mut buffer = [0u8; 8192];
        let mut pending = Vec::<u8>::new();
        loop {
            match reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => {
                    pending.extend_from_slice(&buffer[..n]);
                    emit_terminal_utf8_chunks(&app, &session_id, &token, &mut pending);
                }
                Err(_) => break,
            }
        }

        if !pending.is_empty() {
            emit_terminal_data(
                &app,
                &session_id,
                &token,
                String::from_utf8_lossy(&pending).to_string(),
            );
        }
    });
}

fn emit_terminal_utf8_chunks(
    app: &AppHandle,
    session_id: &str,
    token: &str,
    pending: &mut Vec<u8>,
) {
    loop {
        match std::str::from_utf8(pending) {
            Ok(valid) => {
                if !valid.is_empty() {
                    emit_terminal_data(app, session_id, token, valid.to_string());
                }
                pending.clear();
                break;
            }
            Err(err) => {
                let valid_up_to = err.valid_up_to();
                if valid_up_to > 0 {
                    let data = String::from_utf8_lossy(&pending[..valid_up_to]).to_string();
                    emit_terminal_data(app, session_id, token, data);
                    pending.drain(..valid_up_to);
                    continue;
                }

                if let Some(error_len) = err.error_len() {
                    let data = String::from_utf8_lossy(&pending[..error_len]).to_string();
                    emit_terminal_data(app, session_id, token, data);
                    pending.drain(..error_len);
                    continue;
                }

                break;
            }
        }
    }
}

fn emit_terminal_data(app: &AppHandle, session_id: &str, token: &str, data: String) {
    if data.is_empty() {
        return;
    }

    let _ = app.emit(
        TERMINAL_DATA_EVENT_NAME,
        TerminalDataEvent {
            session_id: session_id.to_string(),
            token: token.to_string(),
            data,
        },
    );
}

fn spawn_terminal_waiter(
    app: AppHandle,
    terminal_sessions: Arc<Mutex<HashMap<String, TerminalProcess>>>,
    session_id: String,
    token: String,
    mut child: Box<dyn Child + Send + Sync>,
) {
    std::thread::spawn(move || {
        let status = child.wait();
        let (exit_code, signal) = match status {
            Ok(status) => (
                Some(status.exit_code()),
                status.signal().map(std::string::ToString::to_string),
            ),
            Err(err) => (None, Some(err.to_string())),
        };

        tauri::async_runtime::spawn(async move {
            let mut sessions = terminal_sessions.lock().await;
            let is_current = sessions
                .get(&session_id)
                .map(|process| process.token == token)
                .unwrap_or(false);
            if !is_current {
                return;
            }

            sessions.remove(&session_id);
            let _ = app.emit(
                TERMINAL_EXIT_EVENT_NAME,
                TerminalExitEvent {
                    session_id,
                    token,
                    exit_code,
                    signal,
                },
            );
        });
    });
}

fn terminate_terminal_process(process: TerminalProcess) {
    if let Ok(mut killer) = process.killer.lock() {
        let _ = killer.kill();
    }
}

fn emit_agent_event(
    app: &AppHandle,
    workspace_id: &str,
    conversation_id: &str,
    event: &AgentEvent,
) -> Result<()> {
    app.emit(
        AGENT_EVENT_NAME,
        ConversationEvent {
            workspace_id: workspace_id.to_string(),
            conversation_id: conversation_id.to_string(),
            event: event.clone(),
        },
    )
    .context("unable to emit agent event")?;
    Ok(())
}

fn emit_agent_file_changes(app: &AppHandle, workspace_id: &str, event: &AgentEvent) {
    match event {
        AgentEvent::ToolFinished { file_changes, .. } => {
            for change in file_changes {
                let _ = app.emit(
                    FILE_CHANGE_EVENT_NAME,
                    WorkspaceFileChangeEvent {
                        workspace_path: workspace_id.to_string(),
                        relative_path: change.relative_path.clone(),
                    },
                );
            }
        }
        AgentEvent::SubAgentEvent { event, .. } => {
            emit_agent_file_changes(app, workspace_id, event);
        }
        _ => {}
    }
}

fn build_user_message(
    text: &str,
    attachments: &[AttachmentInput],
    workspace_root: &Path,
    plan_control: Option<PlanControlInput>,
    message_visibility: MessageVisibilityInput,
) -> ChatMessage {
    let mut parts = Vec::new();
    let mut context_blocks = Vec::new();
    let mut context_attachments = Vec::new();

    for attachment in attachments.iter().take(8) {
        let path = resolve_attachment_path(workspace_root, &attachment.path);
        let label = attachment_label(attachment, &path);
        let attachment_meta = json!({
            "path": path.display().to_string(),
            "name": label.clone(),
        });
        match prepare_attachment(&path, &label) {
            PreparedAttachment::Image(mut image) => {
                if let Part::Image { meta, .. } = &mut image {
                    *meta = Some(json!({ "attachment": attachment_meta }));
                }
                parts.push(image);
            }
            PreparedAttachment::Context(block) => {
                context_blocks.push(block);
                context_attachments.push(attachment_meta);
            }
        }
    }

    parts.push(Part::Text {
        text: text.to_string(),
        meta: match message_visibility {
            MessageVisibilityInput::Normal => None,
            MessageVisibilityInput::SystemReminder => Some(json!({ "system_reminder": true })),
        },
    });

    if matches!(plan_control, Some(PlanControlInput::StopQuestions)) {
        parts.push(Part::Text {
            text: "\n\n<plan_mode_control action=\"stop_questions\">\nThe user clicked Send and stop questions. Do not ask more questions in this turn. Produce the complete Markdown plan now and do not implement it.\n</plan_mode_control>".to_string(),
            meta: Some(json!({ "plan_control": "stop_questions" })),
        });
    }

    if !context_blocks.is_empty() {
        parts.push(Part::Text {
            text: format!(
                "\n\nAttached file context:\n\n{}",
                context_blocks.join("\n\n")
            ),
            meta: Some(json!({
                "attachment_context": true,
                "attachments": context_attachments,
            })),
        });
    }

    ChatMessage {
        role: Role::User,
        parts,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlanReference {
    path: String,
    title: Option<String>,
}

fn plan_implementation_turn_reminder(
    workspace_root: &Path,
    workflow: &PlanWorkflowState,
    attachments: &[AttachmentInput],
    control: Option<PlanControlInput>,
) -> std::result::Result<Option<String>, String> {
    if !matches!(control, Some(PlanControlInput::ImplementPlan)) {
        return Ok(None);
    }

    let plan = match workflow {
        PlanWorkflowState::PlanReady { artifact } => {
            Some(plan_reference_from_artifact(workspace_root, artifact))
        }
        _ => plan_reference_from_attachments(workspace_root, attachments),
    }
    .ok_or_else(|| "plan implementation requires an attached plan".to_string())?;

    let mut lines = vec![
        "You are implementing this plan for the current turn.".to_string(),
        format!("Plan path: {}", plan.path),
    ];
    if let Some(title) = plan.title.filter(|title| !title.trim().is_empty()) {
        lines.push(format!("Plan title: {}", title.trim()));
    }
    lines.extend([
        "Treat the plan as the source of truth for this implementation run.".to_string(),
        "Use the ToDoList tool to track implementation progress when the plan has multiple steps, and keep it updated until the plan is complete.".to_string(),
        "Read the plan file when you need details, keep changes aligned with it, and complete the implementation before your final response.".to_string(),
    ]);

    Ok(Some(lines.join("\n")))
}

fn with_turn_plan_reminder(base: &str, reminder: Option<String>) -> String {
    let Some(reminder) = reminder else {
        return base.to_string();
    };
    format!("{base}\n\n<plan_implementation_turn>\n{reminder}\n</plan_implementation_turn>")
}

fn plan_reference_from_artifact(
    workspace_root: &Path,
    artifact: &PlanArtifactState,
) -> PlanReference {
    let path = if !artifact.path.trim().is_empty() {
        artifact.path.clone()
    } else {
        artifact
            .absolute_path
            .as_deref()
            .map(|path| plan_display_path(workspace_root, path))
            .unwrap_or_else(|| "attached plan".to_string())
    };
    PlanReference {
        path,
        title: artifact.title.clone(),
    }
}

fn plan_reference_from_attachments(
    workspace_root: &Path,
    attachments: &[AttachmentInput],
) -> Option<PlanReference> {
    let attachment = attachments
        .iter()
        .find(|attachment| attachment_looks_like_plan(attachment))
        .or_else(|| attachments.first())?;
    Some(PlanReference {
        path: plan_display_path(workspace_root, &attachment.path),
        title: attachment.name.clone(),
    })
}

fn attachment_looks_like_plan(attachment: &AttachmentInput) -> bool {
    let path = attachment.path.to_ascii_lowercase();
    let name = attachment
        .name
        .as_deref()
        .unwrap_or_default()
        .to_ascii_lowercase();
    path.ends_with(".md")
        || path.contains(".sinew/plans/")
        || name.ends_with(".md")
        || name.contains("plan")
}

fn plan_display_path(workspace_root: &Path, raw: &str) -> String {
    let resolved = resolve_attachment_path(workspace_root, raw);
    resolved
        .strip_prefix(workspace_root)
        .ok()
        .filter(|relative| !relative.as_os_str().is_empty())
        .map(|relative| relative.display().to_string())
        .unwrap_or_else(|| {
            if raw.trim().is_empty() {
                resolved.display().to_string()
            } else {
                raw.to_string()
            }
        })
}

fn attach_latest_plan_artifact(
    workspace_root: &Path,
    conversation_id: &str,
    history: &mut Vec<ChatMessage>,
    turn_user_history_index: usize,
) -> Result<Option<PlanArtifactState>> {
    if turn_has_question_tool(history, turn_user_history_index) {
        return Ok(None);
    }

    let Some(assistant_index) = latest_assistant_index_after(history, turn_user_history_index)
    else {
        return Ok(None);
    };
    let plan_text = assistant_plan_text(&history[assistant_index]);
    if plan_text.trim().is_empty() {
        return Ok(None);
    }

    let relative_path = latest_plan_artifact_path(history)
        .filter(|path| is_safe_plan_path(path))
        .unwrap_or_else(|| new_plan_relative_path(conversation_id, &plan_text));
    let plan_path = workspace_root.join(&relative_path);
    if let Some(parent) = plan_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("unable to create plan directory {}", parent.display()))?;
    }

    let plan_text = ensure_trailing_newline(plan_text.trim());
    fs::write(&plan_path, &plan_text)
        .with_context(|| format!("unable to write plan {}", plan_path.display()))?;

    mark_plan_source(&mut history[assistant_index]);

    let title = plan_title(&plan_text).unwrap_or_else(|| "Plan created".to_string());
    let updated_at_ms = now_ms();
    let artifact = PlanArtifactState {
        path: relative_path,
        absolute_path: Some(plan_path.display().to_string()),
        title: Some(title),
        updated_at_ms: Some(updated_at_ms),
    };
    history[assistant_index].parts.push(Part::Text {
        text: String::new(),
        meta: Some(json!({
            "plan_artifact": {
                "path": artifact.path.clone(),
                "absolutePath": artifact.absolute_path.clone(),
                "title": artifact.title.clone(),
                "updatedAtMs": artifact.updated_at_ms,
            }
        })),
    });

    Ok(Some(artifact))
}

fn turn_has_question_tool(history: &[ChatMessage], turn_user_history_index: usize) -> bool {
    history
        .iter()
        .skip(turn_user_history_index.saturating_add(1))
        .flat_map(|message| &message.parts)
        .any(|part| {
            matches!(
                part,
                Part::ToolCall { name, .. } if name == "Question"
            )
        })
}

fn latest_assistant_index_after(
    history: &[ChatMessage],
    turn_user_history_index: usize,
) -> Option<usize> {
    let start = turn_user_history_index.saturating_add(1);
    (start..history.len())
        .rev()
        .find(|index| matches!(history[*index].role, Role::Assistant))
}

fn assistant_plan_text(message: &ChatMessage) -> String {
    message
        .parts
        .iter()
        .filter_map(|part| match part {
            Part::Text { text, .. } if !text.trim().is_empty() => Some(text.trim()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn mark_plan_source(message: &mut ChatMessage) {
    for part in &mut message.parts {
        let Part::Text { text, meta } = part else {
            continue;
        };
        if text.trim().is_empty() {
            continue;
        }
        insert_meta(meta, "plan_source", Value::Bool(true));
    }
}

fn latest_plan_artifact_path(history: &[ChatMessage]) -> Option<String> {
    for message in history.iter().rev() {
        for part in message.parts.iter().rev() {
            let Some(path) = part_meta(part)
                .and_then(|meta| meta.get("plan_artifact"))
                .and_then(|artifact| artifact.get("path"))
                .and_then(Value::as_str)
            else {
                continue;
            };
            return Some(path.to_string());
        }
    }
    None
}

fn is_safe_plan_path(path: &str) -> bool {
    if !path.starts_with(".sinew/plans/") || !path.ends_with(".md") {
        return false;
    }
    Path::new(path)
        .components()
        .all(|component| matches!(component, Component::Normal(_)))
}

fn new_plan_relative_path(conversation_id: &str, plan_text: &str) -> String {
    let title = plan_title(plan_text).unwrap_or_else(|| "plan".to_string());
    let slug = slugify(&title);
    let short_id = conversation_id.chars().take(8).collect::<String>();
    format!(".sinew/plans/{}-{}-{}.md", now_ms(), short_id, slug)
}

fn plan_title(plan_text: &str) -> Option<String> {
    plan_text.lines().find_map(|line| {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return None;
        }
        let title = trimmed.trim_start_matches('#').trim();
        (!title.is_empty()).then(|| {
            if title.chars().count() > 80 {
                let mut shortened = title.chars().take(77).collect::<String>();
                shortened.push_str("...");
                shortened
            } else {
                title.to_string()
            }
        })
    })
}

fn slugify(value: &str) -> String {
    let mut slug = String::new();
    let mut last_dash = false;
    for ch in value.chars() {
        let lower = ch.to_ascii_lowercase();
        if lower.is_ascii_alphanumeric() {
            slug.push(lower);
            last_dash = false;
        } else if !last_dash && !slug.is_empty() {
            slug.push('-');
            last_dash = true;
        }
        if slug.len() >= 48 {
            break;
        }
    }
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() {
        "plan".to_string()
    } else {
        slug
    }
}

fn ensure_trailing_newline(mut value: &str) -> String {
    value = value.trim_end();
    let mut output = value.to_string();
    output.push('\n');
    output
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

fn insert_meta(meta: &mut Option<Value>, key: &str, value: Value) {
    let mut map = match meta.take() {
        Some(Value::Object(map)) => map,
        Some(previous) => {
            let mut map = serde_json::Map::new();
            map.insert("previous_meta".into(), previous);
            map
        }
        None => serde_json::Map::new(),
    };
    map.insert(key.to_string(), value);
    *meta = Some(Value::Object(map));
}

fn tool_descriptors_for_workspace(
    workspace_root: &Path,
    mode: AgentMode,
    skill_settings: &SkillSettings,
) -> Vec<ToolDescriptor> {
    let bash = BashTool::new(workspace_root);
    let mut tools = vec![
        bash.descriptor(),
        bash.input_descriptor(),
        GlobTool::new(workspace_root).descriptor(),
        GrepTool::new(workspace_root).descriptor(),
        ReadTool::new(workspace_root).descriptor(),
        clean_context_descriptor(),
        ToDoListTool::new().descriptor(),
        QuestionTool::new().descriptor(),
        WebSearchTool::new().descriptor(),
        WebFetchTool::new().descriptor(),
    ];
    if let Some(descriptor) =
        SkillTool::with_settings(workspace_root, skill_settings.clone()).descriptor()
    {
        tools.push(descriptor);
    }
    if mode != AgentMode::Plan {
        tools.insert(4, ApplyPatchTool::new(workspace_root).descriptor());
        tools.push(CreateImageTool::new(workspace_root).descriptor());
    }
    tools
}

fn configurable_tool_catalog(workspace_root: &Path) -> Vec<ToolDescriptor> {
    let mut tools =
        tool_descriptors_for_workspace(workspace_root, AgentMode::Act, &SkillSettings::default());
    tools.retain(|tool| tool.name != "skill");
    tools.extend(TeamTool::descriptors_static());
    tools.extend(TeamTool::agent_descriptors_static());
    tools
}

fn system_prompt_for_workspace(workspace_root: &Path, base: &str) -> Result<String> {
    let mut sections = Vec::new();

    if let Some(instructions) =
        read_workspace_prompt_file(workspace_root, WORKSPACE_INSTRUCTIONS_FILE)?
    {
        sections.push(format!(
            "# Workspace instructions\n\nThe following instructions come from the current workspace and should be treated as the project source of truth.\n\n{instructions}"
        ));
    }

    if let Some(design) = read_workspace_prompt_file(workspace_root, WORKSPACE_DESIGN_FILE)? {
        sections.push(format!(
            "# Workspace design context\n\nThe following design guidance comes from the current workspace and should guide product, UX, visual, and frontend decisions.\n\n{design}"
        ));
    }

    if sections.is_empty() {
        return Ok(base.to_string());
    }

    Ok(format!("{base}\n\n{}", sections.join("\n\n")))
}

fn read_workspace_prompt_file(workspace_root: &Path, file_name: &str) -> Result<Option<String>> {
    let path = workspace_root.join(file_name);
    let contents = match fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(None);
        }
        Err(err) => {
            return Err(err).with_context(|| {
                format!("unable to read workspace prompt file at {}", path.display())
            });
        }
    };

    let contents = contents.trim();
    if contents.is_empty() {
        return Ok(None);
    }

    Ok(Some(contents.to_string()))
}

impl ContextTokenUsage {
    fn from_input_tokens(input_tokens: u32) -> Self {
        Self {
            input_tokens,
            total_tokens: input_tokens,
            ..Self::default()
        }
    }

    fn from_stream_usage(usage: &Value) -> Self {
        let input_tokens = usage_u32(usage, "input_tokens").unwrap_or(0);
        let output_tokens = usage_u32(usage, "output_tokens").unwrap_or(0);
        let reasoning_tokens = usage_u32(usage, "reasoning_tokens").unwrap_or(0);
        let cache_read_tokens = usage_u32(usage, "cache_read_tokens").unwrap_or(0);
        let cache_creation_tokens = usage_u32(usage, "cache_creation_tokens").unwrap_or(0);
        let total_tokens = usage_u32(usage, "total_tokens").unwrap_or(0);
        Self {
            input_tokens,
            output_tokens,
            reasoning_tokens,
            cache_read_tokens,
            cache_creation_tokens,
            total_tokens,
        }
    }

    fn total(self) -> u32 {
        if self.total_tokens > 0 {
            self.total_tokens
        } else {
            self.input_tokens
                .saturating_add(self.output_tokens)
                .saturating_add(self.reasoning_tokens)
                .saturating_add(self.cache_read_tokens)
                .saturating_add(self.cache_creation_tokens)
        }
    }
}

fn latest_stream_context_usage(
    history: &[ChatMessage],
    provider: &str,
    model: &str,
) -> Option<ContextTokenUsage> {
    for message in history.iter().rev() {
        if !matches!(message.role, Role::Assistant) {
            continue;
        }

        for part in message.parts.iter().rev() {
            if let Some(usage) = token_usage_meta(part) {
                if usage.get("source").and_then(Value::as_str) != Some("stream") {
                    continue;
                }
                if usage.get("provider").and_then(Value::as_str) != Some(provider) {
                    continue;
                }
                if usage.get("model").and_then(Value::as_str) != Some(model) {
                    continue;
                }
                let usage = ContextTokenUsage::from_stream_usage(usage);
                if usage.total() > 0 {
                    return Some(usage);
                }
            }
        }
    }

    None
}

fn context_usage_breakdown(usage: ContextTokenUsage) -> Option<Vec<ContextBreakdownItem>> {
    let mut items = Vec::new();
    push_context_usage_breakdown(&mut items, "input", "Input", usage.input_tokens);
    push_context_usage_breakdown(&mut items, "output", "Output", usage.output_tokens);
    push_context_usage_breakdown(&mut items, "reasoning", "Reasoning", usage.reasoning_tokens);
    push_context_usage_breakdown(&mut items, "cache", "Cache read", usage.cache_read_tokens);
    push_context_usage_breakdown(
        &mut items,
        "cache_write",
        "Cache write",
        usage.cache_creation_tokens,
    );
    (!items.is_empty()).then_some(items)
}

fn push_context_usage_breakdown(
    items: &mut Vec<ContextBreakdownItem>,
    key: &'static str,
    label: &'static str,
    tokens: u32,
) {
    if tokens > 0 {
        items.push(ContextBreakdownItem {
            key: key.to_string(),
            label: label.to_string(),
            tokens,
        });
    }
}

fn token_usage_meta(part: &Part) -> Option<&Value> {
    part_meta(part)?.get("token_usage")
}

fn usage_u32(usage: &Value, key: &str) -> Option<u32> {
    usage
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
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

fn tool_name_set(tools: &[ToolDescriptor]) -> HashSet<String> {
    tools.iter().map(|tool| tool.name.clone()).collect()
}

fn workspace_rules_weight(workspace_root: &Path) -> Result<u64> {
    let mut weight = 0;

    if let Some(instructions) =
        read_workspace_prompt_file(workspace_root, WORKSPACE_INSTRUCTIONS_FILE)?
    {
        weight += context_text_weight(&format!(
            "# Workspace instructions\n\nThe following instructions come from the current workspace and should be treated as the project source of truth.\n\n{instructions}"
        ));
    }

    if let Some(design) = read_workspace_prompt_file(workspace_root, WORKSPACE_DESIGN_FILE)? {
        weight += context_text_weight(&format!(
            "# Workspace design context\n\nThe following design guidance comes from the current workspace and should guide product, UX, visual, and frontend decisions.\n\n{design}"
        ));
    }

    Ok(weight)
}

fn context_breakdown_weights(
    system_prompt: &str,
    workspace_rules_weight: u64,
    history: &[ChatMessage],
    tools: &[ToolDescriptor],
    mcp_tool_names: &HashSet<String>,
    sub_agent_tool_names: &HashSet<String>,
) -> Vec<ContextBreakdownWeight> {
    let mut weights = Vec::new();
    let system_weight = context_text_weight(system_prompt).saturating_sub(workspace_rules_weight);

    push_context_weight(&mut weights, "system", "System prompt", system_weight);
    push_context_weight(&mut weights, "rules", "Rules", workspace_rules_weight);

    let mut base_tools_weight = 0;
    let mut skills_weight = 0;
    let mut mcp_weight = 0;
    let mut sub_agents_weight = 0;

    for tool in tools {
        let weight = tool_descriptor_weight(tool);
        if mcp_tool_names.contains(&tool.name) {
            mcp_weight += weight;
        } else if sub_agent_tool_names.contains(&tool.name) {
            sub_agents_weight += weight;
        } else if tool.name == "skill" {
            skills_weight += weight;
        } else {
            base_tools_weight += weight;
        }
    }

    push_context_weight(&mut weights, "tools", "Tools", base_tools_weight);
    push_context_weight(&mut weights, "skills", "Skills", skills_weight);
    push_context_weight(&mut weights, "mcp", "MCP", mcp_weight);
    push_context_weight(&mut weights, "subagents", "Subagents", sub_agents_weight);
    push_context_weight(
        &mut weights,
        "conversation",
        "Conversation",
        history_weight(history),
    );

    weights
}

fn push_context_weight(
    weights: &mut Vec<ContextBreakdownWeight>,
    key: &'static str,
    label: &'static str,
    weight: u64,
) {
    if weight > 0 {
        weights.push(ContextBreakdownWeight { key, label, weight });
    }
}

fn tool_descriptor_weight(tool: &ToolDescriptor) -> u64 {
    let schema = serde_json::to_string(&tool.input_schema).unwrap_or_default();
    96 + context_text_weight(&tool.name)
        + context_text_weight(&tool.description)
        + context_text_weight(&schema)
}

fn history_weight(history: &[ChatMessage]) -> u64 {
    history.iter().map(message_weight).sum()
}

fn message_weight(message: &ChatMessage) -> u64 {
    48 + message.parts.iter().map(part_weight).sum::<u64>()
}

fn part_weight(part: &Part) -> u64 {
    match part {
        Part::Text { text, .. } | Part::Thinking { text, .. } => 24 + context_text_weight(text),
        Part::Image {
            media_type, data, ..
        } => 96 + context_text_weight(media_type) + image_weight(data),
        Part::ToolCall {
            id, name, input, ..
        } => {
            let input = serde_json::to_string(input).unwrap_or_default();
            80 + context_text_weight(id) + context_text_weight(name) + context_text_weight(&input)
        }
        Part::ToolResult {
            tool_call_id,
            content,
            images,
            ..
        } => {
            80 + context_text_weight(tool_call_id)
                + context_text_weight(content)
                + images
                    .iter()
                    .map(|image| {
                        80 + context_text_weight(&image.media_type)
                            + image
                                .path
                                .as_deref()
                                .map(context_text_weight)
                                .unwrap_or_default()
                            + image_weight(&image.data)
                    })
                    .sum::<u64>()
        }
    }
}

fn context_text_weight(value: &str) -> u64 {
    value.chars().count() as u64
}

fn image_weight(data: &str) -> u64 {
    1_200 + ((data.len() as u64) / 2_048).min(3_200)
}

fn local_context_token_estimate(weights: &[ContextBreakdownWeight]) -> u32 {
    let total_weight = weights
        .iter()
        .fold(0_u64, |sum, item| sum.saturating_add(item.weight));
    let estimate = total_weight.saturating_add(3) / 4;
    estimate.max(1).min(u32::MAX as u64) as u32
}

fn scale_context_breakdown(
    used_tokens: u32,
    weights: Vec<ContextBreakdownWeight>,
) -> Vec<ContextBreakdownItem> {
    let weights: Vec<_> = weights.into_iter().filter(|item| item.weight > 0).collect();
    if used_tokens == 0 || weights.is_empty() {
        return Vec::new();
    }

    let total_weight: u64 = weights.iter().map(|item| item.weight).sum();
    if total_weight == 0 {
        return Vec::new();
    }

    let mut scaled = weights
        .into_iter()
        .map(|item| {
            let raw = (item.weight as f64 / total_weight as f64) * used_tokens as f64;
            let tokens = raw.floor() as u32;
            let remainder = raw - tokens as f64;
            (item, tokens, remainder)
        })
        .collect::<Vec<_>>();

    let assigned = scaled
        .iter()
        .fold(0_u32, |sum, (_, tokens, _)| sum.saturating_add(*tokens));
    let mut remaining = used_tokens.saturating_sub(assigned);
    let mut order = (0..scaled.len()).collect::<Vec<_>>();
    order.sort_by(|a, b| {
        scaled[*b]
            .2
            .partial_cmp(&scaled[*a].2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for index in order {
        if remaining == 0 {
            break;
        }
        scaled[index].1 = scaled[index].1.saturating_add(1);
        remaining -= 1;
    }

    scaled
        .into_iter()
        .filter(|(_, tokens, _)| *tokens > 0)
        .map(|(item, tokens, _)| ContextBreakdownItem {
            key: item.key.to_string(),
            label: item.label.to_string(),
            tokens,
        })
        .collect()
}

fn resolve_attachment_path(workspace_root: &Path, raw: &str) -> std::path::PathBuf {
    let path = Path::new(raw);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace_root.join(path)
    }
}

fn attachment_label(attachment: &AttachmentInput, path: &Path) -> String {
    attachment
        .name
        .clone()
        .or_else(|| {
            path.file_name()
                .and_then(|value| value.to_str())
                .map(|value| value.to_string())
        })
        .unwrap_or_else(|| attachment.path.clone())
}

enum PreparedAttachment {
    Image(Part),
    Context(String),
}

fn prepare_attachment(path: &Path, label: &str) -> PreparedAttachment {
    let Some(media_type) = supported_image_media_type(path) else {
        return PreparedAttachment::Context(read_attachment_block(path, label));
    };

    let intro = format!("<attachment path=\"{}\">", path.display());
    match fs::read(path) {
        Ok(bytes) => {
            if bytes.len() > MAX_IMAGE_BYTES {
                return PreparedAttachment::Context(format!(
                    "{intro}\n[Image too large to send visually: {label}]\n</attachment>"
                ));
            }

            PreparedAttachment::Image(Part::Image {
                media_type: media_type.to_string(),
                data: BASE64_STANDARD.encode(bytes),
                meta: None,
            })
        }
        Err(err) => PreparedAttachment::Context(format!(
            "{intro}\n[Unable to read image {label}: {err}]\n</attachment>"
        )),
    }
}

fn supported_image_media_type(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

fn clipboard_image_type(
    media_type: &str,
    name: Option<&str>,
) -> Option<(&'static str, &'static str)> {
    let normalized = media_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    match normalized.as_str() {
        "image/png" => return Some(("image/png", "png")),
        "image/jpeg" | "image/jpg" => return Some(("image/jpeg", "jpg")),
        "image/gif" => return Some(("image/gif", "gif")),
        "image/webp" => return Some(("image/webp", "webp")),
        _ => {}
    }

    let ext = Path::new(name?).extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "png" => Some(("image/png", "png")),
        "jpg" | "jpeg" => Some(("image/jpeg", "jpg")),
        "gif" => Some(("image/gif", "gif")),
        "webp" => Some(("image/webp", "webp")),
        _ => None,
    }
}

fn clipboard_image_display_name(name: Option<&str>, extension: &str) -> String {
    let raw = name
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("pasted-image");
    let stem = Path::new(raw)
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("pasted-image");
    format!("{}.{}", safe_temp_file_stem(stem), extension)
}

fn safe_temp_file_stem(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if out.len() >= 72 {
            break;
        }
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else if matches!(ch, '-' | '_') {
            out.push(ch);
        } else if ch.is_whitespace() && !out.ends_with('-') {
            out.push('-');
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        "pasted-image".to_string()
    } else {
        out
    }
}

fn read_attachment_block(path: &Path, label: &str) -> String {
    let intro = format!("<attachment path=\"{}\">", path.display());

    match fs::read(path) {
        Ok(bytes) => {
            if bytes.contains(&0) || std::str::from_utf8(&bytes).is_err() {
                return format!("{intro}\n[Binary file attached: {label}]\n</attachment>");
            }

            let truncated = bytes.len() > MAX_ATTACHMENT_BYTES;
            let slice = &bytes[..bytes.len().min(MAX_ATTACHMENT_BYTES)];
            let mut content = String::from_utf8_lossy(slice).into_owned();
            if truncated {
                content.push_str("\n\n[truncated]");
            }

            format!("{intro}\n{content}\n</attachment>")
        }
        Err(err) => format!("{intro}\n[Unable to read {label}: {err}]\n</attachment>"),
    }
}

fn error_to_string(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn create_new_window(app: &AppHandle) -> Result<()> {
    let label = next_window_label(app);
    let mut builder =
        WebviewWindowBuilder::new(app, label, WebviewUrl::App(PathBuf::from(NEW_WINDOW_URL)))
            .title("Sinew")
            .inner_size(1500.0, 940.0)
            .min_inner_size(1100.0, 720.0)
            .resizable(true)
            .center();

    #[cfg(target_os = "macos")]
    {
        builder = builder
            .title_bar_style(tauri::TitleBarStyle::Overlay)
            .hidden_title(true)
            .traffic_light_position(tauri::LogicalPosition::new(14.0, 18.0));
    }

    let window = builder.build().context("unable to create new window")?;
    let _ = window.set_focus();
    Ok(())
}

fn create_new_window_detached(app: &AppHandle) {
    let app = app.clone();
    std::thread::spawn(move || {
        if let Err(err) = create_new_window(&app) {
            tracing::warn!(%err, "unable to create new window");
        }
    });
}

fn focus_existing_window(app: &AppHandle) -> bool {
    let mut windows = app.webview_windows();
    let window = windows
        .remove("main")
        .or_else(|| windows.into_values().next());

    if let Some(window) = window {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
        return true;
    }

    false
}

fn next_window_label(app: &AppHandle) -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();

    for index in 0..1000 {
        let label = format!("{NEW_WINDOW_LABEL_PREFIX}-{millis}-{index}");
        if app.get_webview_window(&label).is_none() {
            return label;
        }
    }

    format!("{NEW_WINDOW_LABEL_PREFIX}-{millis}-fallback")
}

#[cfg(target_os = "macos")]
fn install_macos_dock_menu(app: &AppHandle) {
    let _ = MACOS_APP_HANDLE.set(app.clone());

    let Some(mtm) = MainThreadMarker::new() else {
        tracing::warn!("unable to install macOS dock menu outside main thread");
        return;
    };

    let ns_app = NSApplication::sharedApplication(mtm);
    let delegate: *mut AnyObject = unsafe { objc2::msg_send![&*ns_app, delegate] };
    if delegate.is_null() {
        tracing::warn!("unable to install macOS dock menu without app delegate");
        return;
    }

    let delegate_class = unsafe { &*delegate }.class() as *const AnyClass as *mut AnyClass;
    unsafe {
        let dock_menu_imp = std::mem::transmute::<
            unsafe extern "C-unwind" fn(&AnyObject, Sel, *mut AnyObject) -> *mut AnyObject,
            Imp,
        >(macos_application_dock_menu);
        let new_window_imp = std::mem::transmute::<
            unsafe extern "C-unwind" fn(&AnyObject, Sel, *mut AnyObject),
            Imp,
        >(macos_new_window_from_dock);

        let _ = class_addMethod(
            delegate_class,
            objc2::sel!(applicationDockMenu:),
            dock_menu_imp,
            b"@@:@\0".as_ptr().cast(),
        );
        let _ = class_addMethod(
            delegate_class,
            objc2::sel!(sinewNewWindowFromDock:),
            new_window_imp,
            b"v@:@\0".as_ptr().cast(),
        );
    }
}

#[cfg(target_os = "macos")]
unsafe extern "C-unwind" fn macos_application_dock_menu(
    target: &AnyObject,
    _cmd: Sel,
    _sender: *mut AnyObject,
) -> *mut AnyObject {
    let Some(mtm) = MainThreadMarker::new() else {
        return std::ptr::null_mut();
    };

    let menu_title = NSString::from_str("Sinew");
    let item_title = NSString::from_str("Nouvelle fenêtre");
    let empty_key = NSString::new();
    let menu = NSMenu::initWithTitle(mtm.alloc(), &menu_title);
    let item = unsafe {
        NSMenuItem::initWithTitle_action_keyEquivalent(
            mtm.alloc(),
            &item_title,
            Some(objc2::sel!(sinewNewWindowFromDock:)),
            &empty_key,
        )
    };

    unsafe {
        item.setTarget(Some(target));
    }
    menu.addItem(&item);

    Retained::autorelease_return(menu) as *mut AnyObject
}

#[cfg(target_os = "macos")]
unsafe extern "C-unwind" fn macos_new_window_from_dock(
    _target: &AnyObject,
    _cmd: Sel,
    _sender: *mut AnyObject,
) {
    if let Some(app) = MACOS_APP_HANDLE.get() {
        create_new_window_detached(app);
    }
}

fn restore_workspace_for_rewrite(
    app: &AppHandle,
    store: &AppStore,
    workspace_root: &Path,
    conversation_id: &str,
    history_index: usize,
) -> Result<()> {
    let checkpoint_records = store
        .load_turn_checkpoints_from(conversation_id, history_index)
        .context("unable to load turn checkpoints")?;
    let checkpoints = checkpoint_records
        .into_iter()
        .map(|record| record.checkpoint)
        .collect::<Vec<_>>();
    let restored_paths = restore_turn_checkpoints(workspace_root, &checkpoints)
        .context("unable to restore workspace checkpoint")?;
    store
        .delete_turn_checkpoints_from(conversation_id, history_index)
        .context("unable to delete old turn checkpoints")?;
    for relative_path in restored_paths {
        emit_workspace_file_change(app, workspace_root, &relative_path);
    }
    Ok(())
}

fn emit_workspace_file_change(app: &AppHandle, workspace_root: &Path, relative_path: &str) {
    let _ = app.emit(
        FILE_CHANGE_EVENT_NAME,
        WorkspaceFileChangeEvent {
            workspace_path: workspace_root.display().to_string(),
            relative_path: relative_path.to_string(),
        },
    );
}

fn reveal_path(path: &Path) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("path does not exist");
    }

    #[cfg(target_os = "macos")]
    {
        let status = Command::new("open")
            .arg("-R")
            .arg(path)
            .status()
            .context("unable to reveal item in Finder")?;
        if !status.success() {
            anyhow::bail!("Finder reveal failed");
        }
        return Ok(());
    }

    #[cfg(target_os = "windows")]
    {
        let status = Command::new("explorer")
            .arg(format!("/select,{}", path.display()))
            .status()
            .context("unable to reveal item in Explorer")?;
        if !status.success() {
            anyhow::bail!("Explorer reveal failed");
        }
        return Ok(());
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let target = if path.is_dir() {
            path
        } else {
            path.parent().unwrap_or(path)
        };
        let status = Command::new("xdg-open")
            .arg(target)
            .status()
            .context("unable to open file manager")?;
        if !status.success() {
            anyhow::bail!("file manager open failed");
        }
        return Ok(());
    }

    #[allow(unreachable_code)]
    Ok(())
}

fn delete_installed_skill(workspace_root: &Path, skill_md: &Path) -> Result<PathBuf> {
    let skill_md = fs::canonicalize(skill_md).context("skill file does not exist")?;
    if skill_md.file_name().and_then(|name| name.to_str()) != Some("SKILL.md") {
        anyhow::bail!("can only delete a SKILL.md file");
    }

    let folder = skill_md
        .parent()
        .ok_or_else(|| anyhow::anyhow!("skill has no parent folder"))?
        .to_path_buf();
    let allowed_roots = skill_roots(workspace_root)
        .into_iter()
        .filter_map(|root| fs::canonicalize(root).ok())
        .collect::<Vec<_>>();
    let allowed = allowed_roots
        .iter()
        .any(|root| folder.parent() == Some(root.as_path()));
    if !allowed {
        anyhow::bail!("skill is outside the configured skill folders");
    }

    fs::remove_dir_all(&folder)
        .with_context(|| format!("unable to delete skill folder {}", folder.display()))?;
    Ok(folder)
}

fn skill_roots(workspace_root: &Path) -> Vec<PathBuf> {
    let mut roots = vec![
        workspace_root.join(".agents/skills"),
        workspace_root.join(".sinew/skills"),
    ];
    if let Some(home) = home_dir() {
        roots.push(home.join(".agents/skills"));
        roots.push(home.join(".sinew/skills"));
    }
    roots
}

fn home_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("USERPROFILE").map(PathBuf::from)
    }

    #[cfg(not(target_os = "windows"))]
    {
        std::env::var_os("HOME").map(PathBuf::from)
    }
}

fn open_external_url(raw_url: &str) -> Result<()> {
    let url = raw_url.trim();
    if !is_safe_external_url(url) {
        anyhow::bail!("only http and https links can be opened");
    }

    #[cfg(target_os = "macos")]
    {
        let status = Command::new("open")
            .arg(url)
            .status()
            .context("unable to open link")?;
        if !status.success() {
            anyhow::bail!("link open failed");
        }
        return Ok(());
    }

    #[cfg(target_os = "windows")]
    {
        let status = Command::new("cmd")
            .args(["/C", "start", "", url])
            .status()
            .context("unable to open link")?;
        if !status.success() {
            anyhow::bail!("link open failed");
        }
        return Ok(());
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let status = Command::new("xdg-open")
            .arg(url)
            .status()
            .context("unable to open link")?;
        if !status.success() {
            anyhow::bail!("link open failed");
        }
        return Ok(());
    }

    #[allow(unreachable_code)]
    Ok(())
}

fn is_safe_external_url(url: &str) -> bool {
    if url.len() > 4096 || url.chars().any(char::is_control) {
        return false;
    }
    let lower = url.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

fn is_workspace_file_event(kind: &EventKind) -> bool {
    matches!(
        kind,
        EventKind::Any | EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
    )
}

fn should_ignore_workspace_event_path(root: &Path, path: &Path) -> bool {
    const IGNORED: &[&str] = &[
        ".git",
        "node_modules",
        "target",
        "dist",
        "build",
        ".next",
        ".turbo",
        ".cache",
        ".idea",
        "__pycache__",
        ".pytest_cache",
        ".venv",
        "venv",
        ".mypy_cache",
        "out",
    ];

    match path.strip_prefix(root) {
        Ok(relative) => relative.components().any(|component| {
            matches!(
                component,
                Component::Normal(value)
                    if IGNORED
                        .iter()
                        .any(|ignored| value.to_string_lossy() == *ignored)
            )
        }),
        Err(_) => false,
    }
}

fn event_relative_path(root: &Path, path: &Path) -> Option<String> {
    path.strip_prefix(root).ok().map(|relative| {
        relative
            .components()
            .filter_map(|component| match component {
                Component::Normal(value) => Some(value.to_string_lossy().into_owned()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("/")
    })
}

fn read_clipboard_file_paths() -> Result<Vec<String>> {
    let mut paths = read_platform_clipboard_file_paths().unwrap_or_default();
    if paths.is_empty() {
        paths = read_clipboard_text_paths().unwrap_or_default();
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}

#[cfg(target_os = "macos")]
fn read_platform_clipboard_file_paths() -> Result<Vec<String>> {
    let script = r#"
use framework "AppKit"
use scripting additions
set pasteboard to current application's NSPasteboard's generalPasteboard()
set urls to pasteboard's readObjectsForClasses:{current application's NSURL} options:(missing value)
if urls is missing value then return ""
set output to {}
repeat with itemUrl in urls
    set itemPath to (itemUrl's |path|()) as text
    if itemPath is not "" then set end of output to itemPath
end repeat
set AppleScript's text item delimiters to linefeed
return output as text
"#;
    let output = Command::new("osascript")
        .arg("-e")
        .arg(script)
        .output()
        .context("unable to read macOS clipboard")?;
    if !output.status.success() {
        return Ok(Vec::new());
    }
    Ok(parse_clipboard_paths(&String::from_utf8_lossy(
        &output.stdout,
    )))
}

#[cfg(not(target_os = "macos"))]
fn read_platform_clipboard_file_paths() -> Result<Vec<String>> {
    Ok(Vec::new())
}

fn read_clipboard_text_paths() -> Result<Vec<String>> {
    let output = clipboard_text_command()
        .and_then(|mut command| command.output().ok())
        .filter(|output| output.status.success());
    let Some(output) = output else {
        return Ok(Vec::new());
    };
    Ok(parse_clipboard_paths(&String::from_utf8_lossy(
        &output.stdout,
    )))
}

fn clipboard_text_command() -> Option<Command> {
    #[cfg(target_os = "macos")]
    {
        let command = Command::new("pbpaste");
        return Some(command);
    }
    #[cfg(target_os = "windows")]
    {
        let mut command = Command::new("powershell");
        command.args(["-NoProfile", "-Command", "Get-Clipboard"]);
        return Some(command);
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let mut command = Command::new("sh");
        command.args([
            "-c",
            "wl-paste 2>/dev/null || xclip -selection clipboard -o 2>/dev/null || xsel -b -o 2>/dev/null",
        ]);
        return Some(command);
    }
    #[allow(unreachable_code)]
    None
}

fn parse_clipboard_paths(raw: &str) -> Vec<String> {
    raw.lines()
        .filter_map(|line| {
            let value = line.trim().trim_matches('"').trim_matches('\'');
            let value = value.strip_prefix("file://").unwrap_or(value);
            if value.is_empty() {
                return None;
            }
            let path = PathBuf::from(percent_decode_path(value));
            if path.exists() {
                Some(path.display().to_string())
            } else {
                None
            }
        })
        .collect()
}

fn percent_decode_path(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut idx = 0;
    while idx < bytes.len() {
        if bytes[idx] == b'%' && idx + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex_value(bytes[idx + 1]), hex_value(bytes[idx + 2])) {
                decoded.push(hi * 16 + lo);
                idx += 3;
                continue;
            }
        }
        decoded.push(bytes[idx]);
        idx += 1;
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_plan_ready() -> PlanWorkflowState {
        PlanWorkflowState::PlanReady {
            artifact: PlanArtifactState {
                path: ".sinew/plans/test.md".into(),
                absolute_path: Some("/workspace/.sinew/plans/test.md".into()),
                title: Some("Test plan".into()),
                updated_at_ms: Some(1),
            },
        }
    }

    #[test]
    fn plan_policy_starts_question_loop_from_idle_plan_mode() {
        let policy = plan_turn_policy(&PlanWorkflowState::Idle, AgentMode::Plan, None).unwrap();

        assert_eq!(policy.mode, AgentMode::Plan);
        assert!(!policy.stop_questions);
        assert!(!policy.attach_plan);
        assert_eq!(policy.next_workflow, PlanWorkflowState::PlanningQuestions);
    }

    #[test]
    fn plan_policy_forces_plan_mode_while_questions_are_active() {
        let policy =
            plan_turn_policy(&PlanWorkflowState::PlanningQuestions, AgentMode::Act, None).unwrap();

        assert_eq!(policy.mode, AgentMode::Plan);
        assert_eq!(policy.next_workflow, PlanWorkflowState::PlanningQuestions);
    }

    #[test]
    fn plan_policy_only_attaches_plan_after_stop_questions() {
        let policy = plan_turn_policy(
            &PlanWorkflowState::PlanningQuestions,
            AgentMode::Plan,
            Some(PlanControlInput::StopQuestions),
        )
        .unwrap();

        assert_eq!(policy.mode, AgentMode::Plan);
        assert!(policy.stop_questions);
        assert!(policy.attach_plan);
        assert_eq!(policy.next_workflow, PlanWorkflowState::PlanningQuestions);
    }

    #[test]
    fn plan_policy_rejects_implementation_before_plan_exists() {
        let err = plan_turn_policy(
            &PlanWorkflowState::PlanningQuestions,
            AgentMode::Act,
            Some(PlanControlInput::ImplementPlan),
        )
        .unwrap_err();

        assert!(err.contains("create the plan"));
    }

    #[test]
    fn plan_policy_allows_card_actions_after_manual_exit() {
        let update_policy = plan_turn_policy(
            &PlanWorkflowState::Idle,
            AgentMode::Act,
            Some(PlanControlInput::UpdatePlan),
        )
        .unwrap();
        assert_eq!(update_policy.mode, AgentMode::Plan);
        assert_eq!(
            update_policy.next_workflow,
            PlanWorkflowState::PlanningQuestions
        );

        let implement_policy = plan_turn_policy(
            &PlanWorkflowState::Idle,
            AgentMode::Act,
            Some(PlanControlInput::ImplementPlan),
        )
        .unwrap();
        assert_eq!(implement_policy.mode, AgentMode::Act);
        assert_eq!(implement_policy.next_workflow, PlanWorkflowState::Idle);
    }

    #[test]
    fn plan_policy_rejects_act_mode_when_plan_is_ready_without_user_action() {
        let err = plan_turn_policy(&sample_plan_ready(), AgentMode::Act, None).unwrap_err();

        assert!(err.contains("choose update plan or implement plan"));
    }

    #[test]
    fn plan_policy_allows_implementation_after_plan_is_ready() {
        let policy = plan_turn_policy(
            &sample_plan_ready(),
            AgentMode::Act,
            Some(PlanControlInput::ImplementPlan),
        )
        .unwrap();

        assert_eq!(policy.mode, AgentMode::Act);
        assert_eq!(policy.next_workflow, PlanWorkflowState::Idle);
        assert!(!policy.attach_plan);
    }

    #[test]
    fn plan_policy_returns_to_question_loop_when_updating_ready_plan() {
        let policy = plan_turn_policy(
            &sample_plan_ready(),
            AgentMode::Plan,
            Some(PlanControlInput::UpdatePlan),
        )
        .unwrap();

        assert_eq!(policy.mode, AgentMode::Plan);
        assert_eq!(policy.next_workflow, PlanWorkflowState::PlanningQuestions);
    }

    #[test]
    fn context_estimate_stays_in_plan_mode_for_active_workflows() {
        assert_eq!(
            plan_estimate_mode(&PlanWorkflowState::PlanningQuestions, AgentMode::Act),
            AgentMode::Plan
        );
        assert_eq!(
            plan_estimate_mode(&sample_plan_ready(), AgentMode::Act),
            AgentMode::Plan
        );
    }

    #[test]
    fn plan_implementation_reminder_uses_ready_plan_artifact() {
        let reminder = plan_implementation_turn_reminder(
            Path::new("/workspace"),
            &sample_plan_ready(),
            &[],
            Some(PlanControlInput::ImplementPlan),
        )
        .unwrap()
        .unwrap();

        assert!(reminder.contains("Plan path: .sinew/plans/test.md"));
        assert!(reminder.contains("Plan title: Test plan"));
        assert!(reminder.contains("current turn"));
        assert!(reminder.contains("Use the ToDoList tool"));
    }

    #[test]
    fn plan_implementation_reminder_uses_attached_plan_after_context_clear() {
        let attachments = vec![AttachmentInput {
            path: "/workspace/.sinew/plans/fresh.md".into(),
            name: Some("fresh.md".into()),
        }];
        let reminder = plan_implementation_turn_reminder(
            Path::new("/workspace"),
            &PlanWorkflowState::Idle,
            &attachments,
            Some(PlanControlInput::ImplementPlan),
        )
        .unwrap()
        .unwrap();

        assert!(reminder.contains("Plan path: .sinew/plans/fresh.md"));
    }

    #[test]
    fn plan_implementation_reminder_is_scoped_to_implement_control() {
        let reminder = plan_implementation_turn_reminder(
            Path::new("/workspace"),
            &sample_plan_ready(),
            &[],
            Some(PlanControlInput::UpdatePlan),
        )
        .unwrap();

        assert!(reminder.is_none());
    }

    #[test]
    fn swarm_completion_event_extracts_structured_responses() {
        let event = AgentEvent::ToolFinished {
            id: "team-run".to_string(),
            output: "Agent Swarm finished".to_string(),
            is_error: false,
            file_changes: Vec::new(),
            images: Vec::new(),
            meta: Some(json!({
                "teamRunStatus": "completed",
                "team": { "name": "team-demo" },
                "agentFinalResponses": [
                    {
                        "agent": "builder",
                        "status": "finished",
                        "lastResponse": "Built the feature."
                    },
                    {
                        "agent": "reviewer",
                        "status": "finished",
                        "lastResponse": "Reviewed the result."
                    }
                ]
            })),
        };

        let completion = agent_swarm_completion_from_event(&event)
            .expect("completed TeamRun event should trigger a swarm completion wake");

        assert_eq!(completion.team_name, "team-demo");
        assert_eq!(completion.responses.len(), 2);
        assert_eq!(completion.responses[0].agent, "builder");
        assert_eq!(completion.responses[0].last_response, "Built the feature.");
    }

    #[test]
    fn swarm_completion_wake_text_mentions_finished_and_agent_responses() {
        let completion = AgentSwarmCompletion {
            team_name: "team-demo".to_string(),
            responses: vec![AgentSwarmFinalResponse {
                agent: "builder".to_string(),
                status: "finished".to_string(),
                last_response: "Built the feature.".to_string(),
                last_error: None,
            }],
        };

        let wake_text = agent_swarm_completion_wake_text(&completion);

        assert!(wake_text.contains("<agent_swarm_finished>"));
        assert!(wake_text.contains("agent: @builder"));
        assert!(wake_text.contains("Built the feature."));
        assert!(wake_text.contains("Agent Swarm a terminé"));
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();
    if let Ok(provider) = AnthropicProvider::from_default_sources() {
        providers.insert("anthropic".into(), Arc::new(provider) as Arc<dyn Provider>);
    }
    if let Ok(provider) = OpenAiProvider::from_default_sources() {
        providers.insert("openai".into(), Arc::new(provider) as Arc<dyn Provider>);
    }
    if let Ok(provider) = GoogleProvider::from_default_sources() {
        providers.insert("google".into(), Arc::new(provider) as Arc<dyn Provider>);
    }
    if let Ok(provider) = KimiProvider::from_default_sources() {
        providers.insert("kimi".into(), Arc::new(provider) as Arc<dyn Provider>);
    }

    let default_model = if providers.contains_key("anthropic") {
        ModelRef::new("anthropic", ANTHROPIC_MODEL_ID).with_effort(Effort::Max)
    } else if providers.contains_key("openai") {
        ModelRef::new("openai", OPENAI_MODEL_ID).with_effort(Effort::Medium)
    } else if providers.contains_key("kimi") {
        ModelRef::new("kimi", KIMI_MODEL_ID).with_effort(Effort::High)
    } else {
        ModelRef::new("google", GOOGLE_MODEL_ID).with_effort(Effort::Medium)
    };

    let state = DesktopState {
        providers: Arc::new(StdMutex::new(providers)),
        store: AppStore::open_default().expect("unable to open app store"),
        default_model,
        system_prompt: DEFAULT_SYSTEM_PROMPT.into(),
        max_tool_rounds: 200,
        active_turns: Arc::new(Mutex::new(HashMap::new())),
        team_runtime: Arc::new(RwLock::new(TeamRuntime::default())),
        file_watchers: Arc::new(Mutex::new(HashMap::new())),
        terminal_sessions: Arc::new(Mutex::new(HashMap::new())),
        openai_login: Arc::new(Mutex::new(None)),
        anthropic_login: Arc::new(Mutex::new(None)),
        google_login: Arc::new(Mutex::new(None)),
        kimi_login: Arc::new(Mutex::new(None)),
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let handle = app.handle();
            #[cfg(target_os = "macos")]
            install_macos_dock_menu(handle);

            let menu = tauri::menu::Menu::default(handle)?;
            let new_window_item =
                tauri::menu::MenuItemBuilder::with_id(NEW_WINDOW_MENU_ID, "New Window")
                    .accelerator("CmdOrCtrl+Shift+N")
                    .build(handle)?;
            let file_menu = tauri::menu::SubmenuBuilder::new(handle, "File")
                .item(&new_window_item)
                .build()?;
            let terminal_menu = tauri::menu::SubmenuBuilder::new(handle, "Terminal")
                .text(TERMINAL_OPEN_MENU_ID, "Open Terminal")
                .build()?;
            menu.append(&file_menu)?;
            menu.append(&terminal_menu)?;
            app.set_menu(menu)?;
            Ok(())
        })
        .on_menu_event(|app, event| {
            if event.id() == NEW_WINDOW_MENU_ID {
                create_new_window_detached(app);
            } else if event.id() == TERMINAL_OPEN_MENU_ID {
                let focused = app
                    .webview_windows()
                    .into_values()
                    .find(|window| window.is_focused().unwrap_or(false));
                if let Some(window) = focused {
                    let _ = window.emit(TERMINAL_OPEN_EVENT_NAME, ());
                } else {
                    let _ = app.emit(TERMINAL_OPEN_EVENT_NAME, ());
                }
            }
        })
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            open_workspace,
            open_new_window,
            watch_workspace_command,
            unwatch_workspace_command,
            list_workspace_entries_command,
            list_workspace_files_command,
            search_workspace_files_command,
            read_workspace_file_command,
            write_workspace_file_command,
            create_workspace_file_command,
            create_workspace_directory_command,
            rename_workspace_entry_command,
            delete_workspace_entry_command,
            trash_workspace_entry_command,
            restore_workspace_deleted_entries_command,
            reveal_workspace_entry_command,
            reveal_absolute_path_command,
            resolve_terminal_path_command,
            read_external_file_command,
            delete_skill_command,
            open_external_url_command,
            copy_workspace_entries_command,
            import_workspace_paths_command,
            save_clipboard_image_attachment_command,
            read_clipboard_file_paths_command,
            list_conversations,
            create_conversation,
            load_conversation,
            rename_conversation,
            delete_conversation,
            set_conversation_mode,
            set_conversation_model_preference,
            list_mcp_settings,
            save_mcp_settings,
            list_tool_settings,
            save_tool_settings,
            list_sub_agent_settings,
            save_sub_agent_settings,
            list_configured_model_providers,
            get_openai_provider_status,
            start_openai_oauth_login,
            cancel_openai_oauth_login,
            disconnect_openai_provider,
            get_anthropic_provider_status,
            start_anthropic_oauth_login,
            cancel_anthropic_oauth_login,
            disconnect_anthropic_provider,
            get_google_provider_status,
            start_google_oauth_login,
            cancel_google_oauth_login,
            disconnect_google_provider,
            get_kimi_provider_status,
            start_kimi_oauth_login,
            cancel_kimi_oauth_login,
            disconnect_kimi_provider,
            probe_mcp_tools,
            list_installed_skills_command,
            save_skill_settings,
            send_message,
            compact_conversation,
            estimate_context,
            estimate_sub_agent_context,
            cancel_turn,
            stop_agent_swarm_command,
            run_terminal_command,
            spawn_terminal,
            write_terminal,
            resize_terminal,
            kill_terminal,
        ])
        .build(tauri::generate_context!())
        .expect("error while building sinew desktop")
        .run(|app, event| {
            #[cfg(target_os = "macos")]
            if let tauri::RunEvent::Reopen { .. } = event {
                if !focus_existing_window(app) {
                    create_new_window_detached(app);
                }
            }
        })
}
