//! Voice dictation (push-to-talk speech-to-text).
//!
//! Hold the Fn key anywhere on macOS: Sinew records the microphone, releases
//! transcribe the audio through the configured engine (dedicated OpenAI API
//! key, the Google subscription, or the OpenRouter API key) and either inserts
//! the text into the focused Sinew composer or pastes it into the active app
//! via a simulated Cmd+V. The transcript is always copied to the clipboard.

use std::{
    path::PathBuf,
    sync::{Arc, Mutex as StdMutex},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{
    image::Image,
    menu::{Menu, MenuEvent, MenuItemBuilder, PredefinedMenuItem},
    tray::{TrayIcon, TrayIconBuilder},
    AppHandle, Emitter, Manager, State, Wry,
};

pub(super) const DICTATION_STATE_EVENT_NAME: &str = "dictation-state";
pub(super) const DICTATION_TEXT_EVENT_NAME: &str = "dictation-text";

/// Recordings shorter than this are treated as accidental Fn taps.
const MIN_RECORDING_MS: u128 = 350;
/// Hard cap so a stuck Fn key cannot record forever (10 minutes).
const MAX_RECORDING_SECS: usize = 60 * 10;
const TARGET_SAMPLE_RATE: u32 = 16_000;

const TRANSCRIBE_PROMPT: &str = "You are a transcription engine. Transcribe the attached audio recording verbatim, in its original spoken language. Output ONLY the raw transcribed text, with correct punctuation, and no quotes, labels, markdown or commentary. If the audio contains no intelligible speech, output an empty string.";
const HISTORY_TTL_MS: i64 = 24 * 60 * 60 * 1_000;
const HISTORY_MENU_LIMIT: usize = 20;
const HISTORY_STORE_LIMIT: usize = 200;
const TRAY_ID: &str = "dictation-history";
const HISTORY_ITEM_PREFIX: &str = "dictation-history-copy-";
const HISTORY_REFRESH_ID: &str = "dictation-history-refresh";
const HISTORY_CLEAR_ID: &str = "dictation-history-clear";

// ---------------------------------------------------------------------------
// Dictation history (24h local fallback)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DictationHistoryEntry {
    id: String,
    text: String,
    created_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
struct DictationHistoryFile {
    entries: Vec<DictationHistoryEntry>,
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(i64::MAX as u128) as i64)
        .unwrap_or_default()
}

fn history_path() -> Result<PathBuf, String> {
    let dirs = ProjectDirs::from("dev", "hyrak", "sinew")
        .ok_or_else(|| "unable to resolve local data directory".to_string())?;
    Ok(dirs.data_local_dir().join("dictation-history.json"))
}

fn load_history() -> Vec<DictationHistoryEntry> {
    let Ok(path) = history_path() else {
        return Vec::new();
    };
    match std::fs::read(&path) {
        Ok(bytes) => serde_json::from_slice::<DictationHistoryFile>(&bytes)
            .map(|file| prune_history(file.entries))
            .unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

fn prune_history(mut entries: Vec<DictationHistoryEntry>) -> Vec<DictationHistoryEntry> {
    let cutoff = now_ms().saturating_sub(HISTORY_TTL_MS);
    entries.retain(|entry| entry.created_at_ms >= cutoff && !entry.text.trim().is_empty());
    entries.sort_by(|a, b| b.created_at_ms.cmp(&a.created_at_ms));
    entries.truncate(HISTORY_STORE_LIMIT);
    entries
}

fn write_history(entries: &[DictationHistoryEntry]) -> Result<(), String> {
    let path = history_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| format!("unable to create history directory: {err}"))?;
    }
    let payload = serde_json::to_vec_pretty(&DictationHistoryFile {
        entries: entries.to_vec(),
    })
    .map_err(|err| format!("unable to serialize dictation history: {err}"))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, payload).map_err(|err| format!("unable to write history: {err}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
    }
    std::fs::rename(&tmp, &path).map_err(|err| format!("unable to write history: {err}"))
}

fn history_item_id(id: &str) -> String {
    format!("{HISTORY_ITEM_PREFIX}{id}")
}

fn truncate_menu_text(text: &str) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out = String::new();
    for (idx, ch) in collapsed.chars().enumerate() {
        if idx >= 72 {
            out.push('…');
            return out;
        }
        out.push(ch);
    }
    if out.is_empty() {
        "(empty transcription)".into()
    } else {
        out
    }
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(super) struct DictationSettings {
    pub enabled: bool,
    /// "openai" | "google" | "openrouter" | "mistral"
    pub engine: String,
    pub openai_api_key: String,
    pub openai_model: String,
    pub mistral_api_key: String,
    pub mistral_model: String,
    pub google_model: String,
    pub openrouter_model: String,
    /// Optional ISO-639-1 language hint (e.g. "fr"). Empty = auto detect.
    pub language: String,
    pub sound_feedback: bool,
}

impl Default for DictationSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            engine: "openai".into(),
            openai_api_key: String::new(),
            openai_model: "gpt-4o-mini-transcribe".into(),
            mistral_api_key: String::new(),
            mistral_model: "voxtral-mini-latest".into(),
            google_model: "gemini-3-flash".into(),
            openrouter_model: "google/gemini-2.5-flash".into(),
            language: String::new(),
            sound_feedback: true,
        }
    }
}

impl DictationSettings {
    fn normalized(mut self) -> Self {
        self.openai_api_key = self.openai_api_key.trim().to_string();
        self.openai_model = self.openai_model.trim().to_string();
        self.mistral_api_key = self.mistral_api_key.trim().to_string();
        self.mistral_model = self.mistral_model.trim().to_string();
        self.google_model = self.google_model.trim().to_string();
        self.openrouter_model = self.openrouter_model.trim().to_string();
        self.language = self.language.trim().to_lowercase();
        let defaults = Self::default();
        if self.openai_model.is_empty() {
            self.openai_model = defaults.openai_model.clone();
        }
        if self.mistral_model.is_empty() {
            self.mistral_model = defaults.mistral_model.clone();
        }
        if self.google_model.is_empty() {
            self.google_model = defaults.google_model.clone();
        }
        if self.openrouter_model.is_empty() {
            self.openrouter_model = defaults.openrouter_model.clone();
        }
        if !matches!(
            self.engine.as_str(),
            "openai" | "google" | "openrouter" | "mistral"
        ) {
            self.engine = defaults.engine;
        }
        self
    }
}

fn settings_path() -> Result<PathBuf, String> {
    let dirs = ProjectDirs::from("dev", "hyrak", "sinew")
        .ok_or_else(|| "unable to resolve local data directory".to_string())?;
    Ok(dirs.data_local_dir().join("dictation.json"))
}

fn load_settings() -> DictationSettings {
    let Ok(path) = settings_path() else {
        return DictationSettings::default();
    };
    match std::fs::read(&path) {
        Ok(bytes) => serde_json::from_slice::<DictationSettings>(&bytes)
            .map(DictationSettings::normalized)
            .unwrap_or_default(),
        Err(_) => DictationSettings::default(),
    }
}

fn save_settings(settings: &DictationSettings) -> Result<(), String> {
    let path = settings_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| format!("unable to create settings directory: {err}"))?;
    }
    let payload = serde_json::to_vec_pretty(settings)
        .map_err(|err| format!("unable to serialize dictation settings: {err}"))?;
    // Atomic write with owner-only permissions: the file may hold an API key.
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &payload).map_err(|err| format!("unable to write settings: {err}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
    }
    std::fs::rename(&tmp, &path).map_err(|err| format!("unable to write settings: {err}"))
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(super) struct DictationState {
    settings: Arc<StdMutex<DictationSettings>>,
    history: Arc<StdMutex<Vec<DictationHistoryEntry>>>,
    tray: Arc<StdMutex<Option<TrayIcon>>>,
    #[cfg(target_os = "macos")]
    recorder: std::sync::mpsc::Sender<macos::RecorderCmd>,
    http: reqwest::Client,
}

impl DictationState {
    pub(super) fn new() -> Self {
        Self {
            settings: Arc::new(StdMutex::new(load_settings())),
            history: Arc::new(StdMutex::new(load_history())),
            tray: Arc::new(StdMutex::new(None)),
            #[cfg(target_os = "macos")]
            recorder: macos::spawn_recorder(),
            http: reqwest::Client::builder()
                .user_agent("sinew/0.1")
                .build()
                .unwrap_or_default(),
        }
    }

    fn settings_snapshot(&self) -> DictationSettings {
        self.settings
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }

    fn history_snapshot(&self) -> Vec<DictationHistoryEntry> {
        self.history
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }

    fn push_history(&self, text: String) -> Result<(), String> {
        let text = text.trim().to_string();
        if text.is_empty() {
            return Ok(());
        }
        let mut history = self
            .history
            .lock()
            .map_err(|_| "dictation history is unavailable".to_string())?;
        let created_at_ms = now_ms();
        let id = format!("{created_at_ms}-{}", history.len());
        history.insert(
            0,
            DictationHistoryEntry {
                id,
                text,
                created_at_ms,
            },
        );
        *history = prune_history(std::mem::take(&mut *history));
        write_history(&history)
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct DictationStatus {
    settings: DictationSettings,
    supported: bool,
    accessibility_trusted: bool,
    google_connected: bool,
    openrouter_connected: bool,
}

fn build_status(settings: DictationSettings) -> DictationStatus {
    let google_connected = sinew_google::load_default_auth_status()
        .map(|status| status.connected)
        .unwrap_or(false);
    let openrouter_connected = sinew_openrouter::load_default_api_key()
        .ok()
        .flatten()
        .is_some();
    DictationStatus {
        settings,
        supported: cfg!(target_os = "macos"),
        accessibility_trusted: accessibility_trusted(),
        google_connected,
        openrouter_connected,
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DictationStatePayload {
    state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DictationTextPayload {
    text: String,
}

fn emit_state(app: &AppHandle, state: &'static str, message: Option<String>) {
    let _ = app.emit(
        DICTATION_STATE_EVENT_NAME,
        DictationStatePayload { state, message },
    );
}

// ---------------------------------------------------------------------------
// Tauri commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub(super) fn get_dictation_status(
    state: State<'_, DictationState>,
) -> Result<DictationStatus, String> {
    Ok(build_status(state.settings_snapshot()))
}

#[tauri::command]
pub(super) fn save_dictation_settings(
    state: State<'_, DictationState>,
    input: DictationSettings,
) -> Result<DictationStatus, String> {
    let settings = input.normalized();
    {
        // Hold the lock across the file write so concurrent saves cannot leave
        // the persisted file and the in-memory settings diverging.
        let mut guard = state
            .settings
            .lock()
            .map_err(|_| "dictation settings are unavailable".to_string())?;
        save_settings(&settings)?;
        *guard = settings.clone();
    }
    Ok(build_status(settings))
}

/// Opens the macOS Accessibility privacy pane so the user can trust Sinew
/// (required for the global Fn key monitor and the synthetic Cmd+V paste).
#[tauri::command]
pub(super) fn open_dictation_permission_settings() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let mut child = std::process::Command::new("open")
            .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")
            .spawn()
            .map_err(|err| format!("unable to open System Settings: {err}"))?;
        // Reap the short-lived `open` process off-thread to avoid zombies.
        std::thread::spawn(move || {
            let _ = child.wait();
        });
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    Err("dictation is only supported on macOS".to_string())
}

// ---------------------------------------------------------------------------
// Push-to-talk flow
// ---------------------------------------------------------------------------
//
// All press/release events are funneled through a single coordinator thread
// (see `macos::spawn_ptt_coordinator`). This serializes the state machine:
// rapid Fn taps cannot interleave start/stop handshakes, so the recorder and
// the coordinator's `recording_since` can never diverge into a stuck hot mic.

#[cfg(target_os = "macos")]
fn run_ptt_loop(app: AppHandle, events: std::sync::mpsc::Receiver<macos::PttEvent>) {
    use std::time::Duration;

    let mut recording_since: Option<Instant> = None;
    while let Ok(event) = events.recv() {
        let state = app.state::<DictationState>();
        match event {
            macos::PttEvent::Pressed => {
                if recording_since.is_some() {
                    continue;
                }
                let settings = state.settings_snapshot();
                if !settings.enabled {
                    continue;
                }
                let (reply_tx, reply_rx) = std::sync::mpsc::channel();
                if state
                    .recorder
                    .send(macos::RecorderCmd::Start(reply_tx))
                    .is_err()
                {
                    emit_state(&app, "error", Some("audio recorder is unavailable".into()));
                    continue;
                }
                match reply_rx.recv_timeout(Duration::from_secs(6)) {
                    Ok(Ok(())) => {
                        recording_since = Some(Instant::now());
                        if settings.sound_feedback {
                            play_sound("Tink");
                        }
                        emit_state(&app, "recording", None);
                    }
                    Ok(Err(err)) => {
                        emit_state(&app, "error", Some(err));
                    }
                    Err(_) => {
                        // The recorder may still be stuck opening the stream
                        // (e.g. the first-run microphone permission prompt).
                        // Queue a fire-and-forget Stop so any session that
                        // eventually starts is immediately discarded, keeping
                        // the recorder in sync with our `recording_since`.
                        let (discard_tx, _discard_rx) = std::sync::mpsc::channel();
                        let _ = state.recorder.send(macos::RecorderCmd::Stop(discard_tx));
                        emit_state(
                            &app,
                            "error",
                            Some("microphone did not start (check microphone permission)".into()),
                        );
                    }
                }
            }
            macos::PttEvent::Released => {
                let Some(started_at) = recording_since.take() else {
                    continue;
                };
                let (reply_tx, reply_rx) = std::sync::mpsc::channel();
                if state
                    .recorder
                    .send(macos::RecorderCmd::Stop(reply_tx))
                    .is_err()
                {
                    emit_state(&app, "error", Some("audio recorder is unavailable".into()));
                    continue;
                }
                let wav = match reply_rx.recv_timeout(Duration::from_secs(6)) {
                    Ok(Ok(wav)) => wav,
                    Ok(Err(err)) => {
                        emit_state(&app, "error", Some(err));
                        continue;
                    }
                    Err(_) => {
                        emit_state(&app, "error", Some("audio recorder did not respond".into()));
                        continue;
                    }
                };

                if started_at.elapsed().as_millis() < MIN_RECORDING_MS {
                    emit_state(&app, "idle", None);
                    continue;
                }

                let settings = state.settings_snapshot();
                let http = state.http.clone();
                let task_app = app.clone();
                emit_state(&app, "transcribing", None);
                tauri::async_runtime::spawn(async move {
                    match transcribe(&http, &settings, wav).await {
                        Ok(text) if text.trim().is_empty() => {
                            emit_state(&task_app, "idle", None);
                        }
                        Ok(text) => {
                            let text = text.trim().to_string();
                            let state = task_app.state::<DictationState>();
                            if let Err(err) = state.push_history(text.clone()) {
                                tracing::warn!(error = %err, "failed to persist dictation history");
                            }
                            refresh_history_tray(&task_app);
                            deliver_text(&task_app, text);
                            if settings.sound_feedback {
                                play_sound("Pop");
                            }
                            emit_state(&task_app, "idle", None);
                        }
                        Err(err) => {
                            if settings.sound_feedback {
                                play_sound("Basso");
                            }
                            emit_state(&task_app, "error", Some(err));
                        }
                    }
                });
            }
        }
    }
}

/// Routes the transcript: clipboard always, then either the focused Sinew
/// window (composer insertion) or a synthetic Cmd+V into the active app.
#[cfg(target_os = "macos")]
fn deliver_text(app: &AppHandle, text: String) {
    let focused = app
        .webview_windows()
        .into_values()
        .find(|window| window.is_focused().unwrap_or(false));

    let clipboard_text = text.clone();
    let needs_paste = focused.is_none();
    let _ = app.run_on_main_thread(move || {
        macos::set_clipboard_text(&clipboard_text);
        if needs_paste {
            // Give the pasteboard a beat to settle before the synthetic Cmd+V.
            std::thread::spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(120));
                macos::send_cmd_v();
            });
        }
    });

    if let Some(window) = focused {
        // Target only the focused window: a plain `emit` would broadcast to
        // every webview and insert the transcript into all open windows.
        let _ = window.emit_to(
            window.label(),
            DICTATION_TEXT_EVENT_NAME,
            DictationTextPayload { text },
        );
    }
}

fn build_history_menu(
    app: &AppHandle,
    entries: &[DictationHistoryEntry],
) -> Result<Menu<Wry>, String> {
    let menu = Menu::new(app).map_err(|err| err.to_string())?;
    let header = MenuItemBuilder::new("History (24h)")
        .enabled(false)
        .build(app)
        .map_err(|err| err.to_string())?;
    menu.append(&header).map_err(|err| err.to_string())?;
    let separator = PredefinedMenuItem::separator(app).map_err(|err| err.to_string())?;
    menu.append(&separator).map_err(|err| err.to_string())?;

    if entries.is_empty() {
        let empty = MenuItemBuilder::new("No dictations yet")
            .enabled(false)
            .build(app)
            .map_err(|err| err.to_string())?;
        menu.append(&empty).map_err(|err| err.to_string())?;
    } else {
        for entry in entries.iter().take(HISTORY_MENU_LIMIT) {
            let item = MenuItemBuilder::with_id(
                history_item_id(&entry.id),
                truncate_menu_text(&entry.text),
            )
            .build(app)
            .map_err(|err| err.to_string())?;
            menu.append(&item).map_err(|err| err.to_string())?;
        }
    }

    let separator = PredefinedMenuItem::separator(app).map_err(|err| err.to_string())?;
    menu.append(&separator).map_err(|err| err.to_string())?;
    let refresh = MenuItemBuilder::with_id(HISTORY_REFRESH_ID, "Refresh History")
        .build(app)
        .map_err(|err| err.to_string())?;
    menu.append(&refresh).map_err(|err| err.to_string())?;
    let clear = MenuItemBuilder::with_id(HISTORY_CLEAR_ID, "Clear History")
        .enabled(!entries.is_empty())
        .build(app)
        .map_err(|err| err.to_string())?;
    menu.append(&clear).map_err(|err| err.to_string())?;
    Ok(menu)
}

fn refresh_history_tray(app: &AppHandle) {
    let state = app.state::<DictationState>();
    let entries = state.history_snapshot();
    let menu = match build_history_menu(app, &entries) {
        Ok(menu) => menu,
        Err(err) => {
            tracing::warn!(error = %err, "failed to build dictation tray menu");
            return;
        }
    };
    let tray = state.tray.lock().ok().and_then(|guard| guard.clone());
    if let Some(tray) = tray {
        if let Err(err) = tray.set_menu(Some(menu)) {
            tracing::warn!(error = %err, "failed to update dictation tray menu");
        }
    }
}

fn copy_history_entry(app: &AppHandle, id: &str) {
    let state = app.state::<DictationState>();
    let Some(entry) = state
        .history_snapshot()
        .into_iter()
        .find(|entry| entry.id == id)
    else {
        return;
    };
    let text = entry.text;
    let _ = app.run_on_main_thread(move || {
        #[cfg(target_os = "macos")]
        macos::set_clipboard_text(&text);
    });
}

fn clear_history(app: &AppHandle) {
    let state = app.state::<DictationState>();
    if let Ok(mut history) = state.history.lock() {
        history.clear();
        if let Err(err) = write_history(&history) {
            tracing::warn!(error = %err, "failed to clear dictation history");
        }
    }
    refresh_history_tray(app);
}

fn dictation_tray_icon() -> Result<Image<'static>, String> {
    Image::from_bytes(include_bytes!("../icons/dictation-history-template.png"))
        .map(|image| image.to_owned())
        .map_err(|err| format!("unable to load dictation tray icon: {err}"))
}

fn install_history_tray(app: &AppHandle) -> Result<(), String> {
    let state = app.state::<DictationState>();
    let entries = state.history_snapshot();
    let menu = build_history_menu(app, &entries)?;
    let mut builder = TrayIconBuilder::<Wry>::with_id(TRAY_ID)
        .menu(&menu)
        .tooltip("Sinew Dictation History")
        .show_menu_on_left_click(true)
        .on_menu_event(|app: &AppHandle<Wry>, event: MenuEvent| {
            let id = event.id().0.as_str();
            if id == HISTORY_REFRESH_ID {
                refresh_history_tray(app);
            } else if id == HISTORY_CLEAR_ID {
                clear_history(app);
            } else if let Some(entry_id) = id.strip_prefix(HISTORY_ITEM_PREFIX) {
                copy_history_entry(app, entry_id);
            }
        });

    builder = builder.icon(dictation_tray_icon()?).icon_as_template(true);

    let tray = builder.build(app).map_err(|err| err.to_string())?;
    let mut guard = state
        .tray
        .lock()
        .map_err(|_| "dictation tray is unavailable".to_string())?;
    *guard = Some(tray);
    Ok(())
}

#[cfg(target_os = "macos")]
fn play_sound(name: &str) {
    let name = name.to_string();
    // `status()` on a detached thread reaps the child and avoids zombies.
    std::thread::spawn(move || {
        let _ = std::process::Command::new("afplay")
            .arg("-v")
            .arg("0.35")
            .arg(format!("/System/Library/Sounds/{name}.aiff"))
            .status();
    });
}

fn accessibility_trusted() -> bool {
    #[cfg(target_os = "macos")]
    {
        macos::ax_is_process_trusted()
    }
    #[cfg(not(target_os = "macos"))]
    false
}

/// Called once from the Tauri setup hook (main thread).
pub(super) fn init(app: &AppHandle) {
    if let Err(err) = install_history_tray(app) {
        tracing::warn!(error = %err, "failed to install dictation history tray");
    }
    #[cfg(target_os = "macos")]
    {
        let events = macos::spawn_ptt_coordinator(app.clone());
        macos::install_fn_monitors(events);
    }
    #[cfg(not(target_os = "macos"))]
    let _ = app;
}

// ---------------------------------------------------------------------------
// Transcription engines
// ---------------------------------------------------------------------------

fn transcription_instructions(language: &str) -> String {
    if language.is_empty() {
        TRANSCRIBE_PROMPT.to_string()
    } else {
        format!("{TRANSCRIBE_PROMPT} The speaker most likely speaks `{language}`.")
    }
}

async fn transcribe(
    http: &reqwest::Client,
    settings: &DictationSettings,
    wav: Vec<u8>,
) -> Result<String, String> {
    match settings.engine.as_str() {
        "openai" => transcribe_openai(http, settings, wav).await,
        "mistral" => transcribe_mistral(http, settings, wav).await,
        "google" => transcribe_google(http, settings, wav).await,
        "openrouter" => transcribe_openrouter(http, settings, wav).await,
        other => Err(format!("unknown dictation engine `{other}`")),
    }
}

async fn transcribe_openai(
    http: &reqwest::Client,
    settings: &DictationSettings,
    wav: Vec<u8>,
) -> Result<String, String> {
    if settings.openai_api_key.is_empty() {
        return Err("add your OpenAI API key in Settings → Dictation".to_string());
    }
    let part = reqwest::multipart::Part::bytes(wav)
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .map_err(|err| format!("unable to build audio upload: {err}"))?;
    let mut form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("model", settings.openai_model.clone());
    if !settings.language.is_empty() {
        form = form.text("language", settings.language.clone());
    }

    let response = http
        .post("https://api.openai.com/v1/audio/transcriptions")
        .bearer_auth(&settings.openai_api_key)
        .multipart(form)
        .send()
        .await
        .map_err(|err| format!("transcription request failed: {err}"))?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format!(
            "OpenAI transcription failed ({status}): {}",
            api_error_message(&body)
        ));
    }
    let value: Value = serde_json::from_str(&body)
        .map_err(|err| format!("invalid transcription response: {err}"))?;
    value["text"]
        .as_str()
        .map(|text| text.to_string())
        .ok_or_else(|| "transcription response did not contain text".to_string())
}

async fn transcribe_mistral(
    http: &reqwest::Client,
    settings: &DictationSettings,
    wav: Vec<u8>,
) -> Result<String, String> {
    if settings.mistral_api_key.is_empty() {
        return Err("add your Mistral API key in Settings → Dictation".to_string());
    }
    let part = reqwest::multipart::Part::bytes(wav)
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .map_err(|err| format!("unable to build audio upload: {err}"))?;
    let mut form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("model", settings.mistral_model.clone());
    if !settings.language.is_empty() {
        form = form.text("language", settings.language.clone());
    }

    let response = http
        .post("https://api.mistral.ai/v1/audio/transcriptions")
        .bearer_auth(&settings.mistral_api_key)
        // Mistral's audio docs show x-api-key while the generated API reference
        // shows Bearer auth. Sending both is harmless for API-key auth and keeps
        // this isolated from Sinew's provider/auth system.
        .header("x-api-key", &settings.mistral_api_key)
        .multipart(form)
        .send()
        .await
        .map_err(|err| format!("transcription request failed: {err}"))?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format!(
            "Mistral transcription failed ({status}): {}",
            api_error_message(&body)
        ));
    }
    let value: Value = serde_json::from_str(&body)
        .map_err(|err| format!("invalid transcription response: {err}"))?;
    value["text"]
        .as_str()
        .map(|text| text.to_string())
        .ok_or_else(|| "transcription response did not contain text".to_string())
}

async fn transcribe_google(
    http: &reqwest::Client,
    settings: &DictationSettings,
    wav: Vec<u8>,
) -> Result<String, String> {
    let credential = sinew_google::auth::Credential::load_default()
        .map_err(|err| err.to_string())?
        .ok_or_else(|| "connect Google in Settings → Providers".to_string())?;
    let token = credential
        .bearer(http)
        .await
        .map_err(|err| err.to_string())?;
    let project = sinew_google::auth::load_default_user_data()
        .ok()
        .flatten()
        .map(|user| user.project_id);

    let body = json!({
        "model": settings.google_model,
        "project": project,
        "request": {
            "contents": [{
                "role": "user",
                "parts": [
                    { "text": transcription_instructions(&settings.language) },
                    { "inlineData": { "mimeType": "audio/wav", "data": B64.encode(&wav) } }
                ]
            }],
            "generationConfig": { "temperature": 0 }
        },
        "requestType": "agent",
        "userAgent": "antigravity",
        "requestId": format!("dictation-{}", sinew_google::generate_state()),
    });

    let bases = [
        "https://daily-cloudcode-pa.googleapis.com/v1internal",
        "https://cloudcode-pa.googleapis.com/v1internal",
    ];
    let mut last_error = "Google transcription failed".to_string();
    for base in bases {
        let response = match http
            .post(format!("{base}:streamGenerateContent"))
            .query(&[("alt", "sse")])
            .bearer_auth(&token)
            .header("user-agent", "antigravity/2.0.0 darwin/arm64")
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .json(&body)
            .send()
            .await
        {
            Ok(response) => response,
            Err(err) => {
                // Transport failure on one base must not skip the fallback.
                last_error = format!("transcription request failed: {err}");
                continue;
            }
        };
        let status = response.status();
        let payload = response.text().await.unwrap_or_default();
        if status.is_success() {
            let text = google_sse_text(&payload);
            if text.trim().is_empty() {
                return Err("Google returned an empty transcription".to_string());
            }
            return Ok(text);
        }
        last_error = format!(
            "Google transcription failed ({status}): {}",
            api_error_message(&payload)
        );
        if !matches!(
            status,
            reqwest::StatusCode::FORBIDDEN | reqwest::StatusCode::NOT_FOUND
        ) {
            break;
        }
    }
    Err(last_error)
}

/// Extracts the concatenated candidate text from a cloudcode SSE payload.
fn google_sse_text(payload: &str) -> String {
    let mut out = String::new();
    for line in payload.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(data.trim()) else {
            continue;
        };
        let response = if value.get("response").is_some() {
            &value["response"]
        } else {
            &value
        };
        let Some(candidates) = response["candidates"].as_array() else {
            continue;
        };
        for candidate in candidates {
            let Some(parts) = candidate["content"]["parts"].as_array() else {
                continue;
            };
            for part in parts {
                if part["thought"].as_bool() == Some(true) {
                    continue;
                }
                if let Some(text) = part["text"].as_str() {
                    out.push_str(text);
                }
            }
        }
    }
    out
}

async fn transcribe_openrouter(
    http: &reqwest::Client,
    settings: &DictationSettings,
    wav: Vec<u8>,
) -> Result<String, String> {
    let api_key = sinew_openrouter::load_default_api_key()
        .map_err(|err| err.to_string())?
        .ok_or_else(|| "connect OpenRouter in Settings → Providers".to_string())?;

    let body = json!({
        "model": settings.openrouter_model,
        "messages": [{
            "role": "user",
            "content": [
                { "type": "text", "text": transcription_instructions(&settings.language) },
                { "type": "input_audio", "input_audio": { "data": B64.encode(&wav), "format": "wav" } }
            ]
        }],
        "temperature": 0
    });

    let response = http
        .post("https://openrouter.ai/api/v1/chat/completions")
        .bearer_auth(&api_key)
        .header("HTTP-Referer", "https://github.com/hyrak/sinew")
        .header("X-OpenRouter-Title", "Sinew")
        .json(&body)
        .send()
        .await
        .map_err(|err| format!("transcription request failed: {err}"))?;
    let status = response.status();
    let payload = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format!(
            "OpenRouter transcription failed ({status}): {}",
            api_error_message(&payload)
        ));
    }
    let value: Value = serde_json::from_str(&payload)
        .map_err(|err| format!("invalid transcription response: {err}"))?;
    value["choices"][0]["message"]["content"]
        .as_str()
        .map(|text| text.to_string())
        .ok_or_else(|| "transcription response did not contain text".to_string())
}

fn api_error_message(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| {
            value["error"]["message"]
                .as_str()
                .or_else(|| value["message"].as_str())
                .map(|message| message.to_string())
        })
        .unwrap_or_else(|| {
            let trimmed = body.trim();
            if trimmed.is_empty() {
                "unknown error".to_string()
            } else {
                trimmed.chars().take(300).collect()
            }
        })
}

// ---------------------------------------------------------------------------
// macOS: microphone recorder, Fn key monitors, clipboard + synthetic paste
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod macos {
    use std::{
        ptr::NonNull,
        sync::{
            atomic::{AtomicBool, Ordering},
            mpsc, Arc, Mutex as StdMutex,
        },
        time::Instant,
    };

    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use objc2_app_kit::{
        NSEvent, NSEventMask, NSEventModifierFlags, NSPasteboard, NSPasteboardTypeString,
    };
    use objc2_foundation::NSString;
    use tauri::AppHandle;

    use super::{MAX_RECORDING_SECS, MIN_RECORDING_MS, TARGET_SAMPLE_RATE};

    const FN_KEY_CODE: u16 = 63;

    // -- Accessibility -----------------------------------------------------

    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn AXIsProcessTrusted() -> bool;
    }

    pub(super) fn ax_is_process_trusted() -> bool {
        unsafe { AXIsProcessTrusted() }
    }

    // -- Synthetic Cmd+V ----------------------------------------------------

    const KCG_HID_EVENT_TAP: u32 = 0;
    const KCG_EVENT_FLAG_MASK_COMMAND: u64 = 1 << 20;
    const KVK_ANSI_V: u16 = 9;

    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGEventCreateKeyboardEvent(
            source: *mut std::ffi::c_void,
            virtual_key: u16,
            key_down: bool,
        ) -> *mut std::ffi::c_void;
        fn CGEventSetFlags(event: *mut std::ffi::c_void, flags: u64);
        fn CGEventPost(tap: u32, event: *mut std::ffi::c_void);
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFRelease(cf: *const std::ffi::c_void);
    }

    pub(super) fn send_cmd_v() {
        unsafe {
            for key_down in [true, false] {
                let event = CGEventCreateKeyboardEvent(std::ptr::null_mut(), KVK_ANSI_V, key_down);
                if event.is_null() {
                    return;
                }
                CGEventSetFlags(event, KCG_EVENT_FLAG_MASK_COMMAND);
                CGEventPost(KCG_HID_EVENT_TAP, event);
                CFRelease(event);
            }
        }
    }

    pub(super) fn set_clipboard_text(text: &str) {
        unsafe {
            let pasteboard = NSPasteboard::generalPasteboard();
            pasteboard.clearContents();
            pasteboard.setString_forType(&NSString::from_str(text), NSPasteboardTypeString);
        }
    }

    // -- Push-to-talk coordinator --------------------------------------------

    pub(super) enum PttEvent {
        Pressed,
        Released,
    }

    /// Single thread that consumes Fn press/release events in order. This is
    /// the only sender on the recorder channel, so start/stop handshakes can
    /// never interleave or reorder (see `super::run_ptt_loop`).
    pub(super) fn spawn_ptt_coordinator(app: AppHandle) -> mpsc::Sender<PttEvent> {
        let (tx, rx) = mpsc::channel::<PttEvent>();
        std::thread::Builder::new()
            .name("sinew-dictation-ptt".into())
            .spawn(move || super::run_ptt_loop(app, rx))
            .expect("unable to spawn dictation coordinator thread");
        tx
    }

    // -- Fn key monitors ----------------------------------------------------

    /// Installs global + local NSEvent monitors for `flagsChanged` events.
    /// Must be called on the main thread. Monitors are intentionally leaked:
    /// they live for the whole app lifetime.
    pub(super) fn install_fn_monitors(events: mpsc::Sender<PttEvent>) {
        let was_down = Arc::new(AtomicBool::new(false));

        let handler = {
            let was_down = Arc::clone(&was_down);
            move |key_code: u16, flags: NSEventModifierFlags| {
                if key_code != FN_KEY_CODE {
                    return;
                }
                let down = flags.contains(NSEventModifierFlags::Function);
                if was_down.swap(down, Ordering::SeqCst) == down {
                    return;
                }
                // Monitors fire on the main thread: forward to the coordinator
                // channel (preserves ordering) instead of doing any work here.
                let _ = events.send(if down {
                    PttEvent::Pressed
                } else {
                    PttEvent::Released
                });
            }
        };

        let global_handler = handler.clone();
        let global_block = block2::RcBlock::new(move |event: NonNull<NSEvent>| {
            let event = unsafe { event.as_ref() };
            global_handler(event.keyCode(), event.modifierFlags());
        });
        let global_monitor = NSEvent::addGlobalMonitorForEventsMatchingMask_handler(
            NSEventMask::FlagsChanged,
            &global_block,
        );
        if let Some(monitor) = global_monitor {
            std::mem::forget(monitor);
        }
        std::mem::forget(global_block);

        let local_block = block2::RcBlock::new(move |event: NonNull<NSEvent>| -> *mut NSEvent {
            let event_ref = unsafe { event.as_ref() };
            handler(event_ref.keyCode(), event_ref.modifierFlags());
            event.as_ptr()
        });
        let local_monitor = unsafe {
            NSEvent::addLocalMonitorForEventsMatchingMask_handler(
                NSEventMask::FlagsChanged,
                &local_block,
            )
        };
        if let Some(monitor) = local_monitor {
            std::mem::forget(monitor);
        }
        std::mem::forget(local_block);
    }

    // -- Microphone recorder -------------------------------------------------

    pub(super) enum RecorderCmd {
        Start(mpsc::Sender<Result<(), String>>),
        Stop(mpsc::Sender<Result<Vec<u8>, String>>),
    }

    struct RecordingSession {
        _stream: cpal::Stream,
        buffer: Arc<StdMutex<Vec<f32>>>,
        sample_rate: u32,
        started: Instant,
    }

    /// The cpal stream is not `Send`, so a dedicated thread owns the whole
    /// recording session and is driven through a command channel.
    pub(super) fn spawn_recorder() -> mpsc::Sender<RecorderCmd> {
        let (tx, rx) = mpsc::channel::<RecorderCmd>();
        std::thread::Builder::new()
            .name("sinew-dictation-recorder".into())
            .spawn(move || {
                let mut session: Option<RecordingSession> = None;
                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        RecorderCmd::Start(reply) => {
                            if session.is_some() {
                                let _ = reply.send(Err("already recording".into()));
                                continue;
                            }
                            match start_session() {
                                Ok(next) => {
                                    session = Some(next);
                                    let _ = reply.send(Ok(()));
                                }
                                Err(err) => {
                                    let _ = reply.send(Err(err));
                                }
                            }
                        }
                        RecorderCmd::Stop(reply) => match session.take() {
                            Some(active) => {
                                let _ = reply.send(finish_session(active));
                            }
                            None => {
                                let _ = reply.send(Err("not recording".into()));
                            }
                        },
                    }
                }
            })
            .expect("unable to spawn dictation recorder thread");
        tx
    }

    fn start_session() -> Result<RecordingSession, String> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| "no microphone available".to_string())?;
        let config = device
            .default_input_config()
            .map_err(|err| format!("unable to read microphone config: {err}"))?;
        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        // Cap measured at the device rate (the buffer holds device-rate samples).
        let max_samples = sample_rate as usize * MAX_RECORDING_SECS;
        let buffer: Arc<StdMutex<Vec<f32>>> = Arc::new(StdMutex::new(Vec::new()));
        let sink = Arc::clone(&buffer);
        let err_fn = |err| tracing::warn!(error = %err, "dictation input stream error");

        let push_frames = move |samples: &[f32]| {
            let Ok(mut guard) = sink.lock() else {
                return;
            };
            if guard.len() >= max_samples {
                return;
            }
            if channels <= 1 {
                guard.extend_from_slice(samples);
            } else {
                for frame in samples.chunks_exact(channels) {
                    let sum: f32 = frame.iter().sum();
                    guard.push(sum / channels as f32);
                }
            }
        };

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config.into(),
                move |data: &[f32], _| push_frames(data),
                err_fn,
                None,
            ),
            cpal::SampleFormat::I16 => device.build_input_stream(
                &config.into(),
                move |data: &[i16], _| {
                    let converted: Vec<f32> = data.iter().map(|s| *s as f32 / 32_768.0).collect();
                    push_frames(&converted);
                },
                err_fn,
                None,
            ),
            cpal::SampleFormat::U16 => device.build_input_stream(
                &config.into(),
                move |data: &[u16], _| {
                    let converted: Vec<f32> = data
                        .iter()
                        .map(|s| (*s as f32 - 32_768.0) / 32_768.0)
                        .collect();
                    push_frames(&converted);
                },
                err_fn,
                None,
            ),
            other => {
                return Err(format!("unsupported microphone sample format: {other:?}"));
            }
        }
        .map_err(|err| format!("unable to open microphone: {err}"))?;
        stream
            .play()
            .map_err(|err| format!("unable to start microphone: {err}"))?;

        Ok(RecordingSession {
            _stream: stream,
            buffer,
            sample_rate,
            started: Instant::now(),
        })
    }

    fn finish_session(session: RecordingSession) -> Result<Vec<u8>, String> {
        let RecordingSession {
            _stream,
            buffer,
            sample_rate,
            started,
        } = session;
        drop(_stream);
        let samples = buffer
            .lock()
            .map(|guard| guard.clone())
            .map_err(|_| "audio buffer is unavailable".to_string())?;
        if samples.is_empty() && started.elapsed().as_millis() >= MIN_RECORDING_MS {
            return Err("no audio captured (check microphone permission)".to_string());
        }
        let mono16k = resample_linear(&samples, sample_rate, TARGET_SAMPLE_RATE);
        Ok(encode_wav_mono16(&mono16k, TARGET_SAMPLE_RATE))
    }

    fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
        if src_rate == dst_rate || samples.is_empty() {
            return samples.to_vec();
        }
        let ratio = src_rate as f64 / dst_rate as f64;
        let out_len = ((samples.len() as f64) / ratio).floor() as usize;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let pos = i as f64 * ratio;
            let idx = pos.floor() as usize;
            let frac = (pos - idx as f64) as f32;
            let a = samples[idx];
            let b = samples.get(idx + 1).copied().unwrap_or(a);
            out.push(a + (b - a) * frac);
        }
        out
    }

    fn encode_wav_mono16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
        let data_len = (samples.len() * 2) as u32;
        let byte_rate = sample_rate * 2;
        let mut wav = Vec::with_capacity(44 + data_len as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            let clamped = (sample.clamp(-1.0, 1.0) * 32_767.0) as i16;
            wav.extend_from_slice(&clamped.to_le_bytes());
        }
        wav
    }
}
