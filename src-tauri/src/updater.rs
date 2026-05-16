//! In-app updater integration around `tauri-plugin-updater`.
//!
//! Exposes three commands to the frontend:
//!   * `updater_check`          → check the configured endpoint for a newer release.
//!   * `updater_download_and_install` → stream the signed bundle, verify, install.
//!   * `updater_restart`        → relaunch the app after an install (frontend-driven).
//!
//! During download the backend emits two events to the focused webview:
//!   * `updater://progress` `{ downloaded: u64, total: Option<u64> }`
//!   * `updater://finished`
//!
//! The plugin is desktop-only (Windows, macOS, Linux). On mobile targets the
//! commands compile to a stub that returns "unsupported", which keeps the
//! `invoke_handler` registration uniform.

use std::sync::{Arc, Mutex};

use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

#[cfg(desktop)]
use tauri_plugin_updater::{Update, UpdaterExt};

/// Snapshot of an available update exposed to the frontend.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateInfo {
    pub available: bool,
    pub current_version: String,
    /// `None` when `available == false`.
    pub version: Option<String>,
    /// Release notes / body returned by the manifest, if any.
    pub notes: Option<String>,
    /// RFC-3339 publish date, if the manifest provided one.
    pub date: Option<String>,
    /// True for this fork/custom build. Official upstream updates should not be
    /// installed over it because they would replace the customized app bundle.
    pub custom_build: bool,
    /// True when installing the discovered update is intentionally blocked.
    pub update_protected: bool,
    /// Human-readable reason shown by the frontend when an update is blocked.
    pub protection_reason: Option<String>,
}

/// Stores the last `Update` returned by a successful `check()` so that
/// `updater_download_and_install` can reuse it without re-hitting the endpoint.
///
/// We keep this in Tauri-managed state (registered in `lib.rs`).
#[derive(Default)]
pub struct UpdaterState {
    #[cfg(desktop)]
    pending: Arc<Mutex<Option<Update>>>,
    #[cfg(not(desktop))]
    _phantom: std::marker::PhantomData<Arc<Mutex<()>>>,
}

impl UpdaterState {
    pub fn new() -> Self {
        Self::default()
    }
}

const PROGRESS_EVENT: &str = "updater://progress";
const FINISHED_EVENT: &str = "updater://finished";
const CUSTOM_BUILD: bool = true;
const CUSTOM_BUILD_PROTECTION_REASON: &str = "This is a custom Sinew build. Installing the official upstream update would replace your custom changes. Merge the new release into your fork and rebuild from your custom source instead.";

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProgressPayload {
    downloaded: u64,
    total: Option<u64>,
}

/// Query the configured update endpoint. Returns `available = false` when the
/// remote manifest reports the current version is already the latest.
#[tauri::command]
pub async fn updater_check(
    app: AppHandle,
    state: State<'_, UpdaterState>,
) -> Result<UpdateInfo, String> {
    let current_version = app.package_info().version.to_string();

    #[cfg(desktop)]
    {
        let pending = state.pending.clone();
        let updater = app
            .updater_builder()
            .build()
            .map_err(|err| format!("failed to build updater: {err}"))?;

        match updater.check().await {
            Ok(Some(update)) => {
                let info = UpdateInfo {
                    available: true,
                    current_version,
                    version: Some(update.version.clone()),
                    notes: update.body.clone(),
                    // `OffsetDateTime`'s Display impl yields an ISO-8601 ish
                    // representation already; that's plenty for the UI.
                    date: update.date.map(|d| d.to_string()),
                    custom_build: CUSTOM_BUILD,
                    update_protected: CUSTOM_BUILD,
                    protection_reason: CUSTOM_BUILD.then(|| CUSTOM_BUILD_PROTECTION_REASON.into()),
                };
                if let Ok(mut guard) = pending.lock() {
                    *guard = Some(update);
                }
                Ok(info)
            }
            Ok(None) => {
                if let Ok(mut guard) = pending.lock() {
                    *guard = None;
                }
                Ok(UpdateInfo {
                    available: false,
                    current_version,
                    version: None,
                    notes: None,
                    date: None,
                    custom_build: CUSTOM_BUILD,
                    update_protected: false,
                    protection_reason: None,
                })
            }
            Err(err) => Err(format!("update check failed: {err}")),
        }
    }

    #[cfg(not(desktop))]
    {
        let _ = state;
        Ok(UpdateInfo {
            available: false,
            current_version,
            version: None,
            notes: None,
            date: None,
            custom_build: CUSTOM_BUILD,
            update_protected: false,
            protection_reason: None,
        })
    }
}

/// Downloads + verifies + installs the update that was previously discovered by
/// `updater_check`. If no pending update is cached we re-run the check first.
///
/// Progress is streamed via `updater://progress` events. When the install
/// completes successfully we emit `updater://finished` so the frontend can show
/// the "Restart" affordance. The caller is responsible for invoking
/// `updater_restart` afterwards.
#[tauri::command]
pub async fn updater_download_and_install(
    app: AppHandle,
    state: State<'_, UpdaterState>,
) -> Result<(), String> {
    #[cfg(desktop)]
    {
        if CUSTOM_BUILD {
            return Err(CUSTOM_BUILD_PROTECTION_REASON.into());
        }

        let pending = state.pending.clone();

        // Pull the cached Update or refresh.
        let update = {
            let guard = pending
                .lock()
                .map_err(|_| "updater state poisoned".to_string())?;
            guard.clone()
        };

        let update = match update {
            Some(u) => u,
            None => {
                let updater = app
                    .updater_builder()
                    .build()
                    .map_err(|err| format!("failed to build updater: {err}"))?;
                let fresh = updater
                    .check()
                    .await
                    .map_err(|err| format!("update check failed: {err}"))?
                    .ok_or_else(|| "no update available".to_string())?;
                if let Ok(mut guard) = pending.lock() {
                    *guard = Some(fresh.clone());
                }
                fresh
            }
        };

        let progress_app = app.clone();
        let mut downloaded: u64 = 0;
        let result = update
            .download_and_install(
                move |chunk, total| {
                    downloaded = downloaded.saturating_add(chunk as u64);
                    let _ =
                        progress_app.emit(PROGRESS_EVENT, ProgressPayload { downloaded, total });
                },
                || {},
            )
            .await;

        match result {
            Ok(()) => {
                let _ = app.emit(FINISHED_EVENT, ());
                Ok(())
            }
            Err(err) => Err(format!("update install failed: {err}")),
        }
    }

    #[cfg(not(desktop))]
    {
        let _ = (app, state);
        Err("updater unsupported on this platform".into())
    }
}

/// Relaunch the application. Called by the frontend once the user accepts the
/// restart prompt that appears after a successful install.
#[tauri::command]
pub fn updater_restart(app: AppHandle) {
    app.restart();
}

/// Returns the current app version. Used by the frontend to render the badge
/// tooltip even when no update is available.
#[tauri::command]
pub fn updater_current_version(app: AppHandle) -> String {
    app.package_info().version.to_string()
}
