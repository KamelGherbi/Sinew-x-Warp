use std::{collections::HashMap, time::{SystemTime, UNIX_EPOCH}};

use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ProviderUsageSummary {
    pub(super) updated_at_ms: i64,
    pub(super) providers: Vec<ProviderUsageStatus>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ProviderUsageStatus {
    pub(super) provider: String,
    pub(super) source: String,
    pub(super) state: ProviderUsageState,
    pub(super) exact: bool,
    pub(super) label: Option<String>,
    pub(super) windows: Vec<ProviderUsageWindow>,
    pub(super) balance: Option<ProviderUsageBalance>,
    pub(super) spend: Option<ProviderUsageSpend>,
    pub(super) error: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) enum ProviderUsageState {
    Available,
    Unavailable,
    Error,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ProviderUsageWindow {
    pub(super) id: String,
    pub(super) label: String,
    pub(super) used_percent: Option<f64>,
    pub(super) remaining_percent: Option<f64>,
    pub(super) used: Option<f64>,
    pub(super) limit: Option<f64>,
    pub(super) remaining: Option<f64>,
    pub(super) unit: Option<String>,
    pub(super) reset_at_ms: Option<i64>,
    pub(super) reset_at: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ProviderUsageBalance {
    pub(super) label: String,
    pub(super) amount: f64,
    pub(super) unit: Option<String>,
    pub(super) currency: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ProviderUsageSpend {
    pub(super) today: Option<f64>,
    pub(super) week: Option<f64>,
    pub(super) month: Option<f64>,
    pub(super) currency: Option<String>,
}

#[tauri::command]
pub(super) async fn provider_usage_summary() -> std::result::Result<ProviderUsageSummary, String> {
    let http = reqwest::Client::builder()
        .user_agent("sinew/0.1")
        .build()
        .map_err(|err| format!("unable to build usage client: {err}"))?;

    let (openai, google, kimi, openrouter) = tokio::join!(
        fetch_openai_codex_usage(http.clone()),
        fetch_google_usage(http.clone()),
        fetch_kimi_usage(http.clone()),
        fetch_openrouter_usage(http),
    );

    Ok(ProviderUsageSummary {
        updated_at_ms: now_ms(),
        providers: vec![openai, google, kimi, openrouter],
    })
}

async fn fetch_openai_codex_usage(http: reqwest::Client) -> ProviderUsageStatus {
    let provider = "openai";
    let source = "codex-oauth";
    let credential = match sinew_openai::Credential::load_default() {
        Ok(Some(credential)) => credential,
        Ok(None) => return unavailable(provider, source, "OpenAI OAuth is not connected."),
        Err(err) => return errored(provider, source, err.to_string()),
    };

    let bearer = match credential.bearer(&http).await {
        Ok(bearer) => bearer,
        Err(err) => return errored(provider, source, err.to_string()),
    };
    if !bearer.is_oauth {
        return unavailable(
            provider,
            source,
            "OpenAI API keys do not expose ChatGPT/Codex subscription quota.",
        );
    }

    let mut request = http
        .get("https://chatgpt.com/backend-api/wham/usage")
        .bearer_auth(&bearer.token)
        .header("accept", "application/json");
    if let Some(account_id) = bearer.account_id {
        request = request.header("chatgpt-account-id", account_id);
    }

    let json = match send_json(request).await {
        Ok(json) => json,
        Err(err) => return errored(provider, source, err),
    };

    let rate_limit = json.get("rate_limit").unwrap_or(&json);
    let mut windows = Vec::new();
    if let Some(primary) = first_child(rate_limit, &["primary_window", "primaryWindow", "primary"]) {
        windows.push(parse_generic_window(primary, "session", "Session / 5h", Some("percent")));
    }
    if let Some(secondary) = first_child(rate_limit, &["secondary_window", "secondaryWindow", "secondary"]) {
        windows.push(parse_generic_window(secondary, "weekly", "Weekly", Some("percent")));
    }
    if let Some(additional) = rate_limit
        .get("additional_rate_limits")
        .or_else(|| rate_limit.get("additionalRateLimits"))
        .and_then(Value::as_array)
    {
        for (index, entry) in additional.iter().enumerate() {
            let id = field_string(entry, &["id", "model", "name"])
                .unwrap_or_else(|| format!("extra-{index}"));
            let label = field_string(entry, &["title", "label", "name", "model"])
                .unwrap_or_else(|| id.clone());
            let window_value = first_child(entry, &["window", "rate_limit", "rateLimit"]).unwrap_or(entry);
            windows.push(parse_generic_window(window_value, &id, &label, Some("percent")));
        }
    }

    let credits = json.get("credits").or_else(|| rate_limit.get("credits"));
    let balance = credits.and_then(|credits| {
        let amount = field_f64(credits, &["balance", "remaining", "amount"])?;
        Some(ProviderUsageBalance {
            label: if field_bool(credits, &["unlimited"]).unwrap_or(false) {
                "Credits (unlimited)".into()
            } else {
                "Credits remaining".into()
            },
            amount,
            unit: Some("credits".into()),
            currency: None,
        })
    });

    ProviderUsageStatus {
        provider: provider.into(),
        source: source.into(),
        state: ProviderUsageState::Available,
        exact: true,
        label: Some("ChatGPT/Codex subscription quota".into()),
        windows,
        balance,
        spend: None,
        error: None,
    }
}

async fn fetch_google_usage(http: reqwest::Client) -> ProviderUsageStatus {
    let provider = "google";
    let source = "google-code-assist-quota";
    let credential = match sinew_google::auth::Credential::load_default() {
        Ok(Some(credential)) => credential,
        Ok(None) => return unavailable(provider, source, "Google OAuth is not connected."),
        Err(err) => return errored(provider, source, err.to_string()),
    };
    let token = match credential.bearer(&http).await {
        Ok(token) => token,
        Err(err) => return errored(provider, source, err.to_string()),
    };
    let user_data = sinew_google::auth::load_default_user_data().ok().flatten();
    let mut body = serde_json::Map::new();
    if let Some(project_id) = user_data.as_ref().map(|user| user.project_id.trim()).filter(|id| !id.is_empty()) {
        body.insert("project".into(), Value::String(project_id.to_string()));
    }
    let json = match send_json(
        http.post("https://cloudcode-pa.googleapis.com/v1internal:retrieveUserQuota")
            .bearer_auth(token)
            .header("content-type", "application/json")
            .header("accept", "application/json")
            .json(&Value::Object(body)),
    )
    .await
    {
        Ok(json) => json,
        Err(err) => return errored(provider, source, err),
    };

    let buckets = json
        .get("buckets")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut by_model: HashMap<String, (f64, Option<String>)> = HashMap::new();
    for bucket in buckets {
        let Some(model_id) = field_string(&bucket, &["modelId", "model_id", "model"]) else {
            continue;
        };
        let Some(remaining_fraction) = field_f64(&bucket, &["remainingFraction", "remaining_fraction"]) else {
            continue;
        };
        let reset_at = field_string(&bucket, &["resetTime", "reset_time", "resetsAt", "resets_at"]);
        let remaining = if remaining_fraction <= 1.0 {
            remaining_fraction * 100.0
        } else {
            remaining_fraction
        };
        match by_model.get_mut(&model_id) {
            Some((existing, existing_reset)) if remaining < *existing => {
                *existing = remaining;
                *existing_reset = reset_at;
            }
            None => {
                by_model.insert(model_id, (remaining, reset_at));
            }
            _ => {}
        }
    }

    let mut windows = Vec::new();
    for (id, label, predicate) in [
        ("pro", "Pro models", ModelFamily::Pro),
        ("flash", "Flash models", ModelFamily::Flash),
        ("flash-lite", "Flash Lite models", ModelFamily::FlashLite),
    ] {
        let selected = by_model
            .iter()
            .filter(|(model, _)| predicate.matches(model))
            .min_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((model, (remaining, reset_at))) = selected {
            windows.push(ProviderUsageWindow {
                id: id.into(),
                label: format!("{label} ({model})"),
                used_percent: Some(clamp_percent(100.0 - remaining)),
                remaining_percent: Some(clamp_percent(*remaining)),
                used: None,
                limit: None,
                remaining: None,
                unit: Some("percent".into()),
                reset_at_ms: None,
                reset_at: reset_at.clone(),
            });
        }
    }

    if windows.is_empty() {
        for (model, (remaining, reset_at)) in by_model.iter().take(5) {
            windows.push(ProviderUsageWindow {
                id: model.clone(),
                label: model.clone(),
                used_percent: Some(clamp_percent(100.0 - remaining)),
                remaining_percent: Some(clamp_percent(*remaining)),
                used: None,
                limit: None,
                remaining: None,
                unit: Some("percent".into()),
                reset_at_ms: None,
                reset_at: reset_at.clone(),
            });
        }
    }

    ProviderUsageStatus {
        provider: provider.into(),
        source: source.into(),
        state: ProviderUsageState::Available,
        exact: true,
        label: user_data
            .and_then(|user| user.user_tier_name.or(user.user_tier))
            .or_else(|| Some("Google Code Assist quota".into())),
        windows,
        balance: None,
        spend: None,
        error: None,
    }
}

async fn fetch_kimi_usage(http: reqwest::Client) -> ProviderUsageStatus {
    let provider = "kimi";
    let source = "kimi-code-usage";
    let credential = match sinew_kimi::Credential::load_default() {
        Ok(Some(credential)) => credential,
        Ok(None) => return unavailable(provider, source, "Kimi OAuth is not connected."),
        Err(err) => return errored(provider, source, err.to_string()),
    };
    let token = match credential.bearer(&http).await {
        Ok(token) => token,
        Err(err) => return errored(provider, source, err.to_string()),
    };
    let json = match send_json(
        http.get("https://api.kimi.com/coding/v1/usages")
            .bearer_auth(token)
            .header("accept", "application/json"),
    )
    .await
    {
        Ok(json) => json,
        Err(err) => return errored(provider, source, err),
    };

    let mut windows = Vec::new();
    if let Some(usage) = json.get("usage") {
        windows.push(parse_kimi_detail(usage, "weekly", "Weekly requests"));
    }
    if let Some(limit) = json
        .get("limits")
        .and_then(Value::as_array)
        .and_then(|limits| limits.first())
        .and_then(|limit| limit.get("detail"))
    {
        windows.push(parse_kimi_detail(limit, "rate-limit", "5h rate limit"));
    }

    ProviderUsageStatus {
        provider: provider.into(),
        source: source.into(),
        state: ProviderUsageState::Available,
        exact: true,
        label: Some("Kimi Coding quota".into()),
        windows,
        balance: None,
        spend: None,
        error: None,
    }
}

async fn fetch_openrouter_usage(http: reqwest::Client) -> ProviderUsageStatus {
    let provider = "openrouter";
    let source = "openrouter-api";
    let api_key = match sinew_openrouter::load_default_api_key() {
        Ok(Some(api_key)) => api_key,
        Ok(None) => return unavailable(provider, source, "OpenRouter API key is not configured."),
        Err(err) => return errored(provider, source, err.to_string()),
    };

    let credits = send_json(
        http.get("https://openrouter.ai/api/v1/credits")
            .bearer_auth(&api_key)
            .header("accept", "application/json"),
    )
    .await;
    let key = send_json(
        http.get("https://openrouter.ai/api/v1/key")
            .bearer_auth(&api_key)
            .header("accept", "application/json"),
    )
    .await;

    let mut errors = Vec::new();
    let mut balance = None;
    if let Ok(json) = credits {
        let data = json.get("data").unwrap_or(&json);
        let total = field_f64(data, &["total_credits", "totalCredits", "credits"]);
        let used = field_f64(data, &["total_usage", "totalUsage", "usage", "used"]);
        let remaining = field_f64(data, &["balance", "remaining"])
            .or_else(|| total.zip(used).map(|(total, used)| total - used));
        if let Some(amount) = remaining {
            balance = Some(ProviderUsageBalance {
                label: "Credit balance".into(),
                amount,
                unit: None,
                currency: Some("USD".into()),
            });
        }
    } else if let Err(err) = credits {
        errors.push(err);
    }

    let mut windows = Vec::new();
    let mut spend = None;
    if let Ok(json) = key {
        let data = json.get("data").unwrap_or(&json);
        let used = field_f64(data, &["usage", "used", "key_usage", "keyUsage"]);
        let limit = field_f64(data, &["limit", "usage_limit", "usageLimit"]);
        let remaining = field_f64(data, &["limit_remaining", "limitRemaining", "remaining"]);
        if used.is_some() || limit.is_some() || remaining.is_some() {
            windows.push(window_from_amounts(
                "key-limit",
                "API key limit",
                used,
                limit,
                remaining,
                Some("USD".into()),
                None,
                None,
            ));
        }
        let today = field_f64(data, &["daily_spend", "dailySpend", "usage_today", "usageToday"]);
        let week = field_f64(data, &["weekly_spend", "weeklySpend", "usage_week", "usageWeek"]);
        let month = field_f64(data, &["monthly_spend", "monthlySpend", "usage_month", "usageMonth"]);
        if today.is_some() || week.is_some() || month.is_some() {
            spend = Some(ProviderUsageSpend {
                today,
                week,
                month,
                currency: Some("USD".into()),
            });
        }
    } else if let Err(err) = key {
        errors.push(err);
    }

    if balance.is_none() && windows.is_empty() && spend.is_none() && !errors.is_empty() {
        return errored(provider, source, errors.join("; "));
    }

    ProviderUsageStatus {
        provider: provider.into(),
        source: source.into(),
        state: ProviderUsageState::Available,
        exact: true,
        label: Some("OpenRouter credits and API key usage".into()),
        windows,
        balance,
        spend,
        error: (!errors.is_empty()).then(|| errors.join("; ")),
    }
}

async fn send_json(request: reqwest::RequestBuilder) -> Result<Value, String> {
    let response = request
        .send()
        .await
        .map_err(|err| format!("request failed: {err}"))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        let cleaned = body.replace('\n', " ");
        let truncated = if cleaned.chars().count() > 400 {
            let prefix = cleaned.chars().take(400).collect::<String>();
            format!("{prefix}…")
        } else {
            cleaned
        };
        return Err(format!("HTTP {status}: {truncated}"));
    }
    response
        .json::<Value>()
        .await
        .map_err(|err| format!("invalid JSON response: {err}"))
}

fn parse_generic_window(value: &Value, id: &str, label: &str, unit: Option<&str>) -> ProviderUsageWindow {
    let used_percent = field_f64(
        value,
        &[
            "used_percent",
            "usedPercent",
            "utilization",
            "usage_percent",
            "usagePercent",
        ],
    )
    .map(percent_from_ratio_or_percent);
    let reset_at = field_string(value, &["resets_at", "resetsAt", "reset_at", "resetAt"]);
    let reset_at_ms = reset_at
        .as_deref()
        .and_then(parse_epoch_ms)
        .or_else(|| field_f64(value, &["reset_at", "resetAt", "resets_at", "resetsAt"]).and_then(number_to_epoch_ms));
    let used = field_f64(value, &["used", "usage"]);
    let limit = field_f64(value, &["limit", "quota", "total"]);
    let remaining = field_f64(value, &["remaining", "available"]);
    let amount_window = window_from_amounts(id, label, used, limit, remaining, unit.map(str::to_string), reset_at_ms, reset_at);
    if used_percent.is_none() {
        return amount_window;
    }
    ProviderUsageWindow {
        used_percent,
        remaining_percent: used_percent.map(|percent| clamp_percent(100.0 - percent)),
        ..amount_window
    }
}

fn parse_kimi_detail(value: &Value, id: &str, label: &str) -> ProviderUsageWindow {
    let used = field_f64(value, &["used"]);
    let limit = field_f64(value, &["limit"]);
    let remaining = field_f64(value, &["remaining"]);
    let reset_at = field_string(value, &["resetTime", "reset_time", "resetsAt", "resets_at"]);
    window_from_amounts(
        id,
        label,
        used,
        limit,
        remaining,
        Some("requests".into()),
        reset_at.as_deref().and_then(parse_epoch_ms),
        reset_at,
    )
}

fn window_from_amounts(
    id: &str,
    label: &str,
    used: Option<f64>,
    limit: Option<f64>,
    remaining: Option<f64>,
    unit: Option<String>,
    reset_at_ms: Option<i64>,
    reset_at: Option<String>,
) -> ProviderUsageWindow {
    let inferred_used = used.or_else(|| limit.zip(remaining).map(|(limit, remaining)| limit - remaining));
    let inferred_remaining = remaining.or_else(|| limit.zip(inferred_used).map(|(limit, used)| limit - used));
    let used_percent = limit
        .zip(inferred_used)
        .filter(|(limit, _)| *limit > 0.0)
        .map(|(limit, used)| clamp_percent((used / limit) * 100.0));
    ProviderUsageWindow {
        id: id.into(),
        label: label.into(),
        used_percent,
        remaining_percent: used_percent.map(|percent| clamp_percent(100.0 - percent)),
        used: inferred_used,
        limit,
        remaining: inferred_remaining,
        unit,
        reset_at_ms,
        reset_at,
    }
}

fn unavailable(provider: &str, source: &str, message: impl Into<String>) -> ProviderUsageStatus {
    ProviderUsageStatus {
        provider: provider.into(),
        source: source.into(),
        state: ProviderUsageState::Unavailable,
        exact: false,
        label: None,
        windows: Vec::new(),
        balance: None,
        spend: None,
        error: Some(message.into()),
    }
}

fn errored(provider: &str, source: &str, message: impl Into<String>) -> ProviderUsageStatus {
    ProviderUsageStatus {
        provider: provider.into(),
        source: source.into(),
        state: ProviderUsageState::Error,
        exact: false,
        label: None,
        windows: Vec::new(),
        balance: None,
        spend: None,
        error: Some(message.into()),
    }
}

fn field_f64(value: &Value, names: &[&str]) -> Option<f64> {
    first_child(value, names).and_then(|value| match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    })
}

fn field_string(value: &Value, names: &[&str]) -> Option<String> {
    first_child(value, names).and_then(|value| match value {
        Value::String(text) if !text.trim().is_empty() => Some(text.trim().to_string()),
        Value::Number(number) => Some(number.to_string()),
        _ => None,
    })
}

fn field_bool(value: &Value, names: &[&str]) -> Option<bool> {
    first_child(value, names).and_then(|value| match value {
        Value::Bool(value) => Some(*value),
        Value::String(text) => text.trim().parse::<bool>().ok(),
        _ => None,
    })
}

fn first_child<'a>(value: &'a Value, names: &[&str]) -> Option<&'a Value> {
    for name in names {
        if let Some(child) = value.get(*name) {
            if !child.is_null() {
                return Some(child);
            }
        }
    }
    None
}

fn percent_from_ratio_or_percent(value: f64) -> f64 {
    if value <= 1.0 {
        clamp_percent(value * 100.0)
    } else {
        clamp_percent(value)
    }
}

fn clamp_percent(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    value.clamp(0.0, 100.0)
}

fn parse_epoch_ms(value: &str) -> Option<i64> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed.parse::<f64>().ok().and_then(number_to_epoch_ms)
}

fn number_to_epoch_ms(value: f64) -> Option<i64> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let ms = if value < 10_000_000_000.0 {
        value * 1000.0
    } else {
        value
    };
    Some(ms.round() as i64)
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

#[derive(Clone, Copy)]
enum ModelFamily {
    Pro,
    Flash,
    FlashLite,
}

impl ModelFamily {
    fn matches(self, model_id: &str) -> bool {
        let model = model_id.to_ascii_lowercase();
        match self {
            Self::Pro => model.contains("pro"),
            Self::Flash => model.contains("flash") && !model.contains("flash-lite"),
            Self::FlashLite => model.contains("flash-lite"),
        }
    }
}
