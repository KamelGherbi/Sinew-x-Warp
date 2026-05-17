use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};
use sinew_core::{
    AppError, ChatMessage, Effort, ModelCapabilities, ModelRef, Part, Provider, ProviderRequest,
    ProviderStream, Result, Role, TokenEstimate, ToolDescriptor,
};
use tokio::sync::Mutex;

use crate::{
    auth::{
        generate_state, load_default_user_data, save_default_user_data, Credential, GoogleUserData,
    },
    model_info,
    stream::map_stream,
    wire,
};

const BASE_URL: &str = "https://cloudcode-pa.googleapis.com/v1internal";
const USER_AGENT: &str = "Google-Gemini-CLI/0.1 Sinew/0.1";
const FREE_TIER_ID: &str = "free-tier";

#[derive(Clone)]
pub struct GoogleConfig {
    pub credential: Credential,
    pub base_url: String,
}

impl GoogleConfig {
    pub fn new(credential: Credential) -> Self {
        Self {
            credential,
            base_url: BASE_URL.into(),
        }
    }

    pub fn from_default_sources() -> Result<Self> {
        if let Some(credential) = Credential::load_default()? {
            return Ok(Self::new(credential));
        }

        Err(AppError::Auth(
            "no google oauth credential found. Connect Google in Settings > Providers.".into(),
        ))
    }
}

pub struct GoogleProvider {
    config: GoogleConfig,
    http: reqwest::Client,
    user_data: Mutex<Option<GoogleUserData>>,
}

impl GoogleProvider {
    pub fn new(config: GoogleConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .tcp_keepalive(std::time::Duration::from_secs(20))
            .pool_idle_timeout(std::time::Duration::from_secs(90))
            .build()
            .map_err(|err| AppError::Network(err.to_string()))?;
        let user_data = load_default_user_data().unwrap_or(None);
        Ok(Self {
            config,
            http,
            user_data: Mutex::new(user_data),
        })
    }

    pub fn from_default_sources() -> Result<Self> {
        Self::new(GoogleConfig::from_default_sources()?)
    }

    async fn post(&self, method: &str) -> Result<reqwest::RequestBuilder> {
        let token = self.config.credential.bearer(&self.http).await?;
        Ok(self
            .http
            .post(self.method_url(method))
            .bearer_auth(token)
            .header("content-type", "application/json")
            .header("accept", "application/json"))
    }

    async fn get_operation(&self, name: &str) -> Result<reqwest::RequestBuilder> {
        let token = self.config.credential.bearer(&self.http).await?;
        Ok(self
            .http
            .get(format!(
                "{}/{}",
                self.config.base_url.trim_end_matches('/'),
                name.trim_start_matches('/')
            ))
            .bearer_auth(token)
            .header("accept", "application/json"))
    }

    fn method_url(&self, method: &str) -> String {
        format!("{}:{method}", self.config.base_url.trim_end_matches('/'))
    }

    async fn ensure_user_data(&self) -> Result<GoogleUserData> {
        if let Some(user_data) = self.user_data.lock().await.clone() {
            return Ok(user_data);
        }

        let user_data = self.setup_user().await?;
        if let Err(err) = save_default_user_data(&user_data) {
            tracing::warn!(error = %err, "failed to persist google code assist user data");
        }
        *self.user_data.lock().await = Some(user_data.clone());
        Ok(user_data)
    }

    async fn setup_user(&self) -> Result<GoogleUserData> {
        let env_project = google_project_from_env()?;
        let load = self.load_code_assist(env_project.clone()).await?;

        if let Some(current_tier) = load.current_tier.clone() {
            if let Some(project_id) = load
                .cloudaicompanion_project
                .clone()
                .or_else(|| env_project.clone())
            {
                return Ok(GoogleUserData {
                    project_id,
                    user_tier: load
                        .paid_tier
                        .as_ref()
                        .and_then(|tier| tier.id.clone())
                        .or(current_tier.id),
                    user_tier_name: load
                        .paid_tier
                        .as_ref()
                        .and_then(|tier| tier.name.clone())
                        .or(current_tier.name),
                });
            }

            return Err(ineligible_or_project_error(&load));
        }

        let tier = default_tier(&load).ok_or_else(|| ineligible_or_project_error(&load))?;
        let tier_id = tier.id.clone().unwrap_or_else(|| FREE_TIER_ID.into());
        let onboard_project = if tier_id == FREE_TIER_ID {
            None
        } else {
            env_project.clone()
        };
        let operation = self
            .onboard_user(Some(tier_id.clone()), onboard_project.clone())
            .await?;
        let operation = self.wait_for_operation(operation).await?;
        let project_id = operation
            .response
            .and_then(|response| response.cloudaicompanion_project)
            .and_then(|project| project.id)
            .or(onboard_project)
            .or(env_project)
            .ok_or_else(|| ineligible_or_project_error(&load))?;

        Ok(GoogleUserData {
            project_id,
            user_tier: Some(tier_id),
            user_tier_name: tier.name,
        })
    }

    async fn load_code_assist(
        &self,
        project_id: Option<String>,
    ) -> Result<wire::LoadCodeAssistResponse> {
        let body = wire::LoadCodeAssistRequest {
            cloudaicompanion_project: project_id.clone(),
            metadata: client_metadata(project_id),
            mode: None,
        };
        let response = self
            .post("loadCodeAssist")
            .await?
            .json(&body)
            .send()
            .await
            .map_err(|err| AppError::Network(err.to_string()))?;
        if !response.status().is_success() {
            return Err(read_http_error(response).await);
        }
        response
            .json()
            .await
            .map_err(|err| AppError::Decode(format!("invalid google loadCodeAssist body: {err}")))
    }

    async fn onboard_user(
        &self,
        tier_id: Option<String>,
        project_id: Option<String>,
    ) -> Result<wire::LongRunningOperationResponse> {
        let body = wire::OnboardUserRequest {
            tier_id,
            cloudaicompanion_project: project_id.clone(),
            metadata: client_metadata(project_id),
        };
        let response = self
            .post("onboardUser")
            .await?
            .json(&body)
            .send()
            .await
            .map_err(|err| AppError::Network(err.to_string()))?;
        if !response.status().is_success() {
            return Err(read_http_error(response).await);
        }
        response
            .json()
            .await
            .map_err(|err| AppError::Decode(format!("invalid google onboardUser body: {err}")))
    }

    async fn wait_for_operation(
        &self,
        mut operation: wire::LongRunningOperationResponse,
    ) -> Result<wire::LongRunningOperationResponse> {
        let Some(name) = operation.name.clone() else {
            return Ok(operation);
        };
        for _ in 0..30 {
            if operation.done {
                return Ok(operation);
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
            let response = self
                .get_operation(&name)
                .await?
                .send()
                .await
                .map_err(|err| AppError::Network(err.to_string()))?;
            if !response.status().is_success() {
                return Err(read_http_error(response).await);
            }
            operation = response
                .json()
                .await
                .map_err(|err| AppError::Decode(format!("invalid google operation body: {err}")))?;
        }
        Err(AppError::Provider(
            "google code assist onboarding timed out".into(),
        ))
    }
}

#[async_trait]
impl Provider for GoogleProvider {
    fn name(&self) -> &str {
        "google"
    }

    fn capabilities(&self, model: &ModelRef) -> Option<ModelCapabilities> {
        if model.provider != "google" {
            return None;
        }
        Some(model_info::capabilities(model))
    }

    async fn estimate_tokens(&self, request: ProviderRequest) -> Result<TokenEstimate> {
        if request.model.provider != "google" {
            return Err(AppError::Unsupported(format!(
                "google provider cannot count model provider {}",
                request.model.provider
            )));
        }
        let contents = to_contents(&request.transcript)?;
        let body = wire::CountTokensRequest {
            request: wire::VertexCountTokensRequest {
                model: format!("models/{}", request.model.name),
                contents,
            },
        };
        let response = self
            .post("countTokens")
            .await?
            .json(&body)
            .send()
            .await
            .map_err(|err| AppError::Network(err.to_string()))?;
        if !response.status().is_success() {
            return Err(read_http_error(response).await);
        }
        let counted: wire::CountTokensResponse = response
            .json()
            .await
            .map_err(|err| AppError::Decode(format!("invalid google countTokens body: {err}")))?;
        Ok(TokenEstimate {
            input_tokens: counted.total_tokens.unwrap_or(0),
            exact: counted.total_tokens.is_some(),
        })
    }

    async fn stream(&self, request: ProviderRequest) -> Result<ProviderStream> {
        if request.model.provider != "google" {
            return Err(AppError::Unsupported(format!(
                "google provider cannot stream model provider {}",
                request.model.provider
            )));
        }
        let caps = model_info::capabilities(&request.model);
        let user_data = self.ensure_user_data().await?;
        let body = build_generate_request(&request, &user_data, &caps)?;
        let response = self
            .post("streamGenerateContent")
            .await?
            .query(&[("alt", "sse")])
            .header("accept", "text/event-stream")
            .json(&body)
            .send()
            .await
            .map_err(|err| AppError::Network(err.to_string()))?;

        if !response.status().is_success() {
            return Err(read_http_error(response).await);
        }

        Ok(map_stream(response.bytes_stream(), request.model.name))
    }
}

fn build_generate_request(
    request: &ProviderRequest,
    user_data: &GoogleUserData,
    caps: &ModelCapabilities,
) -> Result<wire::CodeAssistGenerateRequest> {
    Ok(wire::CodeAssistGenerateRequest {
        model: request.model.name.clone(),
        project: Some(user_data.project_id.clone()),
        user_prompt_id: generate_state(),
        request: wire::VertexGenerateContentRequest {
            contents: to_contents(&request.transcript)?,
            system_instruction: system_instruction(request.system_prompt.as_deref()),
            tools: to_tools(&request.tools),
            generation_config: Some(generation_config(request, caps)),
            session_id: request.cache_key.clone(),
        },
    })
}

fn system_instruction(text: Option<&str>) -> Option<wire::Content> {
    let text = text?.trim();
    if text.is_empty() {
        return None;
    }
    Some(wire::Content {
        role: "user".into(),
        parts: vec![wire::Part::Text {
            text: text.to_string(),
            thought: None,
            thought_signature: None,
        }],
    })
}

fn generation_config(
    request: &ProviderRequest,
    caps: &ModelCapabilities,
) -> wire::GenerationConfig {
    wire::GenerationConfig {
        temperature: request.temperature.or(Some(1.0)),
        top_p: Some(0.95),
        top_k: Some(64),
        max_output_tokens: Some(
            request
                .max_output_tokens
                .unwrap_or(caps.max_output_tokens)
                .min(caps.max_output_tokens),
        ),
        thinking_config: Some(thinking_config(
            request.effective_effort(),
            &request.model.name,
        )),
    }
}

fn thinking_config(effort: Option<Effort>, model: &str) -> wire::ThinkingConfig {
    if model.starts_with("gemini-3") {
        return wire::ThinkingConfig {
            include_thoughts: Some(!matches!(effort, Some(Effort::None))),
            thinking_budget: None,
            thinking_level: match effort.unwrap_or(Effort::High) {
                Effort::None => None,
                Effort::Low => Some("LOW"),
                Effort::Medium => Some("MEDIUM"),
                Effort::High | Effort::Xhigh | Effort::Max => Some("HIGH"),
            },
        };
    }

    wire::ThinkingConfig {
        include_thoughts: Some(!matches!(effort, Some(Effort::None))),
        thinking_budget: Some(match effort.unwrap_or(Effort::Medium) {
            Effort::None => 0,
            Effort::Low => 1024,
            Effort::Medium => 8192,
            Effort::High => 16_384,
            Effort::Xhigh | Effort::Max => 32_768,
        }),
        thinking_level: None,
    }
}

fn to_tools(tools: &[ToolDescriptor]) -> Vec<wire::Tool> {
    if tools.is_empty() {
        return Vec::new();
    }
    vec![wire::Tool {
        function_declarations: tools
            .iter()
            .map(|tool| wire::FunctionDeclaration {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters_json_schema: tool.input_schema.clone(),
            })
            .collect(),
    }]
}

fn to_contents(transcript: &[ChatMessage]) -> Result<Vec<wire::Content>> {
    let mut contents = Vec::new();
    for message in transcript {
        let role = match message.role {
            Role::User => "user",
            Role::Assistant => "model",
        };
        let mut parts = Vec::new();
        for part in &message.parts {
            if part_is_ui_only(part) {
                continue;
            }
            match part {
                Part::Text { text, .. } => {
                    if !text.is_empty() {
                        parts.push(wire::Part::Text {
                            text: text.clone(),
                            thought: None,
                            thought_signature: None,
                        });
                    }
                }
                Part::Image {
                    media_type, data, ..
                } => {
                    if matches!(message.role, Role::User) && !data.trim().is_empty() {
                        parts.push(wire::Part::InlineData {
                            inline_data: wire::InlineData {
                                mime_type: media_type.clone(),
                                data: data.clone(),
                            },
                        });
                    }
                }
                Part::Thinking { .. } => {}
                Part::ToolCall {
                    id, name, input, ..
                } => {
                    let (_, raw_id) = split_tool_id(name, id);
                    parts.push(wire::Part::FunctionCall {
                        function_call: wire::FunctionCall {
                            name: name.clone(),
                            id: Some(raw_id),
                            args: input.clone(),
                        },
                    });
                }
                Part::ToolResult {
                    tool_call_id,
                    content,
                    images,
                    is_error,
                    ..
                } => {
                    let (name, raw_id) = split_prefixed_tool_id(tool_call_id);
                    let mut response = json!({
                        "output": content,
                    });
                    if *is_error {
                        response["error"] = json!(true);
                    }
                    let image_count = images
                        .iter()
                        .filter(|image| !image.data.trim().is_empty())
                        .count();
                    if image_count > 0 {
                        response["images"] = json!(format!(
                            "{image_count} image attachment{} returned by the tool.",
                            if image_count == 1 { "" } else { "s" }
                        ));
                    }
                    parts.push(wire::Part::FunctionResponse {
                        function_response: wire::FunctionResponse {
                            name,
                            id: Some(raw_id),
                            response,
                        },
                    });
                }
            }
        }
        if !parts.is_empty() {
            contents.push(wire::Content {
                role: role.into(),
                parts,
            });
        }
    }
    Ok(contents)
}

fn split_tool_id(name: &str, id: &str) -> (String, String) {
    let prefix = format!("{name}__");
    if let Some(raw) = id.strip_prefix(&prefix) {
        (name.to_string(), raw.to_string())
    } else {
        (name.to_string(), id.to_string())
    }
}

fn split_prefixed_tool_id(id: &str) -> (String, String) {
    if let Some((name, raw_id)) = id.split_once("__") {
        if !name.trim().is_empty() && !raw_id.trim().is_empty() {
            return (name.to_string(), raw_id.to_string());
        }
    }
    ("generic_tool".into(), id.to_string())
}

fn part_is_ui_only(part: &Part) -> bool {
    part_meta(part)
        .and_then(|meta| meta.get("ui_only"))
        .and_then(|value| value.as_bool())
        == Some(true)
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

fn client_metadata(project_id: Option<String>) -> wire::ClientMetadata {
    wire::ClientMetadata {
        ide_type: "IDE_UNSPECIFIED",
        platform: "PLATFORM_UNSPECIFIED",
        plugin_type: "GEMINI",
        duet_project: project_id,
    }
}

fn google_project_from_env() -> Result<Option<String>> {
    let project = std::env::var("GOOGLE_CLOUD_PROJECT")
        .ok()
        .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT_ID").ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    if let Some(project) = &project {
        if project.chars().all(|c| c.is_ascii_digit()) {
            return Err(AppError::InvalidRequest(
                "GOOGLE_CLOUD_PROJECT must be a string project id, not a numeric project number"
                    .into(),
            ));
        }
    }
    Ok(project)
}

fn default_tier(load: &wire::LoadCodeAssistResponse) -> Option<wire::GeminiUserTier> {
    load.allowed_tiers
        .iter()
        .find(|tier| tier.is_default == Some(true))
        .cloned()
        .or_else(|| load.allowed_tiers.first().cloned())
}

fn ineligible_or_project_error(load: &wire::LoadCodeAssistResponse) -> AppError {
    if !load.ineligible_tiers.is_empty() {
        let reasons = load
            .ineligible_tiers
            .iter()
            .filter_map(|tier| {
                tier.reason_message
                    .clone()
                    .or_else(|| tier.tier_name.clone())
            })
            .collect::<Vec<_>>()
            .join(", ");
        if !reasons.is_empty() {
            return AppError::Auth(reasons);
        }
    }
    AppError::Auth("google code assist requires GOOGLE_CLOUD_PROJECT for this account".into())
}

async fn read_http_error(response: reqwest::Response) -> AppError {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    let parsed: std::result::Result<wire::ApiErrorEnvelope, _> = serde_json::from_str(&body);
    let message = parsed
        .ok()
        .map(|payload| {
            if payload.error.status.is_empty() {
                payload.error.message
            } else if let Some(code) = payload.error.code {
                format!(
                    "{} ({code}): {}",
                    payload.error.status, payload.error.message
                )
            } else {
                format!("{}: {}", payload.error.status, payload.error.message)
            }
        })
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(body);

    if status == reqwest::StatusCode::UNAUTHORIZED {
        AppError::Auth(message)
    } else if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        AppError::RateLimit(message)
    } else if status.is_client_error() {
        if message.contains("context") || message.contains("token") && message.contains("limit") {
            AppError::ContextLength(message)
        } else {
            AppError::InvalidRequest(message)
        }
    } else {
        AppError::Provider(format!("HTTP {status}: {message}"))
    }
}
