use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::stream::Stream;
use futures::{stream, StreamExt};
use futures_util::{FutureExt, SinkExt};
use serde_json::Value;
use sinew_core::{
    AppError, ChatMessage, Effort, ModelCapabilities, ModelRef, Part, Provider, ProviderRequest,
    ProviderStream, Result, Role, StreamEvent, TokenEstimate, ToolDescriptor,
};
use tokio::{
    net::TcpStream,
    sync::{Mutex, OwnedMutexGuard},
};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{client::IntoClientRequest, http::HeaderValue, Message},
    MaybeTlsStream, WebSocketStream,
};

use crate::{
    auth::{BearerToken, Credential},
    model_info,
    stream::EventParser,
    wire,
};

const API_BASE_URL: &str = "https://api.openai.com/v1";
const CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const USER_AGENT: &str = "sinew/0.1";
const FALLBACK_INSTRUCTIONS: &str = "You are Sinew, a concise coding assistant.";
const WEBSOCKET_TTL: Duration = Duration::from_secs(55 * 60);

type ResponsesWsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

#[derive(Clone)]
struct WebSocketConnectSpec {
    url: String,
    token: String,
    account_id: Option<String>,
    is_oauth: bool,
}

struct ResponsesWsSession {
    socket: Option<ResponsesWsStream>,
    connected_at: Option<Instant>,
    model_name: Option<String>,
    previous_response_id: Option<String>,
    covered_message_count: usize,
    websocket_disabled: bool,
}

impl ResponsesWsSession {
    fn new() -> Self {
        Self {
            socket: None,
            connected_at: None,
            model_name: None,
            previous_response_id: None,
            covered_message_count: 0,
            websocket_disabled: false,
        }
    }

    fn reset_chain(&mut self) {
        self.previous_response_id = None;
        self.covered_message_count = 0;
    }

    fn clear_socket(&mut self) {
        self.socket = None;
    }

    fn reset_socket(&mut self) {
        self.clear_socket();
        self.connected_at = None;
    }

    fn reset_connection(&mut self) {
        self.reset_socket();
        self.model_name = None;
        self.reset_chain();
    }

    fn disable_websocket(&mut self) {
        self.websocket_disabled = true;
        self.reset_connection();
    }

    fn expire_if_needed(&mut self) {
        if self
            .connected_at
            .map(|connected| connected.elapsed() >= WEBSOCKET_TTL)
            .unwrap_or(false)
        {
            self.reset_connection();
        }
    }
}

#[derive(Clone)]
pub struct OpenAiConfig {
    pub credential: Credential,
    pub api_base_url: String,
    pub codex_base_url: String,
}

impl OpenAiConfig {
    pub fn new(credential: Credential) -> Self {
        Self {
            credential,
            api_base_url: API_BASE_URL.into(),
            codex_base_url: CODEX_BASE_URL.into(),
        }
    }

    pub fn from_default_sources() -> Result<Self> {
        if let Some(credential) = Credential::load_default()? {
            return Ok(Self::new(credential));
        }

        Err(AppError::Auth(
            "no openai oauth credential found. Connect OpenAI in Settings > Providers".into(),
        ))
    }
}

pub struct OpenAiProvider {
    config: OpenAiConfig,
    http: reqwest::Client,
    websocket_sessions: Mutex<HashMap<String, Arc<Mutex<ResponsesWsSession>>>>,
}

impl OpenAiProvider {
    pub fn new(config: OpenAiConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .build()
            .map_err(|err| AppError::Network(err.to_string()))?;
        Ok(Self {
            config,
            http,
            websocket_sessions: Mutex::new(HashMap::new()),
        })
    }

    pub fn from_default_sources() -> Result<Self> {
        Self::new(OpenAiConfig::from_default_sources()?)
    }

    async fn post(&self, route: &str) -> Result<reqwest::RequestBuilder> {
        let bearer = self.config.credential.bearer(&self.http).await?;
        let base_url = if bearer.is_oauth {
            &self.config.codex_base_url
        } else {
            &self.config.api_base_url
        };
        let mut request = self
            .http
            .post(format!("{}{}", base_url.trim_end_matches('/'), route))
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {}", bearer.token));

        if bearer.is_oauth {
            request = request.header("openai-beta", "responses=experimental");
            if let Some(account_id) = bearer.account_id {
                request = request.header("chatgpt-account-id", account_id);
            }
        }

        Ok(request)
    }

    async fn websocket_session(&self, key: &str) -> Arc<Mutex<ResponsesWsSession>> {
        let mut sessions = self.websocket_sessions.lock().await;
        sessions
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(ResponsesWsSession::new())))
            .clone()
    }

    async fn websocket_enabled_for(&self, cache_key: &str) -> bool {
        let session = self.websocket_session(cache_key).await;
        let session = session.lock().await;
        !session.websocket_disabled
    }

    async fn stream_responses(&self, request: ProviderRequest) -> Result<ProviderStream> {
        if let Some(cache_key) = request.cache_key.as_deref() {
            if self.websocket_enabled_for(cache_key).await {
                let fallback = SseFallbackContext {
                    config: self.config.clone(),
                    http: self.http.clone(),
                    request: request.clone(),
                };
                match self.stream_websocket(request).await {
                    Ok(stream) => return Ok(websocket_stream_with_sse_fallback(stream, fallback)),
                    Err(err) if should_fallback_from_websocket_error(&err) => {
                        tracing::warn!(error = %err, "openai websocket failed; falling back to SSE");
                        return fallback.open().await;
                    }
                    Err(err) => return Err(err),
                }
            }
        }

        self.stream_sse(request).await
    }

    async fn stream_sse(&self, request: ProviderRequest) -> Result<ProviderStream> {
        stream_sse_request(&self.config, &self.http, request).await
    }

    async fn stream_websocket(&self, request: ProviderRequest) -> Result<ProviderStream> {
        let cache_key = request.cache_key.as_deref().ok_or_else(|| {
            AppError::Unsupported("openai websocket mode requires a cache key".into())
        })?;
        let bearer = self.config.credential.bearer(&self.http).await?;
        let is_oauth = bearer.is_oauth;
        let session = self.websocket_session(cache_key).await;
        let mut session = session.lock_owned().await;

        session.expire_if_needed();
        refresh_websocket_session(&mut session).await;
        if session.socket.is_none() {
            session.reset_connection();
        }
        if session.model_name.as_deref() != Some(request.model.name.as_str()) {
            session.reset_chain();
            session.model_name = Some(request.model.name.clone());
        }

        let delta_start = session
            .previous_response_id
            .as_deref()
            .filter(|_| session.covered_message_count <= request.transcript.len())
            .map(|previous_response_id| (session.covered_message_count, previous_response_id));

        let (input, previous_response_id) = match delta_start {
            Some((start, previous_response_id)) => {
                (&request.transcript[start..], Some(previous_response_id))
            }
            None => {
                session.reset_chain();
                (&request.transcript[..], None)
            }
        };

        let body = build_responses_request(
            &request,
            input,
            previous_response_id,
            is_oauth,
            Some(false),
            Some(true),
        )?;
        let payload = websocket_create_payload(&body)?;
        let connect = self.websocket_connect_spec(&bearer)?;

        if let Err(err) = send_websocket_payload(&mut session, &connect, &payload).await {
            session.disable_websocket();
            return Err(err);
        }

        Ok(websocket_provider_stream(
            session,
            request.model.name.to_string(),
            request.transcript.len(),
            connect,
            payload,
        ))
    }

    fn websocket_connect_spec(&self, bearer: &BearerToken) -> Result<WebSocketConnectSpec> {
        Ok(WebSocketConnectSpec {
            url: websocket_url(if bearer.is_oauth {
                &self.config.codex_base_url
            } else {
                &self.config.api_base_url
            })?,
            token: bearer.token.clone(),
            account_id: bearer.account_id.clone(),
            is_oauth: bearer.is_oauth,
        })
    }
}

async fn connect_websocket(connect: &WebSocketConnectSpec) -> Result<ResponsesWsStream> {
    let mut request = connect
        .url
        .clone()
        .into_client_request()
        .map_err(|err| AppError::Network(err.to_string()))?;
    let headers = request.headers_mut();
    headers.insert(
        "authorization",
        HeaderValue::from_str(&format!("Bearer {}", connect.token))
            .map_err(|err| AppError::Auth(err.to_string()))?,
    );
    headers.insert(
        "user-agent",
        HeaderValue::from_str(USER_AGENT).map_err(|err| AppError::Network(err.to_string()))?,
    );
    if connect.is_oauth {
        headers.insert(
            "openai-beta",
            HeaderValue::from_static("responses=experimental"),
        );
        if let Some(account_id) = &connect.account_id {
            headers.insert(
                "chatgpt-account-id",
                HeaderValue::from_str(account_id).map_err(|err| AppError::Auth(err.to_string()))?,
            );
        }
    }

    let (socket, _) = connect_async(request)
        .await
        .map_err(|err| AppError::Network(format!("openai websocket connect failed: {err}")))?;
    Ok(socket)
}

async fn send_websocket_payload(
    session: &mut ResponsesWsSession,
    connect: &WebSocketConnectSpec,
    payload: &str,
) -> Result<()> {
    let mut last_connect_error = None;

    for attempt in 0..=1 {
        if session.socket.is_none() {
            match connect_websocket(connect).await {
                Ok(socket) => {
                    session.socket = Some(socket);
                    session.connected_at = Some(Instant::now());
                }
                Err(err) if attempt == 0 => {
                    session.reset_socket();
                    last_connect_error = Some(err);
                    continue;
                }
                Err(err) => return Err(err),
            }
        }

        let Some(socket) = session.socket.as_mut() else {
            continue;
        };

        match socket.send(Message::Text(payload.to_string().into())).await {
            Ok(()) => return Ok(()),
            Err(err) if attempt == 0 => {
                tracing::warn!(error = %err, "openai websocket send failed; reconnecting once");
                session.reset_socket();
            }
            Err(err) => {
                session.reset_connection();
                return Err(AppError::Network(format!(
                    "openai websocket send failed after reconnect: {err}"
                )));
            }
        }
    }

    Err(last_connect_error
        .unwrap_or_else(|| AppError::Network("openai websocket unavailable".into())))
}

async fn refresh_websocket_session(session: &mut ResponsesWsSession) {
    for _ in 0..8 {
        let Some(next) = session
            .socket
            .as_mut()
            .and_then(|socket| socket.next().now_or_never())
        else {
            return;
        };

        match next {
            Some(Ok(Message::Ping(payload))) => {
                let Some(socket) = session.socket.as_mut() else {
                    return;
                };
                if socket.send(Message::Pong(payload)).await.is_err() {
                    session.reset_connection();
                    return;
                }
            }
            Some(Ok(Message::Pong(_))) => {}
            Some(Ok(Message::Close(_))) | Some(Err(_)) | None => {
                session.reset_connection();
                return;
            }
            Some(Ok(Message::Text(_))) | Some(Ok(Message::Binary(_))) => {
                // A valid response can still leave a trailing data frame on the
                // socket after the terminal event has been consumed. Treat those
                // as stale frames to drain instead of throwing away the whole
                // WebSocket session, otherwise every following OpenAI turn pays
                // a fresh reconnect/handshake penalty.
                continue;
            }
            Some(Ok(_)) => {}
        }
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn capabilities(&self, model: &ModelRef) -> Option<ModelCapabilities> {
        if model.provider != "openai" {
            return None;
        }
        Some(model_info::capabilities(model))
    }

    async fn estimate_tokens(&self, request: ProviderRequest) -> Result<TokenEstimate> {
        if request.model.provider != "openai" {
            return Err(AppError::Unsupported(format!(
                "openai provider cannot count model provider {}",
                request.model.provider
            )));
        }

        let is_oauth = self.config.credential.is_oauth();
        if is_oauth {
            return Ok(TokenEstimate {
                input_tokens: rough_token_estimate(&request),
                exact: false,
            });
        }

        let instructions = request
            .system_prompt
            .as_deref()
            .filter(|value| !value.trim().is_empty());

        let body = wire::InputTokensRequest {
            model: &request.model.name,
            instructions,
            input: to_input_items(&request.transcript, !is_oauth)?,
            tools: request.tools.iter().map(to_wire_tool).collect(),
        };

        let response = self
            .post("/responses/input_tokens")
            .await?
            .header("accept", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|err| AppError::Network(err.to_string()))?;

        if !response.status().is_success() {
            return Err(read_http_error(response).await);
        }

        let counted: wire::InputTokensResponse = response
            .json()
            .await
            .map_err(|err| AppError::Decode(err.to_string()))?;
        Ok(TokenEstimate {
            input_tokens: counted.input_tokens,
            exact: true,
        })
    }

    async fn stream(&self, request: ProviderRequest) -> Result<ProviderStream> {
        if request.model.provider != "openai" {
            return Err(AppError::Unsupported(format!(
                "openai provider cannot run model provider {}",
                request.model.provider
            )));
        }

        self.stream_responses(request).await
    }
}

struct WsStreamState {
    session: OwnedMutexGuard<ResponsesWsSession>,
    default_model: String,
    connect: WebSocketConnectSpec,
    payload: String,
    parser: EventParser,
    pending: Vec<StreamEvent>,
    response_id: Option<String>,
    request_message_count: usize,
    emitted: bool,
    retry_available: bool,
    finished: bool,
    invalidated: bool,
}

impl WsStreamState {
    fn new(
        session: OwnedMutexGuard<ResponsesWsSession>,
        default_model: String,
        request_message_count: usize,
        connect: WebSocketConnectSpec,
        payload: String,
    ) -> Self {
        Self {
            session,
            default_model: default_model.clone(),
            connect,
            payload,
            parser: EventParser::new(default_model),
            pending: Vec::new(),
            response_id: None,
            request_message_count,
            emitted: false,
            retry_available: true,
            finished: false,
            invalidated: false,
        }
    }

    fn invalidate(&mut self) {
        self.session.reset_connection();
        self.invalidated = true;
        self.finished = true;
    }

    fn fallback_to_sse(&mut self) {
        self.session.disable_websocket();
        self.invalidated = true;
        self.finished = true;
    }

    fn complete(&mut self) {
        self.session.model_name = Some(self.default_model.clone());
        if let Some(response_id) = self.response_id.clone() {
            self.session.previous_response_id = Some(response_id);
            self.session.covered_message_count = self.request_message_count + 1;
        } else {
            self.session.previous_response_id = None;
            self.session.covered_message_count = 0;
        }
        self.finished = true;
    }

    async fn retry_before_start(&mut self) -> Result<bool> {
        if self.emitted || !self.retry_available {
            return Ok(false);
        }

        self.retry_available = false;
        self.session.reset_socket();
        self.parser = EventParser::new(self.default_model.clone());
        self.pending.clear();
        self.response_id = None;
        self.finished = false;
        self.invalidated = false;
        send_websocket_payload(&mut self.session, &self.connect, &self.payload).await?;
        Ok(true)
    }

    fn push_event(&mut self, event: Value) -> Result<()> {
        if let Some(response_id) = event_response_id(&event) {
            self.response_id = Some(response_id.to_string());
        }
        let terminal = is_terminal_response_event(&event);
        let mut produced = self.parser.push(event)?;
        if terminal {
            self.complete();
        }
        produced.reverse();
        self.pending.extend(produced);
        Ok(())
    }
}

impl Drop for WsStreamState {
    fn drop(&mut self) {
        if !self.finished || self.invalidated {
            self.session.reset_connection();
        }
    }
}

fn websocket_provider_stream(
    state: OwnedMutexGuard<ResponsesWsSession>,
    default_model: String,
    request_message_count: usize,
    connect: WebSocketConnectSpec,
    payload: String,
) -> ProviderStream {
    let state = WsStreamState::new(
        state,
        default_model,
        request_message_count,
        connect,
        payload,
    );
    stream::unfold(state, |mut state| async move {
        loop {
            if let Some(next) = state.pending.pop() {
                state.emitted = true;
                return Some((Ok(next), state));
            }
            if state.finished {
                return None;
            }

            let Some(socket) = state.session.socket.as_mut() else {
                let err = AppError::Network("openai websocket disconnected".into());
                match state.retry_before_start().await {
                    Ok(true) => continue,
                    Ok(false) => {
                        state.fallback_to_sse();
                        return Some((Err(err), state));
                    }
                    Err(retry_err) => {
                        state.fallback_to_sse();
                        return Some((Err(retry_err), state));
                    }
                }
            };

            let message = socket.next().await;
            match message {
                Some(Ok(Message::Text(text))) => {
                    match serde_json::from_str::<Value>(text.as_ref()) {
                        Ok(event) => {
                            if let Err(err) = state.push_event(event) {
                                state.invalidate();
                                return Some((Err(err), state));
                            }
                        }
                        Err(err) => {
                            state.fallback_to_sse();
                            return Some((
                                Err(AppError::Decode(format!(
                                    "bad openai websocket event: {err}"
                                ))),
                                state,
                            ));
                        }
                    }
                }
                Some(Ok(Message::Binary(bytes))) => match serde_json::from_slice::<Value>(&bytes) {
                    Ok(event) => {
                        if let Err(err) = state.push_event(event) {
                            state.invalidate();
                            return Some((Err(err), state));
                        }
                    }
                    Err(err) => {
                        state.fallback_to_sse();
                        return Some((
                            Err(AppError::Decode(format!(
                                "bad openai websocket event: {err}"
                            ))),
                            state,
                        ));
                    }
                },
                Some(Ok(Message::Ping(payload))) => {
                    if let Some(socket) = state.session.socket.as_mut() {
                        if let Err(err) = socket.send(Message::Pong(payload)).await {
                            let err =
                                AppError::Network(format!("openai websocket pong failed: {err}"));
                            match state.retry_before_start().await {
                                Ok(true) => continue,
                                Ok(false) => {
                                    state.fallback_to_sse();
                                    return Some((Err(err), state));
                                }
                                Err(retry_err) => {
                                    state.fallback_to_sse();
                                    return Some((Err(retry_err), state));
                                }
                            }
                        }
                    }
                }
                Some(Ok(Message::Pong(_))) => {}
                Some(Ok(Message::Close(_))) => {
                    let err = AppError::Stream(
                        "openai websocket closed before response completed".into(),
                    );
                    match state.retry_before_start().await {
                        Ok(true) => continue,
                        Ok(false) => {
                            state.fallback_to_sse();
                            return Some((Err(err), state));
                        }
                        Err(retry_err) => {
                            state.fallback_to_sse();
                            return Some((Err(retry_err), state));
                        }
                    }
                }
                Some(Ok(_)) => {}
                Some(Err(err)) => {
                    let err = AppError::Stream(format!("openai websocket error: {err}"));
                    match state.retry_before_start().await {
                        Ok(true) => continue,
                        Ok(false) => {
                            state.fallback_to_sse();
                            return Some((Err(err), state));
                        }
                        Err(retry_err) => {
                            state.fallback_to_sse();
                            return Some((Err(retry_err), state));
                        }
                    }
                }
                None => {
                    let err =
                        AppError::Stream("openai websocket ended before response completed".into());
                    match state.retry_before_start().await {
                        Ok(true) => continue,
                        Ok(false) => {
                            state.fallback_to_sse();
                            return Some((Err(err), state));
                        }
                        Err(retry_err) => {
                            state.fallback_to_sse();
                            return Some((Err(retry_err), state));
                        }
                    }
                }
            }
        }
    })
    .boxed()
}

#[derive(Clone)]
struct SseFallbackContext {
    config: OpenAiConfig,
    http: reqwest::Client,
    request: ProviderRequest,
}

impl SseFallbackContext {
    async fn open(&self) -> Result<ProviderStream> {
        stream_sse_request(&self.config, &self.http, self.request.clone()).await
    }
}

enum TransportStreamState {
    WebSocket(ProviderStream),
    Sse(ProviderStream),
    Done,
}

fn websocket_stream_with_sse_fallback(
    stream: ProviderStream,
    fallback: SseFallbackContext,
) -> ProviderStream {
    stream::unfold(
        (TransportStreamState::WebSocket(stream), fallback, false),
        |(mut transport, fallback, mut emitted)| async move {
            loop {
                match transport {
                    TransportStreamState::WebSocket(ref mut stream) => match stream.next().await {
                        Some(Ok(event)) => {
                            emitted = true;
                            return Some((Ok(event), (transport, fallback, emitted)));
                        }
                        Some(Err(err))
                            if !emitted && should_fallback_from_websocket_error(&err) =>
                        {
                            tracing::warn!(
                                error = %err,
                                "openai websocket stream failed before output; falling back to SSE"
                            );
                            match fallback.open().await {
                                Ok(sse_stream) => {
                                    transport = TransportStreamState::Sse(sse_stream);
                                    continue;
                                }
                                Err(fallback_err) => {
                                    return Some((
                                        Err(fallback_err),
                                        (TransportStreamState::Done, fallback, emitted),
                                    ));
                                }
                            }
                        }
                        Some(Err(err)) => {
                            return Some((
                                Err(err),
                                (TransportStreamState::Done, fallback, emitted),
                            ));
                        }
                        None => return None,
                    },
                    TransportStreamState::Sse(ref mut stream) => {
                        return stream
                            .next()
                            .await
                            .map(|event| (event, (transport, fallback, true)));
                    }
                    TransportStreamState::Done => return None,
                }
            }
        },
    )
    .boxed()
}

async fn stream_sse_request(
    config: &OpenAiConfig,
    http: &reqwest::Client,
    request: ProviderRequest,
) -> Result<ProviderStream> {
    let bearer = config.credential.bearer(http).await?;
    let is_oauth = bearer.is_oauth;
    let body = build_responses_request(
        &request,
        &request.transcript,
        None,
        is_oauth,
        Some(false),
        Some(true),
    )?;
    let base_url = if bearer.is_oauth {
        &config.codex_base_url
    } else {
        &config.api_base_url
    };
    let mut builder = http
        .post(format!("{}/responses", base_url.trim_end_matches('/')))
        .header("accept", "text/event-stream")
        .header("content-type", "application/json")
        .header("authorization", format!("Bearer {}", bearer.token));

    if bearer.is_oauth {
        builder = builder.header("openai-beta", "responses=experimental");
        if let Some(account_id) = bearer.account_id {
            builder = builder.header("chatgpt-account-id", account_id);
        }
    }

    let response = builder
        .json(&body)
        .send()
        .await
        .map_err(|err| AppError::Network(err.to_string()))?;

    if !response.status().is_success() {
        return Err(read_http_error(response).await);
    }

    Ok(sse_provider_stream(
        response.bytes_stream(),
        request.model.name.clone(),
    ))
}

fn sse_provider_stream<S, E>(body: S, default_model: String) -> ProviderStream
where
    S: Stream<Item = std::result::Result<bytes::Bytes, E>> + Send + 'static,
    E: std::error::Error + Send + Sync + 'static,
{
    let source = Box::pin(body.eventsource());
    let parser = EventParser::new(default_model);

    stream::unfold(
        (source, parser, Vec::<StreamEvent>::new(), false),
        |(mut source, mut parser, mut pending, mut done)| async move {
            loop {
                if let Some(next) = pending.pop() {
                    return Some((Ok(next), (source, parser, pending, done)));
                }
                if done {
                    return None;
                }

                match source.next().await {
                    Some(Ok(event)) => {
                        let data = event.data.trim();
                        if data == "[DONE]" {
                            done = true;
                            continue;
                        }

                        let event = match serde_json::from_str::<Value>(&event.data) {
                            Ok(event) => event,
                            Err(err) => {
                                return Some((
                                    Err(AppError::Decode(format!("bad openai SSE event: {err}"))),
                                    (source, parser, pending, true),
                                ));
                            }
                        };
                        let terminal = is_terminal_response_event(&event);
                        match parser.push(event) {
                            Ok(mut produced) => {
                                if terminal {
                                    done = true;
                                }
                                produced.reverse();
                                pending.extend(produced);
                            }
                            Err(err) => return Some((Err(err), (source, parser, pending, true))),
                        }
                    }
                    Some(Err(err)) => {
                        return Some((
                            Err(AppError::Stream(format!("openai SSE error: {err}"))),
                            (source, parser, pending, true),
                        ));
                    }
                    None => {
                        let mut produced = parser.finish_if_needed();
                        if produced.is_empty() {
                            return None;
                        }
                        produced.reverse();
                        pending.extend(produced);
                        done = true;
                    }
                }
            }
        },
    )
    .boxed()
}

fn should_fallback_from_websocket_error(err: &AppError) -> bool {
    match err {
        AppError::Network(_) | AppError::Stream(_) => true,
        AppError::Decode(message) => message.contains("websocket"),
        AppError::Auth(_)
        | AppError::InvalidRequest(_)
        | AppError::RateLimit(_)
        | AppError::ContextLength(_)
        | AppError::Unsupported(_)
        | AppError::Provider(_) => false,
    }
}

fn build_responses_request<'a>(
    request: &'a ProviderRequest,
    transcript: &'a [ChatMessage],
    previous_response_id: Option<&'a str>,
    is_oauth: bool,
    store: Option<bool>,
    stream: Option<bool>,
) -> Result<wire::ResponsesRequest<'a>> {
    let caps = model_info::capabilities(&request.model);
    Ok(wire::ResponsesRequest {
        model: &request.model.name,
        instructions: response_instructions(request, is_oauth),
        previous_response_id,
        input: to_input_items(transcript, !is_oauth && previous_response_id.is_none())?,
        tools: request.tools.iter().map(to_wire_tool).collect(),
        prompt_cache_key: (!is_oauth)
            .then_some(request.cache_key.as_deref())
            .flatten(),
        max_output_tokens: (!is_oauth).then_some(
            request
                .max_output_tokens
                .unwrap_or(caps.max_output_tokens)
                .min(caps.max_output_tokens),
        ),
        reasoning: effort_to_reasoning(request.effective_effort()),
        temperature: request.temperature,
        store,
        stream,
        generate: None,
    })
}

fn response_instructions(request: &ProviderRequest, is_oauth: bool) -> Option<&str> {
    let instructions = request
        .system_prompt
        .as_deref()
        .filter(|value| !value.trim().is_empty());
    if is_oauth {
        Some(instructions.unwrap_or(FALLBACK_INSTRUCTIONS))
    } else {
        instructions
    }
}

fn websocket_create_payload(body: &wire::ResponsesRequest<'_>) -> Result<String> {
    let mut value = serde_json::to_value(body)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| AppError::Decode("openai websocket body was not an object".into()))?;
    object.insert("type".into(), Value::String("response.create".into()));
    Ok(value.to_string())
}

fn websocket_url(base_url: &str) -> Result<String> {
    let base_url = base_url.trim_end_matches('/');
    if let Some(rest) = base_url.strip_prefix("https://") {
        Ok(format!("wss://{rest}/responses"))
    } else if let Some(rest) = base_url.strip_prefix("http://") {
        Ok(format!("ws://{rest}/responses"))
    } else {
        Err(AppError::InvalidRequest(format!(
            "unsupported openai base url for websocket: {base_url}"
        )))
    }
}

fn event_response_id(event: &Value) -> Option<&str> {
    event
        .get("response")
        .and_then(|response| response.get("id"))
        .and_then(Value::as_str)
        .or_else(|| event.get("response_id").and_then(Value::as_str))
}

fn is_terminal_response_event(event: &Value) -> bool {
    matches!(
        event.get("type").and_then(Value::as_str),
        Some("response.completed" | "response.incomplete")
    )
}

fn effort_to_reasoning(effort: Option<Effort>) -> Option<wire::ReasoningConfig> {
    Some(wire::ReasoningConfig {
        effort: match effort.unwrap_or(Effort::Medium) {
            Effort::None => "none",
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::Xhigh | Effort::Max => "xhigh",
        },
        summary: "auto",
    })
}

fn to_wire_tool(tool: &ToolDescriptor) -> wire::WireTool<'_> {
    wire::WireTool {
        kind: "function",
        name: &tool.name,
        description: &tool.description,
        parameters: &tool.input_schema,
    }
}

fn to_input_items(
    transcript: &[ChatMessage],
    include_response_items: bool,
) -> Result<Vec<wire::InputItem<'_>>> {
    let mut items = Vec::new();
    for message in transcript {
        let mut content = Vec::new();
        for part in &message.parts {
            if part_is_ui_only(part) {
                continue;
            }
            match part {
                Part::Text { text, .. } => {
                    if text.is_empty() {
                        continue;
                    }
                    let item = match message.role {
                        Role::User => wire::InputContent::InputText { text },
                        Role::Assistant => wire::InputContent::OutputText { text },
                    };
                    content.push(item);
                }
                Part::Image {
                    media_type, data, ..
                } => {
                    if matches!(message.role, Role::User) {
                        content.push(wire::InputContent::InputImage {
                            image_url: format!("data:{media_type};base64,{data}"),
                        });
                    }
                }
                Part::Thinking { meta, .. } => {
                    flush_message_content(message.role, &mut content, &mut items);
                    if include_response_items {
                        if let Some(item) = openai_response_item(meta) {
                            items.push(wire::InputItem::ResponseItem(item));
                        }
                    }
                }
                Part::ToolCall {
                    id,
                    name,
                    input,
                    meta: _,
                } => {
                    flush_message_content(message.role, &mut content, &mut items);
                    items.push(wire::InputItem::FunctionCall {
                        kind: "function_call",
                        call_id: id,
                        name,
                        arguments: input.to_string(),
                    });
                }
                Part::ToolResult {
                    tool_call_id,
                    content: text,
                    images,
                    ..
                } => {
                    flush_message_content(message.role, &mut content, &mut items);
                    let inline_images = images
                        .iter()
                        .filter(|image| !image.data.trim().is_empty())
                        .collect::<Vec<_>>();
                    let output = if inline_images.is_empty() {
                        wire::ToolOutput::Text(text)
                    } else {
                        let mut blocks = Vec::new();
                        if !text.trim().is_empty() {
                            blocks.push(wire::ToolOutputBlock::InputText { text });
                        }
                        blocks.extend(inline_images.into_iter().map(|image| {
                            wire::ToolOutputBlock::InputImage {
                                image_url: format!(
                                    "data:{};base64,{}",
                                    image.media_type, image.data
                                ),
                            }
                        }));
                        wire::ToolOutput::Blocks(blocks)
                    };
                    items.push(wire::InputItem::FunctionCallOutput {
                        kind: "function_call_output",
                        call_id: tool_call_id,
                        output,
                    });
                }
            }
        }

        flush_message_content(message.role, &mut content, &mut items);
    }

    Ok(items)
}

fn flush_message_content<'a>(
    role: Role,
    content: &mut Vec<wire::InputContent<'a>>,
    items: &mut Vec<wire::InputItem<'a>>,
) {
    if content.is_empty() {
        return;
    }
    items.push(wire::InputItem::Message {
        role: match role {
            Role::User => "user",
            Role::Assistant => "assistant",
        },
        content: std::mem::take(content),
    });
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

fn openai_response_item(meta: &Option<serde_json::Value>) -> Option<&serde_json::Value> {
    let meta = meta.as_ref()?;
    if meta.get("provider").and_then(|value| value.as_str()) != Some("openai") {
        return None;
    }
    meta.get("item")
}

fn rough_token_estimate(request: &ProviderRequest) -> u32 {
    let mut chars: usize = 0;
    if let Some(system) = &request.system_prompt {
        chars += system.chars().count();
    }
    for message in &request.transcript {
        for part in &message.parts {
            if part_is_ui_only(part) {
                continue;
            }
            match part {
                Part::Text { text, .. } | Part::Thinking { text, .. } => {
                    chars += text.chars().count()
                }
                Part::Image { data, .. } => {
                    chars += if data.trim().is_empty() { 0 } else { 4_000 };
                }
                Part::ToolCall { name, input, .. } => {
                    chars += name.chars().count();
                    chars += input.to_string().chars().count();
                }
                Part::ToolResult {
                    content, images, ..
                } => {
                    chars += content.chars().count();
                    chars += images
                        .iter()
                        .filter(|image| !image.data.trim().is_empty())
                        .count()
                        * 4_000;
                }
            }
        }
    }
    for tool in &request.tools {
        chars += tool.name.chars().count();
        chars += tool.description.chars().count();
        chars += tool.input_schema.to_string().chars().count();
    }
    ((chars / 4).max(1)).min(u32::MAX as usize) as u32
}

async fn read_http_error(response: reqwest::Response) -> AppError {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    let parsed: std::result::Result<wire::ApiErrorEnvelope, _> = serde_json::from_str(&body);
    let message = parsed
        .ok()
        .and_then(|payload| {
            let code = payload.error.code.unwrap_or_default();
            let kind = payload.error.kind.trim();
            let error_message = payload.error.message.trim();
            if code.is_empty() && kind.is_empty() && error_message.is_empty() {
                None
            } else if code.is_empty() {
                Some(format!("{kind}: {error_message}").trim().to_string())
            } else {
                Some(
                    format!("{kind} ({code}): {error_message}")
                        .trim()
                        .to_string(),
                )
            }
        })
        .filter(|message| !message.trim().is_empty())
        .unwrap_or_else(|| {
            let body = body.trim();
            if body.is_empty() {
                format!("HTTP {status}")
            } else {
                body.to_string()
            }
        });

    if status == reqwest::StatusCode::UNAUTHORIZED {
        AppError::Auth(message)
    } else if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        AppError::RateLimit(message)
    } else if status.is_client_error() {
        let lower = message.to_ascii_lowercase();
        if lower.contains("context") || lower.contains("too long") {
            AppError::ContextLength(message)
        } else {
            AppError::InvalidRequest(message)
        }
    } else {
        AppError::Provider(format!("HTTP {status}: {message}"))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use sinew_core::{ChatMessage, ModelRef, Part, ProviderRequest, Role, ToolResultImage};

    use super::{build_responses_request, to_input_items, websocket_create_payload, websocket_url};

    #[test]
    fn image_tool_result_uses_responses_input_block_types() {
        let transcript = vec![ChatMessage {
            role: Role::User,
            parts: vec![Part::ToolResult {
                tool_call_id: "call_read".into(),
                content: "path: image.png\n\n[Image attached visually.]".into(),
                images: vec![ToolResultImage {
                    media_type: "image/png".into(),
                    data: "iVBORw0KGgo=".into(),
                    path: None,
                }],
                is_error: false,
                meta: None,
            }],
        }];

        let items = to_input_items(&transcript, true).expect("tool result should serialize");
        let value = serde_json::to_value(items).expect("items should be json");

        assert_eq!(
            value,
            json!([
                {
                    "type": "function_call_output",
                    "call_id": "call_read",
                    "output": [
                        {
                            "type": "input_text",
                            "text": "path: image.png\n\n[Image attached visually.]"
                        },
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,iVBORw0KGgo="
                        }
                    ]
                }
            ])
        );
    }

    #[test]
    fn websocket_payload_uses_create_event_with_stream_flag() {
        let request = ProviderRequest::new(
            ModelRef::new("openai", "gpt-5.5"),
            vec![ChatMessage::user_text("hello")],
        )
        .with_system("be helpful")
        .with_cache_key("conversation-1");
        let body = build_responses_request(
            &request,
            &request.transcript,
            None,
            false,
            Some(false),
            Some(true),
        )
        .expect("body should serialize");
        let payload = websocket_create_payload(&body).expect("payload should serialize");
        let value: serde_json::Value =
            serde_json::from_str(&payload).expect("payload should be json");

        assert_eq!(value["type"], "response.create");
        assert_eq!(value["store"], false);
        assert_eq!(value["stream"], true);
        assert_eq!(value["input"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn websocket_delta_payload_sends_only_new_items() {
        let transcript = vec![
            ChatMessage::user_text("old question"),
            ChatMessage::assistant_text("old answer"),
            ChatMessage::user_text("new question"),
        ];
        let request = ProviderRequest::new(ModelRef::new("openai", "gpt-5.5"), transcript)
            .with_cache_key("c");
        let body = build_responses_request(
            &request,
            &request.transcript[2..],
            Some("resp_123"),
            false,
            Some(false),
            None,
        )
        .expect("body should serialize");
        let value = serde_json::to_value(&body).expect("body should be json");

        assert_eq!(value["previous_response_id"], "resp_123");
        assert_eq!(value["input"].as_array().unwrap().len(), 1);
        assert_eq!(value["input"][0]["content"][0]["text"], "new question");
    }

    #[test]
    fn websocket_helpers_match_openai_shapes() {
        assert_eq!(
            websocket_url("https://api.openai.com/v1").unwrap(),
            "wss://api.openai.com/v1/responses"
        );
    }
}
