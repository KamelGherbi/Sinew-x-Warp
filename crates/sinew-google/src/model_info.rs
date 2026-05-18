use sinew_core::{EffortMode, ModelCapabilities, ModelRef};

pub const MODEL_ID: &str = "gemini-3.1-pro-preview";
pub const MODEL_WINDOW: u32 = 1_048_576;
pub const MODEL_MAX_OUTPUT: u32 = 65_536;

pub fn capabilities(model: &ModelRef) -> Option<ModelCapabilities> {
    if model.name != MODEL_ID {
        return None;
    }
    Some(ModelCapabilities {
        model: model.clone(),
        context_window: MODEL_WINDOW,
        preferred_window: 950_000,
        max_output_tokens: MODEL_MAX_OUTPUT,
        supports_thinking: true,
        visible_thinking: true,
        supports_tools: true,
        supports_images: true,
        effort_mode: EffortMode::Tier,
    })
}
