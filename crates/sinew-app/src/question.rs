use serde::Deserialize;
use serde_json::{json, Value};
use sinew_core::ToolDescriptor;

use crate::tool_run::ToolRunResult;

#[derive(Debug, Clone, Default)]
pub struct QuestionTool;

impl QuestionTool {
    pub fn new() -> Self {
        Self
    }

    pub fn descriptor(&self) -> ToolDescriptor {
        let question_schema = json!({
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question shown to the user."
                },
                "type": {
                    "type": "string",
                    "enum": ["single_choice", "multiple_choice"],
                    "description": "single_choice lets the user select one option. multiple_choice lets the user select several options."
                },
                "options": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Stable optional option id."
                            },
                            "label": {
                                "type": "string",
                                "description": "Visible option label."
                            },
                            "description": {
                                "type": "string",
                                "description": "Optional short supporting text for the option."
                            }
                        },
                        "required": ["label"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["question", "type", "options"],
            "additionalProperties": false
        });
        ToolDescriptor {
            name: "Question".into(),
            description: "Ask the user one or more questions in the chat UI. Use this when you need choices, preferences, or clarifications before continuing. You may pass either a single question with question/type/options, or multiple independent questions with questions[]. After calling it, wait for the user's next message.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question shown to the user."
                    },
                    "type": {
                        "type": "string",
                        "enum": ["single_choice", "multiple_choice"],
                        "description": "single_choice lets the user select one option. multiple_choice lets the user select several options."
                    },
                    "options": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Stable optional option id."
                                },
                                "label": {
                                    "type": "string",
                                    "description": "Visible option label."
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Optional short supporting text for the option."
                                }
                            },
                            "required": ["label"],
                            "additionalProperties": false
                        }
                    },
                    "questions": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 6,
                        "description": "Multiple independent questions to show in one UI block. Each question can have its own type and options.",
                        "items": question_schema
                    }
                },
                "additionalProperties": false
            }),
        }
    }

    pub async fn run(&self, input: Value) -> ToolRunResult {
        let parsed: QuestionInput = match serde_json::from_value(input) {
            Ok(value) => value,
            Err(err) => {
                return ToolRunResult::err(format!("invalid Question input: {err}"), Vec::new())
            }
        };

        let questions = match parsed.into_questions() {
            Ok(questions) => questions,
            Err(err) => return ToolRunResult::err(err, Vec::new()),
        };

        for question in &questions {
            if question.question.trim().is_empty() {
                return ToolRunResult::err("question is required", Vec::new());
            }
            if question.options.is_empty() {
                return ToolRunResult::err("at least one option is required", Vec::new());
            }
            for option in &question.options {
                if option.label.trim().is_empty() {
                    return ToolRunResult::err("option labels cannot be empty", Vec::new());
                }
            }
        }

        if questions.len() == 1 {
            let question = &questions[0];
            let option_count = question.options.len();
            return ToolRunResult::ok(
                format!(
                    "Question shown to the user ({}, {option_count} option{}): {}\nWait for the user's next message before continuing.",
                    question.kind.label(),
                    if option_count == 1 { "" } else { "s" },
                    question.question.trim(),
                ),
                Vec::new(),
            );
        }

        ToolRunResult::ok(
            format!(
                "{} questions shown to the user:\n{}\nWait for the user's next message before continuing.",
                questions.len(),
                questions
                    .iter()
                    .enumerate()
                    .map(|(idx, question)| format!(
                        "{}. {} ({}, {} option{})",
                        idx + 1,
                        question.question.trim(),
                        question.kind.label(),
                        question.options.len(),
                        if question.options.len() == 1 { "" } else { "s" },
                    ))
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            Vec::new(),
        )
    }
}

#[derive(Debug, Deserialize)]
struct QuestionInput {
    question: Option<String>,
    #[serde(rename = "type")]
    kind: Option<QuestionKind>,
    #[serde(default)]
    options: Vec<QuestionOption>,
    #[serde(default)]
    questions: Vec<QuestionPrompt>,
}

impl QuestionInput {
    fn into_questions(self) -> Result<Vec<QuestionPrompt>, String> {
        if !self.questions.is_empty() {
            return Ok(self.questions);
        }
        let question = self
            .question
            .ok_or_else(|| "question is required".to_string())?;
        let kind = self.kind.ok_or_else(|| "type is required".to_string())?;
        Ok(vec![QuestionPrompt {
            question,
            kind,
            options: self.options,
        }])
    }
}

#[derive(Debug, Deserialize)]
struct QuestionPrompt {
    question: String,
    #[serde(rename = "type")]
    kind: QuestionKind,
    options: Vec<QuestionOption>,
}

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum QuestionKind {
    SingleChoice,
    MultipleChoice,
}

impl QuestionKind {
    fn label(self) -> &'static str {
        match self {
            Self::SingleChoice => "single choice",
            Self::MultipleChoice => "multiple choice",
        }
    }
}

#[derive(Debug, Deserialize)]
struct QuestionOption {
    #[allow(dead_code)]
    id: Option<String>,
    label: String,
    #[allow(dead_code)]
    description: Option<String>,
}
