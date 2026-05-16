use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use directories::BaseDirs;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_core::ToolDescriptor;

use crate::tool_run::ToolRunResult;

const SKILL_TOOL_NAME: &str = "skill";
const SKILL_FILE_NAME: &str = "SKILL.md";

#[derive(Debug, Clone)]
pub struct SkillTool {
    workspace_root: PathBuf,
    settings: SkillSettings,
}

#[derive(Debug, Clone)]
struct SkillEntry {
    name: String,
    path: PathBuf,
}

impl SkillTool {
    pub fn new(workspace_root: impl Into<PathBuf>) -> Self {
        Self {
            workspace_root: workspace_root.into(),
            settings: SkillSettings::default(),
        }
    }

    pub fn with_settings(workspace_root: impl Into<PathBuf>, settings: SkillSettings) -> Self {
        Self {
            workspace_root: workspace_root.into(),
            settings: settings.normalized(),
        }
    }

    pub fn descriptor(&self) -> Option<ToolDescriptor> {
        let skills = self.discover();
        if skills.is_empty() {
            return None;
        }

        Some(ToolDescriptor {
            name: SKILL_TOOL_NAME.into(),
            description: format!(
                "Load one skill by name before using it. Available skills:\n{}",
                skills
                    .iter()
                    .map(|skill| format!("- {}", skill.name))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact skill name from the available skills list."
                    }
                },
                "required": ["name"],
                "additionalProperties": false
            }),
        })
    }

    pub async fn run(&self, input: Value) -> ToolRunResult {
        match self.load(input) {
            Ok(output) => ToolRunResult::ok(output, Vec::new()),
            Err(err) => ToolRunResult::err(err.to_string(), Vec::new()),
        }
    }

    fn load(&self, input: Value) -> Result<String> {
        let parsed: SkillInput = serde_json::from_value(input)
            .map_err(|err| anyhow::anyhow!("invalid skill input: {err}"))?;
        let name = parsed.name.trim();
        if name.is_empty() {
            bail!("skill name is required");
        }

        let skills = self.discover();
        let Some(skill) = skills.iter().find(|skill| skill.name == name) else {
            let available = skills
                .iter()
                .map(|skill| skill.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            bail!("unknown skill `{name}`. Available skills: {available}");
        };

        let content = fs::read_to_string(&skill.path)
            .with_context(|| format!("unable to read skill `{name}`"))?;
        Ok(format!(
            "Loaded skill `{name}`:\n\n<skill name=\"{name}\">\n{}\n</skill>",
            content.trim()
        ))
    }

    fn discover(&self) -> Vec<SkillEntry> {
        let mut roots = vec![
            self.workspace_root.join(".agents/skills"),
            self.workspace_root.join(".sinew/skills"),
        ];
        if let Some(base_dirs) = BaseDirs::new() {
            let home = base_dirs.home_dir();
            roots.push(home.join(".agents/skills"));
            roots.push(home.join(".sinew/skills"));
        }
        roots.extend(builtin_skill_roots());

        let mut seen = HashSet::new();
        let mut skills = Vec::new();
        for root in roots {
            for skill in scan_skill_root(&root) {
                if self.settings.is_enabled(&skill.name) && seen.insert(skill.name.clone()) {
                    skills.push(skill);
                }
            }
        }
        skills.sort_by(|left, right| left.name.cmp(&right.name));
        skills
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillSettings {
    #[serde(default)]
    pub skills: Vec<SkillConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillConfig {
    pub name: String,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

impl SkillSettings {
    pub fn normalized(mut self) -> Self {
        let mut seen = HashSet::new();
        self.skills = self
            .skills
            .into_iter()
            .filter_map(|mut skill| {
                skill.name = skill.name.trim().to_string();
                if skill.name.is_empty() || !seen.insert(skill.name.clone()) {
                    return None;
                }
                Some(skill)
            })
            .collect();
        self
    }

    pub fn is_enabled(&self, name: &str) -> bool {
        self.skills
            .iter()
            .find(|skill| skill.name == name)
            .map(|skill| skill.enabled)
            .unwrap_or(true)
    }
}

fn scan_skill_root(root: &Path) -> Vec<SkillEntry> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };

    let mut skills = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let skill_path = path.join(SKILL_FILE_NAME);
        if !skill_path.is_file() {
            continue;
        }

        let fallback_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("skill")
            .to_string();
        let name = fs::read_to_string(&skill_path)
            .ok()
            .and_then(|content| parse_skill_name(&content))
            .unwrap_or(fallback_name);

        skills.push(SkillEntry {
            name,
            path: skill_path,
        });
    }

    skills
}

fn parse_skill_name(content: &str) -> Option<String> {
    parse_frontmatter(content)
        .remove("name")
        .filter(|value| !value.is_empty())
}

fn parse_frontmatter(content: &str) -> HashMap<String, String> {
    let mut fields = HashMap::new();
    let mut lines = content.lines();
    if lines.next().map(str::trim) != Some("---") {
        return fields;
    }
    for line in lines {
        let trimmed = line.trim();
        if trimmed == "---" {
            break;
        }
        let Some((key, value)) = trimmed.split_once(':') else {
            continue;
        };
        fields.insert(
            key.trim().to_string(),
            clean_yaml_string(value.trim()).to_string(),
        );
    }
    fields
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum SkillSource {
    Workspace,
    Global,
    Builtin,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledSkill {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub source: SkillSource,
    pub root_label: String,
    pub absolute_path: String,
    pub content: String,
    pub enabled: bool,
}

pub fn list_installed_skills(
    workspace_root: impl AsRef<Path>,
    settings: &SkillSettings,
) -> Vec<InstalledSkill> {
    let workspace_root = workspace_root.as_ref();
    let home_dir = BaseDirs::new().map(|base| base.home_dir().to_path_buf());

    let mut roots: Vec<(SkillSource, PathBuf)> = vec![
        (
            SkillSource::Workspace,
            workspace_root.join(".agents/skills"),
        ),
        (SkillSource::Workspace, workspace_root.join(".sinew/skills")),
    ];
    if let Some(home) = home_dir.as_ref() {
        roots.push((SkillSource::Global, home.join(".agents/skills")));
        roots.push((SkillSource::Global, home.join(".sinew/skills")));
    }
    for root in builtin_skill_roots() {
        roots.push((SkillSource::Builtin, root));
    }

    let mut seen = HashSet::new();
    let mut skills = Vec::new();

    for (source, root) in roots {
        let Ok(entries) = fs::read_dir(&root) else {
            continue;
        };
        let root_label = format_root_label(&root, workspace_root, home_dir.as_deref());

        for entry in entries.flatten() {
            let dir = entry.path();
            if !dir.is_dir() {
                continue;
            }
            let skill_path = dir.join(SKILL_FILE_NAME);
            if !skill_path.is_file() {
                continue;
            }

            let Ok(content) = fs::read_to_string(&skill_path) else {
                continue;
            };
            let frontmatter = parse_frontmatter(&content);
            let fallback_name = dir
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("skill")
                .to_string();
            let name = frontmatter
                .get("name")
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .unwrap_or(fallback_name);

            if !seen.insert(name.clone()) {
                continue;
            }

            let description = frontmatter
                .get("description")
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty());
            let enabled = settings.is_enabled(&name);

            skills.push(InstalledSkill {
                name,
                description,
                source,
                root_label: root_label.clone(),
                absolute_path: skill_path.display().to_string(),
                content,
                enabled,
            });
        }
    }

    skills.sort_by(|left, right| left.name.cmp(&right.name));
    skills
}

fn format_root_label(root: &Path, workspace_root: &Path, home_dir: Option<&Path>) -> String {
    if let Ok(rel) = root.strip_prefix(workspace_root) {
        return rel.display().to_string();
    }
    if let Some(home) = home_dir {
        if let Ok(rel) = root.strip_prefix(home) {
            return format!("~/{}", rel.display());
        }
    }
    root.display().to_string()
}

fn builtin_skill_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    let manifest_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(repo_root) = manifest_root.parent().and_then(|path| path.parent()) {
        roots.push(repo_root.join("resources/skills"));
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            roots.push(exe_dir.join("resources/skills"));
            roots.push(exe_dir.join("skills"));
            roots.push(exe_dir.join("../Resources/resources/skills"));
            roots.push(exe_dir.join("../Resources/skills"));
            roots.push(exe_dir.join("../resources/skills"));
            roots.push(exe_dir.join("../skills"));
        }
    }

    let mut seen = HashSet::new();
    roots
        .into_iter()
        .filter(|root| root.is_dir())
        .filter(|root| seen.insert(root.clone()))
        .collect()
}

fn clean_yaml_string(value: &str) -> &str {
    value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .or_else(|| {
            value
                .strip_prefix('\'')
                .and_then(|value| value.strip_suffix('\''))
        })
        .unwrap_or(value)
}

fn default_enabled() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct SkillInput {
    name: String,
}
