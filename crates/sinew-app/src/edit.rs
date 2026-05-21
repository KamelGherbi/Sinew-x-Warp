use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use sinew_core::ToolDescriptor;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{
    read::{fingerprint_path, ReadFingerprint},
    tool_run::{diff_snapshots, snapshot_workspace_paths, ToolRunResult},
    workspace::normalize_workspace_relative_path,
};

const MAX_EDIT_COUNT: usize = 128;
const MAX_TOTAL_CONTENT_BYTES: usize = 2 * 1024 * 1024;

const EDIT_FILE_DESCRIPTION: &str = r#"Edit existing workspace text files by line number. This tool only edits files; it does not create, delete, rename, or move files.

Input is an array of edits. Each edit has:
- path: relative to the workspace root, or an absolute path inside the workspace.
- lines: 1-based line numbers from the last successful read. "10-15" means lines 10 through 15 inclusive; "10" means line 10.
- mode: required. One of "replace", "insert_before", or "insert_after".
- content: the new text. For replace, content may be empty to remove the selected lines. For insert modes, content must be non-empty.

Rules:
- You must read a file successfully before editing it. edit_file refuses to write if the file changed since that read.
- For multiple edits in the same file, line numbers always refer to the original file as last read; edits are applied from bottom to top automatically.
- Prefer one edit_file call with multiple edits for multiple changes in the same file.
- For appending, use mode "insert_after" with the last line number shown by read. For prepending, use mode "insert_before" with line "1".
- Overlapping edits are rejected. If edits touch the same area, combine them into one replace edit.
"#;

#[derive(Debug, Clone)]
pub struct EditFileTool {
    workspace_root: PathBuf,
    write_lock: Option<Arc<Semaphore>>,
}

impl EditFileTool {
    pub fn new(workspace_root: impl Into<PathBuf>) -> Self {
        Self {
            workspace_root: workspace_root.into(),
            write_lock: None,
        }
    }

    pub fn with_workspace_write_lock(mut self, write_lock: Arc<Semaphore>) -> Self {
        self.write_lock = Some(write_lock);
        self
    }

    pub fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            name: "edit_file".into(),
            description: EDIT_FILE_DESCRIPTION.into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_EDIT_COUNT,
                        "description": "The file edits to apply in one operation.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "File path to edit. Relative paths are resolved from the workspace root; absolute paths must be inside the workspace."
                                },
                                "lines": {
                                    "type": "string",
                                    "description": "1-based inclusive line number or range, e.g. '7' or '4-34'."
                                },
                                "mode": {
                                    "type": "string",
                                    "enum": ["replace", "insert_before", "insert_after"],
                                    "description": "Required edit mode."
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Replacement or inserted content."
                                }
                            },
                            "required": ["path", "lines", "mode", "content"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["edits"],
                "additionalProperties": false
            }),
        }
    }

    pub async fn run(
        &self,
        input: Value,
        read_fingerprints: &HashMap<String, ReadFingerprint>,
    ) -> ToolRunResult {
        match self.edit(input, read_fingerprints).await {
            Ok(output) => output,
            Err(err) => ToolRunResult::err(err.to_string(), Vec::new()),
        }
    }

    async fn edit(
        &self,
        input: Value,
        read_fingerprints: &HashMap<String, ReadFingerprint>,
    ) -> Result<ToolRunResult> {
        let parsed: EditFileInput = serde_json::from_value(input)
            .map_err(|err| anyhow::anyhow!("invalid edit_file input: {err}"))?;
        if parsed.edits.is_empty() {
            bail!("edits must contain at least one edit");
        }
        if parsed.edits.len() > MAX_EDIT_COUNT {
            bail!("too many edits in one call; maximum is {MAX_EDIT_COUNT}");
        }
        let total_content_bytes = parsed
            .edits
            .iter()
            .map(|edit| edit.content.len())
            .sum::<usize>();
        if total_content_bytes > MAX_TOTAL_CONTENT_BYTES {
            bail!("edit content is too large to apply safely");
        }

        let resolved = parsed
            .edits
            .into_iter()
            .enumerate()
            .map(|(index, edit)| self.resolve_edit(index, edit))
            .collect::<Result<Vec<_>>>()?;
        let affected_paths = resolved
            .iter()
            .map(|edit| edit.relative_path.clone())
            .collect::<Vec<_>>();

        let _write_permit = self.acquire_write_permit().await?;
        let mut grouped = group_edits(resolved);
        let mut summaries = Vec::new();
        let mut writes = Vec::new();

        for group in grouped.values_mut() {
            let expected = read_fingerprints.get(&group.relative_path).ok_or_else(|| {
                anyhow::anyhow!(
                    "edit_file requires a successful read of {} before editing it",
                    group.relative_path
                )
            })?;
            let current = fingerprint_path(&self.workspace_root, &group.absolute_path)?;
            if !fingerprints_match(expected, &current) {
                bail!(
                    "{} changed since the last successful read; run read on this file before edit_file",
                    group.relative_path
                );
            }

            let original = fs::read_to_string(&group.absolute_path)
                .with_context(|| format!("unable to read file {}", group.relative_path))?;
            let original_lines = split_logical_lines(&original);
            let plan = plan_group_edits(&group.relative_path, &original_lines, &group.edits)?;
            let updated_lines = apply_planned_edits(original_lines.clone(), &plan.operations);
            summaries.push(format_group_summary(
                &group.relative_path,
                original_lines.len(),
                updated_lines.len(),
                &plan.summaries,
            ));
            writes.push((
                group.relative_path.clone(),
                group.absolute_path.clone(),
                join_lines(&updated_lines),
            ));
        }

        let before = snapshot_workspace_paths(&self.workspace_root, &affected_paths);
        for (relative_path, absolute_path, content) in &writes {
            fs::write(absolute_path, content)
                .with_context(|| format!("unable to write file {relative_path}"))?;
        }
        let after = snapshot_workspace_paths(&self.workspace_root, &affected_paths);
        let file_changes = diff_snapshots(before, after);
        let updated_fingerprints = writes
            .iter()
            .map(|(_, absolute_path, _)| fingerprint_path(&self.workspace_root, absolute_path))
            .collect::<Result<Vec<_>>>()?;

        let content = if summaries.is_empty() {
            "No edits applied.".to_string()
        } else {
            format!(
                "Edited {} file{}.

{}",
                summaries.len(),
                if summaries.len() == 1 { "" } else { "s" },
                summaries.join("\n")
            )
        };

        let meta = if updated_fingerprints.len() == 1 {
            json!({
                "read_fingerprint": updated_fingerprints[0],
                "read_fingerprints": updated_fingerprints,
            })
        } else {
            json!({ "read_fingerprints": updated_fingerprints })
        };
        Ok(ToolRunResult::ok_with_meta(content, file_changes, meta))
    }

    fn resolve_edit(&self, index: usize, edit: EditInput) -> Result<ResolvedEdit> {
        if edit.path.trim().is_empty() {
            bail!("edit {}: path is required", index + 1);
        }
        let (relative_path, absolute_path) = resolve_existing_workspace_file(&self.workspace_root, &edit.path)
            .with_context(|| format!("edit {}: invalid path {}", index + 1, edit.path))?;
        let mode = edit.mode;
        let lines = parse_line_spec(&edit.lines)
            .with_context(|| format!("edit {}: invalid lines '{}'", index + 1, edit.lines))?;
        if mode != EditMode::Replace && lines.start != lines.end {
            bail!(
                "edit {}: {} requires a single reference line, not a range",
                index + 1,
                mode.as_str()
            );
        }
        let new_lines = split_logical_lines(&edit.content);
        if mode != EditMode::Replace && new_lines.is_empty() {
            bail!("edit {}: content cannot be empty for {}", index + 1, mode.as_str());
        }
        Ok(ResolvedEdit {
            input_index: index,
            relative_path,
            absolute_path,
            original_lines: edit.lines,
            mode,
            lines,
            new_lines,
        })
    }

    async fn acquire_write_permit(&self) -> Result<Option<OwnedSemaphorePermit>> {
        let Some(write_lock) = &self.write_lock else {
            return Ok(None);
        };
        write_lock
            .clone()
            .acquire_owned()
            .await
            .map(Some)
            .map_err(|_| anyhow::anyhow!("workspace write lock is closed"))
    }
}

#[derive(Debug, Deserialize)]
struct EditFileInput {
    edits: Vec<EditInput>,
}

#[derive(Debug, Deserialize)]
struct EditInput {
    path: String,
    lines: String,
    mode: EditMode,
    content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum EditMode {
    Replace,
    InsertBefore,
    InsertAfter,
}

impl EditMode {
    fn as_str(self) -> &'static str {
        match self {
            EditMode::Replace => "replace",
            EditMode::InsertBefore => "insert_before",
            EditMode::InsertAfter => "insert_after",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LineSpec {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone)]
struct ResolvedEdit {
    input_index: usize,
    relative_path: String,
    absolute_path: PathBuf,
    original_lines: String,
    mode: EditMode,
    lines: LineSpec,
    new_lines: Vec<String>,
}

#[derive(Debug)]
struct EditGroup {
    relative_path: String,
    absolute_path: PathBuf,
    edits: Vec<ResolvedEdit>,
}

#[derive(Debug, Clone)]
struct PlannedEdit {
    input_index: usize,
    original_lines: String,
    mode: EditMode,
    start_index: usize,
    old_len: usize,
    new_lines: Vec<String>,
}

#[derive(Debug)]
struct PlannedGroup {
    operations: Vec<PlannedEdit>,
    summaries: Vec<EditSummary>,
}

#[derive(Debug)]
struct EditSummary {
    input_index: usize,
    mode: EditMode,
    original_lines: String,
    now_start: Option<usize>,
    now_end: Option<usize>,
}

fn group_edits(edits: Vec<ResolvedEdit>) -> BTreeMap<String, EditGroup> {
    let mut grouped = BTreeMap::new();
    for edit in edits {
        grouped
            .entry(edit.relative_path.clone())
            .or_insert_with(|| EditGroup {
                relative_path: edit.relative_path.clone(),
                absolute_path: edit.absolute_path.clone(),
                edits: Vec::new(),
            })
            .edits
            .push(edit);
    }
    grouped
}

fn plan_group_edits(
    relative_path: &str,
    original_lines: &[String],
    edits: &[ResolvedEdit],
) -> Result<PlannedGroup> {
    let total_lines = original_lines.len();
    let mut operations = edits
        .iter()
        .map(|edit| plan_edit(relative_path, total_lines, edit))
        .collect::<Result<Vec<_>>>()?;
    validate_no_overlaps(relative_path, &operations)?;

    let summaries = operations
        .iter()
        .map(|operation| summary_for_operation(operation, &operations))
        .collect::<Vec<_>>();

    operations.sort_by(|left, right| {
        right
            .start_index
            .cmp(&left.start_index)
            .then_with(|| right.old_len.cmp(&left.old_len))
            .then_with(|| left.input_index.cmp(&right.input_index))
    });
    Ok(PlannedGroup {
        operations,
        summaries,
    })
}

fn plan_edit(relative_path: &str, total_lines: usize, edit: &ResolvedEdit) -> Result<PlannedEdit> {
    match edit.mode {
        EditMode::Replace => {
            if total_lines == 0 {
                bail!("{} is empty; replace edits require existing lines", relative_path);
            }
            if edit.lines.end > total_lines {
                bail!(
                    "{} has {total_lines} line{}; replace range {} is out of bounds",
                    relative_path,
                    if total_lines == 1 { "" } else { "s" },
                    edit.original_lines
                );
            }
            Ok(PlannedEdit {
                input_index: edit.input_index,
                original_lines: edit.original_lines.clone(),
                mode: edit.mode,
                start_index: edit.lines.start - 1,
                old_len: edit.lines.end - edit.lines.start + 1,
                new_lines: edit.new_lines.clone(),
            })
        }
        EditMode::InsertBefore => {
            if total_lines == 0 {
                if edit.lines.start != 1 {
                    bail!("{} is empty; use lines '1' to insert into an empty file", relative_path);
                }
            } else if edit.lines.start > total_lines {
                bail!(
                    "{} has {total_lines} line{}; insert_before line {} is out of bounds",
                    relative_path,
                    if total_lines == 1 { "" } else { "s" },
                    edit.lines.start
                );
            }
            Ok(PlannedEdit {
                input_index: edit.input_index,
                original_lines: edit.original_lines.clone(),
                mode: edit.mode,
                start_index: edit.lines.start.saturating_sub(1),
                old_len: 0,
                new_lines: edit.new_lines.clone(),
            })
        }
        EditMode::InsertAfter => {
            if total_lines == 0 {
                bail!("{} is empty; insert_after requires an existing line", relative_path);
            }
            if edit.lines.start > total_lines {
                bail!(
                    "{} has {total_lines} line{}; insert_after line {} is out of bounds",
                    relative_path,
                    if total_lines == 1 { "" } else { "s" },
                    edit.lines.start
                );
            }
            Ok(PlannedEdit {
                input_index: edit.input_index,
                original_lines: edit.original_lines.clone(),
                mode: edit.mode,
                start_index: edit.lines.start,
                old_len: 0,
                new_lines: edit.new_lines.clone(),
            })
        }
    }
}

fn validate_no_overlaps(relative_path: &str, operations: &[PlannedEdit]) -> Result<()> {
    for (left_index, left) in operations.iter().enumerate() {
        for right in &operations[left_index + 1..] {
            if operations_conflict(left, right) {
                bail!(
                    "overlapping edits in {relative_path}: edit {} ({}) conflicts with edit {} ({})",
                    left.input_index + 1,
                    left.original_lines,
                    right.input_index + 1,
                    right.original_lines
                );
            }
        }
    }
    Ok(())
}

fn operations_conflict(left: &PlannedEdit, right: &PlannedEdit) -> bool {
    let left_end = left.start_index + left.old_len;
    let right_end = right.start_index + right.old_len;
    match (left.old_len == 0, right.old_len == 0) {
        (true, true) => left.start_index == right.start_index,
        (false, false) => left.start_index < right_end && right.start_index < left_end,
        (true, false) => left.start_index >= right.start_index && left.start_index <= right_end,
        (false, true) => right.start_index >= left.start_index && right.start_index <= left_end,
    }
}

fn summary_for_operation(operation: &PlannedEdit, operations: &[PlannedEdit]) -> EditSummary {
    let shift = operations
        .iter()
        .filter(|other| other.input_index != operation.input_index)
        .filter(|other| other.start_index < operation.start_index)
        .map(|other| other.new_lines.len() as isize - other.old_len as isize)
        .sum::<isize>();
    let now_start_index = (operation.start_index as isize + shift).max(0) as usize;
    let (now_start, now_end) = if operation.new_lines.is_empty() {
        (None, None)
    } else {
        (
            Some(now_start_index + 1),
            Some(now_start_index + operation.new_lines.len()),
        )
    };
    EditSummary {
        input_index: operation.input_index,
        mode: operation.mode,
        original_lines: operation.original_lines.clone(),
        now_start,
        now_end,
    }
}

fn apply_planned_edits(mut lines: Vec<String>, operations: &[PlannedEdit]) -> Vec<String> {
    for operation in operations {
        lines.splice(
            operation.start_index..operation.start_index + operation.old_len,
            operation.new_lines.clone(),
        );
    }
    lines
}

fn format_group_summary(
    relative_path: &str,
    old_count: usize,
    new_count: usize,
    summaries: &[EditSummary],
) -> String {
    let mut output = format!(
        "{relative_path}: {old_count} -> {new_count} line{}",
        if new_count == 1 { "" } else { "s" }
    );
    let mut summaries = summaries.iter().collect::<Vec<_>>();
    summaries.sort_by_key(|summary| summary.input_index);
    for summary in summaries {
        let now = match (summary.now_start, summary.now_end) {
            (Some(start), Some(end)) if start == end => format!("now {start}"),
            (Some(start), Some(end)) => format!("now {start}-{end}"),
            _ => "now removed".to_string(),
        };
        output.push_str(&format!(
            "\n  [{}] {} {} -> {now}",
            summary.input_index + 1,
            summary.mode.as_str(),
            summary.original_lines
        ));
    }
    output
}

fn parse_line_spec(raw: &str) -> Result<LineSpec> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("lines is required");
    }
    let Some((start, end)) = trimmed.split_once('-') else {
        let line = parse_positive_line(trimmed)?;
        return Ok(LineSpec {
            start: line,
            end: line,
        });
    };
    if end.contains('-') {
        bail!("expected 'N' or 'N-M'");
    }
    let start = parse_positive_line(start.trim())?;
    let end = parse_positive_line(end.trim())?;
    if start > end {
        bail!("range start must be less than or equal to range end");
    }
    Ok(LineSpec { start, end })
}

fn parse_positive_line(raw: &str) -> Result<usize> {
    let value = raw
        .parse::<usize>()
        .with_context(|| format!("invalid line number '{raw}'"))?;
    if value == 0 {
        bail!("line numbers are 1-based and must be greater than 0");
    }
    Ok(value)
}

fn split_logical_lines(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut lines = text
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line).to_string())
        .collect::<Vec<_>>();
    if lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    lines
}

fn join_lines(lines: &[String]) -> String {
    if lines.is_empty() {
        String::new()
    } else {
        let mut content = lines.join("\n");
        content.push('\n');
        content
    }
}

fn resolve_existing_workspace_file(root: &Path, raw: &str) -> Result<(String, PathBuf)> {
    let trimmed = raw.trim();
    let candidate = Path::new(trimmed);
    let absolute = if candidate.is_absolute() {
        candidate
            .canonicalize()
            .with_context(|| format!("unable to resolve path {}", candidate.display()))?
    } else {
        let normalized = normalize_workspace_relative_path(trimmed)?;
        if normalized.is_empty() {
            bail!("path cannot be empty");
        }
        root.join(normalized)
            .canonicalize()
            .with_context(|| format!("unable to resolve path {trimmed}"))?
    };
    let root = root
        .canonicalize()
        .with_context(|| format!("unable to resolve workspace root {}", root.display()))?;
    let metadata = fs::metadata(&absolute)
        .with_context(|| format!("unable to read file metadata {}", absolute.display()))?;
    if !metadata.is_file() {
        bail!("path is not a file");
    }
    let relative = absolute
        .strip_prefix(&root)
        .with_context(|| format!("{} is outside the workspace", absolute.display()))?
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => Some(value.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/");
    if relative.is_empty() {
        bail!("path cannot be the workspace root");
    }
    Ok((relative, absolute))
}

fn fingerprints_match(expected: &ReadFingerprint, current: &ReadFingerprint) -> bool {
    expected.size == current.size
        && expected.modified_ms == current.modified_ms
        && expected.sha256 == current.sha256
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use uuid::Uuid;

    use super::*;

    #[tokio::test]
    async fn replace_existing_lines() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "one\ntwo\nthree\nfour\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        let result = tool
            .edit(
                json!({
                    "edits": [{
                        "path": "app.rs",
                        "lines": "2-3",
                        "mode": "replace",
                        "content": "deux\ntrois"
                    }]
                }),
                &fingerprints,
            )
            .await
            .expect("edit should apply");

        assert!(!result.is_error);
        assert_eq!(fs::read_to_string(root.join("app.rs")).unwrap(), "one\ndeux\ntrois\nfour\n");
        assert!(result.content.contains("app.rs: 4 -> 4 lines"));
        assert!(result.content.contains("[1] replace 2-3 -> now 2-3"));
        assert_eq!(result.file_changes.len(), 1);
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn insert_before_and_after_lines() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "one\ntwo\nthree\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        tool.edit(
            json!({
                "edits": [
                    {"path": "app.rs", "lines": "1", "mode": "insert_before", "content": "zero"},
                    {"path": "app.rs", "lines": "2", "mode": "insert_after", "content": "two point five"}
                ]
            }),
            &fingerprints,
        )
        .await
        .expect("edit should apply");

        assert_eq!(
            fs::read_to_string(root.join("app.rs")).unwrap(),
            "zero\none\ntwo\ntwo point five\nthree\n"
        );
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn multi_edits_use_original_line_numbers() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "a\nb\nc\nd\ne\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        tool.edit(
            json!({
                "edits": [
                    {"path": "app.rs", "lines": "2", "mode": "replace", "content": "B1\nB2"},
                    {"path": "app.rs", "lines": "5", "mode": "insert_after", "content": "f"}
                ]
            }),
            &fingerprints,
        )
        .await
        .expect("edit should apply");

        assert_eq!(fs::read_to_string(root.join("app.rs")).unwrap(), "a\nB1\nB2\nc\nd\ne\nf\n");
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn rejects_overlapping_edits_before_writing() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "a\nb\nc\nd\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        let error = tool
            .edit(
                json!({
                    "edits": [
                        {"path": "app.rs", "lines": "2-3", "mode": "replace", "content": "x"},
                        {"path": "app.rs", "lines": "3", "mode": "insert_before", "content": "y"}
                    ]
                }),
                &fingerprints,
            )
            .await
            .expect_err("overlap should fail");

        assert!(error.to_string().contains("overlapping edits"));
        assert_eq!(fs::read_to_string(root.join("app.rs")).unwrap(), "a\nb\nc\nd\n");
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn rejects_stale_read_fingerprint() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "a\nb\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);
        fs::write(root.join("app.rs"), "a\nchanged\n").expect("modify file");

        let error = tool
            .edit(
                json!({
                    "edits": [{"path": "app.rs", "lines": "2", "mode": "replace", "content": "B"}]
                }),
                &fingerprints,
            )
            .await
            .expect_err("stale fingerprint should fail");

        assert!(error.to_string().contains("changed since the last successful read"));
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn inserts_into_empty_file_with_insert_before_one() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("empty.txt"), "").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["empty.txt"]);

        tool.edit(
            json!({
                "edits": [{"path": "empty.txt", "lines": "1", "mode": "insert_before", "content": "hello"}]
            }),
            &fingerprints,
        )
        .await
        .expect("insert should apply");

        assert_eq!(fs::read_to_string(root.join("empty.txt")).unwrap(), "hello\n");
        fs::remove_dir_all(root).ok();
    }

    fn fingerprints(root: &Path, paths: &[&str]) -> HashMap<String, ReadFingerprint> {
        paths
            .iter()
            .map(|path| {
                let fingerprint = fingerprint_path(root, &root.join(path)).expect("fingerprint file");
                ((*path).to_string(), fingerprint)
            })
            .collect()
    }

    fn unique_temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!("sinew-edit-test-{}", Uuid::new_v4()))
    }
}
