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

const EDIT_FILE_DESCRIPTION: &str = r#"Use this tool to edit files."#;

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
                    "files": {
                        "type": "array",
                        "minItems": 1,
                        "description": "The file edit groups to apply in one operation. ALWAYS consolidate edits to the same file under one entry. Do not create multiple entries sharing the same path.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "File path to edit. Relative paths and absolute are tolerated. Prefere relative."
                                },
                                "edits": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": MAX_EDIT_COUNT,
                                    "description": "Exact replacements to apply to this file. Replacements in the same file must target disjoint regions in the original content.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "oldContent": {
                                                "type": "string",
                                                "description": "Exact text to replace. ALWAYS use the shortest old_content that is still unique in the file. Do not include surrounding context beyond what is strictly needed for an unambiguous match."
                                            },
                                            "newContent": {
                                                "type": "string",
                                                "description": "Replacement text. May be empty to delete oldContent."
                                            },
                                            "replaceAll": {
                                                "type": "boolean",
                                                "description": "When true, replace every non-overlapping occurrence of oldContent. Defaults to false, which requires oldContent to match exactly once."
                                            }
                                        },
                                        "required": ["oldContent", "newContent"],
                                        "additionalProperties": false
                                    }
                                }
                            },
                            "required": ["path", "edits"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["files"],
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
        validate_edit_input(&parsed)?;

        let resolved = parsed
            .files
            .into_iter()
            .enumerate()
            .map(|(index, file)| self.resolve_file_group(index, file))
            .collect::<Result<Vec<_>>>()?;
        let mut grouped = group_file_edits(resolved)?;
        let affected_paths = grouped.keys().cloned().collect::<Vec<_>>();

        let _write_permit = self.acquire_write_permit().await?;
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
            let normalized_original = normalize_file_text(&original);
            let plan = plan_file_edits(
                &group.relative_path,
                &normalized_original.content,
                &group.edits,
            )?;
            let updated_content = normalized_original.restore(&plan.updated_content);
            summaries.push(FileEditSummary {
                relative_path: group.relative_path.clone(),
                replacement_count: plan.replacement_count,
            });
            writes.push((
                group.relative_path.clone(),
                group.absolute_path.clone(),
                updated_content,
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

        let content = format_edit_output(&summaries);

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

    fn resolve_file_group(&self, _index: usize, file: FileEditInput) -> Result<ResolvedFileEdit> {
        if file.path.trim().is_empty() {
            bail!("Could not edit file: <empty>. Error code: path cannot be empty");
        }
        let (relative_path, absolute_path) =
            resolve_existing_workspace_file(&self.workspace_root, &file.path).map_err(|err| {
                anyhow::anyhow!("Could not edit file: {}. Error code: {err}", file.path)
            })?;
        Ok(ResolvedFileEdit {
            relative_path,
            absolute_path,
            edits: file.edits,
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
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct EditFileInput {
    #[serde(default)]
    files: Vec<FileEditInput>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FileEditInput {
    path: String,
    #[serde(default)]
    edits: Vec<ReplacementInput>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReplacementInput {
    #[serde(alias = "oldText")]
    old_content: String,
    #[serde(alias = "newText")]
    new_content: String,
    #[serde(default)]
    replace_all: bool,
}

#[derive(Debug)]
struct ResolvedFileEdit {
    relative_path: String,
    absolute_path: PathBuf,
    edits: Vec<ReplacementInput>,
}

#[derive(Debug)]
struct EditGroup {
    relative_path: String,
    absolute_path: PathBuf,
    edits: Vec<ReplacementInput>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineEnding {
    Lf,
    Crlf,
}

#[derive(Debug)]
struct NormalizedFileText {
    bom: bool,
    line_ending: LineEnding,
    content: String,
}

impl NormalizedFileText {
    fn restore(&self, content: &str) -> String {
        let mut restored = restore_line_endings(content, self.line_ending);
        if self.bom {
            restored.insert(0, '\u{FEFF}');
        }
        restored
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReplacementMatch {
    start: usize,
    len: usize,
}

#[derive(Debug)]
struct FuzzyNormalizedText {
    text: String,
    start_boundaries: Vec<Option<usize>>,
    end_boundaries: Vec<Option<usize>>,
    next_trimmed_boundaries: Vec<Option<usize>>,
}

impl FuzzyNormalizedText {
    fn original_range(&self, start: usize, end: usize) -> Option<(usize, usize)> {
        if start >= end {
            return None;
        }
        let original_start = *self.start_boundaries.get(start)?.as_ref()?;
        let original_end = *self.end_boundaries.get(end)?.as_ref()?;
        if original_start <= original_end {
            Some((original_start, original_end))
        } else {
            None
        }
    }

    fn original_range_with_trimmed_suffix(
        &self,
        start: usize,
        end: usize,
    ) -> Option<(usize, usize)> {
        let (original_start, original_end) = self.original_range(start, end)?;
        let extended_end = self
            .next_trimmed_boundaries
            .get(end)
            .and_then(|value| *value)
            .unwrap_or(original_end);
        Some((original_start, extended_end))
    }
}

#[derive(Debug, Clone)]
struct PlannedReplacement {
    edit_index: usize,
    start: usize,
    old_len: usize,
    new_content: String,
}

impl PlannedReplacement {
    fn end(&self) -> usize {
        self.start + self.old_len
    }
}

#[derive(Debug)]
struct PlannedFileEdit {
    updated_content: String,
    replacement_count: usize,
}

struct FileEditSummary {
    relative_path: String,
    replacement_count: usize,
}

fn validate_edit_input(input: &EditFileInput) -> Result<()> {
    let replacement_count = input
        .files
        .iter()
        .map(|file| file.edits.len())
        .sum::<usize>();
    if replacement_count == 0 || input.files.iter().any(|file| file.edits.is_empty()) {
        bail!("Edit tool input is invalid. files must contain at least one file group, and each file group must contain at least one edit.");
    }
    if replacement_count > MAX_EDIT_COUNT {
        bail!("too many replacements in one call; maximum is {MAX_EDIT_COUNT}");
    }
    let total_content_bytes = input
        .files
        .iter()
        .flat_map(|file| &file.edits)
        .map(|edit| edit.old_content.len() + edit.new_content.len())
        .sum::<usize>();
    if total_content_bytes > MAX_TOTAL_CONTENT_BYTES {
        bail!("edit content is too large to apply safely");
    }
    Ok(())
}

fn group_file_edits(files: Vec<ResolvedFileEdit>) -> Result<BTreeMap<String, EditGroup>> {
    let mut grouped = BTreeMap::new();
    for file in files {
        if grouped.contains_key(&file.relative_path) {
            bail!(
                "duplicate file entry for {}; merge replacements for each path into one edits array",
                file.relative_path
            );
        }
        grouped.insert(
            file.relative_path.clone(),
            EditGroup {
                relative_path: file.relative_path,
                absolute_path: file.absolute_path,
                edits: file.edits,
            },
        );
    }
    Ok(grouped)
}

fn plan_file_edits(
    relative_path: &str,
    original: &str,
    edits: &[ReplacementInput],
) -> Result<PlannedFileEdit> {
    let multiple = edits.len() > 1;
    let mut replacements = Vec::with_capacity(edits.len());

    for (index, edit) in edits.iter().enumerate() {
        let old_content = normalize_line_endings(&edit.old_content);
        let new_content = normalize_line_endings(&edit.new_content);

        if old_content.is_empty() {
            bail!(
                "{} must not be empty in {relative_path}.",
                old_content_label(index, multiple)
            );
        }

        let matched = if edit.replace_all {
            find_all_replacement_matches(relative_path, original, &old_content, index, multiple)?
        } else {
            vec![find_unique_replacement_match(
                relative_path,
                original,
                &old_content,
                index,
                multiple,
            )?]
        };

        if old_content == new_content {
            bail!(
                "No changes made to {relative_path}. The replacement produced identical content. The oldContent and newContent are the same."
            );
        }

        for matched in matched {
            replacements.push(PlannedReplacement {
                edit_index: index,
                start: matched.start,
                old_len: matched.len,
                new_content: new_content.clone(),
            });
        }
    }

    validate_no_overlaps(relative_path, &replacements)?;
    let replacement_count = replacements.len();
    let updated_content = apply_replacements(original, &replacements);
    if updated_content == original {
        bail!(
            "No changes made to {relative_path}. The replacement produced identical content. The oldContent and newContent are the same."
        );
    }

    Ok(PlannedFileEdit {
        updated_content,
        replacement_count,
    })
}

fn old_content_label(index: usize, multiple: bool) -> String {
    if multiple {
        format!("edits[{index}].oldContent")
    } else {
        "oldContent".to_string()
    }
}

fn find_unique_replacement_match(
    relative_path: &str,
    original: &str,
    old_content: &str,
    edit_index: usize,
    multiple: bool,
) -> Result<ReplacementMatch> {
    let exact_occurrences = find_occurrences(original, old_content);
    let fuzzy = fuzzy_normalize_with_map(original);
    let fuzzy_old_content = normalize_for_fuzzy_match(old_content);
    if fuzzy_old_content.is_empty() {
        not_found_error(relative_path, edit_index, multiple)?;
    }
    let fuzzy_occurrences = find_occurrences(&fuzzy.text, &fuzzy_old_content);

    if fuzzy_occurrences.len() > 1 {
        return duplicate_match_error(relative_path, edit_index, multiple, fuzzy_occurrences.len());
    }

    match exact_occurrences.len() {
        1 => {
            return Ok(ReplacementMatch {
                start: exact_occurrences[0],
                len: old_content.len(),
            });
        }
        count if count > 1 => {
            return duplicate_match_error(relative_path, edit_index, multiple, count)
        }
        _ => {}
    }

    match fuzzy_occurrences.len() {
        0 => not_found_error(relative_path, edit_index, multiple),
        1 => {
            let start = fuzzy_occurrences[0];
            let end = start + fuzzy_old_content.len();
            let (original_start, original_end) = fuzzy
                .original_range_with_trimmed_suffix(start, end)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Could not find the exact text in {relative_path}. The old content must match exactly including all whitespace and newlines."
                    )
                })?;
            Ok(ReplacementMatch {
                start: original_start,
                len: original_end - original_start,
            })
        }
        count => duplicate_match_error(relative_path, edit_index, multiple, count),
    }
}

fn find_all_replacement_matches(
    relative_path: &str,
    original: &str,
    old_content: &str,
    edit_index: usize,
    multiple: bool,
) -> Result<Vec<ReplacementMatch>> {
    let exact_occurrences = find_non_overlapping_occurrences(original, old_content);
    if !exact_occurrences.is_empty() {
        return Ok(exact_occurrences
            .into_iter()
            .map(|start| ReplacementMatch {
                start,
                len: old_content.len(),
            })
            .collect());
    }

    let fuzzy = fuzzy_normalize_with_map(original);
    let fuzzy_old_content = normalize_for_fuzzy_match(old_content);
    if fuzzy_old_content.is_empty() {
        return not_found_error(relative_path, edit_index, multiple).map(|_| Vec::new());
    }

    let fuzzy_occurrences = find_non_overlapping_occurrences(&fuzzy.text, &fuzzy_old_content);
    if fuzzy_occurrences.is_empty() {
        return not_found_error(relative_path, edit_index, multiple).map(|_| Vec::new());
    }

    fuzzy_occurrences
        .into_iter()
        .map(|start| {
            let end = start + fuzzy_old_content.len();
            let (original_start, original_end) = fuzzy
                .original_range_with_trimmed_suffix(start, end)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Could not find the exact text in {relative_path}. The old content must match exactly including all whitespace and newlines."
                    )
                })?;
            Ok(ReplacementMatch {
                start: original_start,
                len: original_end - original_start,
            })
        })
        .collect()
}

fn not_found_error(
    relative_path: &str,
    edit_index: usize,
    multiple: bool,
) -> Result<ReplacementMatch> {
    if multiple {
        bail!(
            "Could not find edits[{edit_index}] in {relative_path}. The oldContent must match exactly including all whitespace and newlines."
        );
    }
    bail!(
        "Could not find the exact text in {relative_path}. The old content must match exactly including all whitespace and newlines."
    );
}

fn duplicate_match_error(
    relative_path: &str,
    edit_index: usize,
    multiple: bool,
    count: usize,
) -> Result<ReplacementMatch> {
    if multiple {
        bail!(
            "Found {count} occurrences of edits[{edit_index}] in {relative_path}. Each oldContent must be unique. Please provide more context to make it unique."
        );
    }
    bail!(
        "Found {count} occurrences of the text in {relative_path}. The text must be unique. Please provide more context to make it unique."
    );
}

fn find_occurrences(haystack: &str, needle: &str) -> Vec<usize> {
    let mut occurrences = Vec::new();
    let mut search_start = 0;

    while search_start <= haystack.len() {
        let Some(relative_match) = haystack[search_start..].find(needle) else {
            break;
        };
        let absolute_match = search_start + relative_match;
        occurrences.push(absolute_match);
        search_start = next_char_boundary(haystack, absolute_match);
    }

    occurrences
}

fn find_non_overlapping_occurrences(haystack: &str, needle: &str) -> Vec<usize> {
    let mut occurrences = Vec::new();
    let mut search_start = 0;

    while search_start <= haystack.len() {
        let Some(relative_match) = haystack[search_start..].find(needle) else {
            break;
        };
        let absolute_match = search_start + relative_match;
        occurrences.push(absolute_match);
        search_start = absolute_match + needle.len();
    }

    occurrences
}

fn next_char_boundary(text: &str, offset: usize) -> usize {
    if offset >= text.len() {
        return text.len() + 1;
    }
    offset
        + text[offset..]
            .chars()
            .next()
            .map(char::len_utf8)
            .unwrap_or(1)
}

fn validate_no_overlaps(relative_path: &str, replacements: &[PlannedReplacement]) -> Result<()> {
    let mut sorted = replacements.iter().collect::<Vec<_>>();
    sorted.sort_by_key(|replacement| (replacement.start, replacement.end()));

    for pair in sorted.windows(2) {
        let left = pair[0];
        let right = pair[1];
        if left.end() > right.start {
            bail!(
                "edits[{}] and edits[{}] overlap in {relative_path}. Merge them into one edit or target disjoint regions.",
                left.edit_index,
                right.edit_index
            );
        }
    }
    Ok(())
}

fn apply_replacements(original: &str, replacements: &[PlannedReplacement]) -> String {
    let mut updated = original.to_string();
    let mut sorted = replacements.iter().collect::<Vec<_>>();
    sorted.sort_by(|left, right| right.start.cmp(&left.start));

    for replacement in sorted {
        updated.replace_range(
            replacement.start..replacement.start + replacement.old_len,
            &replacement.new_content,
        );
    }

    updated
}

fn format_edit_output(summaries: &[FileEditSummary]) -> String {
    if summaries.len() == 1 {
        let summary = &summaries[0];
        return format!(
            "Edited {} ({} replacement{}).",
            summary.relative_path,
            summary.replacement_count,
            if summary.replacement_count == 1 {
                ""
            } else {
                "s"
            }
        );
    }

    let mut output = format!("Edited {} files:", summaries.len());
    for summary in summaries {
        output.push_str(&format!(
            "\n- {} ({} replacement{})",
            summary.relative_path,
            summary.replacement_count,
            if summary.replacement_count == 1 {
                ""
            } else {
                "s"
            }
        ));
    }
    output
}
fn normalize_file_text(text: &str) -> NormalizedFileText {
    let (bom, without_bom) = strip_utf8_bom(text);
    let line_ending = detect_line_ending(without_bom);
    NormalizedFileText {
        bom,
        line_ending,
        content: normalize_line_endings(without_bom),
    }
}

fn strip_utf8_bom(text: &str) -> (bool, &str) {
    text.strip_prefix('\u{FEFF}')
        .map(|stripped| (true, stripped))
        .unwrap_or((false, text))
}

fn detect_line_ending(text: &str) -> LineEnding {
    let crlf_index = text.find("\r\n");
    let lf_index = text.find('\n');
    match (crlf_index, lf_index) {
        (Some(crlf), Some(lf)) if crlf == lf.saturating_sub(1) => LineEnding::Crlf,
        _ => LineEnding::Lf,
    }
}

fn normalize_line_endings(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

fn restore_line_endings(text: &str, line_ending: LineEnding) -> String {
    match line_ending {
        LineEnding::Lf => text.to_string(),
        LineEnding::Crlf => text.replace('\n', "\r\n"),
    }
}

fn normalize_for_fuzzy_match(text: &str) -> String {
    fuzzy_normalize_with_map(text).text
}

fn fuzzy_normalize_with_map(text: &str) -> FuzzyNormalizedText {
    let mut normalized = String::new();
    let mut start_boundaries = vec![Some(0)];
    let mut end_boundaries = vec![Some(0)];
    let mut next_trimmed_boundaries = vec![Some(0)];
    let mut text_offset = 0;

    for segment in text.split_inclusive('\n') {
        let has_newline = segment.ends_with('\n');
        let line = if has_newline {
            &segment[..segment.len() - 1]
        } else {
            segment
        };
        let trimmed = line.trim_end_matches(char::is_whitespace);
        let trimmed_end = text_offset + trimmed.len();
        let line_end = text_offset + line.len();

        for (local_offset, ch) in trimmed.char_indices() {
            emit_fuzzy_char(
                &mut normalized,
                &mut start_boundaries,
                &mut end_boundaries,
                &mut next_trimmed_boundaries,
                text_offset + local_offset,
                text_offset + local_offset + ch.len_utf8(),
                fuzzy_char(ch),
            );
        }

        if line_end > trimmed_end {
            let boundary = normalized.len();
            if next_trimmed_boundaries.len() <= boundary {
                next_trimmed_boundaries.resize(boundary + 1, None);
            }
            next_trimmed_boundaries[boundary] = Some(line_end);
        }

        if has_newline {
            let newline_offset = text_offset + segment.len() - 1;
            emit_fuzzy_char(
                &mut normalized,
                &mut start_boundaries,
                &mut end_boundaries,
                &mut next_trimmed_boundaries,
                newline_offset,
                newline_offset + 1,
                '\n',
            );
        }

        text_offset += segment.len();
    }

    if text.is_empty() {
        start_boundaries[0] = Some(0);
        end_boundaries[0] = Some(0);
        next_trimmed_boundaries[0] = Some(0);
    }

    FuzzyNormalizedText {
        text: normalized,
        start_boundaries,
        end_boundaries,
        next_trimmed_boundaries,
    }
}

fn emit_fuzzy_char(
    normalized: &mut String,
    start_boundaries: &mut Vec<Option<usize>>,
    end_boundaries: &mut Vec<Option<usize>>,
    next_trimmed_boundaries: &mut Vec<Option<usize>>,
    original_start: usize,
    original_end: usize,
    ch: char,
) {
    let normalized_start = normalized.len();
    normalized.push(ch);
    let normalized_end = normalized.len();
    let required_len = normalized_end + 1;
    if start_boundaries.len() < required_len {
        start_boundaries.resize(required_len, None);
    }
    if end_boundaries.len() < required_len {
        end_boundaries.resize(required_len, None);
    }
    if next_trimmed_boundaries.len() < required_len {
        next_trimmed_boundaries.resize(required_len, None);
    }
    start_boundaries[normalized_start] = Some(original_start);
    end_boundaries[normalized_end] = Some(original_end);
}

fn fuzzy_char(ch: char) -> char {
    match ch {
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
        '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
        | '\u{2212}' => '-',
        '\u{00A0}' | '\u{2002}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
        _ => ch,
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
    async fn replaces_exact_text_in_one_file() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "one\ntwo\nthree\nfour\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        let result = tool
            .edit(
                json!({
                    "files": [{
                        "path": "app.rs",
                        "edits": [{
                            "oldContent": "two\nthree",
                            "newContent": "deux\ntrois"
                        }]
                    }]
                }),
                &fingerprints,
            )
            .await
            .expect("edit should apply");

        assert!(!result.is_error);
        assert_eq!(
            fs::read_to_string(root.join("app.rs")).unwrap(),
            "one\ndeux\ntrois\nfour\n"
        );
        assert_eq!(result.content, "Edited app.rs (1 replacement).");
        assert_eq!(result.file_changes.len(), 1);
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn applies_multiple_disjoint_replacements_in_one_file() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "a\nb\nc\nd\ne\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        tool.edit(
            json!({
                "files": [{
                    "path": "app.rs",
                    "edits": [
                        {"oldContent": "b", "newContent": "B1\nB2"},
                        {"oldContent": "e", "newContent": "E"}
                    ]
                }]
            }),
            &fingerprints,
        )
        .await
        .expect("edit should apply");

        assert_eq!(
            fs::read_to_string(root.join("app.rs")).unwrap(),
            "a\nB1\nB2\nc\nd\nE\n"
        );
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn applies_replacements_across_multiple_files() {
        let root = unique_temp_dir();
        fs::create_dir_all(root.join("src")).expect("create temp workspace");
        fs::write(root.join("src/a.rs"), "fn old() {}\n").expect("write a");
        fs::write(root.join("src/b.rs"), "old();\n").expect("write b");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["src/a.rs", "src/b.rs"]);

        let result = tool
            .edit(
                json!({
                    "files": [
                        {"path": "src/a.rs", "edits": [{"oldContent": "fn old", "newContent": "fn new"}]},
                        {"path": "src/b.rs", "edits": [{"oldContent": "old();", "newContent": "new();"}]}
                    ]
                }),
                &fingerprints,
            )
            .await
            .expect("edit should apply");

        assert_eq!(
            fs::read_to_string(root.join("src/a.rs")).unwrap(),
            "fn new() {}\n"
        );
        assert_eq!(
            fs::read_to_string(root.join("src/b.rs")).unwrap(),
            "new();\n"
        );
        assert_eq!(
            result.content,
            "Edited 2 files:\n- src/a.rs (1 replacement)\n- src/b.rs (1 replacement)"
        );
        assert_eq!(result.file_changes.len(), 2);
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn rejects_empty_replacement_list() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        let tool = EditFileTool::new(&root);

        let error = tool
            .edit(json!({ "files": [] }), &HashMap::new())
            .await
            .expect_err("empty edits should fail");

        assert_eq!(
            error.to_string(),
            "Edit tool input is invalid. files must contain at least one file group, and each file group must contain at least one edit."
        );
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn rejects_inaccessible_file() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        let tool = EditFileTool::new(&root);

        let error = tool
            .edit(
                json!({
                    "files": [{
                        "path": "missing.rs",
                        "edits": [{"oldContent": "a", "newContent": "b"}]
                    }]
                }),
                &HashMap::new(),
            )
            .await
            .expect_err("missing file should fail");

        assert!(error
            .to_string()
            .starts_with("Could not edit file: missing.rs. Error code:"));
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn rejects_empty_old_content_for_single_edit() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "a\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        let error = tool
            .edit(
                json!({
                    "files": [{
                        "path": "app.rs",
                        "edits": [{"oldContent": "", "newContent": "b"}]
                    }]
                }),
                &fingerprints,
            )
            .await
            .expect_err("empty old content should fail");

        assert_eq!(error.to_string(), "oldContent must not be empty in app.rs.");
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn rejects_empty_old_content_for_multiple_edits() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "a\nb\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        let error = tool
            .edit(
                json!({
                    "files": [{
                        "path": "app.rs",
                        "edits": [
                            {"oldContent": "a", "newContent": "A"},
                            {"oldContent": "", "newContent": "B"}
                        ]
                    }]
                }),
                &fingerprints,
            )
            .await
            .expect_err("empty old content should fail");

        assert_eq!(
            error.to_string(),
            "edits[1].oldContent must not be empty in app.rs."
        );
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn rejects_missing_old_content_for_single_edit() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "a\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        let error = tool
            .edit(
                json!({
                    "files": [{
                        "path": "app.rs",
                        "edits": [{"oldContent": "missing", "newContent": "b"}]
                    }]
                }),
                &fingerprints,
            )
            .await
            .expect_err("missing old content should fail");

        assert_eq!(
            error.to_string(),
            "Could not find the exact text in app.rs. The old content must match exactly including all whitespace and newlines."
        );
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn rejects_missing_old_content_for_multiple_edits_without_writing() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "a\nb\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        let error = tool
            .edit(
                json!({
                    "files": [{
                        "path": "app.rs",
                        "edits": [
                            {"oldContent": "a", "newContent": "A"},
                            {"oldContent": "missing", "newContent": "B"}
                        ]
                    }]
                }),
                &fingerprints,
            )
            .await
            .expect_err("missing old content should fail");

        assert_eq!(
            error.to_string(),
            "Could not find edits[1] in app.rs. The oldContent must match exactly including all whitespace and newlines."
        );
        assert_eq!(fs::read_to_string(root.join("app.rs")).unwrap(), "a\nb\n");
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn rejects_non_unique_old_content_for_single_edit() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "same\nsame\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        let error = tool
            .edit(
                json!({
                    "files": [{
                        "path": "app.rs",
                        "edits": [{"oldContent": "same", "newContent": "other"}]
                    }]
                }),
                &fingerprints,
            )
            .await
            .expect_err("non unique old content should fail");

        assert_eq!(
            error.to_string(),
            "Found 2 occurrences of the text in app.rs. The text must be unique. Please provide more context to make it unique."
        );
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn rejects_non_unique_old_content_for_multiple_edits() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "first\nsame\nsame\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        let error = tool
            .edit(
                json!({
                    "files": [{
                        "path": "app.rs",
                        "edits": [
                            {"oldContent": "first", "newContent": "changed"},
                            {"oldContent": "same", "newContent": "other"}
                        ]
                    }]
                }),
                &fingerprints,
            )
            .await
            .expect_err("non unique old content should fail");

        assert_eq!(
            error.to_string(),
            "Found 2 occurrences of edits[1] in app.rs. Each oldContent must be unique. Please provide more context to make it unique."
        );
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
                    "files": [{
                        "path": "app.rs",
                        "edits": [
                            {"oldContent": "b\nc", "newContent": "x"},
                            {"oldContent": "c\nd", "newContent": "y"}
                        ]
                    }]
                }),
                &fingerprints,
            )
            .await
            .expect_err("overlap should fail");

        assert_eq!(
            error.to_string(),
            "edits[0] and edits[1] overlap in app.rs. Merge them into one edit or target disjoint regions."
        );
        assert_eq!(
            fs::read_to_string(root.join("app.rs")).unwrap(),
            "a\nb\nc\nd\n"
        );
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn rejects_identical_replacement() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "a\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        let error = tool
            .edit(
                json!({
                    "files": [{
                        "path": "app.rs",
                        "edits": [{"oldContent": "a", "newContent": "a"}]
                    }]
                }),
                &fingerprints,
            )
            .await
            .expect_err("identical replacement should fail");

        assert_eq!(
            error.to_string(),
            "No changes made to app.rs. The replacement produced identical content. The oldContent and newContent are the same."
        );
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn replace_all_replaces_every_exact_occurrence() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "foo bar foo\nfoo\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        let result = tool
            .edit(
                json!({
                    "files": [{
                        "path": "app.rs",
                        "edits": [{"oldContent": "foo", "newContent": "baz", "replaceAll": true}]
                    }]
                }),
                &fingerprints,
            )
            .await
            .expect("replaceAll edit should apply");

        assert_eq!(
            fs::read_to_string(root.join("app.rs")).unwrap(),
            "baz bar baz\nbaz\n"
        );
        assert_eq!(result.content, "Edited app.rs (3 replacements).");
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn replace_all_false_still_requires_unique_old_content() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "foo\nfoo\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        let error = tool
            .edit(
                json!({
                    "files": [{
                        "path": "app.rs",
                        "edits": [{"oldContent": "foo", "newContent": "bar", "replaceAll": false}]
                    }]
                }),
                &fingerprints,
            )
            .await
            .expect_err("non replaceAll duplicate should fail");

        assert_eq!(
            error.to_string(),
            "Found 2 occurrences of the text in app.rs. The text must be unique. Please provide more context to make it unique."
        );
        assert_eq!(
            fs::read_to_string(root.join("app.rs")).unwrap(),
            "foo\nfoo\n"
        );
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn replace_all_uses_non_overlapping_occurrences() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "aaaa\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        let result = tool
            .edit(
                json!({
                    "files": [{
                        "path": "app.rs",
                        "edits": [{"oldContent": "aa", "newContent": "b", "replaceAll": true}]
                    }]
                }),
                &fingerprints,
            )
            .await
            .expect("replaceAll edit should apply");

        assert_eq!(fs::read_to_string(root.join("app.rs")).unwrap(), "bb\n");
        assert_eq!(result.content, "Edited app.rs (2 replacements).");
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn replace_all_rejects_overlap_with_other_edits() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "foo\nfoo\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        let error = tool
            .edit(
                json!({
                    "files": [{
                        "path": "app.rs",
                        "edits": [
                            {"oldContent": "foo", "newContent": "bar", "replaceAll": true},
                            {"oldContent": "foo\nfoo", "newContent": "baz"}
                        ]
                    }]
                }),
                &fingerprints,
            )
            .await
            .expect_err("overlap should fail");

        assert_eq!(
            error.to_string(),
            "edits[0] and edits[1] overlap in app.rs. Merge them into one edit or target disjoint regions."
        );
        assert_eq!(
            fs::read_to_string(root.join("app.rs")).unwrap(),
            "foo\nfoo\n"
        );
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn replace_all_supports_fuzzy_matches() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("copy.txt"), "title: “Hello”   \ntitle: “Hello”\n")
            .expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["copy.txt"]);

        let result = tool
            .edit(
                json!({
                    "files": [{
                        "path": "copy.txt",
                        "edits": [{"oldContent": "title: \"Hello\"", "newContent": "title: hi", "replaceAll": true}]
                    }]
                }),
                &fingerprints,
            )
            .await
            .expect("fuzzy replaceAll edit should apply");

        assert_eq!(
            fs::read_to_string(root.join("copy.txt")).unwrap(),
            "title: hi\ntitle: hi\n"
        );
        assert_eq!(result.content, "Edited copy.txt (2 replacements).");
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
                    "files": [{
                        "path": "app.rs",
                        "edits": [{"oldContent": "b", "newContent": "B"}]
                    }]
                }),
                &fingerprints,
            )
            .await
            .expect_err("stale fingerprint should fail");

        assert!(error
            .to_string()
            .contains("changed since the last successful read"));
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn descriptor_uses_files_as_top_level_field() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        let tool = EditFileTool::new(&root);
        let schema = tool.descriptor().input_schema;

        assert!(schema["properties"].get("files").is_some());
        assert!(schema["properties"].get("edits").is_none());
        assert_eq!(schema["required"], json!(["files"]));
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn rejects_legacy_top_level_edits_field() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "alpha\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        let error = tool
            .edit(
                json!({
                    "edits": [{
                        "path": "app.rs",
                        "edits": [{"oldContent": "alpha", "newContent": "beta"}]
                    }]
                }),
                &fingerprints,
            )
            .await
            .expect_err("top-level edits should be rejected");

        assert!(error.to_string().contains("unknown field `edits`"));
        assert_eq!(fs::read_to_string(root.join("app.rs")).unwrap(), "alpha\n");
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn accepts_old_text_new_text_aliases() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "alpha\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        tool.edit(
            json!({
                "files": [{
                    "path": "app.rs",
                    "edits": [{"oldText": "alpha", "newText": "beta"}]
                }]
            }),
            &fingerprints,
        )
        .await
        .expect("alias edit should apply");

        assert_eq!(fs::read_to_string(root.join("app.rs")).unwrap(), "beta\n");
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn preserves_bom_and_crlf_line_endings() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("app.rs"), "\u{FEFF}one\r\ntwo\r\nthree\r\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["app.rs"]);

        tool.edit(
            json!({
                "files": [{
                    "path": "app.rs",
                    "edits": [{"oldContent": "two\nthree", "newContent": "deux\ntrois"}]
                }]
            }),
            &fingerprints,
        )
        .await
        .expect("edit should apply");

        assert_eq!(
            fs::read_to_string(root.join("app.rs")).unwrap(),
            "\u{FEFF}one\r\ndeux\r\ntrois\r\n"
        );
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn fuzzy_fallback_handles_smart_punctuation_and_trailing_whitespace() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(
            root.join("copy.txt"),
            "title: “Hello”—world   \nstatus: fine\n",
        )
        .expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["copy.txt"]);

        tool.edit(
            json!({
                "files": [{
                    "path": "copy.txt",
                    "edits": [{"oldContent": "title: \"Hello\"-world", "newContent": "title: \"Hello\" - world"}]
                }]
            }),
            &fingerprints,
        )
        .await
        .expect("fuzzy edit should apply");

        assert_eq!(
            fs::read_to_string(root.join("copy.txt")).unwrap(),
            "title: \"Hello\" - world\nstatus: fine\n"
        );
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn fuzzy_fallback_still_requires_unique_match() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("create temp workspace");
        fs::write(root.join("copy.txt"), "title: “Hello”\ntitle: \"Hello\"\n").expect("write file");
        let tool = EditFileTool::new(&root);
        let fingerprints = fingerprints(&root, &["copy.txt"]);

        let error = tool
            .edit(
                json!({
                    "files": [{
                        "path": "copy.txt",
                        "edits": [{"oldContent": "title: \"Hello\"", "newContent": "title: hi"}]
                    }]
                }),
                &fingerprints,
            )
            .await
            .expect_err("non unique fuzzy match should fail");

        assert_eq!(
            error.to_string(),
            "Found 2 occurrences of the text in copy.txt. The text must be unique. Please provide more context to make it unique."
        );
        fs::remove_dir_all(root).ok();
    }

    fn fingerprints(root: &Path, paths: &[&str]) -> HashMap<String, ReadFingerprint> {
        paths
            .iter()
            .map(|path| {
                let fingerprint =
                    fingerprint_path(root, &root.join(path)).expect("fingerprint file");
                ((*path).to_string(), fingerprint)
            })
            .collect()
    }

    fn unique_temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!("sinew-edit-test-{}", Uuid::new_v4()))
    }
}
