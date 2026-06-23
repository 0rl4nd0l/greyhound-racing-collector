"""Shared output-directory guard for report-only artifact builders."""

from __future__ import annotations

from pathlib import Path


def assert_prefixed_report_output_dir(
    output_dir: Path,
    *,
    repo_root: Path,
    repo_prefix: str,
    artifact_prefix: str,
    prefix_error: str,
    evidence_root: Path | None = None,
) -> Path:
    logical = output_dir if output_dir.is_absolute() else repo_root / output_dir
    candidate = logical.absolute()
    repo_base = repo_root.absolute()
    try:
        relative = candidate.relative_to(repo_base)
    except ValueError as exc:
        if evidence_root is None:
            raise ValueError("output_dir_must_be_inside_repo") from exc
    else:
        if ".." in relative.parts:
            raise ValueError("output_dir_must_not_contain_parent_traversal")
        if relative.as_posix().startswith(repo_prefix):
            return candidate
        raise ValueError(f"{prefix_error}:{relative}")

    evidence_base = evidence_root if evidence_root.is_absolute() else repo_root / evidence_root
    try:
        relative = candidate.relative_to(evidence_base.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo_or_evidence_root") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if relative.parts and relative.parts[0].startswith(artifact_prefix):
        return candidate
    raise ValueError(f"{prefix_error}:{relative}")
