#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Static gate for the rebased branch.
#
# This is NOT redundant with CI: .github/workflows/pre-commit.yml only
# triggers on pull_request against main/master, so a PR based on
# rfc/elastic_scaling gets no pre-commit run at all. Run this locally.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

BASE_REF="${BASE_REF:-2bede02}" # tip of rfc/elastic_scaling
HEAD_REF="${HEAD_REF:-HEAD}"

if command -v uv >/dev/null 2>&1; then
    runner=(uv run --group dev pre-commit)
elif command -v pre-commit >/dev/null 2>&1; then
    runner=(pre-commit)
else
    echo "pre-commit not available: run 'uv sync --group dev' first" >&2
    exit 1
fi

echo "== pre-commit ${BASE_REF}..${HEAD_REF} =="
"${runner[@]}" run --from-ref "${BASE_REF}" --to-ref "${HEAD_REF}"

# Hooks autofix (ruff-format, markdownlint, trailing whitespace). If anything
# changed, fold it back into the single elastic commit before pushing:
#   git add -A && git commit --amend --no-edit
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo
    echo "NOTE: hooks modified files. Review them, then amend into the commit."
    git status --short
fi
