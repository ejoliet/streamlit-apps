---
name: feature-implementation-update
description: Workflow command scaffold for feature-implementation-update in streamlit-apps.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /feature-implementation-update

Use this workflow when working on **feature-implementation-update** in `streamlit-apps`.

## Goal

Implements or updates a feature by modifying core app logic and supporting files such as README and requirements.

## Common Files

- `s3-explorer/app.py`
- `s3-explorer/auth.py`
- `s3-explorer/firefly_connector.py`
- `s3-explorer/requirements.txt`
- `s3-explorer/README.md`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit or create main app logic file (e.g., app.py)
- Update or create supporting modules (e.g., auth.py, firefly_connector.py)
- Update requirements.txt if dependencies change
- Update README.md to document the feature

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.