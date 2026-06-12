---
name: documentation-update
description: Workflow command scaffold for documentation-update in streamlit-apps.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /documentation-update

Use this workflow when working on **documentation-update** in `streamlit-apps`.

## Goal

Updates documentation files to reflect new features, fixes, or usage instructions.

## Common Files

- `s3-explorer/README.md`
- `s3-explorer/AGENTS.md`
- `s3-explorer/CLAUDE.md`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit README.md or other documentation files (e.g., AGENTS.md, CLAUDE.md)

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.