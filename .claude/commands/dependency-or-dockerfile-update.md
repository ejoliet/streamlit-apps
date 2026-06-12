---
name: dependency-or-dockerfile-update
description: Workflow command scaffold for dependency-or-dockerfile-update in streamlit-apps.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /dependency-or-dockerfile-update

Use this workflow when working on **dependency-or-dockerfile-update** in `streamlit-apps`.

## Goal

Updates dependencies or Dockerfile to support new features or deployment requirements.

## Common Files

- `s3-explorer/requirements.txt`
- `s3-explorer/Dockerfile`
- `gcn-voevent-monitor/.env.example`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit requirements.txt to add or update dependencies
- Edit or add Dockerfile as needed
- Optionally update .env.example or related config files

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.