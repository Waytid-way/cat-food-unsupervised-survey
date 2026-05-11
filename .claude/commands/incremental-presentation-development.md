---
name: incremental-presentation-development
description: Workflow command scaffold for incremental-presentation-development in cat-food-unsupervised-survey.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /incremental-presentation-development

Use this workflow when working on **incremental-presentation-development** in `cat-food-unsupervised-survey`.

## Goal

Add new sections or slides to the main presentation file, typically in batches (e.g., slides 6-10, 11-15), iteratively building up the presentation content.

## Common Files

- `presentation/full-presentation-slides.html`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit 'presentation/full-presentation-slides.html' to add new slides or sections.
- Commit changes with a message indicating which slides or sections were added.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.