```markdown
# cat-food-unsupervised-survey Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns and workflows used in the `cat-food-unsupervised-survey` repository. The codebase is written in Python and focuses on unsupervised learning survey tasks, with a strong emphasis on incremental content development (notably, building up presentations). The repository follows clear coding conventions and structured commit messages to ensure maintainability and clarity.

## Coding Conventions

### File Naming
- Use **snake_case** for all file names.
  - Example: `data_loader.py`, `model_utils.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .utils import preprocess_data
    ```

### Export Style
- Use **named exports** (i.e., explicitly define what is exported from a module).
  - Example:
    ```python
    __all__ = ['train_model', 'evaluate_model']
    ```

### Commit Messages
- Follow **conventional commit** style.
- Use the `feat` prefix for new features.
  - Example: `feat: add data normalization to preprocessing pipeline`

## Workflows

### Incremental Presentation Development
**Trigger:** When you want to add new content or sections to the main presentation slides.  
**Command:** `/add-presentation-slides`

1. Open `presentation/full-presentation-slides.html`.
2. Add new slides or sections as needed (e.g., slides 6-10, 11-15).
3. Save your changes.
4. Commit with a message indicating which slides or sections were added.
   - Example: `feat: add slides 11-15 on clustering evaluation`
5. Push your changes to the repository.

**Example HTML Slide Addition:**
```html
<!-- Slide 11 -->
<section>
  <h2>Clustering Evaluation Metrics</h2>
  <ul>
    <li>Silhouette Score</li>
    <li>Davies-Bouldin Index</li>
  </ul>
</section>
```

## Testing Patterns

- **Framework:** Unknown (not detected in the repository).
- **File Pattern:** Test files are named with the `.test.ts` suffix, suggesting a TypeScript-based testing approach for some components, though the main codebase is Python.
- **Best Practice:** Place test files alongside the modules they test, following the `*.test.ts` pattern.

## Commands

| Command                   | Purpose                                                |
|---------------------------|--------------------------------------------------------|
| /add-presentation-slides  | Add new slides or sections to the main presentation.   |
```
