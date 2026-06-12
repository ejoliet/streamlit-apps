```markdown
# streamlit-apps Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill provides guidance on contributing to the `streamlit-apps` repository, a Python-based collection of Streamlit applications. It covers coding conventions, common development workflows, and testing patterns observed in the codebase. Whether you're implementing new features, updating dependencies, or improving documentation, this guide will help you follow established patterns and streamline your contributions.

## Coding Conventions

### File Naming
- Use **PascalCase** for file names.
  - Example: `FireflyConnector.py`, `App.py`

### Imports
- Use **relative imports** within modules.
  - Example:
    ```python
    from .auth import authenticate_user
    from .firefly_connector import FireflyConnector
    ```

### Exports
- Use **named exports** (i.e., define specific functions/classes for import elsewhere).
  - Example:
    ```python
    def authenticate_user(...):
        ...
    class FireflyConnector:
        ...
    ```

### Commit Messages
- No strict format; freeform style.
- Average length: ~31 characters.
- Prefixes are not required.

## Workflows

### Feature Implementation or Update
**Trigger:** When adding a new feature or improving an existing one in the app  
**Command:** `/feature-impl`

1. Edit or create the main app logic file (e.g., `app.py`).
2. Update or create supporting modules (e.g., `auth.py`, `firefly_connector.py`).
3. If dependencies change, update `requirements.txt`.
4. Update `README.md` to document the new or updated feature.

**Example:**
```python
# s3-explorer/app.py
from .auth import authenticate_user
from .firefly_connector import FireflyConnector

def main():
    user = authenticate_user()
    connector = FireflyConnector()
    # ... feature logic ...
```

### Dependency or Dockerfile Update
**Trigger:** When adding/updating dependencies or Dockerfile for deployment  
**Command:** `/update-deps`

1. Edit `requirements.txt` to add or update dependencies.
2. Edit or add `Dockerfile` as needed for deployment.
3. Optionally update `.env.example` or related config files.

**Example:**
```dockerfile
# s3-explorer/Dockerfile
FROM python:3.10
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

### Documentation Update
**Trigger:** When documenting a new feature or updating usage instructions  
**Command:** `/update-docs`

1. Edit `README.md` or other documentation files (e.g., `AGENTS.md`, `CLAUDE.md`).

**Example:**
```markdown
## New Feature: S3 Explorer
- Added authentication and Firefly integration.
- See usage instructions below.
```

## Testing Patterns

- Test files follow the pattern: `*.test.*`
- Testing framework is **unknown**; check for files like `app.test.py` or `firefly_connector.test.py`.
- To add a test:
  - Create a file named `ModuleName.test.py` alongside the module.
  - Follow the project's existing test structure.

**Example:**
```python
# s3-explorer/firefly_connector.test.py
def test_firefly_connection():
    connector = FireflyConnector()
    assert connector.connect() is True
```

## Commands

| Command        | Purpose                                               |
|----------------|-------------------------------------------------------|
| /feature-impl  | Start a new feature implementation or update workflow |
| /update-deps   | Update dependencies or Dockerfile                     |
| /update-docs   | Update documentation files                            |
```
