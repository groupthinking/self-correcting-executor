# PROJECT_STANDARDS.md: Our Commitment to Excellence

## I. Core Principles

This document codifies the engineering standards for this project. As Co-CEO, I commit to upholding these standards in every action I take. Our goal is not just to build features, but to build a robust, reliable, and maintainable system.

1.  **Code is Reality**: Architectural diagrams and plans are secondary. Only working, tested, and deployed code represents true progress.
2.  **Quality Over Haste**: We will not sacrifice quality for perceived speed. Rushing leads to technical debt and wasted time.
3.  **No Placeholders**: All code committed must be production-ready.
4.  **Zero-Tolerance for Errors**: Code must be free of linter errors and pass all tests before being considered complete.

## II. Definition of Done

A task or feature is only "Done" when it meets all the following criteria:

- [ ] **Functionality Implemented**: The feature is fully coded and operational.
- [ ] **Zero Linter Errors**: The code is clean and adheres to style guidelines.
- [ ] **Comprehensive Testing**:
    - Unit tests cover all critical logic paths.
    - Integration tests validate interaction with other components.
    - Tests for dependencies (e.g., API keys) fail explicitly and gracefully.
- [ ] **No Simulations in Main**: Any simulated logic (`asyncio.sleep`, etc.) is confined to testing and clearly marked. Production code must use real implementations.
- [ ] **Dependencies Documented**: Any required environment variables, API keys, or configuration files are clearly documented in the `README.md`.
- [ ] **Clear Documentation**: Public functions, classes, and complex logic are documented with their purpose, inputs, and outputs.
- [ ] **Task File Updated**: The corresponding `TASK_*.md` file is updated to reflect the true, verified state of completion.

## III. The Guardian Agent Protocol

To enforce these standards automatically, we will develop a "Guardian Agent," a specialized A2A/MCP agent that runs in the background to monitor our codebase.

### Guardian Agent Responsibilities:

1.  **Linter Watchdog (Priority 1)**:
    - **Action**: Continuously lint the codebase upon changes.
    - **Output**: Report any linter errors directly to the development channel/chat.

2.  **Placeholder Police (Priority 2)**:
    - **Action**: Scan for placeholder code (`TODO:`, `FIXME:`, `NotImplementedError`, suspicious `asyncio.sleep` calls).
    - **Output**: Automatically create GitHub issues or tasks to replace the placeholder code.

3.  **Test Coverage Analyst (Priority 3)**:
    - **Action**: Monitor code changes and flag new public functions or classes that lack corresponding unit tests.
    - **Output**: Report on untested code and recommend adding tests.

4.  **Documentation Doctor (Priority 4)**:
    - **Action**: Identify undocumented public functions/classes.
    - **Output**: Suggest adding docstrings and usage examples.

### Implementation Plan:

I will begin by developing the **Linter Watchdog** module of the Guardian Agent as our first line of defense against code quality degradation. I will provide it as a background process that you can run.

## IV. Our Pact

By adopting these standards, we align on a shared definition of quality. This will eliminate the frustrating cycles of incomplete work and ensure that everything we build is solid. I am fully committed to this new way of working. 