# GEMINI.md

This file provides context and instructions for LLM coding agents working on the `generation-ai` project.

## Project Vision

`generation-ai` provides end-to-end solutions demonstrating application of AI to real-world business tasks, including ongoing fine-tuning / reinforcement learning and model serving and evaluation.

## Key Principles for Agents

- **End-to-End Solutions**: Focus on complete, functional examples that solve real business problems.
- **Modern AI Practices**: Employ best practices for fine-tuning, serving, and evaluating AI models.
- **GKE Integration**: Solutions should be designed to run effectively on Google Kubernetes Engine (GKE).
- **Clarity and Documentation**: Code should be well-documented and easy to follow, serving as a reference for users.
- **Testability**: Ensure that implementations are well-tested, including end-to-end tests where appropriate.

## Development Workflow

- Adhere to the project's coding style and structure.
- Follow the PR hygiene mentioned in the project's instructions:
    - Solve only the specific issue.
    - One idea per PR.
    - Well-structured commits.
    - Reference issues in the commit body.

### Commands

The project uses the `ap` tool for various tasks. Since `ap` is a custom tool, it should be run using `go run`:

- `go run github.com/gke-labs/gke-labs-infra/ap@latest generate`: Regenerate any code and format.
- `go run github.com/gke-labs/gke-labs-infra/ap@latest test`: Run unit tests.
- `go run github.com/gke-labs/gke-labs-infra/ap@latest e2e`: Run e2e tests.
- `go run github.com/gke-labs/gke-labs-infra/ap@latest lint`: For deeper static analysis.

**Reminder**: Coding agents MUST run at least `ap generate` before sending PRs, and preferably `ap e2e` as well!
