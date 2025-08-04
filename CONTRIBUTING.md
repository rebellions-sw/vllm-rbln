# Contributing to vllm-rbln

Welcome! 🎉 Thank you for your interest in contributing to **vllm-rbln**, a plugin extension for the[vLLM project](https://github.com/vllm-project/vllm). As an open-source project, we rely on the support and involvement of the community to help make this project better for everyone. This document outlines our contribution process, coding guidelines, and community standards.

We value transparency, collaboration, and a safe environment for contributors. All contributions are expected to follow these guidelines.

------

## Getting Started

### Contributors

1. **Fork the repository** and create your branch from main.
2. Make your changes with clear and concise commits.
3. Ensure that your code follows the style and linting rules.
4. If relevant, update or add new tests and documentation.
5. Open a pull request with a detailed description of your changes.

### Core Contributors & Collaborators

1. **Create your branch** and work on branches within the repository.
2. Make your changes with clear and concise commits.
3. Ensure that your code follows the style and linting rules.
4. If relevant, update or add new tests and documentation.
5. Open a pull request with a detailed description of your changes.

All contributors must use **English** for issues, comments, and code.

------

## 💡 How You Can Contribute

One of the best ways to contribute to the project is by creating issues — whether you're reporting a bug, suggesting a new idea, implementing nice features, or asking a question.

If you’ve found something that needs attention or improvement, we’d love to hear from you!
Your input helps us make **vllm-rbln** better for everyone.

When creating an issue, please provide as much detail as possible and select the appropriate label to help us triage and respond efficiently. 🙏

### General Issue

These issues are used to discuss general suggestions, requests, bug reports, and other topics.

- proposal: Suggest enhancements or new functionality that would benefit the vllm-rbln.
- request: Request a specific development task that you think should be implemented.
- bug: Help us identify and fix issues by reporting bugs with clear reproduction steps.
- question: Ask general questions about using the project, understanding behavior, or seeking clarification. Ideal for newcomers or anyone unsure about how something works.
- discussion: Start open-ended conversations about design decisions, optimization features, etc. Useful for gathering community feedback before moving to a proposal.
- help wanted: Highlight tasks where contributor support is requested. Often used in combination with other labels like bug or question.

### Development-related Issue

These issue types represent development tasks that are typically addressed through pull requests.

- feature: Develop a new capability or functionality in the codebase. Should be scoped and accompanied by acceptance criteria or use cases if possible.
- model: Issues related to adding, modifying, or improving support for specific ML models. Include model details (e.g., architecture).
- core: Changes that impact core engine components such as worker, model runner, scheduler, memory management, or plugin infrastructure. These usually require in-depth review and testing.
- bug-fix: Tracks the resolution of known bugs.
- perf: Implement improvements focused on performance, such as latency reduction, memory usage, or throughput. Include benchmarks or measurement methodology if available.
- refactor: Improve readability, maintainability, or consistency without altering external behavior. Includes renaming, code modularization, or dependency cleanup.
- docs: Improve or add to documentation. Includes README, usage guides, code comments, and tutorial examples. Helpful for improving project onboarding and understanding.
- other: Any development-related task that doesn't fit the above categories. Use this label sparingly, and consider proposing a new label if recurring themes emerge.

Please choose labels appropriately when opening an issue.

------

## Pull Request Guidelines

All pull requests **must**:

- Have a corresponding issue: refer to Development-related Issue
- Include a clear title following[**Conventional Commits v1.0**](https://www.conventionalcommits.org/en/v1.0.0).
- Contain the following in the description:
  - Purpose and detailed explanation
  - Related issue number
  - Affected modules (e.g., Platform, Worker, Runner, Attn, Models, optimum)
  - Type of change (use Labels): Feature / Bug Fix / Refactor / ....
  - Describe How to Test and a summary of expected results

💡 Individual commit messages in PR branches do not need to strictly follow Conventional Commits, but should remain readable and descriptive.

------

## Merge Policy

All of the following must be satisfied for a PR to be merged:

- ✅ All CI tests must pass
- ✅ At least one approval from the relevant team
- ✅ Allow only **Squash and merge**

------

## 🙌 Thank You

Thank you for taking the time to contribute to **vllm-rbln**!
Whether you're submitting a pull request, opening an issue, improving documentation, or simply asking thoughtful questions — your effort helps strengthen the project and the community around it.
We believe that great software is built in the open, by people who care. We're excited to have you on board, and we look forward to your contributions. 🚀
