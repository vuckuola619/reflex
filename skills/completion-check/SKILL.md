---
name: completion-check
description: Decide whether an agent task is complete from explicit acceptance criteria, tests, requested artifacts, and unresolved failures.
---
# Completion check

1. Convert the user's request into explicit acceptance criteria.
2. Evaluate deterministic criteria first: required files exist, tests pass, commands exited successfully, and requested artifacts were produced.
3. Use semantic judgment only for criteria that cannot be checked directly.
4. Return complete only if every required criterion passes; otherwise return incomplete and identify the unresolved criterion in the harness trace.
