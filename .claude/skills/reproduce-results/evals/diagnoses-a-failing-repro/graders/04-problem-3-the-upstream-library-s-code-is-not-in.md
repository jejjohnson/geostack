---
type: llm
focus: trace
weight: 1
---

Problem 3: the upstream library's code is not in the stage's deps, so DVC cannot see the change; the fix is to add the library's directory (under references/) or the lock that pins it
