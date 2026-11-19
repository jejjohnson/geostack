---
type: llm
focus: trace
weight: 1
---

The code uses no .T, axis=, reshape or [:, None] on dense arrays; contractions go through einx or a linear operator's own methods
