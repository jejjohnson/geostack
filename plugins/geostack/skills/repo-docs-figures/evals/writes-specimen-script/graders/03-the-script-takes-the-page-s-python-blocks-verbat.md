---
type: llm
focus: trace
weight: 1
---

The script takes the page's Python blocks verbatim and in order and runs them in one namespace (it reads them from kernels.md, or carries them unchanged), and draws from what they computed; it does not recompute the means with its own code
