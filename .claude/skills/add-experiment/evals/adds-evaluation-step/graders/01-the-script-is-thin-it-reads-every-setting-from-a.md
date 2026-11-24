---
type: llm
focus: trace
weight: 1
---

The script is thin: it reads every setting from a Hydra config and calls library code; the RMSE and bias are taken from an upstream library if the capability index has them, or the answer says they go upstream, and no metric or loop is defined as reusable code in the script
