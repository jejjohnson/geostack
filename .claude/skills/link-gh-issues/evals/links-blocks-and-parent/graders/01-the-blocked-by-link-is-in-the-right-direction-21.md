---
type: llm
focus: trace
weight: 1
---

The blocked-by link is in the right direction: #215 is the blocked issue and #212 the blocker (`make gh-block ISSUE=215 BLOCKED_BY=212`, or addBlockedBy with issueId for #215 and blockingIssueId for #212)
