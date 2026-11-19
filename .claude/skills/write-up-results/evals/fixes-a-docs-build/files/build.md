I added docs/notes/results.md and ran `make docs`. Output (trimmed):

```
⛔️ myst.yml The template "book-theme" could not be downloaded: 403 Forbidden
⚠️  docs/notes/results.md Could not link citation with label "okafor2031".
⚠️  docs/notes/results.md:14 Cross reference target was not found: fig-rmse
📖 Built docs/notes/background.md in 12 ms.
📖 Built docs/notes/methods.md in 9 ms.
```

The results page does not show up in the site's left-hand navigation at all. The page has this in it:

```markdown
The emulator reaches an RMSE of 0.12 ppb [@okafor2031], see {numref}`fig-rmse`.

![](../../results/figures/rmse.png)
```

I got the 0.12 from a run in my terminal yesterday.
