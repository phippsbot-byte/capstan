# Local model lifecycle

A model is not "installed" until all of this is true:

1. Artifacts are identified and classified.
2. Runtime is known.
3. Start command is captured in a manifest.
4. Preflight passes.
5. Server reaches readiness.
6. Smoke test passes.
7. Cleanup candidates are documented.
8. Stop/restart works.

## Artifact classes

- **Active model artifact**: do not delete unless retiring the model.
- **Secondary sidecar / cache lane**: active only for specific runtimes; document it.
- **Runtime cache**: usually safe to delete after stopping the server.
- **Download cache**: often safe if the real model dir exists elsewhere.
- **Old experiment**: unsafe until a human signs off.

## Promotion gates

Minimum useful gate:

```bash
modelctl preflight -m modelctl.toml
modelctl start -m modelctl.toml --wait
modelctl smoke -m modelctl.toml
modelctl status -m modelctl.toml
```

For bigger models, add a soak outside this CLI for now:

- exact JSON x5
- normal chat x3
- long prompt x1
- repeated prefix x2
- swap sampling before/after
