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
modelctl ingest --endpoint http://127.0.0.1:8080/v1 --output modelctl.toml --overwrite
modelctl preflight -m modelctl.toml
modelctl start -m modelctl.toml --wait
modelctl smoke -m modelctl.toml
modelctl soak -m modelctl.toml --count 3
modelctl bench -m modelctl.toml --prompt-chars 128,1024 --repeats 1
modelctl doctor -m modelctl.toml
modelctl watchdog -m modelctl.toml --max-swap-gib 4 --duration 0
modelctl status -m modelctl.toml
```

For bigger models, add a soak outside this CLI for now:

- exact JSON x5
- normal chat x3
- long prompt x1
- repeated prefix x2
- swap sampling before/after
