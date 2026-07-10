# Promotion receipts

Capstan can require a hash-pinned benchmark receipt before a candidate promotion mutates lifecycle state.

## Stable policy

The stable manifest can make receipts mandatory for every candidate:

```toml
[promotion]
require_receipt = true
```

Operators can enforce the same rule per invocation with `promote --require-receipt`.

## Candidate manifest

```toml
[promotion.receipt]
path = "/Volumes/ModelSSD/logs/evals/candidate-receipt.json"
sha256 = "64-character-sha256-of-the-exact-receipt-file"
max_age_sec = 86400
require_decision = "promote"
required_gates = ["logit", "quality"]
```

Manifests without `[promotion.receipt]` retain the pre-v0.24.5 behavior unless the stable manifest or CLI requires one. Once configured or required, both promotion planning and execution fail closed unless the receipt validates. Execution revalidates the receipt under the promotion lifecycle lock immediately before rotation.

## Receipt schema

```json
{
  "schema": "capstan-promotion-receipt-v1",
  "generated_at": "2026-07-10T14:35:00+00:00",
  "decision": "promote",
  "candidate_fingerprint": "...",
  "gates": {
    "logit": {"pass": true, "strict_pass": true},
    "quality": {"pass": true, "score": 4.2, "blocking": []}
  }
}
```

Each required gate must be an object with an exact JSON boolean `"pass": true`. A rejected decision, false/missing gate, stale timestamp, candidate drift, or receipt-byte change blocks promotion before `Popen`.

## Workflow

1. Get the binding expected for the exact candidate launch and required artifacts:

   ```bash
   capstan -m candidate.toml receipt fingerprint > candidate-binding.json
   ```

2. Run the evaluator against that candidate configuration. The evaluator writes `capstan-promotion-receipt-v1` with the returned `candidate_fingerprint`, its gate details, and an explicit decision.

3. Pin the exact receipt bytes in the candidate manifest:

   ```bash
   shasum -a 256 /path/to/candidate-receipt.json
   ```

4. Validate without mutation, then plan or execute promotion:

   ```bash
   capstan -m candidate.toml receipt validate
   capstan -m stable.toml promote --candidate candidate.toml
   capstan -m stable.toml promote --candidate candidate.toml --execute
   ```

## Trust and drift checks

Capstan requires the receipt to be:

- a regular, non-symlink, single-link file owned by the current user;
- not group/world writable;
- no larger than 1 MiB;
- byte-for-byte equal to the SHA-256 pinned in the manifest;
- fresh, timezone-aware, and not materially future-dated;
- bound to the candidate model ID, endpoint, launch command/CWD, hashed `start.env`, and current `[preflight].required_paths` content fingerprints;
- backed by at least one receipt-bound required artifact; each must be a regular file no larger than 128 MiB and is fully SHA-256 hashed.

Use model config files, sidecar manifests, compact indexes, or wrapper scripts as receipt-bound artifacts. Do not list model directories or giant shards for receipt binding; list the small digest manifest that identifies them.

The receipt file can live on a writable external volume because its bytes are hash-pinned in the manifest and read with no-follow/inode consistency checks.
