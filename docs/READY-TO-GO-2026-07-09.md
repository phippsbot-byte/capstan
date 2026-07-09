# Capstan ready-to-go plan — 2026-07-09

## Operating decision

Capstan remains a **lifecycle and governance product for giant local models**.

The next release is not another model runtime. It is the evidence layer that turns runtime experiments into defensible promotion decisions.

## Rules from this point forward

1. `main` is the released lifecycle product.
2. Model-specific runtime research stays in a lab branch/repo until it meets explicit gates.
3. Raw benchmark blobs do not land in `main`; keep compact summaries and external/release artifacts.
4. Every explicit lifecycle/benchmark run and state-changing supervisor event produces a Capstan receipt; routine health-loop samples are aggregated, not emitted as thousands of files.
5. “Promoted” must mean the candidate passed an incumbent comparison policy, not merely that it started successfully.
6. Integrate existing servers and routers; do not rebuild them.
7. No further Phipps quality evaluation without Nate’s explicit sign-off.
8. Cache-policy micro-tuning on Hy3 is closed. Future Hy3 work must attack kernel/runtime architecture.
9. Receipts are content-addressed and append-only by default; edited or stale evidence must never authorize promotion.

## Phase 0 — contain the current branch

### Actions

- [ ] Keep PR #15 in Draft.
- [ ] Add a prominent lab-only/non-merge warning to PR #15 and its top-level experiment README.
- [ ] Create a compact research index: hypothesis, command, result summary, verdict, next/stop decision.
- [ ] Move large NPZ/raw JSON artifacts to release assets, object storage, or a dedicated lab artifact location.
- [ ] Keep only small reproducibility fixtures and compact result summaries in Git.
- [ ] Extract reusable code only through focused PRs with a product caller and tests.
- [ ] Close or supersede stale experiment status text that predates the latest direct-q4/cache results.

### Exit gate

PR #15 is preserved as evidence, but no longer looks like a candidate merge into `main`.

## Phase 0.5 — v0.24.4 Safety Gate

Ship this before the evidence work.

### Actions

- [ ] Centralize preflight enforcement before ordinary `start`, executed fleet recovery, daemon-triggered restart, rotate target start, rotate rollback, and promotion rollback.
- [ ] Make `start --wait` return non-zero when readiness fails.
- [ ] Preserve a narrowly scoped stable-port exception only for the target behind the same readiness-gated rotate/promote deployment identity; all other preflight checks still apply to target and rollback starts.
- [ ] Reject malformed/loosely coerced manifest booleans, negative/non-finite limits, and incompatible fields.
- [ ] Replace cleanup's trusted `safe = true` behavior with canonical-root containment, protected-path rules, symlink-safe deletion, and active-manifest/PID checks.
- [ ] Keep `--force` explicit, but do not let it delete protected system, home, or active-model roots without a separate exact-path confirmation contract.
- [ ] Gate release-asset and PyPI publication on the same passing test/build job.

### Exit gate

Every path that starts or restarts a heavyweight model applies the same preflight contract, readiness failure is visible to automation, and cleanup cannot delete arbitrary manifest-selected paths merely because they are marked `safe`.

## Phase 1 — v0.25 Evidence Spine

Target: 1–2 focused weeks.

### 1. Extend manifest provenance

Add optional, backward-compatible sections:

```toml
[artifact]
source = "huggingface|local|custom"
repo = "org/model"
revision = "commit-or-release"
format = "gguf|mlx|safetensors|custom"
quantization = "..."
paths = ["..."]
identity = "operator-supplied digest or artifact ID"

[runtime]
name = "llama.cpp|mlx_lm|custom"
repo = "org/repo"
revision = "commit-or-release"
binary = "/path/to/binary"
version_command = ["/path/to/binary", "--version"]

[policy]
role = "incumbent|candidate|dormant|inspect"
comparison = "policies/giant-local.toml"
```

Do not hash 282 GiB on every run. Create an explicit artifact-lock operation that records immutable upstream revisions plus a checksum manifest (or operator-supplied trusted digest set) once, then cheaply verifies path/stat drift during routine runs. Runtime binaries, launch scripts, manifests, policies, and evaluator code/config are small enough to hash for each decision receipt.

### 2. Add canonical run receipts

Every explicit receipt-bearing operation should write a receipt containing:

- receipt schema version and execution `run_id`;
- timestamp and command;
- allowlisted effective-manifest fields;
- manifest digest;
- artifact/runtime identity;
- artifact-lock digest and verification status;
- hardware and OS snapshot;
- Capstan version/commit;
- command name plus explicitly allowlisted arguments and environment fields; secret-bearing values are omitted or replaced by a digest;
- adapter-schema-allowlisted output evidence and exactness result;
- prompt/completion latency and throughput;
- swap before/after/delta;
- adapter-schema-allowlisted custom metrics; unknown fields are rejected from the persisted receipt;
- policy and evaluator implementation/config digests;
- final verdict and issues/warnings.

Store under the compatibility state root initially:

`~/.local/state/modelctl/runs/<model>/<receipt-id>.json`

Receipt rules:

- assign `run_id` independently when execution starts;
- derive `receipt_id` as SHA-256 of the canonical receipt payload with the `receipt_id` field omitted; `run_id` remains inside the hashed payload;
- store the completed object as `<receipt-id>.json`, include `receipt_id` in the stored envelope, and verify by removing that field and recomputing the payload digest on read;
- create with no-overwrite semantics and verify the digest on read;
- use owner-only filesystem permissions;
- select persisted environment/manifest fields from an explicit allowlist rather than trying to guess and redact every possible secret name;
- store bounded, schema-allowlisted output evidence or digests by default, not unbounded raw prompts/responses;
- separate explicit runs from supervisor samples;
- emit supervisor receipts only for incidents, restarts, recoveries, state transitions, and at most one periodic rollup per model per hour;
- retain ordinary receipts for 90 days subject to per-model caps of 10,000 files and 2 GiB; prune only through a dry-run-first explicit command;
- never auto-prune promotion, rollback, rejection, incident, or other decision receipts—or artifact locks and receipts they reference; removing pinned evidence requires an explicit unlink/delete workflow with dependency reporting;
- require promotion-time revalidation that manifest, artifact lock, runtime binary/script, policy, evaluator, hardware policy, and freshness window still match the passing receipt.

Receipt-bearing operations are explicit preflight/validate, start/stop, rotate/promote/rollback, smoke/soak/bench/health/evaluator runs, `compare`, cleanup execution, fleet recovery, and service state changes. Pure reads such as status/list/show do not create receipts. Supervisor loops aggregate routine samples and emit only incidents, restarts, recoveries, state transitions, and bounded rollups.

### 3. Add evaluator adapters

A generic adapter contract should accept a command that emits JSON metrics. This lets Capstan ingest:

- Phipps results;
- llama.cpp timings/metrics;
- Hy3 cache/read/parity telemetry;
- custom correctness probes;
- future standardized benchmarks.

Capstan owns the receipt and policy result, not the evaluator implementation.

### 4. Add comparison policies

Create a versioned policy schema with:

- hard safety floors;
- correctness requirements;
- maximum regression ceilings;
- latency/throughput thresholds;
- memory/swap thresholds;
- optional quality metrics;
- required evidence count and repeatability.

Keep two concepts explicit:

- **qualification/comparison** — evaluate one lane independently or compare different models only when the policy declares metrics compatible;
- **promotion** — replace an incumbent behind the same deployment role/stable API identity, with rotate/rollback eligibility verified separately.

Command shape:

```bash
capstan compare \
  --baseline <receipt-or-run-set> \
  --candidate <receipt-or-run-set> \
  --policy policies/giant-local.toml
```

Qualification output: `qualified`, `archive_only`, `rejected`, or `insufficient_evidence`. A separate same-role eligibility check may produce `promotion_eligible`; cross-model comparison can never produce that verdict by itself.

### 5. Fix promotion semantics

`capstan promote` requires a fresh `promotion_eligible` receipt. An emergency deployment path must use a distinct `forced_deployment` action/status, record operator reason plus rollback target, and must not update the canonical incumbent or describe the result as promoted.

Promotion receipt must include:

- incumbent and candidate identities;
- comparison/policy identity;
- rotate/readiness/post-health results;
- canonical incumbent update;
- rollback target;
- final decision.

Add:

```bash
capstan explain <model>
```

It should answer why the model is live and point to the promotion receipt.

### v0.25 acceptance criteria

- [ ] Existing v0.24 manifests still load unchanged.
- [ ] DS4 can emit a full receipt without model-specific Capstan code.
- [ ] A synthetic second runtime can emit the same schema.
- [ ] Repeated runs are queryable by model, runtime, date, and verdict.
- [ ] Comparison blocks a known latency or correctness regression.
- [ ] Promotion records rollback lineage.
- [ ] Tests are split into focused modules rather than growing one giant test file.
- [ ] CI covers Python 3.11 and 3.12; macOS remains the service-integration lane.
- [ ] Release-asset and PyPI publication jobs depend on a passing test/build gate instead of publishing independently.

## Phase 2 — connect the lab to the product

Target: 1 focused week after v0.25.

### Actions

- [ ] Define a `capstan-labs` result adapter or separate `capstan-hy3` lab repo.
- [ ] Convert existing DS4 and Hy3 summary artifacts into canonical Capstan receipts.
- [ ] Keep the Hy3 runtime out of the core package.
- [ ] Create one Capstan campaign file for each candidate shape.
- [ ] Make every future spike end with one of: integrate, archive, reject.
- [ ] Capture runtime commit/binary identity for the DS4 custom llama.cpp fork.
- [ ] Capture the exact primary/secondary sidecar artifact identities.

### Exit gate

Capstan can ingest DS4 and Hy3 receipts under the same schema, qualify each independently, and compare only their policy-declared common metrics without hard-coding either model. This does not make Hy3 eligible to replace DS4 automatically.

## Phase 3 — second giant-model lane

Target: 2–4 weeks, only after the evidence spine exists.

The point is not necessarily to promote Hy3. The point is to prove Capstan can govern a second unrelated 100B+ runtime.

Candidate choices:

1. Hy3, if a credible kernel path remains.
2. Another sparse model with existing upstream runtime support.
3. A second DS4 configuration only as an integration baseline—not as proof of runtime generality.

### Hy3 continue gate

Allow at most two focused compute experiments, each with a written hypothesis before implementation.

Allowed directions:

- tiled/simdgroup direct q4 Metal kernel;
- deeper fused route/down/weighting kernel;
- MPS-backed batch path;
- upstream MLX/llama.cpp support that removes custom hot-loop overhead.

Disallowed directions:

- more LRU ordering tweaks;
- larger cache guesses without trace evidence;
- server/UI polish before runtime viability;
- broad rewrites without a benchmark gate.

### Hy3 measurement contract

Pin the existing evidence, but be honest: the old run did not record a complete invocation, so it is a historical baseline—not yet a fully reproducible gate.

- research commit: `42b45812a1d23e8fabc2c423d17dba6cded631f2`;
- summary: `experiments/hy3-mlx-canary/results/20260709-cpp-route-packed-cache-summary.json`;
- summary SHA-256: `d1dd3083b5cac2a7b90a0fbda01078d7e37604423a2b178a17b92641e9e51627`;
- forced-logit evidence: `experiments/hy3-mlx-canary/results/20260708-cpp-route-q4-forced-logits-delta-summary.json`;
- forced-logit SHA-256: `e6b24b126c4140dbbc5a2f6fc2c876ba13272c7bef69012a8a350cc931483f21`.

Historical baseline shape and results:

- exact 30-token benchmark-note prompt embedded in the summary;
- `topk=8`, direct q4, 16 GiB packed cache, four generated tokens;
- first step **73.242s**;
- four-step sum **82.298s**;
- median decode step **2.994s**;
- decode rate **0.334 tok/s**, defined as `1 / median(decode_step_seconds)` for one-token decode steps after the first step;
- reads **75.493 GiB**;
- swap delta **0.0 GiB**.

Before either allowed kernel experiment, add one versioned `hy3_gate` command on the research branch. It must emit full allowlisted argv/environment, source commit, model/layout/index identities, prompt text and digest, forced-sequence fixture identities, runtime/cache/page-cache state, raw per-step timings, generated IDs, logit deltas, read accounting, and swap samples into one receipt. No next experiment runs until that command exists.

For each candidate kernel:

- run through that canonical gate command on the same 96 GiB Studio with DS4 stopped, no competing model server, and starting swap ≤ 4 GiB;
- define “runtime cold” as a fresh daemon with empty dense/packed runtime caches; record OS page-cache state and never call a result physical-cold unless the page cache was deliberately purged and recorded;
- use three fresh-process runs and gate on their median;
- use the same prompt, top-k, cache budgets, four-token generation shape, and pinned forced-logit evidence set;
- report first-step wall, four-step sum, median decode-step seconds, decode tok/s using the formula above, routed compute, bytes/read calls, swap delta, generated IDs, and parity/logit deltas;
- require median four-step sum ≤ **61.724s** (25% below 82.298s) before integrating a candidate kernel into the runtime;
- treat any changed benchmark shape as a new lane, not as a comparison to this baseline.

### Hy3 R&D viability gate

Continue toward candidate status only if the end-to-end top-8 lane reaches all of:

- median short-prompt first step ≤ 30s under the measurement contract;
- median decode rate ≥ 1.0 tok/s using `1 / median(decode_step_seconds)`;
- swap delta ≤ 4 GiB on the bounded test;
- stable repeated requests without daemon/protocol failure;
- top-1/logit behavior within an accepted correctness policy over the pinned fixture set;
- passes the candidate-kernel integration threshold in the measurement contract above.

### Hy3 promotion-candidate gate

Before any real quality evaluation or promotion attempt:

- median short-prompt first step ≤ 20s under the same measurement contract;
- median decode rate ≥ 2.0 tok/s using the same formula;
- 20-request soak passes;
- exact JSON/tool scaffolding passes;
- no unacceptable argmax drift under the approved policy;
- quality evaluation plan receives Nate’s sign-off;
- all results exist as Capstan receipts.

If two serious kernel experiments fail the R&D gate, freeze Hy3 and wait for upstream runtime progress. A research proof is still a win; an eternal custom runtime is not.

## Phase 4 — fleet control plane

Only after receipts/comparison work.

### High-value additions

- remote/peer fleet inventory for Studio + Mini;
- runtime version drift detection;
- service/manifest/run-history API;
- concise TUI or read-only dashboard built from canonical receipts;
- optional llama-swap/llama.cpp-router adapter;
- fleet-wide receipt summaries built from the v0.25 retention contract.

### Do not build yet

- chat UI;
- model downloader/catalog;
- proxy/router;
- training pipeline;
- recommendation engine based on scraped benchmark vibes;
- Kubernetes/cloud orchestration;
- general dashboard before the data model is trustworthy.

## Packaging and product cleanup

After v0.25 evidence work is stable:

- [ ] Decide the `local-modelctl` → Capstan distribution migration.
- [ ] Keep `modelctl` console/state/env compatibility for at least one major release.
- [ ] Move implementation modules toward `capstan.*` incrementally.
- [ ] Add GitHub topics, a roadmap, and issue templates.
- [ ] Publish one opinionated tutorial: “Promote a 100B+ custom local model without bricking your Mac.”
- [ ] Ask 3–5 serious local-model operators to try the lifecycle, not the Hy3 runtime.

## v1.0 definition of done

- [ ] Three heterogeneous giant-model manifests work without core special cases.
- [ ] Two runtimes are real, not synthetic.
- [ ] Every explicit run has provenance and a content-addressed, tamper-evident receipt.
- [ ] Candidate comparison has blocked at least one known regression.
- [ ] Promotion and rollback have been exercised live.
- [ ] `capstan explain` answers why the incumbent is live.
- [ ] Fleet status/health/doctor remain truthful under dormant/down/invalid states.
- [ ] One fresh machine can reproduce a lane from docs plus declared external artifacts.
- [ ] No destructive command is implicit.
- [ ] Raw lab artifacts are outside the core Git history.

## Success metrics for the next 90 days

- 20+ canonical run receipts across at least two runtimes;
- 3+ explicit candidate decisions (`promote`, `archive_only`, or `reject`);
- zero unexplained service restarts or swap incidents caused by Capstan;
- one proven rollback from a failed candidate;
- one second 100B+ lane represented without core model-specific code;
- one external operator completes the lifecycle and reports where it breaks.

## Kill criteria for the product thesis

Reassess Capstan if any of these remain true after v0.25/v0.26:

1. A second runtime requires model-specific changes in core lifecycle code.
2. Receipts cannot reproduce or explain promotion decisions.
3. The tool drifts into duplicating llama-swap/Ollama/LM Studio instead of governing them.
4. Operators still need shell archaeology to answer what is live and why.
5. We keep producing bespoke benchmark files that cannot be compared.

## Exact next ten actions

1. Keep PR #15 Draft and label it lab-only.
2. Open and ship a focused `v0.24.4-safety-gate` branch from `main`.
3. Enforce preflight/readiness exit semantics and cleanup path containment.
4. Gate release publication on tests.
5. Open `v0.25-evidence-spine` from the hardened mainline.
6. Add backward-compatible provenance and artifact-lock sections.
7. Implement content-addressed receipts, redaction, aggregation, and retention.
8. Implement evaluator adapters plus qualification/comparison policies.
9. Make promotion consume a fresh passing receipt and preserve rollback lineage.
10. Capture DS4, import Hy3 evidence, then approve or reject at most two Hy3 kernel experiments.
