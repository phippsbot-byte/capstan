# Capstan current status — 2026-07-09

## Executive verdict

Capstan is a **useful alpha lifecycle controller** and a **promising giant-model research program**, but those are currently two loosely connected projects.

The released product reliably answers:

> What model lanes are registered, alive, healthy, dormant, or down on this machine?

It does **not** yet answer the higher-value questions:

> Why is this model the incumbent? Which exact model/runtime bits produced it? What evidence beat the previous candidate? What regressed? Can the decision be reproduced and rolled back from receipts?

That missing evidence/governance layer is the next product—not another server, proxy, router, model downloader, or chat UI.

## Authority chain

Use this order when sources disagree:

1. **Live Capstan operator output and live manifests** — current runtime truth.
2. **Source and tests at the latest release (`v0.24.3`)** — released product behavior.
3. **`README.md`, `docs/model-lifecycle.md`, and `CHANGELOG.md` on `main`** — operator guidance and release history; source wins if guidance drifts.
4. **Draft PR #15 / `hy3-lazy-sidecar-canary`** — research evidence only; not released product behavior.
5. **Experiment status files, local scripts, logs, and old session notes** — supporting evidence, never product authority.

Do not treat the Hy3 branch HEAD as released Capstan. Do not treat a registered dormant lane as broken. Do not treat a green smoke as evidence that a candidate is better than the incumbent.

## Product thesis

**Capstan should be the evidence-backed control plane for turning weird, giant, high-risk local-model runtimes into reproducible services.**

Primary user:

- an advanced local-AI operator running 100B+ or larger sparse models, custom runtime forks, external SSD sidecars, and mixed MLX/llama.cpp/custom servers on Apple Silicon;
- first customer: us;
- secondary users: researchers and workstation operators with heterogeneous local inference fleets.

The painful job Capstan owns:

- capture the exact runtime/model/hardware configuration;
- preflight destructive or memory-heavy actions;
- start, observe, smoke, soak, and benchmark consistently;
- compare candidate against incumbent under explicit gates;
- promote or reject with a content-addressed, tamper-evident receipt;
- roll back safely;
- answer, in one command, **what is live and why**.

### What Capstan should not become

- another inference engine;
- another Ollama/LM Studio clone;
- another hot-swap proxy competing with llama-swap or llama.cpp router mode;
- a general chat UI;
- a model-training framework;
- an unbounded dumping ground for model-specific research artifacts.

Capstan should orchestrate those systems, not rebuild them.

## Verified now

| Area | Verified state |
|---|---|
| Public repo | `phippsbot-byte/capstan`, MIT, public |
| Latest release | `v0.24.3` from 2026-06-23 |
| Audit base before this strategy-doc change | `main` at `a167405` |
| Active research | draft PR #15, 37 commits ahead of `main` |
| Release tests | 49 Python tests; package builds on macOS/Python 3.11 |
| Research tests | 2 CTests plus real sidecar smokes on PR #15 |
| Installed CLI | editable `local-modelctl 0.24.3` sourced from the active Hy3 draft checkout; `capstan` and compatibility `modelctl` both work |
| Live fleet | 1 ready lane, 2 explicitly dormant lanes |
| Fleet doctor | clean: 0 issues, 0 warnings |
| Live service drift | DS4 LaunchAgent matches desired manifest |
| Live DS4 health | explicit exact JSON smoke passes; swap delta 0.0 GiB; routine health does not enable semantic smoke or latency ceilings |
| DS4 smoke performance | 15.44s wall for 17 prompt + 6 completion tokens; ~1.15 prompt tok/s and ~8.33 completion tok/s |
| Saved report history | only 1 report, dated 2026-06-15 |
| GitHub adoption, checked 2026-07-09 | 0 issues, 0 stars, 0 forks |

The live DS4 lane references roughly **145 GiB** of primary package data plus **137 GiB** of secondary sidecar data. Capstan keeps it healthy, but does not fingerprint the runtime commit, binary, artifact revision, or decision evidence that made this exact configuration the incumbent.

## What is genuinely working

### 1. Guarded local lifecycle

Capstan has a coherent dependency-free manifest and command path for:

- validate and preflight;
- start, wait, status, stop;
- exact-output smoke and repeated soak;
- synthetic latency/swap benchmark artifacts;
- health and doctor checks;
- guarded cleanup;
- launchd install/diff/control;
- readiness-gated rotate and rollback.

Several safety defaults are good: dry-run first, explicit `--execute`, explicit restart behavior, readiness gates, and swap ceilings/deltas. But safety is **not** enforced end to end: ordinary `start`, fleet recovery, and daemon restart can bypass preflight; `start --wait` can report failed readiness with exit code 0; cleanup trusts a manifest-declared `safe = true` path without root containment or an in-use check.

### 2. Fleet truth

The released fleet surface correctly distinguishes:

- ready;
- down;
- dormant;
- invalid.

`fleet status`, `fleet health`, `fleet doctor`, `fleet intake`, and `fleet recover` form a real operator loop. The live intake scan found no unregistered model endpoints on the Studio.

### 3. Release discipline

The project shipped 28 releases in its first nine days, with packaging, GitHub Actions, GitHub release artifacts, and compatibility-preserving rebranding. Core test density is respectable: about 4.3k source lines and 2.5k test lines.

### 4. Giant-model R&D

The Hy3 work proved that a model whose flat MLX form swaps itself to death can be split into:

- 4.612 GiB resident core;
- 149.977 GiB routed expert sidecar;
- 6.249 GiB cold active expert reads per native top-8 token.

The research also produced reusable ideas: packed sidecars, route traces, compact indexes, native q4 kernels, persistent daemon integration, cache telemetry, parity fixtures, and bounded expert-bank caching.

That is real technical progress. It is not yet a product lane.

## Hy3 reality

### Current best evidence

- Flat MLX: dead on this machine; ~94.7 GiB swap before useful generation.
- Stable Python canary: top-5/slot-16 clear mode, but Phipps slice composite only **2.32**, average latency **94.6s**, output ~**0.6 tok/s**.
- Native direct top-8 packed-cache run:
  - first step: **73.242s**;
  - median decode step: **2.994s** (~0.33 tok/s);
  - four-token step sum: **82.298s**;
  - reads: **75.493 GiB**;
  - swap delta: **0.0 GiB**;
  - same four greedy token IDs as baseline.
- Direct/hybrid modes are not default-safe across all tested top-k shapes because some forced sequences show knife-edge argmax drift.
- Cache-policy work is exhausted as a major lever. The status evidence already says the next credible lever is a real tiled/simdgroup q4 kernel, deeper fusion, MPS-backed batching, or upstream runtime support.

### Branch shape

PR #15 adds **118,219 lines across 130 files**. Every changed file except the CI workflow is under `experiments/`. The experiment tree is about 10.7k source lines versus 4.3k lines in the released core, plus 67 tracked result artifacts and several ~900 KiB NPZ files.

Verdict: **do not merge PR #15 into `main` in its current form.** Preserve it as a research branch, split reusable infrastructure from raw evidence, and promote only small, product-shaped pieces.

## What is missing

### Critical product gaps

1. **Start/recovery/cleanup safety**
   - preflight is optional and is not enforced by ordinary `start`, fleet recovery, or daemon restart;
   - `start --wait` can return shell success when readiness fails;
   - cleanup parses `safe = "false"` as truthy because manifest booleans are loosely coerced; a temporary-file deletion was reproduced without `--force`;
   - cleanup has a manifest-declared safety flag, but no canonical-root containment, protected-path rules, symlink-race defense, or active-model check.

2. **Supervisor policy and observability**
   - routine semantic health/latency gates are not enabled for DS4, so green does not mean usable;
   - the daemon does not emit incremental durable health/restart events during its long-running loop;
   - restart can trigger on any non-`ok` verdict, including warnings based on machine-global swap, with no attribution, hysteresis, cooldown, or retry budget;
   - service drift checks require the operator to remember install-time flags instead of reading manifest-owned policy;
   - fleet service state does not expose loaded/running state, service PID, last exit, or restart history.

3. **Provenance**
   - no model source/revision/hash fields;
   - no runtime repo/commit/version/binary fingerprint;
   - no automatic hardware/OS snapshot;
   - no manifest digest embedded in runs.
   - the live CLI/service imports an editable draft checkout, and the DS4 launch script lives outside Git.

4. **Durable run receipts**
   - smoke/soak/bench results are not automatically saved into one canonical run store;
   - the live report store has one stale report despite extensive experimentation;
   - experiment artifacts use bespoke schemas and paths.

5. **Comparison and policy gates**
   - no incumbent-vs-candidate comparison command;
   - no reusable latency, throughput, memory, correctness, quality, or regression policy;
   - no explicit promote/archive/reject result.

6. **Promotion semantics**
   - current `promote` is safe deployment rotation with post-health rollback;
   - it does not prove the candidate beat the incumbent;
   - it does not persist a promotion receipt or canonical incumbent pointer.

7. **Research integration**
   - model-specific experiments do not emit Capstan-native receipts;
   - no adapter contract for custom metrics such as expert reads, cache hits, parity error, or quality evaluator output;
   - raw experiment artifacts overwhelm the branch.

### Important but secondary gaps

- implementation package/state/env names remain `modelctl` for compatibility;
- only macOS/Python 3.11 CI is exercised;
- tests live mostly in one 2,457-line file;
- GitHub release/PyPI build workflows are not dependency-gated on the test job, even though tag push CI also runs;
- no issue-backed execution backlog or milestone; this dated roadmap is the first published strategy artifact;
- no remote fleet/peer model yet;
- no external user validation;
- no automatic runtime update/version drift detection.
- `main` has no branch protection/ruleset, vulnerability alerts are disabled, and stale merged branches remain;
- PyPI publication has never run; `local-modelctl` is absent and the `capstan` package name belongs to another project.

## Competitive reality

| Tool | What it already owns | Capstan should do instead |
|---|---|---|
| Ollama | model import, load/unload, local serving, running-model inventory | govern custom/non-Ollama runtimes and evidence |
| LM Studio / llmster | download, load/unload, headless service, JIT loading, runtime management | avoid GUI/server competition |
| llama.cpp | inference, router mode, multi-model management, health/slots/metrics | consume its APIs and capture reproducible runtime receipts |
| llama-swap | heterogeneous server launch, proxying, hot-swap, TTL, groups, logs/metrics | integrate as a managed backend; do not build another proxy |
| Capstan | preflight, safety, health, fleet truth, rotation/rollback | become the governance and promotion layer above all of them |

## Current blockers

- No evidence schema connecting a model artifact, runtime build, hardware state, benchmark, and decision.
- The Hy3 branch accumulated work without a previously accepted performance/correctness gate; the linked roadmap now proposes a reproducible stop/continue contract.
- PR #15 is too large and artifact-heavy to review or merge responsibly.
- Prior public docs did not define a durable release-vs-research authority chain; this status update establishes one that future docs must preserve.
- We have not demonstrated that the manifest/evidence model works for two unrelated giant-model runtimes without core special cases.

## Recommended next move

Stop optimizing Hy3 for one sprint.

First ship a narrow **v0.24.4 Safety Gate**: enforce preflight before every start/recovery path, make command exit status match structured failures, strictly validate manifests, harden cleanup path containment/in-use checks, and make restart policy attributable and bounded.

Then build **Capstan v0.25: Evidence Spine** on `main`:

1. content-addressed, tamper-evident run receipts;
2. artifact/runtime/hardware provenance;
3. automatic smoke/soak/bench/health persistence;
4. incumbent-vs-candidate comparison;
5. explicit promotion policies and decision receipts;
6. one command that answers: `why is this model live?`

Then import DS4 and Hy3 evidence into that system. If the schema cannot represent both cleanly, fix the schema before doing more kernel work.

## End state

Capstan v1.0 is successful when an operator can:

1. register a giant local model and its exact runtime/artifacts;
2. prove the machine can safely run it;
3. execute reproducible correctness/performance/quality checks;
4. compare it with the incumbent under an explicit policy;
5. promote, reject, or archive it with a content-addressed, tamper-evident receipt;
6. recover or roll back without archaeology;
7. inspect the entire local fleet and understand **what is live, why, and from which bits**.

The moat is not model serving. The moat is **safe, evidence-backed giant-model operations**.
