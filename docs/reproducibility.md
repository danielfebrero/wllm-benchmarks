# Reproducibility checklist

Record and retain:

- benchmark Git SHA and wllm Git SHA/version;
- task revision, upstream-selection salt when applicable, and fixture digest;
- agent CLI version, absolute executable path/hash and exact command;
- committed config/analysis-plan Git commit and SHA-256 values;
- `attested-immutable` model snapshot status and exact provider snapshot IDs;
- requested model, effort and topology, plus provider-observed identities and
  realized delegation evidence where available;
- arm order, brief budget, timeout, no-build policy, permission/sandbox profile
  and environment overrides;
- factual cache regime plus OS, CPU, memory, storage and network/machine regime;
- upstream dataset commit or snapshot and every selected instance ID;
- constrained dependency set and tokenizer/cache revision or file hashes;
- Docker image digest and official grader revision when applicable;
- raw event stream, stderr, briefing, patch, grade and normalized report;
- missing/unsupported telemetry fields as `null` with a reason;
- fixture-verification artifacts and invalid-run failure categories/diagnostics.

For wall-time comparisons use `--jobs 1`, close other heavy workloads, and keep
both arms of each pair on the same machine. Alternate arm order to reduce drift.
Use an even number of repetitions so the alternating order is balanced.
Treat warm-cache experiments as a separate regime and disclose all pre-indexing
costs; the default headline is cold end-to-end time.

The native runner records CLI versions, commands, fixtures and raw artifacts;
its machine fingerprint, benchmark/wllm Git SHAs, provider
telemetry-completeness evidence, and bootstrap confidence intervals remain
external publication metadata/analysis. The RepoQA adapter additionally records
benchmark revision when available, executable paths/hashes/versions, its pinned
tokenizer provenance and the resolved Python environment. Retain any remaining
external metadata alongside the result.

Before publishing:

1. Run the no-model unit suite.
2. Run one smoke pair for every selected agent adapter.
3. Run `upstream.py doctor` for every upstream suite in scope and every
   adapter-specific doctor (for RepoQA, `repoqa_ab.py doctor`).
4. Freeze configs and instance IDs before inspecting outcomes. Replace every
   template model and binary path, set factual machine/cache regimes, retain
   `arm: both`, budget, timeout, runs/jobs and `no_build: true`, then set
   `model_snapshot_status` to `attested-immutable`.
5. Run `analysis.py freeze-plan`, review its three fixed primary cells and
   execution-protocol hashes, then commit the frozen config and plan together.
   `matrix.py --analysis-plan` must start from those clean `HEAD` bytes.
6. Verify hidden graders are outside each agent-visible workspace.
7. Export all invalid runs rather than deleting them.
8. Confirm infrastructure-invalid pairs are excluded while agent/brief timeout
   outcomes remain in intention-to-treat quality and time analysis.
9. For native multi-agent cells, distinguish requested topology from realized
   delegation and publish token totals only with an explicit complete
   parent-and-children coverage attestation.
10. Retain the exact `artifact-index.json` printed by the attested matrix and run
   `analysis.py run --matrix-index` against it. Exit `2` or
   `publication_ready: false` means the declared family is not publishable; do
   not remove missing, invalid, negative-control, or unfavorable cells.
11. Treat only the three predeclared Codex/medium/single primary cells as
   confirmatory, apply Bonferroni across them, require complete metric coverage,
   and label every other cell exploratory.
