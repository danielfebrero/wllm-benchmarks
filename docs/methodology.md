# Methodology

## Estimand

The primary estimand is the paired product effect of `wllm` versus `baseline`:
a task-conditioned initial brief plus on-demand runtime CLI access, compared
with neither capability. Two predeclared mechanism contrasts decompose it:
`brief-only / baseline` estimates the initial-brief contribution, while
`wllm / brief-only` estimates the incremental contribution of runtime access.
These mechanism contrasts are exploratory; the confirmatory efficiency claim
remains `wllm / baseline`. No arm receives hidden hints and models never differ
within a cell.

The two brief arms independently compute a cold, identically budgeted briefing
before the first turn and append it to the common prompt. Only the full `wllm`
arm also receives a PATH-first CLI shim that forwards to the exact pinned binary.
Runtime use is optional agent behavior and every invocation is recorded.
Product-level `wllm init` hook validation remains a separate experiment; this
harness does not label the controlled CLI intervention as a hook result.

## Pairing and isolation

For every repetition the fixture generator runs exactly once, then copies that
source into every requested arm before either agent starts. A canonical SHA-256
digest covers relative paths, entry types, permission modes, symlink targets and
file bytes across the whole tree, including `.git`; the harness records each
digest and refuses agent execution unless all copies match. The public task and
common efficiency instruction are identical. The baseline never receives a
briefing or wllm directive. `brief-only` receives the briefing but is explicitly
denied runtime wllm. The full arm receives the briefing and targeted-use
directive. Generated `.wllm` state is restored before the agent starts, then
the complete workspace is re-hashed;
agent execution is infrastructure-invalid if it no longer matches the fixture.

The harness disables user-configured integrations (`--ignore-user-config` and
`--ignore-rules` for Codex, safe mode for Claude, and MCP-tool denial for Grok).
It prepends a deny shim named `wllm` in control-arm PATHs, so a global install
cannot leak into them and an attempted call becomes a recorded protocol failure.
The full arm gets an allow shim pointing only to the copied benchmark binary.
Thus CLI runtime capability is provider-neutral; MCP-specific product adapters
can be studied separately. The agent is told to stay inside its workspace.
Publication runs should add an
OS-level boundary (container or sandbox mount) because prompt instructions are
not an isolation mechanism. The grader, gold behavior and private tests live
outside that mount.

## Outcomes

Quality is primary. Report the external-grader score, exact solve rate and
failure taxonomy. Efficiency is reported only alongside quality:

- total and uncached input tokens where the provider exposes them;
- output and reasoning tokens as separate fields;
- cold briefing time and token estimate;
- agent time and end-to-end time;
- tool actions and any provider-exposed delegation evidence;
- timeout, process failure and telemetry validity.

Provider token semantics differ. Never pool token counts across agents or
providers. An unavailable metric is `null` and excluded from that metric's
aggregate. Agent exits, protocol failures and agent or briefing timeouts are
intention-to-treat outcomes: they retain score zero and observed or
timeout-censored end-to-end time. They remain in quality and paired time
analysis. Fixture, grader, spawn and harness failures are infrastructure-invalid
records and are excluded from estimands. Both kinds retain diagnostics and do
not stop later arms or repetitions.

## Predeclared comparison

Analyze paired cells within `(agent, model, effort, topology, task, machine
regime)`. Report score delta and geometric mean ratios for input tokens and
end-to-end time for all three pairwise contrasts. A ratio below one favors the
numerator arm. Use a repetition count divisible by three so each arm occupies
every execution position equally often.

A publication matrix must come from a committed config whose
`model_snapshot_status` is `attested-immutable`. That value is an explicit
operator attestation, not automatic provider-ID recognition. The harness
requires exact model and absolute executable mappings for all selected agents,
an absolute wllm executable, explicit arm/budget/timeout/run/job/no-build values,
and non-template machine/cache regime labels. The frozen execution protocol
includes executable, task and harness hashes. Protocol-changing CLI overrides
are rejected. The checked-in `configs/publication.json` remains a `template` and
is not publication-eligible.

Only pairs whose relevant arm records are both valid contribute to pairwise
ratios or score deltas; complete but invalid pairs are counted and
listed separately.

For a publication claim use at least six repetitions per cell, divisible by
three. The harness
emits descriptive paired ratios; compute paired bootstrap confidence intervals
in a separately versioned analysis retained with the result. A defensible win
requires:

1. quality non-inferiority under a predeclared margin;
2. an input-token ratio confidence interval below one;
3. an end-to-end ratio confidence interval below one.

Do not collapse these into an opaque composite score. Preserve negative
controls, failed cells and task-level heterogeneity.

### Deterministic paired-bootstrap analysis

`analysis.py` consumes each native or RepoQA report as a separate experimental
cell. `freeze-plan` binds the statistical parameters to the frozen native
matrix config SHA, full execution-protocol SHA, salt, run count, complete
expected cell family, three-cell primary family and fixed decision rule before
outcomes exist. The config and plan must be clean tracked files in the same Git
commit before `matrix.py --analysis-plan` will collect an attested outcome.
Native cells
resample complete `run` pairs. RepoQA first aggregates repeated pairs within
each `instance_id` (arithmetic mean for quality deltas, geometric mean for
ratios), then resamples instances so an instance with more repetitions cannot
receive more inferential weight. It reports percentile intervals for the mean
clustered quality delta and geometric-mean clustered input-token and
end-to-end-time ratios. Every metric carries both its analyzable-pair and
cluster counts; missing token telemetry does not remove that pair from quality
or time.

Pairs with an explicitly invalid arm, including a RepoQA fixture-integrity
failure, are counted but excluded. Valid intention-to-treat failures retain
their score-zero quality and recorded or substituted timeout duration. Time
intervals containing substituted timeout limits are descriptive paired-cost
summaries, not survival-analysis inference. Analyze cells independently and do
not pool agents, models, efforts, topologies, tasks or machine regimes.

For the native publication family, undeclared or duplicate reports are errors.
Missing or infrastructure-invalid declared cells remain in the accounting and
withhold publication readiness. Model IDs must carry the explicit
`attested-immutable` status. The primary confirmatory family is fixed to the
Codex, medium-effort, single-agent cells for `release-evidence`,
`config-precedence`, and `migration-lineage`. Bonferroni divides the family
alpha by three. All other declared cells are exploratory and cannot produce a
confirmatory win. Each primary requires all three predeclared criteria and full
pair/cluster coverage for quality, input tokens and end-to-end time; missing
input-token telemetry therefore withholds the claim. The analyzer deliberately
emits no pooled/global win. Publication mode accepts one attested matrix index,
not a loose collection of report paths, and verifies report hashes and embedded
provenance against the frozen protocol. RepoQA is a separate decision family;
using the native statistical template with
`--exploratory` is visibly non-publishable, not a substitute for freezing a
RepoQA-specific plan before outcomes.

## Topologies

`single` disables native subagents only when the selected host exposes and the
harness verifies such a switch. `native-multi-agent` makes the host mechanism
available and requests bounded delegation; it does not guarantee that the model
will delegate. These are requested policy strata, not interchangeable
implementations or proof of realized delegation. Provider-reported usage is
retained as emitted, but the harness does not currently expose a normalized,
cross-provider subagent-call count and does not claim that every provider
includes all child-agent usage. Report requested topology separately from any
provider-specific delegation evidence. Treat multi-agent token results as
unavailable unless the recorded CLI version's telemetry contract establishes
complete parent-and-child coverage. A claimable report must attest
`input_token_coverage.scope = parent-and-children` and `complete = true`;
requested topology alone is insufficient.

## Upstream benchmarks

An upstream suite is publishable only when the exact dataset revision,
instance IDs, container digests, agent command and official grader are pinned.
Small subsets are selected before results by sorting instance IDs on
`SHA256("wllm-bench-v1" + instance_id)`, optionally within declared language or
repository strata. Manual post-result selection is forbidden.

The implemented RepoQA-SNF adapter is deliberately labeled a derived
context-efficiency A/B. Baseline receives RepoQA's official 16k long context;
treatment receives one cold wllm brief built only from the natural-language
description and repository source bytes. Needle metadata and gold locations
remain outside both agent workspaces and prompts. The retrieval repository is
deleted before either agent starts; both arms execute from separate empty Git
workspaces, and any detected tool use or unavailable tool telemetry is a
protocol failure. Answers are scored with
`repoqa.compute_score.needle_evaluator` at threshold 0.8. The deterministic
subset greedily balances language, repository and position-decile counts with
a salted SHA-256 rank as its final tie-breaker. Because the context intervention
differs from the official protocol, these results are not leaderboard scores.
RepoQA CodeLlama context tokens and provider-reported input tokens are reported
as different units.

The adapter pins the RepoQA harness, canonical dataset, CodeLlama tokenizer
revision, relevant upstream source-file hashes, executable hashes, and the
resolved Python environment in its report. Publication runs must retain the
report and the constrained environment. The generic
`upstream.py doctor repoqa-snf` checks only manifest-level dependencies; the
adapter-specific `repoqa_ab.py doctor` is additionally required.
