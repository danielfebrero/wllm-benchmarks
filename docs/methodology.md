# Methodology

## Estimand

The primary estimand is the paired effect of replacing normal first-turn
repository discovery with a single task-conditioned `wllm context` briefing.
It is not the effect of giving the treatment hidden hints, and it is not a
comparison between different models.

The headline treatment is `prefetch`: the harness computes one briefing before
the agent's first turn and appends it to the common prompt. This isolates the
retrieval intervention. `init-hook` is a secondary product-validation mode;
it exercises the integration installed by `wllm init` but mixes retrieval with
host hook behavior.

## Pairing and isolation

For every repetition the fixture generator creates two byte-identical Git
workspaces. The public prompt and efficiency instruction are identical. The
baseline never receives the briefing. The treatment's generated `.wllm` state
is removed before the agent starts, so only the briefing differs.

The agent is told to stay inside its workspace. Publication runs should add an
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
- tool actions and subagent calls;
- timeout, process failure and telemetry validity.

Provider token semantics differ. Never pool token counts across agents or
providers. An unavailable metric is `null` and excluded from that metric's
aggregate, while the run remains in quality/time aggregates.

## Predeclared comparison

Analyze paired cells within `(agent, model, effort, topology, task, seed,
machine regime)`. Report score delta and geometric mean ratios for input tokens
and end-to-end time (`wllm / baseline`). A ratio below one favors wllm.

For a publication claim use at least five repetitions per cell and paired
bootstrap confidence intervals. A defensible win requires:

1. quality non-inferiority under a predeclared margin;
2. an input-token ratio confidence interval below one;
3. an end-to-end ratio confidence interval below one.

Do not collapse these into an opaque composite score. Preserve negative
controls, failed cells and task-level heterogeneity.

## Topologies

`single` disables native subagents when the host exposes such a switch.
`native-multi-agent` enables the host's own subagent mechanism and asks for
bounded delegation. These are separate experimental strata, not interchangeable
implementations. All root and child usage must be included when the host emits
it; otherwise the report flags the topology's token accounting as incomplete.

## Upstream benchmarks

An upstream suite is publishable only when the exact dataset revision,
instance IDs, container digests, agent command and official grader are pinned.
Small subsets are selected before results by sorting instance IDs on
`SHA256("wllm-bench-v1" + instance_id)`, optionally within declared language or
repository strata. Manual post-result selection is forbidden.
