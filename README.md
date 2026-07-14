# wllm-benchmarks

Reproducible, paired evaluations of coding agents with and without one bounded
`wllm context` briefing. The primary question is causal and narrow: can a
task-conditioned workspace map replace enough agent discovery to reduce input
tokens and end-to-end time without reducing correctness?

This repository keeps unfavorable controls, counts cold briefing time, and
never treats missing telemetry as zero. Comparisons are only valid within the
same agent, model, effort, topology, task instance, and machine regime.

## What runs locally

The native suite is self-contained and uses external graders that are never
placed in the agent workspace:

| Task | Shape | Expected role |
|---|---|---|
| `release-evidence` | large cross-format, cross-file repair | positive retrieval case |
| `config-precedence` | code + defaults + environment contract | cross-file case |
| `migration-lineage` | schema + migrations + compatibility | cross-file case |
| `webhook-rotation` | small, obvious target | negative control |
| `single-file-control` | named file, local fix | negative control |

Run one paired cell:

```bash
python3 run.py --task release-evidence --runs 3 \
  --agent codex --model gpt-5.6-sol --reasoning medium \
  --wllm-bin /absolute/path/to/wllm
```

Run a preregistered matrix serially (required for publishable wall-time
comparisons):

```bash
python3 matrix.py \
  --tasks release-evidence,config-precedence,migration-lineage,webhook-rotation,single-file-control \
  --agents codex,claude,grok \
  --model codex=gpt-5.6-sol \
  --model claude=sonnet \
  --model grok=grok-4.5 \
  --efforts low,medium,high \
  --topologies single,native-multi-agent \
  --runs 5 --jobs 1 \
  --wllm-bin /absolute/path/to/wllm
```

Use `--jobs > 1` only for exploratory quality/token runs. Concurrent cells
contend for CPU, disk and network, so their wall times are not publication
quality.

Run harness tests without model calls:

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
```

## Mainstream suites

`suites/` registers upstream provenance, license, immutable-selection policy,
grader type and integration state. A registered suite is not automatically an
implemented A/B adapter. `python3 upstream.py list` and
`python3 upstream.py doctor SUITE` make that distinction explicit.

| Family | Current integration | Purpose |
|---|---|---|
| SWE-bench Pro | protocol registered; official Docker adapter required | long-horizon repairs |
| SWE-bench Live | protocol registered; pinned snapshots required | contamination/freshness control |
| Multi-SWE-bench | protocol registered; official harness required | multi-language repairs |
| RepoBench-R | protocol registered; retrieval adapter required | retrieval diagnostic |
| CrossCodeEval | protocol registered; retrieval adapter required | cross-file completion |
| RepoQA SNF | official runner documented; wllm A/B adapter required | long-context function retrieval |
| Long Code Arena | protocol registered; retrieval/summarization adapters required | bug localization and module understanding |

No upstream score is called comparable until its manifest says
`comparability: official-grader` and `ab_adapter: implemented`. Datasets are
downloaded on demand and are never vendored here.

## Scientific contract

- Byte-identical baseline and treatment workspaces per pair.
- Same public task, model, effort, topology, timeout and permissions.
- The treatment gets exactly one briefing derived only from the public prompt.
- Brief construction is inside end-to-end time and its text is naturally part
  of model input.
- Hidden graders and gold patches remain outside the agent workspace.
- Arm order alternates; failures and timeouts remain in the report.
- Missing provider telemetry is `null`, never `0`.
- All task/control results are retained, including results unfavorable to wllm.

See [methodology](docs/methodology.md) and
[reproducibility](docs/reproducibility.md).

## Repository relationship

The main `wllm` repository pins this repository as the `bench/agent`
submodule. A recursive clone preserves the exact benchmark revision:

```bash
git clone --recurse-submodules https://github.com/danielfebrero/wllm.git
```

Apache-2.0 covers this harness and its original fixtures. Upstream datasets
and graders retain their own licenses; see `THIRD_PARTY.md`.
