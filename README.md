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
python3 run.py --task release-evidence --runs 2 \
  --agent codex --model gpt-5.6-sol --reasoning medium \
  --wllm-bin /absolute/path/to/wllm
```

Run the full matrix shape serially when exploring wall-time behavior. The model
IDs below are examples, not an immutable publication declaration:

```bash
python3 matrix.py \
  --tasks release-evidence,config-precedence,migration-lineage,webhook-rotation,single-file-control \
  --agents codex,claude,grok \
  --model codex=gpt-5.6-sol \
  --model claude=claude-sonnet-5 \
  --model grok=grok-4.5 \
  --efforts low,medium,high \
  --topologies single,native-multi-agent \
  --runs 6 --jobs 1 \
  --wllm-bin /absolute/path/to/wllm
```

Use provider model IDs that resolve immutably for publication; moving aliases
such as `sonnet` are suitable only for exploratory runs. Record the agent CLI
version and provider-reported model identity with every result.

Before committing to the full matrix, run a bounded Codex-only slice. This
example is four cells and therefore 48 agent calls plus 24 treatment briefs:

```bash
python3 matrix.py \
  --tasks release-evidence,config-precedence \
  --agents codex --model codex=gpt-5.6-sol \
  --efforts medium \
  --topologies single,native-multi-agent \
  --runs 6 --jobs 1 \
  --wllm-bin /absolute/path/to/wllm
```

The checked-in publication configuration is deliberately a non-publishable
template. Its `model_snapshot_status` is `template`, and generated plans record
`model_snapshot_publication_eligible: false`. It can be inspected directly:

```bash
python3 matrix.py --config configs/smoke.json --dry-run
python3 matrix.py --config configs/publication.json --dry-run
```

Before collecting publishable results, copy `configs/publication.json`, replace
every `models` value with an immutable provider snapshot ID, and set
`model_snapshot_status` to `attested-immutable`. Replace every `agent_bins` and
`wllm_bin` placeholder with an absolute path to the exact executable. Give
`cache_regime` and `machine_regime` factual, non-template labels describing the
enforced cache state and machine environment. Keep `arm: both`, the declared
brief budget and timeout, `jobs: 1`, and `no_build: true`; the latter prevents a
surprise wllm rebuild. These labels are attestations, not cache or machine
controls by themselves. The frozen plan records binary, task and harness hashes.

The harness validates exact agent/model/binary coverage but cannot infer whether
an arbitrary provider model ID is truly immutable. Once the config is marked
`attested-immutable`, protocol-changing CLI overrides are rejected: make such
changes in the config, regenerate the plan, and recommit both files instead.

The smoke config contains only negative controls and checks harness plumbing;
it is not evidence of an efficiency win. The publication config expands to 90
cells: 1,080 agent calls and 540 treatment brief constructions. At the default
900-second agent timeout, agent-call timeout ceilings alone total 270 hours.
Inspect the dry-run plan, provider quotas and expected spend before removing
`--dry-run`. Run one authenticated smoke pair for each selected agent CLI first.

An explicitly supplied CLI option replaces the corresponding config field;
for example, `--agents codex` replaces the complete configured `agents` list.
Relative `wllm_bin`, `output_dir`, and path-like `agent_bins` values in JSON
are resolved from the config file's directory. Relative CLI paths are resolved
from the invocation directory. Configs use schema version `1.0`; unknown fields,
wrong JSON types, unknown agents, and unknown topologies fail before any output
directory or agent process is created. The publication-only `selection_salt`
field and model-snapshot status are validated and recorded as protocol metadata.
Optional `cold` and `retain_failures` declarations are likewise metadata, not
enforcement switches;
the runner always retains failures, while cache regime still requires external
control and disclosure.

Use `--jobs > 1` only for exploratory quality/token runs. Concurrent cells
contend for CPU, disk and network, so their wall times are not publication
quality.

Run harness tests without model calls:

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
```

Before any publication outcome exists, freeze and commit the exact protocol.
The config and plan must be tracked, byte-identical to the same Git `HEAD`, and
clean when the real matrix starts:

```bash
cp configs/publication.json configs/publication-frozen.json
${EDITOR:-vi} configs/publication-frozen.json

python3 analysis.py freeze-plan \
  --template configs/analysis-plan.json \
  --config configs/publication-frozen.json \
  --family-id native-publication-v1 \
  --output configs/native-publication-plan.json

# Validate the exact frozen family without model calls.
python3 matrix.py \
  --config configs/publication-frozen.json \
  --analysis-plan configs/native-publication-plan.json \
  --dry-run

git add configs/publication-frozen.json configs/native-publication-plan.json
git commit --only -m "Freeze native wllm publication protocol" -- \
  configs/publication-frozen.json configs/native-publication-plan.json
test -z "$(git status --porcelain -- \
  configs/publication-frozen.json configs/native-publication-plan.json)"

# Collect outcomes only after that commit. Preserve the printed matrix path.
mkdir -p results
MATRIX_LOG="$PWD/results/native-publication-matrix.stderr.log"
python3 matrix.py \
  --config configs/publication-frozen.json \
  --analysis-plan configs/native-publication-plan.json \
  2>"$MATRIX_LOG"
MATRIX_STATUS=$?
cat "$MATRIX_LOG" >&2
printf 'matrix exit status: %s\n' "$MATRIX_STATUS"
MATRIX_DIR="$(sed -n 's/^Matrix artifacts: //p' "$MATRIX_LOG" | tail -n 1)"
MATRIX_INDEX="$MATRIX_DIR/artifact-index.json"
test -f "$MATRIX_INDEX"

python3 analysis.py run \
  --plan configs/native-publication-plan.json \
  --matrix-index "$MATRIX_INDEX" \
  --output "$MATRIX_DIR/analysis.json"
```

`matrix.py` verifies the committed config/plan, hashes the complete execution
protocol before the first outcome, injects that provenance into every report,
and writes it to the captured matrix index. Publication analysis accepts exactly
one `--matrix-index`; loose `--report` and `--reports-root` inputs are reserved
for `--exploratory` analysis. Duplicate, undeclared, missing or
infrastructure-invalid cells produce exit status `2` and
`publication_ready: false`.

The confirmatory family is exactly three cells: `release-evidence`,
`config-precedence`, and `migration-lineage`, each with the frozen Codex model,
`effort=medium`, and `topology=single`. With family alpha `0.05`, Bonferroni sets
each confirmatory cell to `0.05 / 3`. Every required metric must cover all valid
ITT pairs and clusters; in particular, missing input-token telemetry makes a
primary cell non-claimable. The other 87 cells are explicitly exploratory and
cannot emit a confirmatory win. No pooled or global win is manufactured.

For any separate native multi-agent token-efficiency statement, provider
telemetry must attest `scope: parent-and-children` and `complete: true`; otherwise
the token result is not claimable. Requested multi-agent topology alone is not
that evidence.

RepoQA uses the same clustered estimator, but it is a separate decision family.
Until a RepoQA-specific plan is frozen externally, run it only as an explicitly
non-publishable diagnostic:

```bash
python3 analysis.py run --exploratory \
  --plan configs/analysis-plan.json \
  --report results/path/to/repoqa/report.json \
  --output results/path/to/repoqa-analysis.json
```

## Mainstream suites

`suites/` registers upstream provenance, license, immutable-selection policy,
grader type and integration state. A registered suite is not automatically an
implemented A/B adapter. `python3 upstream.py list` and
`python3 upstream.py doctor SUITE` make that distinction explicit at the
manifest/dependency level. Runnable adapters can require a stricter,
adapter-specific doctor as well.

| Family | Current integration | Purpose |
|---|---|---|
| SWE-bench Pro | protocol registered; official Docker adapter required | long-horizon repairs |
| SWE-bench Live | protocol registered; pinned snapshots required | contamination/freshness control |
| Multi-SWE-bench | protocol registered; official harness required | multi-language repairs |
| RepoBench-R | protocol registered; retrieval adapter required | retrieval diagnostic |
| CrossCodeEval | protocol registered; retrieval adapter required | cross-file completion |
| RepoQA SNF | first runnable upstream adapter; derived A/B with official instances + grader | long-context function retrieval |
| Long Code Arena | protocol registered; retrieval/summarization adapters required | bug localization and module understanding |

An implemented adapter still carries its exact comparability label. The
RepoQA-SNF result is a derived context-efficiency diagnostic, not an official
leaderboard score: it changes the context-construction intervention while
retaining RepoQA instances and the official needle evaluator. Upstream datasets
are never vendored or downloaded implicitly by this harness.

### RepoQA-SNF derived A/B

Create a dedicated environment, install the constrained context-construction
dependencies, then install the pinned Apache-2.0 RepoQA harness without letting
it replace those constraints. Explicitly obtain the pinned dataset (the helper
may download it when absent):

```bash
mkdir -p "$HOME/.venvs"
python3 -m venv "$HOME/.venvs/wllm-repoqa-2024-06-23"
source "$HOME/.venvs/wllm-repoqa-2024-06-23/bin/activate"
python3 -m pip install --upgrade pip
python3 -m pip install -r constraints/repoqa.txt
python3 -m pip install --no-deps \
  "git+https://github.com/evalplus/repoqa.git@ae876deb1365dbf5a15b0533723c8ed123eee586"
DATASET=$(python3 -c \
  'from repoqa.data import _get_repoqa_data_ready_path; print(_get_repoqa_data_ready_path())')
EXPECTED_SHA256=bd3f7cab47283cdeccee20daea31af587b680cf8f9db192ab4da1037730cd6e2
SHA256=$(shasum -a 256 "$DATASET" | awk '{print $1}')
test "$SHA256" = "$EXPECTED_SHA256"
```

Freeze a balanced subset before looking at outcomes, inspect prerequisites,
then run the paired diagnostic:

```bash
python3 repoqa_ab.py plan --dataset "$DATASET" \
  --expect-dataset-sha256 "$SHA256" --count 25 \
  --output results/repoqa-selection.json
python3 repoqa_ab.py doctor --dataset "$DATASET" \
  --expect-dataset-sha256 "$SHA256" --agent codex --wllm-bin wllm \
  --allow-model-download
# Final offline preflight after the exact tokenizer snapshot is cached:
python3 repoqa_ab.py doctor --dataset "$DATASET" \
  --expect-dataset-sha256 "$SHA256" --agent codex --wllm-bin wllm
python3 repoqa_ab.py run --dataset "$DATASET" \
  --expect-dataset-sha256 "$SHA256" --count 25 --repetitions 2 \
  --agent codex --model gpt-5.6-sol --effort medium \
  --wllm-bin /absolute/path/to/wllm
```

RepoQA's official baseline context uses the CodeLlama tokenizer; provider input
tokens are recorded separately and must not be conflated with that 16k context
setting. The adapter forces tokenizer revision
`22cb240e0292b0b5ab4c17ccd97aa3a2f799cbed`, records loaded-file hashes and the
complete Python environment, and defaults to offline cache-only operation.
`--allow-model-download` is therefore an explicit, pinned preflight action.
`repoqa_ab.py doctor` validates the dataset, exact RepoQA source bytes, API,
tokenizer, and local binaries; live agent authentication is checked only by
`run`. Before either model call, the treatment brief is constructed, the
retrieval repository is deleted, and both arms start in independent empty Git
workspaces; tool use or unavailable tool telemetry fails the context-only
protocol. The command above performs 100 agent calls and 50 treatment briefs
(`25` instances x `2` repetitions x `2` arms).
Because the current Grok parser exposes only best-effort tool telemetry, Grok
RepoQA cells are retained as `tool_telemetry_unverified` protocol failures; do
not use them for a context-only efficiency claim. Grok remains available in the
native repair matrix, where repository tools are part of the task.

## Scientific contract

- One fixture preparation per repetition, followed by byte-identical arm copies
  whose whole-tree SHA-256 digests (including `.git`) must match before execution.
- Same public task, model, effort, topology, timeout and permissions.
- The treatment gets exactly one briefing derived only from the public prompt.
- Brief construction is inside end-to-end time and its text is naturally part
  of model input.
- Hidden graders and gold patches remain outside the agent workspace.
- Arm order alternates. Agent/protocol failures and timeouts are score-zero
  intention-to-treat outcomes with observed or censored time; later arms still
  continue. Only infrastructure-invalid pairs are excluded from estimands.
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
