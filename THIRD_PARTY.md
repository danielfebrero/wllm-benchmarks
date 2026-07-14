# Third-party benchmarks

Nothing listed here is vendored. Fetch upstream artifacts on demand and obey
the license at the pinned revision.

| Suite | Canonical source | License noted upstream | Integration |
|---|---|---|---|
| SWE-bench Pro | https://github.com/scaleapi/SWE-bench_Pro-os | MIT harness; dataset terms upstream | registered |
| SWE-bench Live | https://github.com/microsoft/SWE-bench-Live | MIT | registered |
| Multi-SWE-bench | https://github.com/multi-swe-bench/multi-swe-bench | Apache-2.0 | registered |
| RepoBench | https://github.com/Leolty/repobench | CC-BY-4.0 | registered |
| CrossCodeEval | https://github.com/amazon-science/cceval | Apache-2.0 | registered |
| RepoQA | https://github.com/evalplus/repoqa | Apache-2.0 | derived A/B adapter; official grader |
| CodeLlama tokenizer | https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf | Llama 2 Community License | pinned context construction; not redistributed |
| Long Code Arena baselines | https://github.com/JetBrains-Research/lca-baselines | Apache-2.0 | registered |

Always re-check upstream license files at the pinned commit. A dataset's
license may differ from its evaluation harness.

The RepoQA-SNF adapter requires a local dataset file and records its SHA-256;
it does not redistribute that file. Repository contents inside the dataset
retain their upstream licenses, which must be audited before redistributing
retained workspaces or result artifacts.
