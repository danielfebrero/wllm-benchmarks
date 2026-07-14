# Reproducibility checklist

Record and retain:

- benchmark Git SHA and wllm Git SHA/version;
- task revision, generated seed and fixture digest;
- agent CLI version and exact command;
- requested and observed model, effort and topology;
- arm order, timeout, permission/sandbox profile and environment overrides;
- OS, CPU, memory, storage and network regime;
- upstream dataset commit or snapshot and every selected instance ID;
- Docker image digest and official grader revision when applicable;
- raw event stream, stderr, briefing, patch, grade and normalized report;
- missing/unsupported telemetry fields as `null` with a reason.

For wall-time comparisons use `--jobs 1`, close other heavy workloads, and keep
both arms of each pair on the same machine. Alternate arm order to reduce drift.
Treat warm-cache experiments as a separate regime and disclose all pre-indexing
costs; the default headline is cold end-to-end time.

Before publishing:

1. Run the no-model unit suite.
2. Run one smoke pair for every selected agent adapter.
3. Run `upstream.py doctor` for every upstream suite in scope.
4. Freeze configs and instance IDs before inspecting outcomes.
5. Verify hidden graders are outside each agent-visible workspace.
6. Export all invalid runs rather than deleting them.
