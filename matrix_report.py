#!/usr/bin/env python3
"""Generate a self-contained HTML infographic for a matrix run directory.

Usage:
  python3 matrix_report.py results/matrix-YYYYMMDDT.../
  # also invoked automatically at the end of matrix.py
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any


ARMS = ("baseline", "brief-only", "wllm")
ARM_LABELS = {
    "baseline": "Baseline (no wllm)",
    "brief-only": "Brief-only",
    "wllm": "Full wllm",
}
ARM_COLORS = {
    "baseline": "#64748b",
    "brief-only": "#0ea5e9",
    "wllm": "#22c55e",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_matrix(matrix_dir: Path) -> dict[str, Any]:
    index_path = matrix_dir / "artifact-index.json"
    plan_path = matrix_dir / "matrix-plan.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing artifact-index.json in {matrix_dir}")
    index = load_json(index_path)
    plan = load_json(plan_path) if plan_path.is_file() else {}
    cells_out: list[dict[str, Any]] = []
    for cell in index.get("cells") or []:
        report_rel = cell.get("report")
        report: dict[str, Any] | None = None
        if report_rel:
            report_path = matrix_dir / report_rel
            if report_path.is_file():
                report = load_json(report_path)
        aggregate = (report or {}).get("aggregate") or {}
        arms: dict[str, Any] = {}
        for arm in ARMS:
            blob = aggregate.get(arm) or {}
            arms[arm] = {
                "solve_rate": blob.get("solve_rate"),
                "median_score": blob.get("median_score"),
                "median_input_tokens": blob.get("median_input_tokens"),
                "median_duration_seconds": blob.get("median_duration_seconds"),
                "median_agent_duration_seconds": blob.get("median_agent_duration_seconds"),
                "median_wllm_brief_tokens": blob.get("median_wllm_brief_tokens"),
                "median_tool_calls": blob.get("median_tool_calls"),
                "valid_runs": blob.get("valid_runs"),
                "invalid_runs": blob.get("invalid_runs"),
            }
        contrasts = aggregate.get("contrasts") or {}
        wllm_vs_base = contrasts.get("wllm_over_baseline") or {}
        brief_vs_base = contrasts.get("brief_only_over_baseline") or {}
        wllm_vs_brief = contrasts.get("wllm_over_brief_only") or {}
        cells_out.append(
            {
                "id": cell.get("id"),
                "number": cell.get("number"),
                "task": cell.get("task") or (report or {}).get("task"),
                "agent": cell.get("agent") or (report or {}).get("agent"),
                "model": cell.get("model") or (report or {}).get("model"),
                "effort": cell.get("effort") or (report or {}).get("reasoning"),
                "topology": cell.get("topology") or (report or {}).get("topology"),
                "duration_seconds": cell.get("duration_seconds"),
                "exit_code": cell.get("exit_code"),
                "timed_out": cell.get("timed_out"),
                "report": report_rel,
                "arms": arms,
                "contrasts": {
                    "wllm_over_baseline": {
                        "input_ratio": wllm_vs_base.get("geometric_mean_input_token_ratio"),
                        "duration_ratio": wllm_vs_base.get("geometric_mean_duration_ratio"),
                        "score_delta": wllm_vs_base.get("median_score_delta"),
                        "valid_pairs": wllm_vs_base.get("valid_pairs"),
                    },
                    "brief_only_over_baseline": {
                        "input_ratio": brief_vs_base.get("geometric_mean_input_token_ratio"),
                        "duration_ratio": brief_vs_base.get("geometric_mean_duration_ratio"),
                        "score_delta": brief_vs_base.get("median_score_delta"),
                        "valid_pairs": brief_vs_base.get("valid_pairs"),
                    },
                    "wllm_over_brief_only": {
                        "input_ratio": wllm_vs_brief.get("geometric_mean_input_token_ratio"),
                        "duration_ratio": wllm_vs_brief.get("geometric_mean_duration_ratio"),
                        "score_delta": wllm_vs_brief.get("median_score_delta"),
                        "valid_pairs": wllm_vs_brief.get("valid_pairs"),
                    },
                },
            }
        )
    cells_out.sort(key=lambda c: int(c.get("number") or 0))
    return {
        "matrix_dir": str(matrix_dir.resolve()),
        "matrix_name": matrix_dir.name,
        "generated_at": (plan.get("generated_at") or index.get("generated_at")),
        "jobs": plan.get("jobs") or index.get("jobs"),
        "cell_count": len(cells_out),
        "timing_comparable": plan.get("timing_comparable"),
        "provenance": plan.get("provenance") or index.get("provenance"),
        "cells": cells_out,
    }


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    cells = payload["cells"]
    ok = sum(1 for c in cells if c.get("exit_code") == 0 and not c.get("timed_out"))
    ratios_in: list[float] = []
    ratios_time: list[float] = []
    score_deltas: list[float] = []
    for cell in cells:
        c = cell["contrasts"]["wllm_over_baseline"]
        if _num(c.get("input_ratio")) is not None:
            ratios_in.append(float(c["input_ratio"]))
        if _num(c.get("duration_ratio")) is not None:
            ratios_time.append(float(c["duration_ratio"]))
        if _num(c.get("score_delta")) is not None:
            score_deltas.append(float(c["score_delta"]))
    def geo(values: list[float]) -> float | None:
        if not values:
            return None
        return math.exp(sum(math.log(v) for v in values if v > 0) / len([v for v in values if v > 0])) if any(v > 0 for v in values) else None

    return {
        "cells_ok": ok,
        "cells_total": len(cells),
        "geo_input_ratio_wllm_baseline": geo(ratios_in),
        "geo_duration_ratio_wllm_baseline": geo(ratios_time),
        "mean_score_delta": (sum(score_deltas) / len(score_deltas)) if score_deltas else None,
        "input_wins": sum(1 for v in ratios_in if v < 1),
        "time_wins": sum(1 for v in ratios_time if v < 1),
        "input_pairs": len(ratios_in),
        "time_pairs": len(ratios_time),
    }


def _svg_grouped_bars(
    labels: list[str],
    series: list[tuple[str, list[float | None], str]],
    *,
    y_max: float | None = None,
    reference_line: float | None = None,
    width: int = 560,
    height: int = 300,
) -> str:
    """Pure-SVG grouped bar chart — no JS / CDN required (works offline + file://)."""
    left, right, top, bottom = 48, 16, 16, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    n_groups = max(len(labels), 1)
    n_series = max(len(series), 1)
    values: list[float] = []
    for _, data, _ in series:
        for value in data:
            number = _num(value)
            if number is not None:
                values.append(number)
    if y_max is None:
        peak = max(values) if values else 1.0
        y_max = peak * 1.12 if peak > 0 else 1.0
    y_max = max(float(y_max), 1e-9)
    if reference_line is not None:
        y_max = max(y_max, float(reference_line) * 1.12)

    group_w = plot_w / n_groups
    bar_gap = 2.0
    bar_w = max(4.0, (group_w - 12) / n_series - bar_gap)

    def y_px(value: float) -> float:
        return top + plot_h * (1.0 - (value / y_max))

    parts: list[str] = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="bar chart" xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="transparent"/>'
    ]
    # grid + y ticks
    for i in range(5):
        frac = i / 4
        value = y_max * (1.0 - frac)
        y = top + plot_h * frac
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="#243049" stroke-width="1"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="#94a3b8" font-size="10" font-family="system-ui,sans-serif">'
            f"{_chart_tick(value)}</text>"
        )
    if reference_line is not None:
        y = y_px(float(reference_line))
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="#fbbf24" stroke-width="1.5" stroke-dasharray="6 4"/>'
            f'<text x="{width - right}" y="{y - 4:.1f}" text-anchor="end" '
            f'fill="#fbbf24" font-size="10" font-family="system-ui,sans-serif">'
            f"parity {reference_line:g}</text>"
        )
    # bars
    for gi, label in enumerate(labels):
        gx = left + gi * group_w
        for si, (_name, data, color) in enumerate(series):
            raw = data[gi] if gi < len(data) else None
            number = _num(raw)
            if number is None:
                continue
            bh = max(0.0, (number / y_max) * plot_h)
            x = gx + 6 + si * (bar_w + bar_gap)
            y = top + plot_h - bh
            title = html.escape(f"{_name}: {_chart_tick(number)}")
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
                f'rx="2" fill="{color}"><title>{title}</title></rect>'
            )
        short = label if len(label) <= 22 else label[:20] + "…"
        parts.append(
            f'<text x="{gx + group_w / 2:.1f}" y="{height - 28}" text-anchor="middle" '
            f'fill="#94a3b8" font-size="10" font-family="system-ui,sans-serif">'
            f"{html.escape(short)}</text>"
        )
    # legend
    lx = left
    for name, _data, color in series:
        parts.append(
            f'<rect x="{lx}" y="{height - 14}" width="10" height="10" rx="2" fill="{color}"/>'
            f'<text x="{lx + 14}" y="{height - 5}" fill="#94a3b8" font-size="11" '
            f'font-family="system-ui,sans-serif">{html.escape(name)}</text>'
        )
        lx += 12 + 7 * len(name) + 18
    parts.append("</svg>")
    return "".join(parts)


def _chart_tick(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _cell_labels(payload: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for cell in payload["cells"]:
        labels.append(f"{cell.get('task') or '?'} · {cell.get('topology') or '?'}")
    return labels


def _arm_series(
    payload: dict[str, Any], field: str, *, scale: float = 1.0
) -> list[tuple[str, list[float | None], str]]:
    out: list[tuple[str, list[float | None], str]] = []
    for arm in ARMS:
        values: list[float | None] = []
        for cell in payload["cells"]:
            number = _num((cell.get("arms") or {}).get(arm, {}).get(field))
            values.append(None if number is None else number * scale)
        short = {"baseline": "Baseline", "brief-only": "Brief-only", "wllm": "wllm"}[arm]
        out.append((short, values, ARM_COLORS[arm]))
    return out


def render_html(payload: dict[str, Any]) -> str:
    summary = summarize(payload)
    data_json = json.dumps(payload, indent=2, sort_keys=True)
    summary_json = json.dumps(summary, indent=2, sort_keys=True)
    title = html.escape(payload["matrix_name"])
    matrix_dir = html.escape(payload["matrix_dir"])
    labels = _cell_labels(payload)

    def fmt_ratio(value: float | None) -> str:
        if value is None:
            return "—"
        return f"{value:.3f}"

    def badge(value: float | None, *, lower_better: bool) -> str:
        if value is None:
            return '<span class="badge muted">n/a</span>'
        good = value < 1 if lower_better else value > 0
        cls = "good" if good else "bad" if (value > 1 if lower_better else value < 0) else "neutral"
        return f'<span class="badge {cls}">{value:.3f}</span>'

    rows = []
    for cell in payload["cells"]:
        arms = cell["arms"]
        c = cell["contrasts"]["wllm_over_baseline"]
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(cell.get('id')))}</code></td>"
            f"<td>{html.escape(str(cell.get('task')))}</td>"
            f"<td>{html.escape(str(cell.get('topology')))}</td>"
            f"<td>{html.escape(str(cell.get('agent')))}/{html.escape(str(cell.get('model')))}</td>"
            f"<td>{html.escape(str(cell.get('effort')))}</td>"
            + "".join(
                f"<td class='num'>{_fmt_pct(arms[arm].get('solve_rate'))}</td>"
                f"<td class='num'>{_fmt_int(arms[arm].get('median_input_tokens'))}</td>"
                f"<td class='num'>{_fmt_sec(arms[arm].get('median_duration_seconds'))}</td>"
                for arm in ARMS
            )
            + f"<td class='num'>{badge(c.get('input_ratio'), lower_better=True)}</td>"
            f"<td class='num'>{badge(c.get('duration_ratio'), lower_better=True)}</td>"
            f"<td class='num'>{_fmt_delta(c.get('score_delta'))}</td>"
            "</tr>"
        )

    chart_solve = _svg_grouped_bars(
        labels, _arm_series(payload, "solve_rate", scale=100.0), y_max=100.0
    )
    chart_tokens = _svg_grouped_bars(labels, _arm_series(payload, "median_input_tokens"))
    chart_time = _svg_grouped_bars(labels, _arm_series(payload, "median_duration_seconds"))
    ratio_series = [
        (
            "Input tokens",
            [
                _num(c["contrasts"]["wllm_over_baseline"].get("input_ratio"))
                for c in payload["cells"]
            ],
            "#a78bfa",
        ),
        (
            "Wall time",
            [
                _num(c["contrasts"]["wllm_over_baseline"].get("duration_ratio"))
                for c in payload["cells"]
            ],
            "#f472b6",
        ),
    ]
    chart_ratios = _svg_grouped_bars(labels, ratio_series, reference_line=1.0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>wllm matrix report — {title}</title>
<style>
  :root {{
    --bg: #0b1220;
    --panel: #121a2b;
    --panel2: #182338;
    --text: #e8eefc;
    --muted: #94a3b8;
    --line: #243049;
    --good: #22c55e;
    --bad: #f87171;
    --accent: #38bdf8;
    --warn: #fbbf24;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: radial-gradient(1200px 600px at 10% -10%, #1e293b 0%, var(--bg) 55%);
    color: var(--text); line-height: 1.45;
  }}
  header {{
    padding: 2rem 1.5rem 1rem; max-width: 1200px; margin: 0 auto;
  }}
  header h1 {{ margin: 0 0 .35rem; font-size: 1.75rem; letter-spacing: -.02em; }}
  header p {{ margin: .2rem 0; color: var(--muted); }}
  main {{ max-width: 1200px; margin: 0 auto; padding: 0 1.5rem 3rem; }}
  .grid {{ display: grid; gap: 1rem; }}
  .kpis {{ grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
  .charts {{ grid-template-columns: 1fr; }}
  @media (min-width: 960px) {{
    .charts {{ grid-template-columns: 1fr 1fr; }}
  }}
  .card {{
    background: linear-gradient(180deg, var(--panel) 0%, var(--panel2) 100%);
    border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1.1rem;
    box-shadow: 0 10px 40px rgba(0,0,0,.25);
  }}
  .card h2 {{ margin: 0 0 .75rem; font-size: 1rem; color: #cbd5e1; font-weight: 600; }}
  .kpi-value {{ font-size: 1.6rem; font-weight: 700; letter-spacing: -.03em; }}
  .kpi-label {{ color: var(--muted); font-size: .85rem; }}
  .infographic {{
    display: grid; gap: 1rem;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  }}
  .step {{
    position: relative; padding: 1rem; border-radius: 14px;
    background: rgba(56,189,248,.06); border: 1px solid rgba(56,189,248,.2);
  }}
  .step .n {{
    display: inline-flex; width: 1.6rem; height: 1.6rem; align-items: center; justify-content: center;
    border-radius: 999px; background: var(--accent); color: #082f49; font-weight: 800; font-size: .85rem;
    margin-bottom: .5rem;
  }}
  .step strong {{ display: block; margin-bottom: .25rem; }}
  .step span {{ color: var(--muted); font-size: .9rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .82rem; }}
  th, td {{ padding: .45rem .4rem; border-bottom: 1px solid var(--line); vertical-align: top; }}
  th {{ text-align: left; color: #cbd5e1; font-weight: 600; position: sticky; top: 0; background: var(--panel); }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .78rem; color: #bae6fd; }}
  .badge {{
    display: inline-block; min-width: 3.5rem; text-align: center; padding: .15rem .4rem;
    border-radius: 999px; font-weight: 700; font-size: .78rem;
  }}
  .badge.good {{ background: rgba(34,197,94,.15); color: #86efac; }}
  .badge.bad {{ background: rgba(248,113,113,.15); color: #fca5a5; }}
  .badge.neutral {{ background: rgba(148,163,184,.15); color: #cbd5e1; }}
  .badge.muted {{ background: rgba(148,163,184,.1); color: var(--muted); }}
  .legend {{ display: flex; flex-wrap: wrap; gap: .75rem; margin: .5rem 0 0; color: var(--muted); font-size: .85rem; }}
  .swatch {{ display: inline-block; width: .75rem; height: .75rem; border-radius: 2px; margin-right: .35rem; vertical-align: middle; }}
  .note {{ color: var(--muted); font-size: .88rem; margin-top: .75rem; }}
  .scroll {{ overflow-x: auto; max-height: 480px; overflow-y: auto; }}
  footer {{ max-width: 1200px; margin: 0 auto; padding: 0 1.5rem 2rem; color: var(--muted); font-size: .8rem; }}
  svg.chart {{ width: 100%; height: auto; display: block; min-height: 260px; }}
</style>
</head>
<body>
<header>
  <h1>wllm agent matrix report</h1>
  <p><strong>{title}</strong></p>
  <p>{matrix_dir}</p>
  <p>Cells OK: {summary['cells_ok']}/{summary['cells_total']}
     · jobs={html.escape(str(payload.get('jobs')))}
     · timing_comparable={html.escape(str(payload.get('timing_comparable')))}
  </p>
</header>
<main>
  <section class="card" style="margin-bottom:1rem">
    <h2>How to read this (3-arm design)</h2>
    <div class="infographic">
      <div class="step"><div class="n">1</div><strong>Baseline</strong><span>No wllm access. Agent discovers the workspace alone (cold discovery tax).</span></div>
      <div class="step"><div class="n">2</div><strong>Brief-only</strong><span>One bounded <code>wllm context</code> briefing, then no further wllm runtime tools.</span></div>
      <div class="step"><div class="n">3</div><strong>Full wllm</strong><span>Same style briefing + runtime access to the pinned <code>wllm</code> CLI.</span></div>
      <div class="step"><div class="n">≈</div><strong>Ratios</strong><span><code>wllm / baseline</code> geometric means. <strong>&lt; 1 favors wllm</strong> for tokens and wall time. Score delta is wllm − baseline (higher is better).</span></div>
    </div>
    <p class="note">Comparisons are only valid within the same task · agent · model · effort · topology · machine regime. Missing telemetry is never treated as zero.</p>
  </section>

  <section class="grid kpis" style="margin-bottom:1rem">
    <div class="card"><div class="kpi-label">Cells completed</div><div class="kpi-value">{summary['cells_ok']}/{summary['cells_total']}</div></div>
    <div class="card"><div class="kpi-label">Geo-mean input tokens (wllm/baseline)</div><div class="kpi-value">{fmt_ratio(summary.get('geo_input_ratio_wllm_baseline'))}</div><div class="kpi-label">wins &lt;1: {summary['input_wins']}/{summary['input_pairs']}</div></div>
    <div class="card"><div class="kpi-label">Geo-mean wall time (wllm/baseline)</div><div class="kpi-value">{fmt_ratio(summary.get('geo_duration_ratio_wllm_baseline'))}</div><div class="kpi-label">wins &lt;1: {summary['time_wins']}/{summary['time_pairs']}</div></div>
    <div class="card"><div class="kpi-label">Mean score Δ (wllm − baseline)</div><div class="kpi-value">{_fmt_delta(summary.get('mean_score_delta'))}</div></div>
  </section>

  <section class="grid charts" style="margin-bottom:1rem">
    <div class="card">
      <h2>Solve rate by arm (%)</h2>
      {chart_solve}
    </div>
    <div class="card">
      <h2>Median input tokens by arm</h2>
      {chart_tokens}
    </div>
    <div class="card">
      <h2>Median end-to-end seconds by arm</h2>
      {chart_time}
    </div>
    <div class="card">
      <h2>Ratios wllm / baseline (&lt;1 favors wllm)</h2>
      {chart_ratios}
      <p class="note">Dashed gold line = parity (1.0). Hover bars for exact values.</p>
    </div>
  </section>

  <section class="card">
    <h2>Per-cell detail</h2>
    <div class="scroll">
      <table>
        <thead>
          <tr>
            <th>Cell</th><th>Task</th><th>Topology</th><th>Agent/Model</th><th>Effort</th>
            <th class="num">B solve</th><th class="num">B tok</th><th class="num">B s</th>
            <th class="num">Br solve</th><th class="num">Br tok</th><th class="num">Br s</th>
            <th class="num">W solve</th><th class="num">W tok</th><th class="num">W s</th>
            <th class="num">tok ratio</th><th class="num">time ratio</th><th class="num">score Δ</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
  </section>
</main>
<footer>
  Generated by <code>bench/agent/matrix_report.py</code> · fully self-contained HTML (inline SVG charts, no CDN) · data embedded below for reproducibility.
</footer>
<script id="matrix-data" type="application/json">{html.escape(data_json)}</script>
<script id="matrix-summary" type="application/json">{html.escape(summary_json)}</script>
</body>
</html>
"""


def _fmt_pct(value: Any) -> str:
    number = _num(value)
    if number is None:
        return "—"
    return f"{number * 100:.0f}%"


def _fmt_int(value: Any) -> str:
    number = _num(value)
    if number is None:
        return "—"
    return f"{int(round(number)):,}"


def _fmt_sec(value: Any) -> str:
    number = _num(value)
    if number is None:
        return "—"
    return f"{number:.1f}"


def _fmt_delta(value: Any) -> str:
    number = _num(value)
    if number is None:
        return "—"
    sign = "+" if number > 0 else ""
    return f"{sign}{number:.3f}"


def write_report(matrix_dir: Path, output: Path | None = None) -> Path:
    payload = collect_matrix(matrix_dir)
    # Also write machine-readable summary next to the HTML.
    summary_path = matrix_dir / "matrix-report.json"
    summary_path.write_text(
        json.dumps({"summary": summarize(payload), "matrix": payload}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    out = output or (matrix_dir / "report.html")
    out.write_text(render_html(payload), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "matrix_dir",
        type=Path,
        help="Path to a matrix-* results directory containing artifact-index.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="HTML output path (default: <matrix_dir>/report.html)",
    )
    args = parser.parse_args(argv)
    matrix_dir = args.matrix_dir.expanduser().resolve()
    try:
        path = write_report(matrix_dir, args.output.expanduser().resolve() if args.output else None)
    except (OSError, json.JSONDecodeError, FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"matrix_report.py: error: {error}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
