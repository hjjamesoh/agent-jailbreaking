"""Human-readable side-by-side steering audit; all content is HTML escaped."""

from __future__ import annotations

import html
import json
from pathlib import Path


def render_review_file(source: str | Path) -> Path:
    source = Path(source)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    sections = []
    for row in rows:
        task_id = html.escape(str(row["task_id"]))
        prompt = html.escape(str(row["prompt"]))
        if "comparisons" in row:
            arms = row["comparisons"]
            cells = []
            for name, payload in sorted(arms.items(), key=lambda item: (
                item[0] != "baseline", item[0])):
                trace = payload["rollout"]
                judgment = payload.get("judgment")
                display = json.dumps({"steps": trace.get("steps"),
                                      "final_answer": trace.get("final_answer"),
                                      "stop_reason": trace.get("stop_reason"),
                                      "local_judgment": judgment},
                                     ensure_ascii=False, indent=2)
                cells.append(f"<section><h4>{html.escape(name)}</h4><pre>{html.escape(display)}</pre></section>")
            priority = html.escape(str(row.get("review_priority", "normal")))
        else:
            cells = []
            for name, rollout in sorted(row["responses"].items(), key=lambda item: (
                item[0] != "baseline", item[0])):
                display = json.dumps({"rollout": rollout,
                                      "local_judgment": row["judgments"].get(name)},
                                     ensure_ascii=False, indent=2)
                cells.append(f"<section><h4>{html.escape(name)}</h4><pre>{html.escape(display)}</pre></section>")
            priority = "normal"
        sections.append(
            f"<article><h3>{task_id} · {priority}</h3><p>{prompt}</p>"
            f"<div class='grid'>{''.join(cells)}</div></article>"
        )
    document = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Agent direction manual review</title>"
        "<style>body{font:15px system-ui;margin:1.5rem;background:#f7f7f7;color:#111}"
        "article{background:white;padding:1rem;margin:1rem 0;border:1px solid #ddd}"
        ".grid{display:flex;gap:1rem;overflow-x:auto}section{min-width:28rem;flex:1}"
        "pre{white-space:pre-wrap;word-break:break-word;background:#f1f1f1;padding:.7rem}"
        "p{white-space:pre-wrap}</style><h1>Steering: baseline vs addition vs ablation</h1>"
        "<p>Judge labels are estimates. Inspect tool calls, tool results and final answers. "
        "No automatic judgment in this file is a human-confirmed conclusion.</p>"
        + "".join(sections) + "</html>"
    )
    output = source.with_suffix(".html")
    output.write_text(document, encoding="utf-8")
    return output
