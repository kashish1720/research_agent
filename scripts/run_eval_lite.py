import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

VQ = "Compare the top 3 open-source vector databases for a startup building RAG products."
QUERIES = [
    ("vector_db", VQ, "", ["DONE", "DEGRADED"]),
    ("multi_agent", "Research the pros and cons of using a multi-agent architecture for customer support automation.", "", ["DONE", "DEGRADED"]),
    ("memory_approaches", "Compare different approaches to adding memory in an AI support agent.", "", ["DONE", "DEGRADED"]),
    ("nonsense", "asdkjh qwpoeiru zxcmnb lkjhg", "", ["FAILED_GRACEFULLY"]),
    ("search_faults", VQ, "timeout:0.6", ["DONE", "DEGRADED", "FAILED_GRACEFULLY"]),
]

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="", help="comma separated ids")
ap.add_argument("--sleep", type=int, default=20)
ap.add_argument("--set", action="append", default=[], help="NAME=VALUE env override for every run")
args = ap.parse_args()
only = {x for x in args.only.split(",") if x}

base_env = dict(os.environ, PYTHONIOENCODING="utf-8")
for item in args.set:
    name, _, value = item.partition("=")
    base_env[name] = value

Path("samples").mkdir(exist_ok=True)
Path("evals").mkdir(exist_ok=True)
rows = []
for qid, query, faults, expected in QUERIES:
    if only and qid not in only:
        continue
    env = dict(base_env)
    if faults:
        env["FAULT_INJECT"] = faults
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "main.py", "--json", query],
                           capture_output=True, text=True, encoding="utf-8", env=env, timeout=900)
        out, err, code = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        out, err, code = "", "timeout after 900s", -1
    wall = round(time.time() - t0, 1)
    row = {"id": qid, "query": query, "faults": faults, "expected": expected,
           "exit_code": code, "wall_s": wall}
    try:
        ans = json.loads(out)
        Path("samples/" + qid + ".json").write_text(json.dumps(ans, indent=2), encoding="utf-8")
        u = ans["usage"]
        row.update(
            status=ans["status"],
            confidence=ans["confidence"]["level"],
            findings=len(ans["key_findings"]),
            sources=len(ans["sources_used"]),
            domains=len({s["domain"] for s in ans["sources_used"]}),
            coverage="{}/{}".format(ans["coverage"]["covered"], ans["coverage"]["total"]),
            reasons=ans["reasons"],
            llm_calls=u["llm_calls"],
            searches=u["searches"],
            fetches=u["fetches"],
            retries=u["tool_retries"],
            short_answer=ans["short_answer"][:200],
        )
        row["pass"] = ans["status"] in expected
    except Exception:
        row.update(status="CRASH", stderr_tail=err[-300:])
        row["pass"] = False
    rows.append(row)
    print(qid, row.get("status"), row.get("confidence", "-"), "PASS" if row["pass"] else "FAIL")
    time.sleep(args.sleep)

Path("evals/results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
lines = ["| id | status | confidence | findings | sources | domains | coverage | retries | wall_s | result |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        r["id"], r.get("status"), r.get("confidence", "-"), r.get("findings", "-"),
        r.get("sources", "-"), r.get("domains", "-"), r.get("coverage", "-"),
        r.get("retries", "-"), r["wall_s"], "PASS" if r["pass"] else "FAIL"))
Path("evals/results.md").write_text("\n".join(lines), encoding="utf-8")
print()
print("\n".join(lines))
