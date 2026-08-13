import json
import sys

sys.path.insert(0, ".")
import main as old_agent

old_route = old_agent._ROUTE


def summarize_route(route):
    ops = {}
    for step in route:
        units = [step.get("farmer") or ["PASS"], *list(step.get("hands") or [])]
        for u in units:
            op = u[0] if u else "PASS"
            ops[op] = ops.get(op, 0) + 1
    return ops


old_ops = summarize_route(old_route)

with open("logs/daily/kaggriculture-episodes-2026-08-12/92164137.json") as f:
    data = json.load(f)

steps = data["steps"]
new_route = []
for t in range(1, len(steps)):
    entry = steps[t][1] if 1 < len(steps[t]) else {}
    action = (entry or {}).get("action") or {}
    new_route.append(
        {
            "farmer": list(action.get("farmer") or ["PASS"]),
            "hands": [list(h or ["PASS"]) for h in (action.get("hands") or [])],
        }
    )

new_ops = summarize_route(new_route)

print("--- OLD ROUTE (Trex) OPS ---")
for k, v in sorted(old_ops.items(), key=lambda item: -item[1]):
    if v > 0:
        print(f"{k}: {v}")

print("\n--- NEW ROUTE (vladee) OPS ---")
for k, v in sorted(new_ops.items(), key=lambda item: -item[1]):
    if v > 0:
        print(f"{k}: {v}")
