#!/usr/bin/env python3
"""Build the v0.4.1 candidate: v0.4.0 chassis + metacounter EXP240 dual-expert router.

The measured gap (logs/september_spy.csv + 2026-09-13 daily batch): v0.4.0 only
has specialized routes for YARN-containing day-6 draws (22% of seeds); on the
other 78% it falls back to route 0 while the metacounter-class field plays 29
EXP240 draw-specialized tapes (routes 100-128, Yusuke Hayashi's Shop Router 0913
lineage, Apache-2.0).

This graft is minimal and reversible:
  * keep v0.4.0's routes 0-12 and its day-27 endgame switch;
  * add the EXP240 payload as a second route table (100-128);
  * replace the day-6 router with the metacounter's dual-expert gate
    (EXP240 when no YARN in first two shops, v0.4.0's table otherwise);
  * keep v0.4.0's R42 opening for the v0.4.0 tapes and metacounter's opening
    for the EXP240 tapes (their routes were authored against it).

Output: scratch/v0_4_1_candidate.py + a one-line summary of what changed.
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
MAIN = os.path.join(PROJECT_ROOT, "main.py")
METACOUNTER = os.path.join(PROJECT_ROOT, "scratch", "public_kernels", "metacounter_agent.py")
OUT = os.path.join(PROJECT_ROOT, "scratch", "v0_4_1_candidate.py")


def main() -> None:
    with open(MAIN) as f:
        v40 = f.read()
    with open(METACOUNTER) as f:
        mc = f.read()

    # 1. Extract metacounter's EXP240 payload literal (the b85 blob) and its shop table.
    m_payload = re.search(r"_R108_DATA=json\.loads\(zlib\.decompress\(base64\.b85decode\('([^']+)'\)\)\)", mc)
    if not m_payload:
        sys.exit("metacounter payload literal not found")
    mc_payload = m_payload.group(1)
    m_shops = re.search(r"_R108_SHOP_ROUTES=\{tuple\(r\['shops'\]\):r\['route'\] for r in _R108_DATA\['shops'\]\}", mc)
    if not m_shops:
        sys.exit("metacounter shop table line not found")

    # 2. Locate v0.4.0's assembly block: from `_ROUTES={0:_PAYLOAD['base']}` to `del _PAYLOAD`.
    start = v40.index("_ROUTES={0:_PAYLOAD['base']}")
    end = v40.index("del _PAYLOAD") + len("del _PAYLOAD")
    assembly = v40[start:end]

    # 3. Compose the new assembly: v0.4.0 routes, then EXP240 routes under ids 100+.
    exp240_block = (
        "_ROUTES.update({int(k)+100:[list(_E['actions'][i]) for i in ids] "
        "for k,ids in ((str(int(r['route'])-100),r['route'] and []) for r in [])})  # placeholder"
    )
    # The EXP240 payload stores full 719-action tapes in ['routes'][str(route_id)] as action indices.
    # Cleanest: decode the payload once and emit tapes with the same shape as _ROUTES values.
    import base64
    import json
    import zlib

    data = json.loads(zlib.decompress(base64.b85decode(mc_payload)))
    routes = {int(k): [data["actions"][i] for i in ids] for k, ids in data["routes"].items()}
    shop_routes = {tuple(r["shops"]): r["route"] for r in data["shops"]}

    # 4. Build the literal EXP240 route dict as compact Python source.
    # JSON-encode each tape: repr() of the decoded structures would flatten the
    # action dicts into bare key lists and crash dict() at import time.
    import json as _json

    lines = ["_EXP_ROUTES={"]
    for rid in sorted(routes):
        if rid < 100:
            continue
        lines.append(f"{rid}:{_json.dumps(routes[rid])},")
    lines.append("}")
    exp_routes_src = "\n".join(lines)
    shops_src = repr(shop_routes)

    # 5. New router: dual-expert gate. EXP240 when no YARN in first two shops.
    router_src = (
        "_EXP_SHOP_ROUTES=" + shops_src + "\n"
        "def _router(observation,step,state):\n"
        "    if step>=144 and not state.get('day6'):\n"
        "        shops=_get(_get(observation,'town',{}),'unlocked_shops',[]) or []\n"
        "        pair=tuple(shops[:2])\n"
        "        if pair.count('YARN_STORE')<=0:\n"
        "            state['expert']='EXP240'\n"
        "            state['route']=_EXP_SHOP_ROUTES.get(pair,100)\n"
        "        else:\n"
        "            state['expert']='V39'\n"
        "            state['route']={('BAKERY', 'YARN_STORE'): 3, ('BRUNCH_SPOT', 'YARN_STORE'): 4, "
        "('FARMERS_MARKET', 'YARN_STORE'): 5, ('ICE_CREAM_SHOP', 'YARN_STORE'): 6, "
        "('PET_CAFE', 'YARN_STORE'): 5, ('PIZZA_SHOP', 'YARN_STORE'): 7, "
        "('SMOOTHIE_SHOP', 'YARN_STORE'): 8, ('YARN_STORE', 'BAKERY'): 9, "
        "('YARN_STORE', 'BRUNCH_SPOT'): 9, ('YARN_STORE', 'FARMERS_MARKET'): 1, "
        "('YARN_STORE', 'ICE_CREAM_SHOP'): 9, ('YARN_STORE', 'PET_CAFE'): 10, "
        "('YARN_STORE', 'PIZZA_SHOP'): 6, ('YARN_STORE', 'SMOOTHIE_SHOP'): 11, "
        "('YARN_STORE', 'YARN_STORE'): 12}.get(pair,0)\n"
        "        state['day6']=True\n"
        "    if step>=648 and not state.get('day27'):\n"
        "        state['route']=2\n"
        "        state['day27']=True\n"
        "    return state.get('route',0)\n"
    )

    # 6. The R42 opening rewrite must only touch the v0.4.0 tapes (0-12):
    opening_src = (
        "_R42_OPENING=[['BUY_PRODUCT', 'WHEAT', 5], ['BUY_PRODUCT', 'WHEAT', 10], ['SELL', 'WHEAT', 60]]\n"
        "for _rid in (0,1,2,3,4,5,6,7,8,9,10,11,12):\n"
        "    _t=_ROUTES.get(_rid)\n"
        "    if _t is not None:\n"
        "        _t[0]=dict(_t[0],market=[list(o) for o in _R42_OPENING])\n"
        "_E42_OPENING=[['BUY_PRODUCT', 'WHEAT', 13], ['BUY_PRODUCT', 'WHEAT', 30], ['SELL', 'WHEAT', 30]]\n"
        "for _rid,_t in _ROUTES.items():\n"
        "    if _rid>=100:\n"
        "        _t[0]=dict(_t[0],market=[list(o) for o in _E42_OPENING])\n"
    )

    # 7. Splice: replace assembly, router, and opening in one block.
    new_v40 = v40
    # replace the assembly block with assembly + EXP routes
    new_assembly = assembly + "\n" + exp_routes_src + "\n_ROUTES.update(_EXP_ROUTES)\ndel _EXP_ROUTES"
    new_v40 = new_v40.replace(assembly, new_assembly)
    # replace router
    r_start = new_v40.index("def _router(observation,step,state):")
    r_end = new_v40.index("_R42_OPENING=")
    new_v40 = new_v40[:r_start] + router_src + "\n" + new_v40[r_end:]
    # replace opening
    o_start = new_v40.index("_R42_OPENING=")
    o_end = new_v40.index("_IMPL=make_agent(")
    # the opening block ends right before _IMPL; find the del line
    o_block = new_v40[o_start:o_end]
    o_block_end = o_block.index("del _r42_tape") + len("del _r42_tape")
    new_v40 = new_v40[:o_start] + opening_src + "\n" + new_v40[o_start + o_block_end :]

    # 8. Version bump + provenance note.
    new_v40 = new_v40.replace('AGENT_VERSION = "0.4.0"', 'AGENT_VERSION = "0.4.1"')
    new_v40 = new_v40.replace(
        "WHY (measured 2026-09-14",
        "v0.4.1: grafts the EXP240 draw-specialized route library (routes 100-128)\n"
        "  and the dual-expert day-6 router from the Apache-2.0 metacounter lineage\n"
        "  (Yusuke Hayashi Shop Router 0913 + lynnsakurai chassis). Non-YARN draws\n"
        "  (78% of seeds) previously fell back to generic route 0.\n\n"
        "WHY (measured 2026-09-14",
        1,
    )

    with open(OUT, "w") as f:
        f.write(new_v40)
    n_exp = sum(1 for r in routes if r >= 100)
    print(f"wrote {OUT}")
    print(f"  v0.4.0 routes kept: {sum(1 for r in routes if r < 100 and r in (0,1,2,3,4,5,6,7,8,9,10,11,12))}/13")
    print(f"  EXP240 routes added: {n_exp}")
    print(f"  shop pairs covered by EXP240: {len(shop_routes)}")


if __name__ == "__main__":
    main()