import os
import sys
import random
import re
from prettytable import PrettyTable
from LLMLP import LLMLP
from utils import *
from prompt_lib import ROLE_PRESETS

# --------- Default demo query (only used when no DQUERY is provided) ----------
QUERY = r"""
You are a team of clinical agents cooperating under the following strict protocol:
- Only the TestPlanner may emit 'REQUEST TEST: <...>'.
- Only the Finalizer may emit 'DIAGNOSIS READY: <...>' (or 'NEED MORE: <...>').
- Keep each turn to 1–3 sentences; be terse and specific.

Case summary:
- CC: fluctuating diplopia and proximal limb weakness for ~1 month.
- Worse with exertion; improves with rest. No fever. No dysphagia.
- Exam: fatigable weakness; stable vitals; no respiratory distress.

Team goal:
- Reach a final diagnosis through minimal high-yield tests with guideline safety.
"""

EXP_NAME = "clinical_demo"
MODEL = os.environ.get("DYLAN_MODEL", "gpt-4o-mini")

ACTIVATION = "listwise"
TYPE = "open-ended"
DIR_NAME = "trial"

# Recommended 7-agent full team (you can switch to clinical_v1_min4)
ROLES = ROLE_PRESETS["clinical_v2_full7"]

def set_rd_seed(seed: int):
    random.seed(seed)

# --- Optional: detect if this is the final turn from an AgentClinic-style prompt ---
_RE_MAX = re.compile(r"You are only allowed to ask\s+(\d+)\s+questions", re.IGNORECASE)
_RE_CUR = re.compile(r"You have asked\s+(\d+)\s+questions\s+so\s+far", re.IGNORECASE)

def _is_final_turn_text(text: str) -> bool:
    mmax = _RE_MAX.search(text or "")
    mcur = _RE_CUR.search(text or "")
    if not (mmax and mcur): 
        return False
    try:
        max_q = int(mmax.group(1))
        cur_q = int(mcur.group(1))
        return (cur_q + 1) >= max_q
    except Exception:
        return False

def answer(query: str, *, model: str = MODEL, roles = None, rounds: int = 3,
           activation: str = ACTIVATION, qtype: str = TYPE):
    roles = roles or ROLES
    assert len(roles) > 0
    set_rd_seed(0)

    # Gentle nudge for final turn formatting when embedded under AgentClinic
    if _is_final_turn_text(query):
        query = (query.rstrip() +
                 "\n\nFINAL TURN: If you are the Finalizer and evidence is sufficient, "
                 "emit exactly one line 'DIAGNOSIS READY: <disease>'. Otherwise emit "
                 "'NEED MORE: <the single most critical missing item>'.")

    llmlp = LLMLP(model, len(roles), roles, rounds, activation, qtype, model)
    llmlp.zero_grad()
    res, resp_cnt, completions, prompt_tokens, completion_tokens = llmlp.forward(query)
    imp_score = llmlp.backward(res)

    importance_by_round = [
        [imp_score[idx] for idx in range(len(roles)*rid, len(roles)*(rid+1))]
        for rid in range(rounds)
    ]

    final_text = (str(res) if res is not None else "").strip()

    meta = {
        "resp_cnt": resp_cnt,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "completions": completions,
        "importance_by_round": importance_by_round,
        "roles": roles,
        "rounds": rounds,
        "model": model,
    }
    return final_text, meta

def main_cli():
    final, meta = answer(QUERY, model=MODEL, roles=ROLES, rounds=3,
                         activation=ACTIVATION, qtype=TYPE)

    pt = PrettyTable()
    pt.add_column("Round", ROLES)
    for rid in range(meta["rounds"]):
        responses = [
            (meta["completions"][idx][rid] if meta["completions"][idx][rid] is not None else "No response.")
            for idx in range(len(ROLES))
        ]
        pt.add_column(str(rid + 1), responses, "l")

    print(r"Query: {}".format(QUERY))
    print(r"#API calls: {}".format(meta["resp_cnt"]))
    print(r"Prompt Tokens: {}".format(meta["prompt_tokens"]))
    print(r"Completion Tokens: {}".format(meta["completion_tokens"]))
    print(pt)
    print(r"Final Answer: {}".format(final))
    print()
    imp_sum = [sum(meta["importance_by_round"][rid][idx] for rid in range(meta["rounds"])) for idx in range(len(ROLES))]
    print(r"Agent Importance Scores: {}".format(imp_sum))

if __name__ == "__main__":
    dquery = os.environ.get("DQUERY")
    if dquery is not None and dquery.strip():
        final, _meta = answer(dquery, model=os.environ.get("DYLAN_MODEL", MODEL))
        print("FINAL_ANSWER: " + str(final).strip())
        sys.exit(0)
    main_cli()
