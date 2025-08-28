import ast
import json
import os
import openai
import random
import sys
from prettytable import PrettyTable
from LLMLP import LLMLP
from utils import *
import re

# Put your query here (fallback when no DQUERY is provided)
QUERY = r"""What 8 letter word can have a letter taken away and it still makes a word..."""

EXP_NAME = "trial_1"
MODEL = "gpt-5-mini"
ACTIVATION = "listwise"
TYPE = "open-ended"
DIR_NAME = "trial"

ROLES = ["Assistant", "Assistant", "Assistant", "Assistant"]

# --- Helpers to detect "final turn" from AgentClinic-style system prompt in the query ---
_RE_MAX = re.compile(r"You are only allowed to ask\s+(\d+)\s+questions", re.IGNORECASE)
_RE_CUR = re.compile(r"You have asked\s+(\d+)\s+questions\s+so\s+far", re.IGNORECASE)

def _is_final_turn_text(query: str) -> bool:
    # 在 DQUERY 里，AgentClinic 的 system_prompt 和用户拼接在一起，我们直接整体解析
    mmax = _RE_MAX.search(query or "")
    mcur = _RE_CUR.search(query or "")
    if not (mmax and mcur): return False
    try:
        max_q = int(mmax.group(1)); cur_q = int(mcur.group(1))
    except Exception:
        return False
    return (cur_q + 1) >= max_q

def set_rd_seed(seed: int):
    random.seed(seed)

def answer(
    query: str,
    *,
    model: str = MODEL,
    roles = None,
    rounds: int = 3,
    activation: str = ACTIVATION,
    qtype: str = TYPE,
):
    """
    Run DyLAN (LLMLP) once and return final answer text.
    Guaranteed non-empty by selecting best candidate from completions if needed.
    Returns: (final_answer, meta_dict)
    """
    roles = roles or ROLES
    assert len(roles) > 0
    set_rd_seed(0)

    # 如果这是末轮，给 query 加一条硬性输出规范
    if _is_final_turn_text(query):
        query = (query.rstrip() + 
                 "\n\nFINAL TURN: Reply strictly with exactly one line: "
                 "DIAGNOSIS READY: <disease>.")

    llmlp = LLMLP(model, len(roles), roles, rounds, activation, qtype, model)
    llmlp.zero_grad()
    res, resp_cnt, completions, prompt_tokens, completion_tokens = llmlp.forward(query)
    imp_score = llmlp.backward(res)  # list length = rounds * roles

    # 结构化 importance: [round][role]
    importance_by_round = [
        [imp_score[idx] for idx in range(len(roles)*rid, len(roles)*(rid+1))]
        for rid in range(rounds)
    ]

    # --- 如果 res 为空：在 DyLAN 内部“自洽选优”确保非空 ---
    def _pick_best_candidate():
        # 按“各角色 across rounds 的总重要性”选一个最佳角色，再取其“最后一轮”的响应
        imp_sum_per_role = [sum(importance_by_round[r][j] for r in range(rounds))
                            for j in range(len(roles))]
        best_role = max(range(len(roles)), key=lambda j: imp_sum_per_role[j])
        last_round = rounds - 1
        cand = completions[best_role][last_round] if completions[best_role][last_round] else ""
        if cand and cand.strip():
            return str(cand).strip()
        # 若该角色为空：退而求其次，取“最后一轮任一非空”
        for j in range(len(roles)):
            c = completions[j][last_round]
            if c and str(c).strip():
                return str(c).strip()
        return ""  # 实在没有（极少发生）

    final_text = (str(res) if res is not None else "").strip()
    if not final_text:
        final_text = _pick_best_candidate()

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
    final, meta = answer(QUERY, model=MODEL, roles=ROLES, rounds=3, activation=ACTIVATION, qtype=TYPE)

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
