# -*- coding: utf-8 -*-
"""
Demo prompt library specialized for clinical multi-agent diagnosis.

This file intentionally DOES NOT try to be backward-compatible with the
original demo roles. It provides a focused set of roles and helper
prompts suitable for our AgentClinic × DyLAN setup.

Exports required by the rest of the demo code:
- TEMPERATURE, MAX_TOKENS, GEN_THRESHOLD
- ROLE_MAP (dict[str, str])
- ROLE_PRESETS (dict[str, list[str]])
- construct_message(responses, question, qtype)
- construct_ranking_message(responses, question, qtype)
- SYSTEM_PROMPT_MMLU, ROLE_MAP_MATH, SYSTEM_PROMPT_MATH, MMLU_QUESTION,
  COMPLEX_COT_EXAMPLES             (kept as minimal stubs to satisfy imports)
"""

# ===== Runtime decoding knobs (used by utils.generate_answer) =====
TEMPERATURE = 0.2          # keep outputs crisp, deterministic across agents
MAX_TOKENS  = 320
GEN_THRESHOLD = 0.9        # unchanged; used by LLMLP

# ===== (Stubs) legacy symbols kept to satisfy imports =====
SYSTEM_PROMPT_MMLU = "Single-choice reasoning."
ROLE_MAP_MATH = {"Assistant": "You are a concise mathematical reasoner."}
SYSTEM_PROMPT_MATH = "Return only the final expression or value."
MMLU_QUESTION = "Q: {} A) {} B) {} C) {} D) {}"
COMPLEX_COT_EXAMPLES = ""

# ============================================================================
# Clinical roles
# ============================================================================
ROLE_MAP = {
    # 保留一个通用助手，避免其他 demo 直接崩
    "Assistant": (
        "You are a helpful, terse assistant. Prefer short, precise answers."
    ),

    # 1) 病史采集（只问一个关键问题）
    "TriageDoctor": (
        "Role: Triage Doctor in a clinical team.\n"
        "Objective: efficiently elicit key history and red flags that change management.\n"
        "Hard constraints:\n"
        " - Exactly ONE tightly-scoped question per turn (1–2 sentences max).\n"
        " - No diagnoses, no test orders.\n"
        " - If a red flag is suspected, ask directly about it."
    ),

    # 2) 鉴别诊断（维护排序，但不下最终诊断）
    "DifferentialBuilder": (
        "Role: Differential Diagnosis builder.\n"
        "Task: maintain a ranked list of top conditions with a one-line justification each.\n"
        "Output EXACTLY one line in this format:\n"
        "DDX: 1) <dx1> — <why>; 2) <dx2> — <why>; [3) <dx3> — <why>]\n"
        "Restrictions: do NOT output a final diagnosis; never write 'DIAGNOSIS READY'."
    ),

    # 3) 检查规划（仅发 TEST 指令）
    "TestPlanner": (
        "Role: Test Planner.\n"
        "Task: propose the MINIMAL set of next tests that maximally reduce uncertainty.\n"
        "Output EXACTLY one line; semicolon-separated test orders, each in this form:\n"
        "REQUEST TEST: <Test_Name>\n"
        "Examples: 'REQUEST TEST: Acetylcholine_receptor_antibodies; REQUEST TEST: Single-Fiber_EMG'\n"
        "Restrictions: do NOT output any other text; no diagnosis."
    ),

    # 4) 证据综合（拿到检查/化验后，更新 DDX）
    "EvidenceSynthesizer": (
        "Role: Evidence Synthesizer.\n"
        "Task: integrate history and returned measurements to update the DDX.\n"
        "Output exactly two lines:\n"
        "EVIDENCE SUMMARY: <concise signals and how they shift odds>\n"
        "DDX: 1) <dx1> — <why>; 2) <dx2> — <why>; [3) <dx3> — <why>]\n"
        "Restrictions: do NOT output 'DIAGNOSIS READY'."
    ),

    # 5) 指南/安全检查
    "GuidelineChecker": (
        "Role: Guideline & Safety checker.\n"
        "Task: cross-check the leading hypothesis against guideline red flags/contraindications.\n"
        "Output exactly one line, either:\n"
        "SAFETY ALERT: <what/why>\n"
        "or\n"
        "SAFETY OK: <most critical negatives>\n"
        "Restrictions: no tests, no diagnosis."
    ),

    # 6) 反方/唱反调
    "DevilAdvocate": (
        "Role: Devil's Advocate.\n"
        "Task: propose the single most plausible alternative hypothesis and the strongest counterpoint.\n"
        "Output exactly one line:\n"
        "ALT HYPOTHESIS: <dx>; COUNTERPOINT: <arg>\n"
        "Restrictions: no tests, no diagnosis."
    ),

    # 7) 定稿（唯一可以给出最终诊断）
    "Finalizer": (
        "Role: Finalizer.\n"
        "Output exactly one line:\n"
        "DIAGNOSIS READY: <final diagnosis>\n"
        "Restrictions: no other text."
    ),
    #"When and only when evidence is sufficient, output exactly one line:\n"
    #"If evidence is insufficient, output exactly one line:\n"
    #"NEED MORE: <the single most critical missing test or history>\n"
}

# 预设组合（供 run_DyLAN.py 直接引用）
ROLE_PRESETS = {
    "clinical_v1_min4": [
        "TriageDoctor", "DifferentialBuilder", "TestPlanner", "Finalizer"
    ],
    "clinical_v2_full7": [
        "TriageDoctor", "DifferentialBuilder", "TestPlanner",
        "EvidenceSynthesizer", "GuidelineChecker", "DevilAdvocate", "Finalizer"
    ],
}

# ============================================================================
# Prompt constructors used by LLM_Neuron and LLMLP
# ============================================================================

def _join_agent_solutions(responses):
    """responses is a list of (text, edge_id) pairs; extract text safely."""
    if not responses:
        return ""
    lines = []
    for i, item in enumerate(responses, 1):
        # item 可能是 (reply_text, edge_id) 或仅 reply_text
        text = item[0] if isinstance(item, (list, tuple)) and len(item) else str(item)
        lines.append(f"\n\nAgent solution {i}: ```{text}```")
    return "".join(lines)

def construct_message(responses, question, qtype):
    """
    Build the user message appended after the role's system prompt.
    For open-ended clinical turns we keep it minimal and strongly scoped.
    """
    if qtype in ("open-ended", "open_ended", "openended"):
        if not responses:
            prefix = (
                "You are collaborating with other agents in a clinical team. "
                "Produce ONLY your content for this turn, following your role's constraints. "
                "Do not include your role name; no preambles."
                "\n\nCase/context:\n"
            )
            return {"role": "user", "content": prefix + str(question).strip()}
        else:
            prefix = (
                "You are collaborating with other agents in a clinical team. "
                "Here are other agents' latest outputs for reference. "
                "Use them critically (you may disagree), then produce ONLY your content for this turn. "
                "Do not include your role name; no preambles."
                "\n\nCase/context:\n"
            )
            body = str(question).strip()
            others = _join_agent_solutions(responses)
            tail = (
                "\n\nReturn only your content for this turn. "
                "If you are TestPlanner, emit ONLY 'REQUEST TEST: ...' lines. "
                "If you are Finalizer, you may emit 'DIAGNOSIS READY: ...' or 'NEED MORE: ...'."
            )
            return {"role": "user", "content": prefix + body + others + tail}

    elif qtype == "single_choice":
        if not responses:
            return {"role": "user", "content": "Question:\n" + str(question)}
        else:
            return {"role": "user", "content": "Question:\n" + str(question) + _join_agent_solutions(responses) +
                    "\n\nPick one option and justify briefly."}

    elif qtype == "math_exp":
        return {"role": "user", "content": "Problem:\n" + str(question) + _join_agent_solutions(responses)}

    else:
        return construct_message(responses, question, "open-ended")

def construct_ranking_message(responses, question, qtype):
    """
    Build a ranking prompt for listwise selection.
    We ask a judge model to pick the top-2 responses that most advance
    reaching a safe, correct diagnosis with minimal tests.
    """
    if qtype in ("open-ended", "open_ended", "openended"):
        prefix = (
            "You are evaluating candidate agent outputs in a clinical team.\n"
            "Case/context:\n" + str(question).strip() + "\n\n"
            "Candidate outputs (order matters only for indexing):"
        )
        body = _join_agent_solutions(responses)
        tail = (
            "\n\nPick the best TWO candidates that most advance a safe, correct diagnosis "
            "with minimal unnecessary testing. Return ONLY a Python-style list of indices, "
            "1-based, such as [2,4]."
        )
        return {"role": "user", "content": prefix + body + tail}

    elif qtype == "single_choice":
        return {"role": "user", "content":
                "Choose the best TWO solutions for the question below and return a list like [1,3].\n\n"
                "Question:\n" + str(question) + _join_agent_solutions(responses)}

    elif qtype == "math_exp":
        return {"role": "user", "content":
                "Choose the best TWO math solutions and return a list like [1,3].\n\n"
                "Problem:\n" + str(question) + _join_agent_solutions(responses)}

    else:
        return construct_ranking_message(responses, question, "open-ended")
