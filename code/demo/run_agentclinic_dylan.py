# -*- coding: utf-8 -*-
"""
内嵌式 DyLAN ⇄ AgentClinic 集成（稳定版）
- 不再走子进程/正则读 stdout
- 直接 import AgentClinic 并 monkey-patch query_model：
  当 model_str == "dylan" 时，用 DyLAN 生成医生一句回复
"""

import os
import re
import sys
import time
import argparse
import pathlib
from pathlib import Path

# 让 "code/*" 可被 import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 让 third_party/AgentClinic 可被 import
AGENTCLINIC_DIR = Path(__file__).resolve().parents[2] / "third_party" / "AgentClinic"
sys.path.insert(0, str(AGENTCLINIC_DIR))

# 推荐强制 UTF-8，避免 Windows 默认 GBK 读 JSONL 出错
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from bridge.dylan_bridge import dylan_answer_once  # 我们之前加的 DQUERY 接口桥
import agentclinic as AC  # 直接引入你贴的 AgentClinic 代码

# 备份原来的 query_model，非 doctor 路径继续走原实现
_ORIG_QUERY_MODEL = AC.query_model

_CLEAN_PREFIX = re.compile(r"^\s*(?:DOCTOR|Doctor)\s*:\s*", re.IGNORECASE)
_LEADING_PUNCT = re.compile(r"^\s*[:\-–—]\s*")

def _strip_outer_quotes(t: str) -> str:
    t = t.strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith('“') and t.endswith('”')) or (t.startswith("'") and t.endswith("'")):
        return t[1:-1].strip()
    return t

def _sanitize(text: str) -> str:
    t = (text or "").strip()
    t = _strip_outer_quotes(t)
    t = _CLEAN_PREFIX.sub("", t)         # 去掉开头的“Doctor:”
    t = _LEADING_PUNCT.sub("", t)        # 去掉孤立的冒号/短横
    t = re.sub(r"\s+", " ", t)
    return t

# 从 AgentClinic 的 system_prompt 中读出“总题数/已问数”
_RE_MAX = re.compile(r"You are only allowed to ask\s+(\d+)\s+questions", re.IGNORECASE)
_RE_CUR = re.compile(r"You have asked\s+(\d+)\s+questions\s+so\s+far", re.IGNORECASE)
# 从文本中抓“怀疑/最可能是 …”
_RE_DX = re.compile(r"\b(?:diagnosis|dx|suspect|most\s+likely)\b[:\s\-–—]*([A-Za-z0-9 \-\(\)\/]+)", re.IGNORECASE)

def _maybe_force_final_dx(system_prompt: str, text: str) -> str:
    """若这是末轮且还没有 DIAGNOSIS READY，则尽量抽取疾病名并生成规范句。"""
    m_max = _RE_MAX.search(system_prompt or "")
    m_cur = _RE_CUR.search(system_prompt or "")
    try:
        max_q = int(m_max.group(1)) if m_max else None
        cur_q = int(m_cur.group(1)) if m_cur else None
    except Exception:
        max_q = cur_q = None

    t = text.strip()
    if "DIAGNOSIS READY:" in t.upper():
        return t  # 已经规范了

    is_final_round = (max_q is not None and cur_q is not None and (cur_q + 1) >= max_q)
    if not is_final_round:
        return t

    # 末轮：尝试从文本里抽提诊断关键词
    m_dx = _RE_DX.search(t)
    if m_dx:
        cand = _sanitize(m_dx.group(1))
        if cand:
            return f"DIAGNOSIS READY: {cand}"
    # 抽取失败，保底把整句作为诊断（便于触发评分）
    return f"DIAGNOSIS READY: {t}"

def _dylan_doctor_once(dylan_root: Path, system_prompt: str, user_prompt: str, timeout_s: int = 180) -> str:
    q = (system_prompt or "").strip() + "\n\n" + (user_prompt or "").strip()

    # —— 若当前是末轮，附加硬性输出规范（与 run_DyLAN.py 内部逻辑一致，双保险）——
    m_max = _RE_MAX.search(system_prompt or "")
    m_cur = _RE_CUR.search(system_prompt or "")
    try:
        is_final = (int(m_cur.group(1)) + 1) >= int(m_max.group(1)) if (m_max and m_cur) else False
    except Exception:
        is_final = False
    if is_final:
        q += "\n\nFINAL TURN: Reply strictly with exactly one line: DIAGNOSIS READY: <disease>."

    ans = dylan_answer_once(dylan_root, q, timeout_s=timeout_s)
    return _sanitize(ans) if ans else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agentclinic_root", type=str, default=str(AGENTCLINIC_DIR))
    ap.add_argument("--agent_dataset", type=str, default="MedQA", choices=["MedQA","MedQA_Ext","NEJM","NEJM_Ext","MIMICIV"])
    ap.add_argument("--total_inferences", type=int, default=8)
    ap.add_argument("--doctor_llm", type=str, default="dylan", help="医生后端；此脚本会强制用 'dylan'")
    ap.add_argument("--patient_llm", type=str, default="gpt-4o-mini")
    ap.add_argument("--measurement_llm", type=str, default="gpt-4o-mini")
    ap.add_argument("--moderator_llm", type=str, default="gpt-4o-mini")
    ap.add_argument("--doctor_image_request", action="store_true", default=False)
    ap.add_argument("--num_scenarios", type=int, default=1)
    ap.add_argument("--openai_api_key", type=str, default=os.environ.get("OPENAI_API_KEY",""))
    ap.add_argument("--openai_api_base", type=str, default=os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL") or "https://api.bianxie.ai/v1")
    ap.add_argument("--openai_default_model", type=str, default=os.environ.get("AGENTCLINIC_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "")
    ap.add_argument("--replicate_api_key", type=str, default=os.environ.get("REPLICATE_API_TOKEN",""))
    ap.add_argument("--anthropic_api_key", type=str, default=os.environ.get("ANTHROPIC_API_KEY",""))
    args, extra = ap.parse_known_args()

    if not args.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is empty. Set env or pass --openai_api_key.")

    # 保证 AgentClinic 读写在其目录下进行（它相对路径读取 *.jsonl）
    os.chdir(args.agentclinic_root)

    # —— 打补丁：当 model_str == 'dylan' 时走 DyLAN，其它走原有 query_model ——
    dylan_root = Path(__file__).resolve().parents[2]
    def _patched_query_model(model_str, prompt, system_prompt, **kwargs):
        if model_str == "dylan":
            raw = _dylan_doctor_once(dylan_root, system_prompt, prompt)
            clean = _sanitize(raw)
            # 不再强行包装为 “DIAGNOSIS READY: …”，把决定权交给 DyLAN（我们已通过末轮硬约束做了自然引导）
            return clean if clean else ""   # 空就让 AgentClinic 正常走下一步（通常会继续一轮或输出测量/患者）
        return _ORIG_QUERY_MODEL(model_str, prompt, system_prompt, **kwargs)

    AC.query_model = _patched_query_model  # monkey-patch

    # —— 调 AgentClinic 主流程（用 LLM 模式，但医生后端是 'dylan'）——
    AC.main(
        api_key=args.openai_api_key,
        replicate_api_key=args.replicate_api_key,
        inf_type="llm",                     # 关键：触发 DoctorAgent.inference_doctor → query_model(...)
        doctor_bias="None",
        patient_bias="None",
        doctor_llm="dylan",                 # 关键：让我们打补丁的分支生效
        patient_llm=args.patient_llm,
        measurement_llm=args.measurement_llm,
        moderator_llm=args.moderator_llm,
        num_scenarios=args.num_scenarios,
        dataset=args.agent_dataset,
        img_request=args.doctor_image_request,
        total_inferences=args.total_inferences,
        anthropic_api_key=args.anthropic_api_key,
        openai_api_base=args.openai_api_base,           # 避免默认走第三方网关
        openai_default_model=args.openai_default_model, # 可留空
    )

if __name__ == "__main__":
    main()
