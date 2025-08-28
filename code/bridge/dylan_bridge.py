# code/bridge/dylan_bridge.py
import os, re, sys, subprocess
from pathlib import Path
from typing import List, Tuple

FINAL_PAT = re.compile(r"(?:FINAL_ANSWER|Final Answer|Answer:)\s*(.*)", re.IGNORECASE)

def _render_query_from_history(history: List[Tuple[str,str]]) -> str:
    lines = []
    lines.append("You are a concise clinician. Reply with ONE short turn.")
    lines.append("If unsure, ask ONE concrete, high-value question OR order ONE test;")
    lines.append("If sufficient evidence, give a FINAL diagnosis with 1-2 key evidences.")
    lines.append("")
    for spk, txt in history[-12:]:
        lines.append(f"{spk}: {txt}")
    lines.append("")
    lines.append("DOCTOR:")
    return "\n".join(lines)

def dylan_answer_once(dylan_repo_root: Path, query: str, timeout_s: int = 300) -> str:
    """
    Call run_DyLAN.py in 'service mode' (env DQUERY is set) and parse FINAL_ANSWER.
    """
    demo_py = dylan_repo_root / "code" / "demo" / "run_DyLAN.py"
    if not demo_py.exists():
        return ""
    env = os.environ.copy()
    env["DQUERY"] = query
    # Optional: allow overriding the model via env DYLAN_MODEL
    # env["DYLAN_MODEL"] = "gpt-4o"
    proc = subprocess.run(
        [sys.executable, str(demo_py)],
        cwd=str(demo_py.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s
    )
    raw = (proc.stdout or "").strip()
    m = FINAL_PAT.search(raw)
    if m:
        return m.group(1).strip()
    # fallback: last non-empty line
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return lines[-1] if lines else ""

def make_doctor_callback(dylan_repo_root: Path):
    def _fn(history: List[Tuple[str,str]]) -> str:
        q = _render_query_from_history(history)
        ans = dylan_answer_once(dylan_repo_root, q)
        # 保底：不要返回空串，避免 TRANSCRIPT 里出现空 DOCTOR
        return ans if ans and ans.strip() else "Could you tell me when the symptoms started and any triggers? I may order basic labs."
    return _fn
