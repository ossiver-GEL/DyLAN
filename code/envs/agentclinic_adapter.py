import os
import re
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

class AgentClinicProcess:
    """
    以 human_doctor 模式把 AgentClinic 跑成交互式子进程。
    关键点：
    - 子进程用 python -u -X utf8，且设置 PYTHONUNBUFFERED/PYTHONUTF8，避免缓冲/编码问题
    - stdout 按“字符流”读取（不是按行），在尾部匹配提示词（即使没有换行也能触发）
    - 兼容多种提示/进度条样式（Doctor[33%]、Question for patient: 等）
    - 达到回合上限后，主动发送最后一次回复并结束子进程（避免它无限循环）
    """
    # —— 兼容更多样式的“医生输入提示”（末尾可无换行）——
    PROMPT_TAIL_PAT = re.compile(
        r"(Your\s+response|Enter\s+your\s+response|"
        r"Doctor(?:\s*\[\d+%])?\s*:?\s*|"
        r"Question\s+for\s+patient\s*:?)\s*$",
        re.IGNORECASE,
    )
    # —— 患者发言（带或不带进度百分比）——
    PATIENT_LINE_PAT = re.compile(
        r"^\s*(?:PATIENT|Patient|P)(?:\s*\[\d+%])?\s*[:：]\s*(.+)$",
        re.IGNORECASE,
    )
    # —— 结束信号（可选，有则优先用）——
    END_PATTERNS = [
        re.compile(r"(?i)final\s+diagnosis"),
        re.compile(r"(?i)case\s+over"),
        re.compile(r"(?i)thank\s+you"),
    ]

    def __init__(
        self,
        agentclinic_repo: Path,
        openai_api_key: str,
        agent_dataset: str = "MedQA",
        doctor_llm: str = "gpt-4o",
        patient_llm: str = "gpt-4o",
        total_inferences: int = 20,
        doctor_image_request: bool = False,
        python_bin: str = sys.executable,
        extra_args: Optional[List[str]] = None,
        measurement_llm: Optional[str] = None,
        moderator_llm: Optional[str] = None,
        prompt_timeout_s: float = 180.0,   # 等待提示的最长时间
    ):
        self.repo = Path(agentclinic_repo)
        if not (self.repo / "agentclinic.py").exists():
            raise FileNotFoundError(f"agentclinic.py not found under: {self.repo}")

        self.env = os.environ.copy()
        if openai_api_key:
            self.env["OPENAI_API_KEY"] = openai_api_key
        # 强制 UTF-8 + 关闭缓冲
        self.env["PYTHONUTF8"] = "1"
        self.env["PYTHONUNBUFFERED"] = "1"

        self.args = [
            python_bin, "-u", "-X", "utf8", "agentclinic.py",
            "--openai_api_key", self.env.get("OPENAI_API_KEY", ""),
            "--inf_type", "human_doctor",
            "--agent_dataset", agent_dataset,
            "--doctor_llm", doctor_llm,
            "--patient_llm", patient_llm,
            "--total_inferences", str(total_inferences),
        ]
        if doctor_image_request:
            self.args += ["--doctor_image_request", "True"]
        if measurement_llm:
            self.args += ["--measurement_llm", measurement_llm]
        if moderator_llm:
            self.args += ["--moderator_llm", moderator_llm]
        if extra_args:
            self.args += list(extra_args)

        self._p: Optional[subprocess.Popen] = None
        self.max_rounds = int(total_inferences)
        self.prompt_timeout_s = float(prompt_timeout_s)

    def start(self):
        self._p = subprocess.Popen(
            self.args,
            cwd=str(self.repo),
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0,   # 0: unbuffered（配合 -u 更稳）
        )

    def terminate(self):
        try:
            if self._p and (self._p.poll() is None):
                self._p.terminate()
        except Exception:
            pass

    def _tail_has_prompt(self, tail: str) -> bool:
        # 只看最后 200 字符，strip 掉右侧空白
        s = tail[-200:].rstrip()
        return bool(self.PROMPT_TAIL_PAT.search(s))

    def _is_end_line(self, line: str) -> bool:
        return any(p.search(line) for p in self.END_PATTERNS)

    def run_episode(self, on_doctor_utterance):
        """
        on_doctor_utterance(history: List[(speaker, text)]) -> str
        返回：[(PATIENT/DOCTOR, text), ...]
        """
        self.start()
        history: List[Tuple[str, str]] = []
        tail = ""         # 最近一段输出的“尾巴”（用于无换行提示匹配）
        cur_line = ""     # 当前正在累积的一行
        rounds = 0
        t_last_output = time.time()

        try:
            assert self._p and self._p.stdout and self._p.stdin
            f_out, f_in = self._p.stdout, self._p.stdin

            # —— 主循环：按字符读取 —— 
            while True:
                ch = f_out.read(1)
                if ch == "" or ch is None:
                    # 子进程已结束
                    break

                t_last_output = time.time()
                tail = (tail + ch)[-512:]  # 保留足够上下文
                # 把输出回显出来，便于观察
                if ch == "\n":
                    print(f"[AgentClinic] {cur_line}")
                    # 处理完整一行（患者发言/结束信号）
                    m = self.PATIENT_LINE_PAT.match(cur_line)
                    if m:
                        history.append(("PATIENT", m.group(1).strip()))
                    if self._is_end_line(cur_line):
                        # 到尾声了：补一次医生总结然后退出
                        reply = on_doctor_utterance(history)
                        f_in.write((reply.rstrip() + "\n")); f_in.flush()
                        history.append(("DOCTOR", reply))
                        break
                    cur_line = ""
                else:
                    cur_line += ch

                # —— 无换行的“提示尾部”检测：出现就发医生回复 —— 
                if self._tail_has_prompt(tail):
                    reply = on_doctor_utterance(history)
                    f_in.write((reply.rstrip() + "\n")); f_in.flush()
                    history.append(("DOCTOR", reply))
                    rounds += 1
                    # 达到回合上限：直接结束（避免 AgentClinic 自己继续循环）
                    if rounds >= self.max_rounds:
                        # 给它一个最终回复的机会（可选）
                        time.sleep(0.3)
                        break
                    # 发完一条，清空 tail，避免重复触发
                    tail = ""

                # —— 超时保护：长时间没任何输出，直接退出 —— 
                if time.time() - t_last_output > self.prompt_timeout_s:
                    print("[AgentClinic] timeout waiting for output; aborting.")
                    break

            # 让子进程优雅退出
            try:
                self._p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass

            # 如果最后一行未打印（没有换行），也输出一下
            if cur_line.strip():
                print(f"[AgentClinic] {cur_line}")

            return history
        finally:
            self.terminate()
