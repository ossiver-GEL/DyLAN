# -*- coding: utf-8 -*-
"""
Summarize AgentClinic × DyLAN runs from plain console logs.

Parses lines like:
  Doctor [33%]: ...
  Patient [66%]: ...
  Measurement [66%]: ...
  Doctor [100%]: DIAGNOSIS READY: ...
  Correct answer: ...
  Scene 0, The diagnosis was  CORRECT 100

Outputs a CSV with per-scene metrics and prints aggregate stats.

Usage:
  python summarize_agentclinic_log.py --log run.log --out summary.csv
  # multiple logs:
  python summarize_agentclinic_log.py --log logs\*.log --out all_runs.csv
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict, Counter
from typing import List, Dict, Any

# --------- Patterns tailored to the actual AgentClinic prints you showed ---------
RE_DOCTOR = re.compile(r"^Doctor\s*\[(\d+)%\]\s*:\s*(.*)$", re.IGNORECASE)
RE_PATIENT = re.compile(r"^Patient\s*\[(\d+)%\]\s*:\s*(.*)$", re.IGNORECASE)
RE_MEASURE = re.compile(r"^Measurement\s*\[(\d+)%\]\s*:\s*(.*)$", re.IGNORECASE)
RE_CORRECT_ANS = re.compile(r"^Correct answer\s*:\s*(.*)$", re.IGNORECASE)
RE_SCENE_END = re.compile(
    r"^Scene\s+(\d+),\s+The diagnosis was\s+(CORRECT|INCORRECT)\s+(\d+)\s*$",
    re.IGNORECASE,
)

RE_DIAG_READY = re.compile(r"DIAGNOSIS\s+READY\s*:\s*(.*)$", re.IGNORECASE)
RE_REQ_TEST = re.compile(r"REQUEST\s+TEST\s*:\s*([^;]+)", re.IGNORECASE)

# Sometimes model prints `Dr. Agent:` or quotes around content
RE_STRIP_DRAGENT = re.compile(r"^\s*(?:Dr\.?\s*Agent)\s*:\s*", re.IGNORECASE)
RE_OUTER_QUOTES = re.compile(r'^\s*["“](.*)["”]\s*$')

def _clean_text(s: str) -> str:
    s = s.strip()
    s = RE_STRIP_DRAGENT.sub("", s)
    m = RE_OUTER_QUOTES.match(s)
    if m:
        s = m.group(1).strip()
    return re.sub(r"\s+", " ", s)

def _word_count(s: str) -> int:
    # crude token proxy
    return len(re.findall(r"\b\w+\b", s))

def _finalize_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    """Compute derived metrics for one scene and return a flat dict."""
    doctor_turns = len(scene["doctor_lines"])
    patient_turns = len(scene["patient_lines"])
    measure_turns = len(scene["measurement_lines"])
    requested_tests = scene["requested_tests"]
    requested_tests = [t.strip() for t in requested_tests if t.strip()]

    # last non-empty DIAGNOSIS READY
    final_dx = ""
    for txt in scene["doctor_lines_rev"]:
        m = RE_DIAG_READY.search(txt)
        if m:
            cand = _clean_text(m.group(1))
            if cand:
                final_dx = cand
                break

    # word counts
    wc_doctor = sum(_word_count(x) for x in scene["doctor_lines"])
    wc_patient = sum(_word_count(x) for x in scene["patient_lines"])
    wc_measure = sum(_word_count(x) for x in scene["measurement_lines"])

    out = {
        "scene_id": scene.get("scene_id"),
        "diagnosis_judged": scene.get("diagnosis_judged"),         # CORRECT / INCORRECT / ""
        "correct_answer": scene.get("correct_answer", ""),
        "predicted_diagnosis": final_dx,
        "has_diagnosis": 1 if final_dx else 0,
        "doctor_turns": doctor_turns,
        "patient_turns": patient_turns,
        "measurement_turns": measure_turns,
        "requested_tests_count": len(requested_tests),
        "requested_tests": "; ".join(requested_tests),
        "doctor_words": wc_doctor,
        "patient_words": wc_patient,
        "measurement_words": wc_measure,
        # Keep raw counts for downstream
        "raw_doctor_lines": " ||| ".join(scene["doctor_lines"]),
        "raw_patient_lines": " ||| ".join(scene["patient_lines"]),
        "raw_measurement_lines": " ||| ".join(scene["measurement_lines"]),
        "source_log": scene.get("source_log", ""),
    }
    return out

def parse_one_log(path: str) -> List[Dict[str, Any]]:
    scenes = []
    cur = None

    def _new_scene():
        return {
            "scene_id": None,
            "correct_answer": "",
            "diagnosis_judged": "",
            "doctor_lines": [],
            "doctor_lines_rev": [],  # for reverse scan
            "patient_lines": [],
            "measurement_lines": [],
            "requested_tests": [],
            "source_log": os.path.basename(path),
        }

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue

            m = RE_DOCTOR.match(line)
            if m:
                if cur is None:
                    cur = _new_scene()
                txt = _clean_text(m.group(2))
                cur["doctor_lines"].append(txt)
                cur["doctor_lines_rev"].insert(0, txt)  # push front for reverse scan
                # collect tests (even if Measurement returns later)
                for tm in RE_REQ_TEST.finditer(txt):
                    cur["requested_tests"].append(tm.group(1).strip())
                continue

            m = RE_PATIENT.match(line)
            if m:
                if cur is None:
                    cur = _new_scene()
                txt = _clean_text(m.group(2))
                cur["patient_lines"].append(txt)
                continue

            m = RE_MEASURE.match(line)
            if m:
                if cur is None:
                    cur = _new_scene()
                txt = _clean_text(m.group(2))
                cur["measurement_lines"].append(txt)
                continue

            m = RE_CORRECT_ANS.match(line)
            if m:
                if cur is None:
                    cur = _new_scene()
                cur["correct_answer"] = _clean_text(m.group(1))
                continue

            m = RE_SCENE_END.match(line)
            if m:
                if cur is None:
                    cur = _new_scene()
                cur["scene_id"] = int(m.group(1))
                cur["diagnosis_judged"] = m.group(2).upper()
                scenes.append(_finalize_scene(cur))
                cur = None
                continue

            # ignore other prints (warnings about torch/TF, etc.)

    # if last scene didn't end with "Scene X, The diagnosis was ..."
    if cur is not None:
        scenes.append(_finalize_scene(cur))

    return scenes

def write_csv(rows: List[Dict[str, Any]], out_path: str):
    if not rows:
        print("No rows parsed; nothing to write.")
        return
    keys = [
        "source_log","scene_id","diagnosis_judged","correct_answer","predicted_diagnosis",
        "has_diagnosis","doctor_turns","patient_turns","measurement_turns",
        "requested_tests_count","requested_tests","doctor_words","patient_words","measurement_words",
        "raw_doctor_lines","raw_patient_lines","raw_measurement_lines"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as wf:
        w = csv.DictWriter(wf, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

def print_aggregate(rows: List[Dict[str, Any]]):
    n = len(rows)
    if n == 0:
        print("No scenes found.")
        return
    acc_known = [r for r in rows if r.get("diagnosis_judged") in ("CORRECT","INCORRECT")]
    acc = sum(1 for r in acc_known if r["diagnosis_judged"] == "CORRECT")
    acc_rate = (acc/len(acc_known)*100) if acc_known else 0.0

    final_rate = sum(r["has_diagnosis"] for r in rows)/n*100.0
    avg_tests = sum(r["requested_tests_count"] for r in rows)/n
    avg_doc_turns = sum(r["doctor_turns"] for r in rows)/n
    avg_pat_turns = sum(r["patient_turns"] for r in rows)/n
    avg_meas_turns = sum(r["measurement_turns"] for r in rows)/n
    avg_doc_words = sum(r["doctor_words"] for r in rows)/n

    print("\n==== Summary ====")
    print(f"Scenes parsed: {n}")
    print(f"Judged accuracy: {acc} / {len(acc_known)} = {acc_rate:.1f}%")
    print(f"Finalized (has 'DIAGNOSIS READY'): {final_rate:.1f}% of scenes")
    print(f"Avg requested tests per scene: {avg_tests:.2f}")
    print(f"Avg turns — Doctor: {avg_doc_turns:.2f}, Patient: {avg_pat_turns:.2f}, Measurement: {avg_meas_turns:.2f}")
    print(f"Avg doctor words per scene: {avg_doc_words:.1f}")

    # Top requested tests (quick glance)
    tests = []
    for r in rows:
        if r.get("requested_tests"):
            tests.extend([t.strip() for t in r["requested_tests"].split(";") if t.strip()])
    if tests:
        cnt = Counter(tests).most_common(10)
        print("Top requested tests:")
        for name, c in cnt:
            print(f"  {name}: {c}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, nargs="+",
                    help="One or more log paths or globs, e.g., run.log or logs\\*.log")
    ap.add_argument("--out", default="summary.csv", help="Output CSV path")
    args = ap.parse_args()

    # expand globs
    paths = []
    for pat in args.log:
        paths.extend(glob.glob(pat))
    paths = [p for p in paths if os.path.isfile(p)]

    if not paths:
        print("No input logs found. Check --log paths.")
        sys.exit(1)

    all_rows = []
    for p in paths:
        rows = parse_one_log(p)
        all_rows.extend(rows)

    write_csv(all_rows, args.out)
    print_aggregate(all_rows)
    print(f"\nWrote {len(all_rows)} rows to: {args.out}")

if __name__ == "__main__":
    main()
