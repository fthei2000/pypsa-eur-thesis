#!/usr/bin/env python3
"""
PyPSA-Eur LSF job monitor agent.

Polls job status every N minutes and:
  - Emails when job transitions PEND → RUN (node acquired)
  - Emails when job completes successfully (DONE)
  - On failure (EXIT): reads logs, asks Claude to diagnose and patch,
    applies the fix, relaunches, and emails a summary.
  - Repeats monitoring for the relaunched job (up to MAX_RETRIES).

Usage:
    python3 monitor_job.py <job_id>

Requirements:
    - pip install --user anthropic
    - ANTHROPIC_API_KEY in environment OR ~/.anthropic_api_key file
    - 'mail' command available on HPC login node
"""

import re
import os
import sys
import time
import subprocess
import textwrap
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

# ── Configuration ─────────────────────────────────────────────────────────────
EMAIL        = "s240459@dtu.dk"
REPO_DIR     = Path("/work3/s240459/pypsa-eur-thesis")
RUN_SCRIPT   = REPO_DIR / "run_L6.sh"
CFG          = REPO_DIR / "config/Myruns/config.L6.yaml"
LOG_DIR      = REPO_DIR / "logs"
POLL_SECS    = 300   # 5 minutes
MAX_RETRIES  = 2
LOG_TAIL_N   = 150   # lines to send to Claude / email

# Claude model for log analysis
CLAUDE_MODEL = "claude-opus-4-6"

# Structured patch instructions Claude must follow
PATCH_SYSTEM = """
You are a PyPSA-Eur / Snakemake HPC expert.
Given a failed LSF job log you must:
1. Identify the root cause.
2. Propose the MINIMAL edit to either run_L6.sh or config.L6.yaml that will fix it.
3. Return ONLY a JSON object — no prose outside the JSON — with keys:
   {
     "error_type": "<MEMORY|LOCK|SOLVER|MISSING_FILE|TIMEOUT|OTHER>",
     "relaunch": true/false,
     "explanation": "<one sentence>",
     "patches": [
       {
         "file": "<relative path from repo root>",
         "old":  "<exact text to replace, or empty string to append>",
         "new":  "<replacement text>"
       }
     ]
   }
If no safe automatic fix exists set relaunch=false and patches=[].
""".strip()
# ──────────────────────────────────────────────────────────────────────────────


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{now()}] {msg}", flush=True)


# ── Email ──────────────────────────────────────────────────────────────────────
def send_email(subject: str, body: str) -> bool:
    try:
        result = subprocess.run(
            ["mail", "-s", subject, EMAIL],
            input=body, text=True, capture_output=True, timeout=30
        )
        ok = result.returncode == 0
        log(f"Email {'sent' if ok else 'FAILED'}: {subject}")
        return ok
    except Exception as e:
        log(f"Email error: {e}")
        return False


# ── LSF helpers ───────────────────────────────────────────────────────────────
def get_job_status(job_id: int) -> Optional[str]:
    """Return LSF STAT string or None if job not found."""
    r = subprocess.run(
        ["bjobs", "-noheader", str(job_id)],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        return None
    line = r.stdout.strip()
    if not line or "not found" in r.stderr.lower():
        return None
    parts = line.split()
    return parts[2] if len(parts) >= 3 else None


def submit_job() -> Optional[int]:
    """bsub < run_L6.sh and return new job ID."""
    r = subprocess.run(
        f"bsub < {RUN_SCRIPT}",
        shell=True, capture_output=True, text=True, cwd=REPO_DIR
    )
    log(f"bsub stdout: {r.stdout.strip()}")
    log(f"bsub stderr: {r.stderr.strip()}")
    m = re.search(r"Job <(\d+)>", r.stdout)
    return int(m.group(1)) if m else None


# ── Log reading ───────────────────────────────────────────────────────────────
def read_log_tail(job_id: int, n: int = LOG_TAIL_N) -> str:
    out, err = [], []
    for stem, store in [(".out", out), (".err", err)]:
        p = LOG_DIR / f"{job_id}{stem}"
        if p.exists():
            store.extend(p.read_text(errors="replace").splitlines()[-n:])
    parts = []
    if out:
        parts.append("=== stdout (last lines) ===\n" + "\n".join(out))
    if err:
        parts.append("=== stderr (last lines) ===\n" + "\n".join(err))
    return "\n\n".join(parts) if parts else "(no log files found yet)"


def job_succeeded_in_log(log_text: str) -> bool:
    return "completed successfully" in log_text.lower()


# ── Claude analysis ───────────────────────────────────────────────────────────
def _get_api_key() -> Optional[str]:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    p = Path.home() / ".anthropic_api_key"
    if p.exists():
        return p.read_text().strip()
    return None


def analyze_with_claude(log_text: str) -> Optional[dict]:
    import json
    key = _get_api_key()
    if not key:
        log("No ANTHROPIC_API_KEY found — skipping Claude analysis.")
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        prompt = (
            f"Here are the last {LOG_TAIL_N} lines of a failed PyPSA-Eur LSF job log.\n\n"
            f"Run script: {RUN_SCRIPT}\n"
            f"Config:     {CFG}\n\n"
            f"{log_text}\n\n"
            "Diagnose the failure and return the JSON patch as instructed."
        )
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1500,
            system=PATCH_SYSTEM,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        log(f"Claude response:\n{raw[:500]}")
        # Extract JSON even if wrapped in ```
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        log(f"Claude analysis error: {e}")
    return None


# ── Patch application ─────────────────────────────────────────────────────────
def apply_patches(patches: List[Dict]) -> List[str]:
    """Apply list of {file, old, new} patches. Return list of applied descriptions."""
    applied = []
    for p in patches:
        path = REPO_DIR / p["file"]
        if not path.exists():
            log(f"Patch target not found: {path}")
            continue
        content = path.read_text()
        old, new = p.get("old", ""), p.get("new", "")
        if old and old not in content:
            log(f"Patch old text not found in {p['file']} — skipping.")
            continue
        if old:
            content = content.replace(old, new, 1)
        else:
            content += "\n" + new
        path.write_text(content)
        applied.append(f"  {p['file']}: replaced {repr(old[:60])} → {repr(new[:60])}")
        log(f"Patched {p['file']}")
    return applied


# ── Fallback pattern-based patches ───────────────────────────────────────────
def fallback_patch(log_text: str) -> dict:
    """Simple pattern matching when Claude is unavailable."""
    txt = log_text.lower()
    if "out of memory" in txt or "oom" in txt or "memory" in txt and "exceeded" in txt:
        script = RUN_SCRIPT.read_text()
        for old_mem in ["mem=24000", "mem=28000", "mem=32000"]:
            if old_mem in script:
                new_mem = {"mem=24000": "mem=32000",
                           "mem=28000": "mem=36000",
                           "mem=32000": "mem=40000"}[old_mem]
                return {
                    "error_type": "MEMORY",
                    "relaunch": True,
                    "explanation": f"OOM detected — bumping memory {old_mem} → {new_mem}.",
                    "patches": [{"file": "run_L6.sh", "old": old_mem, "new": new_mem}]
                }
    if "directory cannot be locked" in txt:
        return {
            "error_type": "LOCK",
            "relaunch": True,
            "explanation": "Stale Snakemake lock — running --unlock before retry.",
            "patches": []  # handled separately below
        }
    return {
        "error_type": "OTHER",
        "relaunch": False,
        "explanation": "Could not automatically diagnose failure.",
        "patches": []
    }


# ── Main monitor loop ─────────────────────────────────────────────────────────
def monitor(job_id: int, retry_count: int = 0):
    log(f"Monitoring job {job_id} (attempt {retry_count + 1}/{MAX_RETRIES + 1})")
    prev_status = None

    while True:
        status = get_job_status(job_id)

        # Transition: PEND → RUN (node acquired)
        if prev_status in (None, "PEND") and status == "RUN":
            send_email(
                f"[PyPSA] Job {job_id} STARTED — node acquired",
                f"Job {job_id} is now RUNNING on a compute node.\n\n"
                f"Started at: {now()}\n"
                f"Config: {CFG}\n"
                f"Monitor this job with: bjobs {job_id}"
            )

        if status in ("DONE", "EXIT") or status is None:
            log_text = read_log_tail(job_id)

            # ── SUCCESS ──────────────────────────────────────────────────────
            if status == "DONE" or (status is None and job_succeeded_in_log(log_text)):
                send_email(
                    f"[PyPSA] Job {job_id} COMPLETED successfully ✓",
                    f"Your L6 myopic 2030-2040-2050 run finished!\n\n"
                    f"Completed at: {now()}\n"
                    f"Results: {REPO_DIR}/results/\n\n"
                    f"--- Log tail ---\n{log_text[-3000:]}"
                )
                log("Job completed. Monitor exiting.")
                return

            # ── FAILURE ──────────────────────────────────────────────────────
            log(f"Job {job_id} FAILED (status={status})")

            if retry_count >= MAX_RETRIES:
                send_email(
                    f"[PyPSA] Job {job_id} FAILED — max retries reached",
                    f"Job {job_id} failed and max retries ({MAX_RETRIES}) exhausted.\n\n"
                    f"Failed at: {now()}\n"
                    f"Manual inspection needed.\n\n"
                    f"--- Log tail ---\n{log_text[-3000:]}"
                )
                log("Max retries reached. Monitor exiting.")
                return

            # Try Claude first, fall back to pattern matching
            analysis = analyze_with_claude(log_text) or fallback_patch(log_text)
            error_type  = analysis.get("error_type", "OTHER")
            explanation = analysis.get("explanation", "")
            patches     = analysis.get("patches", [])
            relaunch    = analysis.get("relaunch", False)

            applied_patches = []

            # Special case: snakemake lock
            if error_type == "LOCK":
                log("Running snakemake --unlock...")
                subprocess.run(
                    ["pixi", "run", "snakemake", "--unlock", "--configfile", str(CFG)],
                    cwd=REPO_DIR
                )
                relaunch = True

            if patches:
                applied_patches = apply_patches(patches)

            if relaunch:
                new_job_id = submit_job()
                if new_job_id:
                    patch_summary = "\n".join(applied_patches) if applied_patches else "  (no file edits)"
                    send_email(
                        f"[PyPSA] Job {job_id} FAILED → patched & relaunched as {new_job_id}",
                        f"Job {job_id} failed.\n\n"
                        f"Error type:  {error_type}\n"
                        f"Explanation: {explanation}\n\n"
                        f"Patches applied:\n{patch_summary}\n\n"
                        f"Relaunched as job {new_job_id} at {now()}.\n\n"
                        f"--- Failed log tail ---\n{log_text[-2000:]}"
                    )
                    return monitor(new_job_id, retry_count + 1)
                else:
                    send_email(
                        f"[PyPSA] Job {job_id} FAILED — relaunch error",
                        f"Job {job_id} failed and the relaunch via bsub also failed.\n\n"
                        f"Error type:  {error_type}\n"
                        f"Explanation: {explanation}\n\n"
                        f"--- Log tail ---\n{log_text[-2000:]}"
                    )
            else:
                send_email(
                    f"[PyPSA] Job {job_id} FAILED — no auto-fix available",
                    f"Job {job_id} failed. No safe automatic fix was identified.\n\n"
                    f"Error type:  {error_type}\n"
                    f"Explanation: {explanation}\n\n"
                    f"Manual inspection needed.\n\n"
                    f"--- Log tail ---\n{log_text[-3000:]}"
                )
            return

        prev_status = status
        time.sleep(POLL_SECS)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <job_id>")
        sys.exit(1)

    job_id = int(sys.argv[1])
    log(f"Starting monitor for job {job_id}, emailing {EMAIL}")
    log(f"Poll interval: {POLL_SECS}s  |  Max retries: {MAX_RETRIES}")

    send_email(
        f"[PyPSA] Monitor started for job {job_id}",
        f"Job monitor is now watching job {job_id}.\n\n"
        f"Config: {CFG}\n"
        f"You will receive emails on: job start, completion, or failure + relaunch.\n\n"
        f"Monitor PID: {os.getpid()}"
    )

    try:
        monitor(job_id)
    except KeyboardInterrupt:
        log("Monitor interrupted by user.")
