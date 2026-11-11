#!/usr/bin/env python3
"""One-click reproduction driver for the PINN beam-shaping experiments.

Runs, in order: exp1 (star / near field), exp2 (star / far field), exp3 (DT /
near field), exp4 (DT / far field, PINN vs GS), and optionally the official
pretrained paper-number check. Existing checkpoints are reused unless --force.

Examples
--------
    python train_all.py                  # train/reuse all 4 + paper check
    python train_all.py --only exp1 exp3
    python train_all.py --force --seed 1 --gpu 0
    python train_all.py --device cpu --only exp1
    python train_all.py --paper-check
    python train_all.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPRO = ROOT / "repro"
MODELS = ROOT / "reproduced_models"
RESULTS = ROOT / "results"
SAVED = ROOT / "saved_models"

EXPERIMENTS = [
    {
        "key": "exp1",
        "script": "exp1_star_nearfield.py",
        "model": MODELS / "star_flat-top-nearfield_model.pth",
        "title": "Exp1  星形平顶 / 近场",
    },
    {
        "key": "exp2",
        "script": "exp2_star_farfield.py",
        "model": MODELS / "star_flat-top-farfield_model.pth",
        "title": "Exp2  星形平顶 / 远场",
    },
    {
        "key": "exp3",
        "script": "exp3_dt_nearfield.py",
        "model": MODELS / "DT_flat-top-nearfield_model.pth",
        "title": "Exp3  DT 字母 / 近场",
    },
    {
        "key": "exp4",
        "script": "exp4_dt_farfield_gs.py",
        "model": MODELS / "DT_flat-top-farfield_model.pth",
        "title": "Exp4  DT 字母 / 远场 (PINN vs GS)",
    },
    {
        "key": "paper",
        "script": "exp4_paper_check.py",
        "model": SAVED / "DT_flat-top-farfield_model.pth",
        "title": "Official 论文数值校验",
    },
]

RE_REL = re.compile(r"Relative RMS:\s*([\d.]+)\s*%")
RE_PINN = re.compile(r"PINN:\s*MSE\s*=\s*([\d.eE+-]+)\s+RMS\s*=\s*([\d.eE+-]+)\s+relative RMS\s*=\s*([\d.eE+-]+)")
RE_GS = re.compile(r"GS:\s*MSE\s*=\s*([\d.eE+-]+)\s+RMS\s*=\s*([\d.eE+-]+)\s+relative RMS\s*=\s*([\d.eE+-]+)")
RE_ETA_PINN = re.compile(r"PINN efficiency:\s*([\d.]+)\s*%")
RE_ETA_GS = re.compile(r"GS efficiency:\s*([\d.]+)\s*%")


def parse_metrics(text: str) -> dict:
    m: dict[str, float] = {}
    if hit := RE_REL.search(text):
        m["rel_rms"] = float(hit.group(1))
    if hit := RE_PINN.search(text):
        m["mse_pinn"] = float(hit.group(1))
        m["rel_rms_pinn"] = float(hit.group(3))
    if hit := RE_GS.search(text):
        m["mse_gs"] = float(hit.group(1))
        m["rel_rms_gs"] = float(hit.group(3))
    if hit := RE_ETA_PINN.search(text):
        m["eta_pinn"] = float(hit.group(1))
    if hit := RE_ETA_GS.search(text):
        m["eta_gs"] = float(hit.group(1))
    return m


def fmt_metrics(m: dict) -> str:
    parts = []
    if "rel_rms" in m:
        parts.append(f"rel-RMS {m['rel_rms']:.2f}%")
    if "rel_rms_pinn" in m:
        parts.append(f"PINN {m['rel_rms_pinn'] * 100:.2f}%")
    if "rel_rms_gs" in m:
        parts.append(f"GS {m['rel_rms_gs'] * 100:.2f}%")
    if "eta_pinn" in m:
        parts.append(f"η_PINN {m['eta_pinn']:.2f}%")
    if "eta_gs" in m:
        parts.append(f"η_GS {m['eta_gs']:.2f}%")
    return ", ".join(parts)


def build_env(args) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return env


def run_one(exp: dict, args, env: dict, summary: dict) -> bool:
    script = REPRO / exp["script"]
    if not script.exists():
        print(f"[skip] missing script {script}")
        summary[exp["key"]] = {"ok": False, "error": "missing script"}
        return False

    model_ready = exp["model"].exists()
    use_checkpoint = model_ready and not args.force
    if args.force and model_ready:
        print(f"[force] retraining {exp['key']} (checkpoint will be overwritten)")

    cmd = [sys.executable, str(script)]
    if use_checkpoint and exp["key"] != "paper":
        cmd.append("--no-train")
    if exp["key"] != "paper":
        cmd += ["--seed", str(args.seed)]

    print("\n" + "=" * 78)
    print(f"▶ {exp['title']}  ({exp['script']})")
    print("  " + " ".join(cmd))
    print("=" * 78, flush=True)

    RESULTS.mkdir(exist_ok=True)
    log_path = RESULTS / f"train_{exp['key']}.log"
    t0 = time.time()
    buf: list[str] = []
    ok = True
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            buf.append(line)
        rc = proc.wait()
    elapsed = time.time() - t0

    if rc != 0:
        ok = False
        print(f"✗ {exp['key']} failed with exit code {rc} (see {log_path})")
    metrics = parse_metrics("".join(buf))
    summary[exp["key"]] = {
        "ok": ok,
        "exit_code": rc,
        "elapsed_s": round(elapsed, 1),
        "checkpoint_used": use_checkpoint and exp["key"] != "paper",
        "metrics": metrics,
        "log": str(log_path.relative_to(ROOT)),
    }
    print(f"✓ {exp['key']} finished in {elapsed:.1f}s  {fmt_metrics(metrics)}")
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="One-click PINN beam-shaping reproduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--only", nargs="+", metavar="KEY",
                        help="run only these keys: exp1 exp2 exp3 exp4 paper")
    parser.add_argument("--force", action="store_true",
                        help="retrain even if a checkpoint already exists")
    parser.add_argument("--paper-check", action="store_true",
                        help="always run the official pretrained paper check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    keys = {e["key"] for e in EXPERIMENTS}
    if args.list:
        for e in EXPERIMENTS:
            print(f"{e['key']:6s} {e['script']:32s} {e['title']}")
        return 0

    selected = [e for e in EXPERIMENTS if not args.only or e["key"] in args.only]
    if args.only:
        unknown = set(args.only) - keys
        if unknown:
            parser.error(f"unknown keys: {sorted(unknown)}")
    if args.paper_check and not any(e["key"] == "paper" for e in selected):
        selected.append(next(e for e in EXPERIMENTS if e["key"] == "paper"))
    if not selected:
        parser.error("nothing selected")

    if args.dry_run:
        for e in selected:
            print(f"would run: {e['script']}")
        return 0

    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("[warn] CUDA unavailable; falling back to CPU")
                args.device = "cpu"
        except Exception:
            args.device = "cpu"

    env = build_env(args)
    if args.device == "cpu":
        env.setdefault("OMP_NUM_THREADS", "1")

    if args.device == "cuda" and not args.force:
        missing = [e["key"] for e in selected if e["key"] != "paper" and not e["model"].exists()]
        if missing:
            print(f"[info] no checkpoint for {missing}: those will train from scratch")

    print(f"Repository : {ROOT}")
    print(f"Device     : {args.device}" + (f" (gpu {args.gpu})" if args.gpu is not None else ""))
    print(f"Experiments: {[e['key'] for e in selected]}")

    summary: dict = {}
    t0 = time.time()
    for exp in selected:
        run_one(exp, args, env, summary)

    total = time.time() - t0
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "train_summary.json"
    out.write_text(json.dumps({"total_s": round(total, 1), "runs": summary}, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"SUMMARY  (total {total / 60:.1f} min)")
    print("=" * 78)
    print(f"{'key':6s} {'status':8s} {'time':>8s}  metrics")
    for key, row in summary.items():
        status = "ok" if row.get("ok") else "FAIL"
        print(f"{key:6s} {status:8s} {row.get('elapsed_s', 0):>7.1f}s  {fmt_metrics(row.get('metrics', {}))}")
    print(f"\nJSON -> {out.relative_to(ROOT)}")

    failed = [k for k, v in summary.items() if not v.get("ok")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
