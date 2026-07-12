#!/usr/bin/env python
"""Thermal-aware benchmark harness for Open-Sora inference on Apple Silicon.

Wraps ``scripts/diffusion/inference.py`` as a subprocess, timestamps the
script's own log markers to split model-load from render time, repeats with
enforced cooldowns (thermal throttling on this machine drifts renders +13-28%
cold->warm), sanity-checks the output pixels (the fp16 bug rendered *black*
frames, silently), and writes a JSON result that ``compare`` can gate against.

Run me WITH the same interpreter that runs inference (it needs ``av`` for the
output check), e.g.:

    $OPENSORA_MPS_PY .claude/skills/mps-bench/scripts/osora_bench.py run \
        --label "13f/20step baseline" --out ../baselines/t2v_13f20s.mps.json

Subcommands: run | compare | show | inspect. See SKILL.md for methodology.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[4]
INFER_SCRIPT = "scripts/diffusion/inference.py"
DEFAULT_CONFIG = "configs/diffusion/inference/256px.py"

# Markers logged by scripts/diffusion/inference.py, in order of appearance.
MARK_BUILD = "Building models..."
MARK_GEN = "Generating video..."
MARK_DONE = "Inference finished."

# The standard probe workload: matches the documented baselines
# (13 frames / 20 steps / seed 42) in docs/apple_silicon.md.
DEFAULT_WORKLOAD = [
    "--prompt", "raining, sea",
    "--num_frames", "13",
    "--num_steps", "20",
    "--seed", "42",
    "--sampling_option.seed", "42",
]


def _git_info() -> dict:
    def _run(*args):
        try:
            return subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except Exception:
            return None

    sha = _run("rev-parse", "--short", "HEAD")
    dirty = bool(_run("status", "--porcelain"))
    return {"git_sha": sha, "git_dirty": dirty}


def _env_info(python: str) -> dict:
    code = (
        "import json,torch,platform;"
        "tv=None\n"
        "try:\n import torchvision as _t; tv=_t.__version__\n"
        "except Exception: pass\n"
        "print(json.dumps({'torch':torch.__version__,'torchvision':tv,"
        "'python':platform.python_version(),"
        "'mps_available':torch.backends.mps.is_available()}))"
    )
    try:
        out = subprocess.run([python, "-c", code], capture_output=True, text=True, check=True)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": f"could not query {python}: {e}"}


def _machine() -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return platform.platform()


def _newest_output(save_dir: Path) -> Path | None:
    exts = {".mp4", ".png"}
    cands = [p for p in save_dir.rglob("*") if p.suffix in exts and p.is_file()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def inspect_output(path: Path) -> dict:
    """Decode with pyav; report luma stats and flag degenerate output.

    Catches the classes of silent failure this port has actually hit:
    black frames (fp16 overflow), flat/constant frames, and unreadable files.
    """
    info: dict = {"file": str(path)}
    try:
        import av  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError as e:
        info["error"] = f"pyav/numpy unavailable in this interpreter: {e}"
        return info
    try:
        means, stds = [], []
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                arr = frame.to_ndarray(format="gray")
                means.append(float(arr.mean()))
                stds.append(float(arr.std()))
        if not means:
            info["error"] = "no decodable frames"
            return info
        info.update(
            frames=len(means),
            luma_mean=round(float(np.mean(means)), 2),
            luma_std=round(float(np.mean(stds)), 2),
        )
        flags = []
        if info["luma_mean"] < 8:
            flags.append("BLACK_FRAMES")  # the fp16 signature
        if info["luma_mean"] > 247:
            flags.append("WHITE_FRAMES")
        if info["luma_std"] < 2:
            flags.append("FLAT_FRAMES")
        info["flags"] = flags
    except Exception as e:  # noqa: BLE001
        info["error"] = str(e)
    return info


def _single_run(cmd: list[str], env: dict, log_path: Path) -> dict:
    """Run one inference subprocess; timestamp the log markers as they arrive."""
    t0 = time.monotonic()
    marks: dict[str, float] = {}
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            now = time.monotonic()
            for mark in (MARK_BUILD, MARK_GEN, MARK_DONE):
                # first occurrence only (GEN repeats per batch; time batch 1)
                if mark in line and mark not in marks:
                    marks[mark] = now
        proc.wait()
    total = time.monotonic() - t0
    res = {
        "returncode": proc.returncode,
        "total_s": round(total, 1),
        "log": str(log_path),
    }
    if MARK_GEN in marks and MARK_DONE in marks:
        res["render_s"] = round(marks[MARK_DONE] - marks[MARK_GEN], 1)
    if MARK_BUILD in marks and MARK_GEN in marks:
        res["load_s"] = round(marks[MARK_GEN] - marks[MARK_BUILD], 1)
    return res


def cmd_run(args: argparse.Namespace) -> int:
    python = args.python or os.environ.get("OPENSORA_MPS_PY") or sys.executable
    workload = args.workload or list(DEFAULT_WORKLOAD)
    save_dir = REPO_ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OPENSORA_DEVICE"] = args.device
    env["HF_HUB_OFFLINE"] = "1"
    # TORCHDYNAMO_DISABLE retired (P5): the timestep_embedding compile is CUDA-only now,
    # so forcing it here is inert and would silently no-op a --compile_mmdit bench.
    if args.no_fallback_net:
        env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)  # smoke test: raise on kernel gaps
    else:
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # NOTE: plain python, never torchrun (see docs/apple_silicon.md). -u so log
    # markers reach the pipe unbuffered and timestamps are honest.
    cmd = [python, "-u", INFER_SCRIPT, args.config, *workload, "--save-dir", str(save_dir)]

    result = {
        "label": args.label,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "machine": _machine(),
        "device": args.device,
        "fallback_net": not args.no_fallback_net,
        "interpreter": python,
        "config": args.config,
        "workload": workload,
        **_git_info(),
        "env": _env_info(python),
        "cooldown_s": args.cooldown,
        "repeats": [],
    }

    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(args.repeats):
        if i > 0 and args.cooldown > 0:
            print(f"[bench] cooldown {args.cooldown}s before repeat {i + 1} "
                  "(thermal discipline; see SKILL.md)...", flush=True)
            time.sleep(args.cooldown)
        print(f"[bench] repeat {i + 1}/{args.repeats}: {' '.join(cmd)}", flush=True)
        run = _single_run(cmd, env, logs_dir / f"{stamp}_r{i}.log")
        if run["returncode"] != 0:
            print(f"[bench] FAILED (exit {run['returncode']}); log tail:", flush=True)
            print("\n".join(Path(run["log"]).read_text().splitlines()[-25:]))
            result["repeats"].append(run)
            result["failed"] = True
            _write_out(result, args.out)
            return run["returncode"] or 1
        out_file = _newest_output(save_dir)
        if out_file:
            run["output"] = inspect_output(out_file)
        result["repeats"].append(run)
        r = run.get("render_s")
        print(f"[bench] repeat {i + 1}: load {run.get('load_s')}s, render {r}s, "
              f"output flags: {run.get('output', {}).get('flags', 'n/a')}", flush=True)

    renders = [r["render_s"] for r in result["repeats"] if "render_s" in r]
    if renders:
        result["render_cold_s"] = renders[0]
        if len(renders) > 1:
            result["render_warm_median_s"] = round(median(renders[1:]), 1)
            result["cold_to_last_drift_pct"] = round(
                100 * (renders[-1] - renders[0]) / renders[0], 1
            )
    result["output_flags"] = sorted(
        {f for r in result["repeats"] for f in r.get("output", {}).get("flags", [])}
    )

    _write_out(result, args.out)
    _print_summary(result)

    if result["output_flags"]:
        print(f"[bench] OUTPUT SANITY FAIL: {result['output_flags']}")
        return 2
    if args.baseline:
        return _compare(json.loads(Path(args.baseline).read_text()), result, args.threshold)
    return 0


def _write_out(result: dict, out: str | None) -> None:
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[bench] wrote {out_path}")


def _print_summary(res: dict) -> None:
    env = res.get("env", {})
    fmt = lambda v, suffix: f"{v}{suffix}" if v is not None else "n/a"
    print(
        f"[bench] {res.get('label') or '(unlabeled)'} | torch {env.get('torch')} | "
        f"{res.get('git_sha')}{'+dirty' if res.get('git_dirty') else ''} | "
        f"cold {fmt(res.get('render_cold_s'), 's')} | "
        f"warm-median {fmt(res.get('render_warm_median_s'), 's')} | "
        f"drift {fmt(res.get('cold_to_last_drift_pct'), '%')}"
    )


def _compare(base: dict, cur: dict, threshold: float) -> int:
    rc = 0
    for k in ("machine", "device"):
        if base.get(k) != cur.get(k):
            print(f"[compare] WARNING: {k} differs ({base.get(k)} vs {cur.get(k)}) — "
                  "cross-machine/device deltas are not regressions.")
    if base.get("env", {}).get("torch") != cur.get("env", {}).get("torch"):
        print("[compare] NOTE: torch versions differ — this is an env A/B, not a "
              "code-regression check.")
    if base.get("workload") != cur.get("workload"):
        print("[compare] WARNING: workloads differ — comparison is meaningless.")
        return 1
    for key in ("render_cold_s", "render_warm_median_s"):
        b, c = base.get(key), cur.get(key)
        if b is None or c is None:
            continue
        delta = 100 * (c - b) / b
        verdict = "ok"
        if delta > 100 * threshold:
            verdict, rc = "REGRESSION", 1
        elif delta < -100 * threshold:
            verdict = "improvement"
        print(f"[compare] {key}: {b}s -> {c}s ({delta:+.1f}%) {verdict}")
    flags = set(cur.get("output_flags", [])) - set(base.get("output_flags", []))
    if flags:
        print(f"[compare] NEW OUTPUT FLAGS: {sorted(flags)} — correctness failure, not noise.")
        rc = 2
    return rc


def cmd_compare(args: argparse.Namespace) -> int:
    return _compare(
        json.loads(Path(args.baseline).read_text()),
        json.loads(Path(args.current).read_text()),
        args.threshold,
    )


def cmd_show(args: argparse.Namespace) -> int:
    res = json.loads(Path(args.result).read_text())
    _print_summary(res)
    print(json.dumps(res, indent=2))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    info = inspect_output(Path(args.file))
    print(json.dumps(info, indent=2))
    return 2 if info.get("flags") else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a timed inference benchmark")
    r.add_argument("--label", default="")
    r.add_argument("--python", default=None, help="interpreter for inference (default: $OPENSORA_MPS_PY or self)")
    r.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    r.add_argument("--config", default=DEFAULT_CONFIG)
    r.add_argument("--save-dir", default="samples/bench")
    r.add_argument("--repeats", type=int, default=2, help="repeat 1 = cold, later = warm (default 2)")
    r.add_argument("--cooldown", type=int, default=180, help="seconds between repeats (default 180)")
    r.add_argument("--no-fallback-net", action="store_true",
                   help="unset PYTORCH_ENABLE_MPS_FALLBACK: kernel gaps raise (smoke test)")
    r.add_argument("--out", default=None, help="write result JSON here")
    r.add_argument("--baseline", default=None, help="compare against this result JSON")
    r.add_argument("--threshold", type=float, default=0.10, help="regression threshold (default 10%%)")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("compare", help="compare two result JSONs")
    c.add_argument("baseline")
    c.add_argument("current")
    c.add_argument("--threshold", type=float, default=0.10)
    c.set_defaults(fn=cmd_compare)

    s = sub.add_parser("show", help="pretty-print a result JSON")
    s.add_argument("result")
    s.set_defaults(fn=cmd_show)

    i = sub.add_parser("inspect", help="sanity-check an output video/image file")
    i.add_argument("file")
    i.set_defaults(fn=cmd_inspect)

    # Everything after a literal `--`, plus any flags argparse doesn't
    # recognize (e.g. --prompt, --num_frames), is the inference workload —
    # passed through verbatim, order preserved.
    argv = sys.argv[1:]
    tail: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, tail = argv[:i], argv[i + 1 :]
    args, extra = p.parse_known_args(argv)
    workload = extra + tail
    if workload and args.fn is not cmd_run:
        p.error(f"unrecognized arguments: {' '.join(workload)}")
    args.workload = workload
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
