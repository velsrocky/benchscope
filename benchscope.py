#!/usr/bin/env python3
"""benchscope - wrapper around stock llama-bench that keeps its output
and adds device / VRAM / RAM / GPU% / CPU% telemetry.

Usage:
  python benchscope.py --list-gpus
  python benchscope.py -- -m model.gguf -p 512 -n 128
  python benchscope.py --llama-bench-bin ./build/bin/llama-bench --interval 0.2 -- -m model.gguf -p 512 -n 128 -r 3
  python benchscope.py --per-test -- -m model.gguf -p 128,512 -n 32,128

Design: stock llama-bench already prints static cpu_info/gpu_info/backends in
json/csv/sql (not in default md table). What it does NOT print is runtime
telemetry (VRAM used, RAM used, utilizations, temp, power). We sample those in
a background thread while the bench subprocess runs, then merge.

Modes:
  single (default): one llama-bench process, telemetry is run-global
    (pp and tg rows share the same numbers).
  --per-test: one process per pp size / tg size / pg pair, so each row gets
    its own isolated telemetry. Costs one model load per test.
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    print("ERROR: psutil required: pip install psutil", file=sys.stderr)
    sys.exit(1)


# ---------------- GPU discovery ----------------

def _read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return None


def _real(p):
    try:
        return os.path.realpath(p)
    except Exception:
        return p


def find_amd_gpus():
    """Enumerate physical AMD DRM devices via sysfs (fast, no subprocess)."""
    gpus = []
    seen = set()
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        # skip connector entries like card1-DP-1
        if "-" in card.name:
            continue
        dev = card / "device"
        total_f = dev / "mem_info_vram_total"
        if not total_f.exists():
            continue
        rp = _real(str(dev))
        if rp in seen:
            continue
        seen.add(rp)
        total = _read_int(total_f)
        if not total:
            continue
        used = _read_int(dev / "mem_info_vram_used")
        busy = _read_int(dev / "gpu_busy_percent")
        # temp / power via hwmon
        temp_c, power_w = None, None
        for hw in sorted((dev / "hwmon").glob("hwmon*")) if (dev / "hwmon").exists() else []:
            t = _read_int(hw / "temp1_input")
            if t is not None:
                temp_c = t / 1000.0
            for pf in ["power1_average", "power1_input"]:
                pw = _read_int(hw / pf)
                if pw is not None:
                    power_w = pw / 1_000_000.0
                    break
            if temp_c is not None:
                break
        # pci id for mapping
        pci = None
        try:
            ue = (dev / "uevent").read_text()
            m = re.search(r"PCI_SLOT_NAME=(\S+)", ue)
            if m:
                pci = m.group(1)
        except Exception:
            pass
        gpus.append({
            "drm": card.name, "sysfs": str(dev), "pci": pci,
            "vram_total": total, "vram_used": used or 0,
            "busy": busy or 0, "temp_c": temp_c, "power_w": power_w,
            "vendor": "AMD",
        })
    # attach product names from rocm-smi once, matched by VRAM total size
    names = _rocm_names_by_vram()
    for g in gpus:
        g["name"] = names.get(g["vram_total"], f"AMD GPU ({g['drm']})")
    # default main GPU = largest VRAM (dGPU over iGPU)
    gpus.sort(key=lambda g: g["vram_total"], reverse=True)
    return gpus


def _rocm_names_by_vram():
    """Map VRAM-total-bytes -> product name via one rocm-smi call."""
    out = {}
    if not shutil.which("rocm-smi"):
        return out
    try:
        p = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=15)
        data = json.loads(p.stdout or "{}")
        for _, info in data.items():
            if not isinstance(info, dict):
                continue
            name = info.get("Card series", info.get("Card Series", "")).strip()
            total = info.get("VRAM Total Memory (B)")
            if total is None:
                continue
            try:
                out[int(str(total))] = name or "AMD GPU"
            except (TypeError, ValueError):
                pass
    except Exception:
        pass
    return out


def find_nvidia_gpus():
    gpus = []
    if not shutil.which("nvidia-smi"):
        return gpus
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        for line in p.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 7:
                continue
            idx, name, mt, mu, util, temp, pw = parts[:7]
            gpus.append({
                "drm": f"nvidia{idx}", "pci": None,
                "vram_total": int(float(mt) * 1024 * 1024),
                "vram_used": int(float(mu) * 1024 * 1024),
                "busy": int(float(util or 0)), "temp_c": float(temp or 0) or None,
                "power_w": float((pw or "0").split()[0] or 0) or None,
                "vendor": "NVIDIA", "name": name,
            })
    except Exception:
        pass
    return gpus


def discover_gpus():
    gpus = find_amd_gpus() + find_nvidia_gpus()
    return gpus


def poll_gpu_snapshot(gpus):
    """Fast re-read of dynamic values (sysfs preferred, no subprocess)."""
    snap = []
    for g in gpus:
        if g["vendor"] == "AMD":
            dev = Path(g["sysfs"])
            total = _read_int(dev / "mem_info_vram_total") or g["vram_total"]
            used = _read_int(dev / "mem_info_vram_used")
            busy = _read_int(dev / "gpu_busy_percent")
            temp_c, power_w = g.get("temp_c"), g.get("power_w")
            for hw in sorted((dev / "hwmon").glob("hwmon*")) if (dev / "hwmon").exists() else []:
                t = _read_int(hw / "temp1_input")
                if t is not None:
                    temp_c = t / 1000.0
                for pf in ["power1_average", "power1_input"]:
                    pw = _read_int(hw / pf)
                    if pw is not None:
                        power_w = pw / 1_000_000.0
                        break
                break
            snap.append({"name": g["name"], "vram_total": total,
                         "vram_used": used if used is not None else g["vram_used"],
                         "busy": busy if busy is not None else 0,
                         "temp_c": temp_c, "power_w": power_w})
        else:
            snap.append({"name": g["name"], "vram_total": g["vram_total"],
                         "vram_used": g["vram_used"], "busy": g.get("busy", 0),
                         "temp_c": g.get("temp_c"), "power_w": g.get("power_w")})
    # NVIDIA live refresh via one nvidia-smi call per tick is slow; refresh
    # opportunistically only if NVIDIA present (interval >= 1s recommended).
    nvidia = [g for g in gpus if g["vendor"] == "NVIDIA"]
    if nvidia and shutil.which("nvidia-smi"):
        try:
            p = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            lines = p.stdout.strip().splitlines()
            # map in order
            k = 0
            for s in snap:
                if k < len(lines):
                    parts = [x.strip() for x in lines[k].split(",")]
                    if len(parts) >= 4:
                        try:
                            s["vram_used"] = int(float(parts[0]) * 1024 * 1024)
                            s["busy"] = int(float(parts[1]))
                            s["temp_c"] = float(parts[2])
                            s["power_w"] = float(parts[3].split()[0])
                        except ValueError:
                            pass
                    k += 1
        except Exception:
            pass
    return snap


# ---------------- sampler ----------------

class Sampler(threading.Thread):
    def __init__(self, gpus, interval=0.2):
        super().__init__(daemon=True)
        self.gpus = gpus
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self.proc = psutil.Process()
        self.t0 = None

    def run(self):
        psutil.cpu_percent(interval=None)  # prime system-wide
        try:
            self.proc.cpu_percent(interval=None)  # prime process
        except Exception:
            pass
        self.t0 = time.time()
        while not self._stop.is_set():
            t = time.time() - self.t0
            try:
                cpu_sys = psutil.cpu_percent(interval=None)
            except Exception:
                cpu_sys = 0.0
            try:
                cpu_proc = self.proc.cpu_percent(interval=None) / (psutil.cpu_count() or 1)
            except Exception:
                cpu_proc = 0.0
            vm = psutil.virtual_memory()
            gs = poll_gpu_snapshot(self.gpus)
            self.samples.append({
                "t": round(t, 3), "cpu_sys": cpu_sys, "cpu_proc": round(cpu_proc, 1),
                "ram_used": vm.used, "ram_total": vm.total,
                "gpus": gs,
            })
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()


def summarize(samples, gpus, gpu_index=0):
    if not samples:
        return {}
    n = len(samples)
    cpu = [s["cpu_sys"] for s in samples]
    ram_peak = max(s["ram_used"] for s in samples)
    ram_total = samples[0]["ram_total"]
    base_vram = samples[0]["gpus"][gpu_index]["vram_used"] if samples[0]["gpus"] else 0
    total_vram = samples[0]["gpus"][gpu_index]["vram_total"] if samples[0]["gpus"] else 0
    busy = [s["gpus"][gpu_index]["busy"] for s in samples if len(s["gpus"]) > gpu_index]
    vram = [s["gpus"][gpu_index]["vram_used"] for s in samples if len(s["gpus"]) > gpu_index]
    temps = [s["gpus"][gpu_index].get("temp_c") or 0 for s in samples if len(s["gpus"]) > gpu_index]
    pwrs = [s["gpus"][gpu_index].get("power_w") or 0 for s in samples if len(s["gpus"]) > gpu_index]
    return {
        "samples": n,
        "duration_s": round(samples[-1]["t"], 2),
        "cpu_sys_avg": round(sum(cpu) / n, 1),
        "cpu_sys_max": round(max(cpu), 1),
        "ram_peak": ram_peak, "ram_total": ram_total,
        "vram_baseline": base_vram, "vram_peak": max(vram) if vram else 0,
        "vram_total": total_vram,
        "vram_delta": (max(vram) - base_vram) if vram else 0,
        "gpu_avg": round(sum(busy) / len(busy), 1) if busy else 0,
        "gpu_max": max(busy) if busy else 0,
        "temp_max": round(max(temps), 1) if temps else None,
        "power_avg": round(sum(pwrs) / len(pwrs), 1) if pwrs and max(pwrs) else None,
        "power_max": round(max(pwrs), 1) if pwrs and max(pwrs) else None,
    }


# ---------------- formatting ----------------

def fmt_gib(b):
    return f"{b / (1024**3):.2f} GiB"


def fmt_mb(b):
    return f"{b / (1024**2):.0f} MiB"


def fmt_ctx(n):
    if n is None:
        return "?"
    if n >= 1024 * 1024 and n % (1024 * 1024) == 0:
        return f"{n // (1024 * 1024)}M"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return str(n)


# ---------------- model context ----------------
# Minimal GGUF KV reader: header KV section sits at file start, so we only
# read a small prefix. No dependency on gguf-py.

_GGUF_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                10: 8, 11: 8, 12: 8}  # type -> byte size (8=string, 9=array)

# llama.cpp KV-cache type -> bytes per element
_KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 1.0,
             "q4_0": 0.5, "q4_1": 0.5, "iq4_nl": 0.5}

_HP_U32 = ("block_count", "embedding_length", "attention.head_count",
           "attention.head_count_kv", "attention.key_length",
           "attention.value_length", "attention.sliding_window")


def read_gguf_hparams(path, _cache={}):
    """Return {arch, context_length, block_count, ...} for a .gguf file, cached.

    Only the file header is read. Missing keys -> None.
    """
    key = str(path)
    if key in _cache:
        return _cache[key]
    hp = {"arch": None, "context_length": None, "block_count": None,
          "embedding_length": None, "head_count": None, "head_count_kv": None,
          "key_length": None, "value_length": None, "sliding_window": 0,
          "swa_pattern": None}
    try:
        import struct
        with open(path, "rb") as f:
            buf = f.read(8 * 1024 * 1024)  # KV block is at file start
        if buf[:4] != b"GGUF":
            _cache[key] = hp
            return hp
        off = 4 + 4  # magic, version; then n_tensors, n_kv follow

        def read_u64(o):
            return struct.unpack_from("<Q", buf, o)[0], o + 8
        def read_u32(o):
            return struct.unpack_from("<I", buf, o)[0], o + 4
        _, off = read_u64(off)  # n_tensors (skip)
        n_kv, off = read_u64(off)

        def read_bytes(o):
            ln, o = read_u64(o)
            return buf[o:o + ln].decode("utf-8", "replace"), o + ln

        def skip_value(o, t):
            if t in _GGUF_SCALAR:
                return o + _GGUF_SCALAR[t] if t != 8 else read_bytes(o)[1]
            if t == 8:
                return read_bytes(o)[1]
            if t == 9:  # array
                et, o = read_u32(o)
                ln, o = read_u64(o)
                if et == 8:
                    for _ in range(ln):
                        _, o = read_bytes(o)
                    return o
                return o + _GGUF_SCALAR.get(et, 0) * ln
            return None

        def read_u32_array(o):
            et, o = read_u32(o)
            ln, o = read_u64(o)
            vals, o2 = [], o
            if et in (4, 5):
                for _ in range(ln):
                    v, o2 = read_u32(o2)
                    vals.append(v)
                return vals, o2
            if et == 7:  # bool
                vals = [bool(b) for b in buf[o2:o2 + ln]]
                return vals, o2 + ln
            return None, skip_value(o - 12, 9)

        arch = None
        for _ in range(n_kv):
            k, off = read_bytes(off)
            t, off = read_u32(off)
            if k == "general.architecture" and t == 8:
                arch, off = read_bytes(off)
                hp["arch"] = arch
                continue
            if arch and t == 4 and k == f"{arch}.context_length":
                hp["context_length"], off = read_u32(off)
                continue
            if arch and k.startswith(arch + "."):
                short = k[len(arch) + 1:]
                if t == 4 and short in _HP_U32:
                    hp[short.split(".")[-1]], off = read_u32(off)
                    continue
                if short == "attention.sliding_window_pattern" and t == 9:
                    vals, off2 = read_u32_array(off)
                    if vals is not None:
                        hp["swa_pattern"] = [bool(v) for v in vals]
                        off = off2
                        continue
            off = skip_value(off, t)
            if off is None or off >= len(buf):
                break
    except Exception:
        pass
    _cache[key] = hp
    return hp


def kv_bytes_per_token(hp, type_k="f16", type_v="f16"):
    """KV-cache bytes per context token for full-attention layers, or None."""
    n = hp.get("block_count")
    heads = hp.get("head_count_kv") or hp.get("head_count")
    emb, nheads = hp.get("embedding_length"), hp.get("head_count")
    head_k = hp.get("key_length") or (emb // nheads if emb and nheads else None)
    head_v = hp.get("value_length") or head_k
    if not n or not heads or not head_k or not head_v:
        return None
    bk = _KV_BYTES.get(str(type_k).lower(), 2.0)
    bv = _KV_BYTES.get(str(type_v).lower(), 2.0)
    return heads * (head_k * bk + head_v * bv), n


def fit_context(hp, weights_bytes, vram_total, reserve_bytes,
                type_k="f16", type_v="f16"):
    """Estimate max ctx tokens fitting on the GPU with full weight offload.

    budget = vram - weights - reserve; SWA layers cost window each, full
    layers cost ctx each. Returns (fit|None, limiter, kv_per_tok|None).
    limiter is 'gpu', 'native' (model ctx is tighter) or 'unknown'.
    """
    native = hp.get("context_length")
    per_tok = kv_bytes_per_token(hp, type_k, type_v)
    if per_tok is None or not vram_total or not weights_bytes:
        return None, "unknown", None
    per_tok_layer, n = per_tok
    pat = hp.get("swa_pattern")
    if isinstance(pat, list) and len(pat) == n:
        n_swa = sum(1 for b in pat if b)  # True = sliding-window layer
        n_full = n - n_swa
    else:
        n_swa, n_full = 0, n
    window = hp.get("sliding_window") or 0
    fixed = n_swa * window * per_tok_layer
    budget = vram_total - weights_bytes - reserve_bytes
    if budget <= 0:
        return 0, "gpu", per_tok_layer
    if n_full <= 0:
        return native, "native", per_tok_layer  # SWA-only: KV is ctx-independent
    fit = int((budget - fixed) // (n_full * per_tok_layer))
    fit = max(fit, 0)
    if native and fit >= native:
        return native, "native", per_tok_layer
    return fit, "gpu", per_tok_layer


def context_used(r):
    """Tokens touched by this test: prefilled depth + prompt + generated."""
    return (r.get("n_depth") or 0) + (r.get("n_prompt") or 0) + (r.get("n_gen") or 0)


def enrich_context(results, vram_total=None, reserve_bytes=0):
    """Attach GPU-fit max context + used context to each result row.

    context_max = estimated max tokens fitting on the GPU with full weight
    offload (capped at the model's native context). context_used = tokens
    touched by the test; context_pct = used/max %.
    """
    by_model = {}
    for r in results:
        mp = r.get("model_filename")
        if mp not in by_model:
            p = Path(mp) if mp and Path(mp).is_absolute() else Path.cwd() / (mp or "")
            by_model[mp] = read_gguf_hparams(p) if p.exists() else read_gguf_hparams("")
        hp = by_model[mp]
        fit, limiter, per_tok = fit_context(
            hp, r.get("model_size") or 0, vram_total or 0, reserve_bytes,
            r.get("type_k", "f16"), r.get("type_v", "f16"))
        used = context_used(r)
        r["context_native"] = hp.get("context_length")
        r["context_fit"] = fit
        r["context_fit_limiter"] = limiter
        r["kv_bytes_per_tok"] = per_tok
        r["context_max"] = fit
        r["context_used"] = used
        r["context_pct"] = round(100.0 * used / fit, 2) if fit else None
    return results


def render_markdown(results, summary, gpu_name):
    lines = []
    lines.append("| model | size | params | backend | ngl | test | t/s | "
                 "ctx max fit | ctx used | "
                 "GPU | VRAM used/total | RAM used/total | GPU% avg/max | CPU% avg/max | Temp max | Power avg |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        tel = r.get("telemetry", summary)
        np_, ng_ = r.get("n_prompt") or 0, r.get("n_gen") or 0
        if np_ and ng_:
            label = f"pg {np_}+{ng_}"
        elif np_:
            label = f"pp {np_}"
        else:
            label = f"tg {ng_}"
        used = r.get("context_used", context_used(r))
        pct = r.get("context_pct")
        ctx_used = f"{used} ({pct:.1f}%)" if pct is not None else str(used)
        fit = r.get("context_max")
        ctx_max = fmt_ctx(fit)
        if r.get("context_fit_limiter") == "native" and fit:
            ctx_max += "*"
        lines.append(
            f"| {r.get('model_type','?')} | {fmt_gib(r.get('model_size',0))} | "
            f"{r.get('model_n_params',0)/1e9:.2f} B | {r.get('backends','?')} | "
            f"{r.get('n_gpu_layers','?')} | "
            f"{label} | "
            f"{r.get('avg_ts',0):.2f} ± {r.get('stddev_ts',0):.2f} | "
            f"{ctx_max} | {ctx_used} | "
            f"{gpu_name} | "
            f"{fmt_mb(tel.get('vram_peak',0))}/{fmt_gib(tel.get('vram_total',1))} | "
            f"{fmt_gib(tel.get('ram_peak',0))}/{fmt_gib(tel.get('ram_total',1))} | "
            f"{tel.get('gpu_avg',0):.0f}/{tel.get('gpu_max',0):.0f} | "
            f"{tel.get('cpu_sys_avg',0):.0f}/{tel.get('cpu_sys_max',0):.0f} | "
            f"{tel.get('temp_max','-')} C | {tel.get('power_avg','-')} W |")
    return "\n".join(lines)


# ---------------- PNG card (16:9) ----------------

def render_png(results, overall, gpu_name, path, mode="single"):
    """Render a 1920x1080 (16:9) summary card PNG. Returns path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox
    from datetime import datetime, timezone

    def mib(b):
        return b / (1024 ** 2)

    header = ["test", "t/s", "ctx fit", "ctx used", "VRAM MiB",
              "RAM GiB", "GPU%", "CPU%", "temp", "power"]
    rows, native_star = [], False
    for r in results:
        tel = r.get("telemetry", overall)
        np_, ng_ = r.get("n_prompt") or 0, r.get("n_gen") or 0
        label = f"pg {np_}+{ng_}" if np_ and ng_ else (f"pp {np_}" if np_ else f"tg {ng_}")
        fit = r.get("context_max")
        star = ""
        if r.get("context_fit_limiter") == "native" and fit:
            star = "*"
            native_star = True
        used = r.get("context_used", context_used(r))
        pct = r.get("context_pct")
        rows.append([
            label,
            f"{r.get('avg_ts', 0):.1f} ± {r.get('stddev_ts', 0):.1f}",
            f"{fmt_ctx(fit)}{star}" if fit else "?",
            f"{used} ({pct:.1f}%)" if pct is not None else str(used),
            f"{mib(tel.get('vram_peak', 0)):.0f}/{mib(tel.get('vram_total', 1)):.0f}",
            f"{tel.get('ram_peak', 0) / 1024**3:.1f}/{tel.get('ram_total', 1) / 1024**3:.1f}",
            f"{tel.get('gpu_avg', 0):.0f}/{tel.get('gpu_max', 0):.0f}",
            f"{tel.get('cpu_sys_avg', 0):.0f}/{tel.get('cpu_sys_max', 0):.0f}",
            f"{tel.get('temp_max', '-')} C",
            f"{tel.get('power_avg', '-')} W",
        ])

    BG, PANEL, ACCENT, TEXT, MUTED = "#0d1117", "#161b22", "#58a6ff", "#f0f6fc", "#8b949e"
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")

    model = results[0].get("model_type", "?") if results else "?"
    backend = results[0].get("backends", "?") if results else "?"
    ax.text(0.5, 0.94, f"{model}  ·  {backend}  ·  {gpu_name}",
            ha="center", va="center", fontsize=26, fontweight="bold", color=TEXT)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ax.text(0.5, 0.885, f"llama-bench + telemetry  ·  {mode} mode  ·  {stamp}",
            ha="center", va="center", fontsize=13, color=MUTED)

    n = max(len(rows), 1)
    fs = max(7, min(14, int(150 / (n + 3))))
    tab_h = min(0.70, (n + 1) * 0.085)
    tab = ax.table(cellText=rows, colLabels=header, loc="center",
                   bbox=Bbox.from_bounds(0.03, 0.80 - tab_h, 0.94, tab_h))
    tab.auto_set_font_size(False)
    tab.set_fontsize(fs)
    for (ri, ci), cell in tab.get_celld().items():
        cell.set_edgecolor(BG)
        cell.set_text_props(color=TEXT)
        if ri == 0:
            cell.set_facecolor("#1f6feb")
            cell.set_text_props(fontweight="bold", color="white")
        else:
            cell.set_facecolor(PANEL if ri % 2 else "#1c2128")
            cell.PAD = 0.04

    if 1 < len(rows) <= 12:
        labels = [r[0] for r in rows]
        y = range(len(labels))
        ts = [float(str(r[1]).split("±")[0]) for r in rows]
        gpu = [float(str(r[6]).split("/")[0]) for r in rows]
        pwr = [float(str(r[9]).split()[0]) if str(r[9]).split()[0] != "-" else 0
               for r in rows]
        ax1 = fig.add_axes((0.05, 0.14, 0.28, 0.26), facecolor=BG)
        ax1.barh(list(y), ts, color=ACCENT, edgecolor="none")
        ax1.set_yticks(list(y), labels, color=TEXT, fontsize=11)
        ax1.set_xlabel("throughput (t/s, log)", color=MUTED, fontsize=11)
        ax1.set_xscale("log")
        ax1.tick_params(colors=MUTED, labelsize=10)
        for s in ax1.spines.values():
            s.set_visible(False)
        ax2 = fig.add_axes((0.38, 0.14, 0.28, 0.26), facecolor=BG)
        ax2.barh(list(y), gpu, color="#3fb950", edgecolor="none")
        ax2.set_yticks(list(y), labels, color=TEXT, fontsize=11)
        ax2.set_xlabel("GPU util avg (%)", color=MUTED, fontsize=11)
        ax2.tick_params(colors=MUTED, labelsize=10)
        for s in ax2.spines.values():
            s.set_visible(False)
        ax3 = fig.add_axes((0.71, 0.14, 0.24, 0.26), facecolor=BG)
        ax3.barh(list(y), pwr, color="#d29922", edgecolor="none")
        ax3.set_yticks(list(y), labels, color=TEXT, fontsize=11)
        ax3.set_xlabel("power avg (W)", color=MUTED, fontsize=11)
        ax3.tick_params(colors=MUTED, labelsize=10)
        for s in ax3.spines.values():
            s.set_visible(False)

    foot = f"{Path(results[0].get('model_filename', '')).name} · ngl={results[0].get('n_gpu_layers', '?')}" \
        if results else ""
    if native_star:
        foot += "   ·   *ctx capped by model native window"
    ax.text(0.5, 0.03, foot + "   ·   benchscope",
            ha="center", va="center", fontsize=11, color=MUTED)
    fig.savefig(path, facecolor=BG, dpi=120)
    plt.close(fig)
    return path


# ---------------- per-test split ----------------

P_DEFAULT, N_DEFAULT = 512, 128
MAX_JOBS = 64

# option spellings: (short flags, long flags with separate value, long= forms)
_SPLIT_OPTS = {
    "pg": (["-pg"], ["--pg"], ["--pg="]),
    "p": (["-p"], ["--n-prompt"], ["--n-prompt="]),
    "n": (["-n"], ["--n-gen"], ["--n-gen="]),
}


def _expand_range_token(tok):
    """Expand 'a-b', 'a-b+s', 'a-b*m' (llama-bench range syntax) to int list."""
    m = re.fullmatch(r"(\d+)-(\d+)([+*](\d+))?", tok.strip())
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    op, step = (m.group(3) or "")[:1], m.group(4)
    if b < a:
        return None
    if not op:
        return list(range(a, b + 1))
    s = int(step)
    if op == "+":
        return list(range(a, b + 1, s)) if s > 0 else None
    # '*mult'
    out, cur = [], a
    while cur <= b:
        out.append(cur)
        if s <= 1:
            break
        cur *= s
    return out


def _expand_int_list(raw_values):
    out = []
    for raw in raw_values:
        for tok in str(raw).split(","):
            tok = tok.strip()
            if not tok:
                continue
            exp = _expand_range_token(tok)
            if exp is not None:
                out.extend(exp)
            else:
                out.append(int(tok))
    return out


def _parse_split_opts(bench_args):
    """Collect -p/-n/-pg values. Returns dict key -> (explicit, [raw values])."""
    found = {"p": [], "n": [], "pg": []}
    i = 0
    # match longest flags first so '-pg...' isn't misread as '-p...'
    while i < len(bench_args):
        tok = bench_args[i]
        hit = None
        for key in ("pg", "p", "n"):
            shorts, longs, longs_eq = _SPLIT_OPTS[key]
            if tok in shorts or tok in longs:
                if i + 1 < len(bench_args):
                    found[key].append(bench_args[i + 1])
                hit = key
                i += 2
                break
            if key == "pg" and tok.startswith("-pg") and tok != "-pg":
                rest = tok[3:].lstrip("=")
                if rest and not rest.startswith("-"):
                    found[key].append(rest)
                    hit = key
                i += 1
                break
            if key == "p" and tok.startswith("-p") and tok not in ("-p", "-pg") \
                    and not tok.startswith("-pg"):
                rest = tok[2:].lstrip("=")
                if rest and not rest.startswith("-"):
                    found[key].append(rest)
                    hit = key
                i += 1
                break
            if key == "n" and tok.startswith("-n") and tok != "-n":
                rest = tok[2:].lstrip("=")
                if rest and not rest.startswith("-"):
                    found[key].append(rest)
                    hit = key
                i += 1
                break
            for le in longs_eq:
                if tok.startswith(le):
                    found[key].append(tok[len(le):])
                    hit = key
                    i += 1
                    break
            if hit:
                break
        if not hit:
            i += 1
    return {k: (bool(v), v) for k, v in found.items()}


def _strip_split_opts(bench_args):
    """Remove all -p/-n/-pg options (and their values) so jobs can set singles."""
    out = []
    i = 0
    while i < len(bench_args):
        tok = bench_args[i]
        consumed = False
        for key in ("pg", "p", "n"):
            shorts, longs, longs_eq = _SPLIT_OPTS[key]
            if tok in shorts or tok in longs:
                i += 2  # drop flag + value
                consumed = True
                break
            if key == "pg" and tok.startswith("-pg") and tok != "-pg":
                i += 1
                consumed = True
                break
            if key == "p" and tok.startswith("-p") and tok not in ("-p", "-pg") \
                    and not tok.startswith("-pg") and len(tok) > 2:
                i += 1
                consumed = True
                break
            if key == "n" and tok.startswith("-n") and tok != "-n" and len(tok) > 2:
                i += 1
                consumed = True
                break
            if any(tok.startswith(le) for le in longs_eq):
                i += 1
                consumed = True
                break
        if not consumed:
            out.append(tok)
            i += 1
    return out


def _force_json(argv):
    """Return argv with output format forced to json (needed for merging)."""
    out, i = [], 0
    forced = False
    while i < len(argv):
        tok = argv[i]
        if tok in ("-o", "--output", "-oe", "--output-err"):
            out += [tok, "json"]
            forced = True
            i += 2
        elif tok.startswith("-o") and len(tok) > 2 and not tok.startswith("-oe"):
            out.append("-o")
            out.append("json")
            forced = True
            i += 1
        elif tok.startswith("--output=") or tok.startswith("--output-err="):
            out.append(tok.split("=", 1)[0] + "=json")
            forced = True
            i += 1
        else:
            out.append(tok)
            i += 1
    if not forced:
        out += ["-o", "json"]
    return out


def build_jobs(bench_args):
    """Expand bench args into [(label, args)] with one test per job.

    pp sizes run as (-p SIZE -n 0), tg sizes as (-p 0 -n SIZE),
    pg pairs as (-p 0 -n 0 -pg PAIR). Zero means 'skip that phase'.
    """
    parsed = _parse_split_opts(bench_args)
    p_exp, p_raw = parsed["p"]
    n_exp, n_raw = parsed["n"]
    pg_exp, pg_raw = parsed["pg"]
    try:
        p_vals = _expand_int_list(p_raw) if p_exp else [P_DEFAULT]
        n_vals = _expand_int_list(n_raw) if n_exp else [N_DEFAULT]
    except ValueError as e:
        print(f"ERROR: cannot parse -p/-n values: {e}", file=sys.stderr)
        sys.exit(2)
    pp_sizes = [v for v in p_vals if v != 0]
    tg_sizes = [v for v in n_vals if v != 0]
    base = _strip_split_opts(bench_args)
    jobs = []
    for s in pp_sizes:
        jobs.append((f"pp {s}", base + ["-p", str(s), "-n", "0"]))
    for s in tg_sizes:
        jobs.append((f"tg {s}", base + ["-p", "0", "-n", str(s)]))
    for pg in pg_raw:
        jobs.append((f"pg {pg}", base + ["-p", "0", "-n", "0", "-pg", pg]))
    if not jobs:
        print("ERROR: nothing to run (-p 0 -n 0 with no -pg?).", file=sys.stderr)
        sys.exit(2)
    if len(jobs) > MAX_JOBS:
        print(f"ERROR: --per-test would run {len(jobs)} jobs (max {MAX_JOBS}). "
              "Narrow your -p/-n ranges.", file=sys.stderr)
        sys.exit(2)
    return jobs


def parse_bench_output(stdout):
    try:
        results = json.loads(stdout)
        if isinstance(results, dict):
            results = [results]
        return results
    except json.JSONDecodeError:
        pass
    return [json.loads(l) for l in stdout.splitlines() if l.strip().startswith("{")]


# ---------------- main ----------------

def next_free(path):
    """Return path, or a numbered sibling (stem_01.ext, ...) if it exists.

    Never overwrites a previous run's output.
    """
    p = Path(path)
    if not p.exists():
        return str(p)
    for i in range(1, 1000):
        q = p.with_name(f"{p.stem}_{i:02d}{p.suffix}")
        if not q.exists():
            print(f"[keep] {p} exists -> writing {q} instead", file=sys.stderr)
            return str(q)
    print(f"ERROR: too many existing files like {p}", file=sys.stderr)
    sys.exit(2)


def resolve_bench(user_bin):
    if user_bin and Path(user_bin).exists():
        return user_bin
    for c in ["llama-bench", "./llama-bench", "./build/bin/llama-bench",
              "/usr/local/bin/llama-bench"]:
        if shutil.which(c) or Path(c).exists():
            return c
    return None


def bench_env():
    """Subprocess env with ROCm lib dir ensured (avoids libhipblas.so.3 errors)."""
    env = dict(os.environ)
    candidates = ["/opt/rocm/lib"]
    if os.environ.get("ROCM_PATH"):
        candidates.append(os.path.join(os.environ["ROCM_PATH"], "lib"))
    ld = env.get("LD_LIBRARY_PATH", "")
    have = [p for p in ld.split(":") if p]
    for d in candidates:
        if Path(d, "libhipblas.so.3").exists() and d not in have:
            have.insert(0, d)
    if have:
        env["LD_LIBRARY_PATH"] = ":".join(have)
    return env


def main():
    ap = argparse.ArgumentParser(description="llama-bench wrapper with system telemetry")
    ap.add_argument("--llama-bench-bin", default=None)
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--gpu-index", type=int, default=0,
                    help="which GPU to report (default 0 = largest VRAM)")
    ap.add_argument("--out-json", default="bench_enriched.json")
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-png", default=None,
                    help="also render a 16:9 (1920x1080) summary card PNG, e.g. --out-png bench.png")
    ap.add_argument("--timeseries", default="bench_timeseries.csv")
    ap.add_argument("--per-test", action="store_true",
                    help="run one llama-bench process per pp/tg/pg test so each "
                         "row gets isolated telemetry (costs one model load per test)")
    ap.add_argument("--ctx-reserve-mib", type=int, default=800,
                    help="VRAM headroom (MiB) kept for compute buffers/fragmentation "
                         "when estimating GPU-fit max context (default 800)")
    ap.add_argument("--list-gpus", action="store_true")
    args, bench_args = ap.parse_known_args()
    # support `--` separator
    if "--" in sys.argv:
        bench_args = sys.argv[sys.argv.index("--") + 1:]

    gpus = discover_gpus()
    if args.list_gpus:
        print(json.dumps([{"name": g["name"], "vendor": g["vendor"],
                           "vram_total": g["vram_total"],
                           "vram_total_h": fmt_gib(g["vram_total"]),
                           "pci": g.get("pci"), "drm": g.get("drm")}
                          for g in gpus], indent=2))
        return

    bench = resolve_bench(args.llama_bench_bin)
    if not bench:
        print("ERROR: llama-bench binary not found. Pass --llama-bench-bin PATH "
              "or install llama.cpp.", file=sys.stderr)
        sys.exit(2)
    if args.gpu_index >= len(gpus):
        print(f"ERROR: --gpu-index {args.gpu_index} out of range ({len(gpus)} GPUs).",
              file=sys.stderr)
        sys.exit(2)

    if args.per_test:
        jobs = build_jobs(bench_args)
        print(f"+ per-test mode: {len(jobs)} jobs", file=sys.stderr)
    else:
        jobs = [("full", bench_args)]
    print(f"+ telemetry: {[g['name'] for g in gpus]} | interval={args.interval}s",
          file=sys.stderr)
    gpu_name = gpus[args.gpu_index]["name"] if gpus else "none"

    all_results, all_series, job_summaries = [], [], []
    for idx, (label, job_args) in enumerate(jobs):
        cmd = [bench] + _force_json(job_args)
        print(f"+ [{idx + 1}/{len(jobs)}] {label}: {' '.join(cmd)}", file=sys.stderr)
        sampler = Sampler(gpus, interval=args.interval)
        sampler.start()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=bench_env())
        finally:
            sampler.stop()
            sampler.join(timeout=5)

        if proc.returncode != 0:
            print(proc.stderr[-3000:], file=sys.stderr)
            print(f"ERROR: llama-bench exited {proc.returncode} on job '{label}'",
                  file=sys.stderr)
            sys.exit(proc.returncode)

        results = parse_bench_output(proc.stdout)
        if not results:
            print(f"ERROR: could not parse llama-bench JSON output for job '{label}'.",
                  file=sys.stderr)
            sys.exit(3)

        summary = summarize(sampler.samples, gpus, args.gpu_index)
        summary["device"] = gpu_name
        summary["label"] = label
        job_summaries.append(summary)
        for r in results:
            r["telemetry"] = dict(summary)
            all_results.append(r)
        for s in sampler.samples:
            all_series.append((idx, label, s))

    overall = summarize([s for _, _, s in all_series], gpus, args.gpu_index) \
        if len(jobs) > 1 else dict(job_summaries[0])
    overall["device"] = gpu_name
    vram_total = gpus[args.gpu_index]["vram_total"] if gpus else 0
    enrich_context(all_results, vram_total,
                   args.ctx_reserve_mib * 1024 * 1024)

    enriched = {"device": gpu_name, "gpus": [{"name": g["name"],
               "vram_total": g["vram_total"]} for g in gpus],
                "mode": "per-test" if args.per_test else "single",
                "telemetry_summary": overall, "job_summaries": job_summaries,
                "results": all_results}
    out_json = next_free(args.out_json)
    Path(out_json).write_text(json.dumps(enriched, indent=2))
    out_ts = next_free(args.timeseries)
    with open(out_ts, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_idx", "label", "t", "cpu_sys", "cpu_proc", "ram_used",
                    "ram_total", "vram_used", "vram_total", "gpu_busy",
                    "temp_c", "power_w"])
        for idx, label, s in all_series:
            g = s["gpus"][args.gpu_index] if len(s["gpus"]) > args.gpu_index else {}
            w.writerow([idx, label, s["t"], s["cpu_sys"], s["cpu_proc"],
                        s["ram_used"], s["ram_total"], g.get("vram_used", 0),
                        g.get("vram_total", 0), g.get("busy", 0),
                        g.get("temp_c", ""), g.get("power_w", "")])

    md = render_markdown(all_results, overall, gpu_name)
    print(md)
    wrote = [out_json, out_ts]
    if args.out_md:
        out_md = next_free(args.out_md)
        Path(out_md).write_text(md + "\n")
        wrote.append(out_md)
    if args.out_png:
        out_png = next_free(args.out_png)
        render_png(all_results, overall, gpu_name, out_png,
                   mode="per-test" if args.per_test else "single")
        wrote.append(out_png)
    print(f"\n[wrote {', '.join(wrote)}]", file=sys.stderr)
    if args.per_test:
        print("[note] per-test mode: each row has isolated telemetry "
              "(one model load per test).", file=sys.stderr)
    else:
        print("[note] single-run mode: telemetry is run-global. "
              "Use --per-test for isolated pp-vs-tg numbers.", file=sys.stderr)


if __name__ == "__main__":
    main()
