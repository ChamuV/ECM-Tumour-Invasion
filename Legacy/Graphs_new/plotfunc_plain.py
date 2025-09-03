# plots_from_runs_plain.py
import os, json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Helpers to find/load runs
# -------------------------
def _approx_eq(a, b, tol=1e-12):
    try:
        return abs(float(a) - float(b)) < tol * max(1.0, abs(float(a)), abs(float(b)))
    except Exception:
        return False

def _scan_float_from_suffix(name: str, prefix: str):
    """
    Extract the float from folder names like 'lambda_0.001' or 'm0_0.5'.
    Returns None if pattern doesn't match or float fails.
    """
    if not name.startswith(prefix):
        return None
    try:
        return float(name[len(prefix):])
    except Exception:
        return None

def _find_run_dir_plain(base_dir: Path, lam: float, m0: float):
    """
    Look for a directory tree:
      base_dir / lambda_* / m0_* / (summary.json, snapshots.npz)
    that matches lam and m0 numerically (approx).
    Returns Path or None.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return None

    # First find a lambda folder that matches
    lam_dir = None
    for cand in base_dir.glob("lambda_*"):
        lam_val = _scan_float_from_suffix(cand.name, "lambda_")
        if lam_val is None: 
            continue
        if _approx_eq(lam_val, lam):
            lam_dir = cand
            break
    if lam_dir is None:
        return None

    # Inside lambda folder, find matching m0 folder
    m_dir = None
    for cand in lam_dir.glob("m0_*"):
        m_val = _scan_float_from_suffix(cand.name, "m0_")
        if m_val is None:
            continue
        if _approx_eq(m_val, m0):
            m_dir = cand
            break
    return m_dir

def _load_summary(run_dir: Path):
    s = run_dir / "summary.json"
    if s.exists():
        with open(s, "r") as f:
            return json.load(f)
    return {}

def _load_snapshots(run_dir: Path):
    z = run_dir / "snapshots.npz"
    if z.exists():
        data = np.load(z)
        return dict(x=data["x"], times=data["times"], N_arr=data["N_arr"], M_arr=data["M_arr"])
    return None

# ---------------------------------------------------------
# 1) Styled panel of u & m profiles across lambda (like yours)
# ---------------------------------------------------------
def plot_u_m_style_grid_from_runs(
    base_dir="speeds_plain_L200N20001",
    m0=0.2, times=(70, 140, 280),
    lambda_vals=(0.1, 0.2, 0.5, 0.8, 1.0, 2, 5, 10),
    xlim=None, ylim=(0, 1.5)
):
    """
    Replicates your panel style: 2x4 of (u,m) profiles for various lambdas,
    using 'snapshots.npz' and 'summary.json' saved by run_grid.
    """
    base_dir = Path(base_dir)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    legend_lines, legend_labels = [], []
    cmap = plt.get_cmap("tab10")

    for i, lam in enumerate(lambda_vals):
        ax = axes[i]
        run_dir = _find_run_dir_plain(base_dir, lam, m0)
        if run_dir is None:
            ax.set_title(rf"$\lambda = {lam:.2g}$" + "\n❌ Missing", fontsize=14)
            ax.grid(True, ls=':')
            continue

        snap = _load_snapshots(run_dir)
        if snap is None:
            ax.set_title(rf"$\lambda = {lam:.2g}$" + "\n❌ No snapshots", fontsize=14)
            ax.grid(True, ls=':')
            continue

        x = snap["x"]; t = snap["times"]; N_arr = snap["N_arr"]; M_arr = snap["M_arr"]
        meta = _load_summary(run_dir)
        speed = meta.get("wave_speed", None)

        for j, tt in enumerate(times):
            idx = int(np.argmin(np.abs(t - tt)))
            color = cmap(j)
            line_n, = ax.plot(x, N_arr[idx], label=f"$t$ = {int(t[idx])}", color=color, linewidth=3)
            ax.plot(x, M_arr[idx], linestyle='--', color=color, linewidth=3, alpha=0.7)
            if i == 0:
                legend_lines.append(line_n)
                legend_labels.append(f"$t$ = {int(t[idx])}")

        ax.set_title(rf"$\lambda = {lam:.2g}$", fontsize=17, fontweight="bold")
        if speed is not None and np.isfinite(speed):
            # place label near top-left of axes
            x_text = x[0] + 0.02*(x[-1]-x[0])
            ax.text(x_text, ylim[1]*0.93, rf"$c = {float(speed):.3f}$", fontsize=15, color='black')

        if xlim is None:
            ax.set_xlim([x.min(), x.max()])
        else:
            ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(axis='both', labelsize=13)
        ax.grid(True, linestyle=':', alpha=0.6)

    # axis labels (bottom row x, left col y)
    for ax in axes[-4:]:
        ax.set_xlabel(r"$x$", fontsize=15)
    for ax in axes[::4]:
        ax.set_ylabel(r"$u(x,t),\, m(x,t)$", fontsize=15)

    # legend & title
    fig.legend(legend_lines, legend_labels,
               loc='lower center', ncol=len(times),
               fontsize=14, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(rf"Tumour and ECM Profiles from runs  ($m_0$ = {m0})", fontsize=20)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

# ---------------------------------------------------------
# 2) Speed vs log10(lambda) for multiple m0 (from summary.json)
# ---------------------------------------------------------
def plot_speed_vs_log_lambda_from_runs(
    base_dir="speeds_plain_L200N20001",
    m0_vals=(0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0),
    lambda_vals=(0.001, 0.01, 0.05, 1, 5, 10, 100, 1000, 1e5, 1e6)
):
    base_dir = Path(base_dir)
    plt.figure(figsize=(7, 5))

    for m0 in m0_vals:
        speeds = []
        for lam in lambda_vals:
            run_dir = _find_run_dir_plain(base_dir, lam, m0)
            if run_dir is None:
                speeds.append(np.nan)
                continue
            meta = _load_summary(run_dir)
            cN = meta.get("wave_speed", np.nan)
            speeds.append(float(cN) if (cN is not None and np.isfinite(cN)) else np.nan)

        log_lambda = np.log10(np.array(lambda_vals, dtype=float))
        plt.plot(log_lambda, speeds, '-o', label=rf"$m_0 = {m0}$")

    plt.xlabel(r"$\log_{10}(\lambda)$", fontsize=13)
    plt.ylabel(r"$c_{\mathrm{num}}$", fontsize=13)
    plt.title("Wave Speed vs Diffusion Rate (from runs)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=11, title="$m_0$")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 3) Speed vs m0 at fixed lambda (from summary.json)
# ---------------------------------------------------------
def plot_speed_vs_m0_fixed_lambda_from_runs(
    base_dir="speeds_plain_L200N20001",
    lambda_fixed=1.0,
    m0_vals=(0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0)
):
    base_dir = Path(base_dir)
    plt.figure(figsize=(7, 5))
    speeds, valid_m0 = [], []

    for m0 in m0_vals:
        run_dir = _find_run_dir_plain(base_dir, lambda_fixed, m0)
        if run_dir is None:
            continue
        meta = _load_summary(run_dir)
        cN = meta.get("wave_speed", np.nan)
        if cN is not None and np.isfinite(cN):
            valid_m0.append(m0)
            speeds.append(float(cN))

    if valid_m0:
        plt.plot(valid_m0, speeds, '-o', linewidth=2, label=rf"$\lambda = {lambda_fixed}$")

    plt.xlabel(r"Initial ECM density $m_0$", fontsize=14)
    plt.ylabel(r"Wave speed $c$", fontsize=14)
    plt.title(rf"Wave Speed vs $m_0$ at $\lambda = {lambda_fixed}$ (from runs)", fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------------- 
# 4) Speed vs m0 for multiple lambda (from summary.json)
# ---------------------------------------------------------

# ---------- small helpers ----------
def _approx_eq(a, b, tol=1e-12):
    a = float(a); b = float(b)
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

def _scan_float_from_suffix(name: str, prefix: str):
    if not name.startswith(prefix):
        return None
    try:
        return float(name[len(prefix):])
    except Exception:
        return None

def _find_run_dir(base_dir: Path, lam: float, m0: float):
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return None
    lam_dir = None
    for cand in base_dir.glob("lambda_*"):
        v = _scan_float_from_suffix(cand.name, "lambda_")
        if v is not None and _approx_eq(v, lam):
            lam_dir = cand
            break
    if lam_dir is None:
        return None
    for cand in lam_dir.glob("m0_*"):
        v = _scan_float_from_suffix(cand.name, "m0_")
        if v is not None and _approx_eq(v, m0):
            return cand
    return None

def _load_speed(run_dir: Path):
    s = run_dir / "summary.json"
    if not s.exists():
        return np.nan
    try:
        with open(s, "r") as f:
            meta = json.load(f)
        c = meta.get("wave_speed", np.nan)
        return float(c) if c is not None else np.nan
    except Exception:
        return np.nan

# ---------- the 1x2 plotting function ----------
def plot_speed_vs_log_lambda_two_panels_from_runs(
    base_dir="speeds_plain_L200N20001",
    m0_list_left=(0.05, 0.1, 0.2, 0.5),
    m0_list_right=(0.5, 0.8, 0.9, 0.95, 1.0),
    lambda_vals=(0.001, 0.01, 0.05, 1, 5, 10, 100, 1000, 1e5, 1e6),
    n0_fixed=1.0,  # kept for title consistency
):
    base_dir = Path(base_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    def _plot_panel(ax, m0_list, title):
        for m0 in m0_list:
            speeds = []
            for lam in lambda_vals:
                run_dir = _find_run_dir(base_dir, lam, m0)
                speed = _load_speed(run_dir) if run_dir is not None else np.nan
                speeds.append(speed)
            x = np.log10(np.asarray(lambda_vals, dtype=float))
            ax.plot(x, speeds, marker='o', label=rf"$m_0 = {m0}$")
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(r"$\log_{10}(\lambda)$", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=10, frameon=False)

    _plot_panel(axes[0], m0_list_left, "Group 1")
    axes[0].set_ylabel(r"Wave speed $c$", fontsize=14)
    _plot_panel(axes[1], m0_list_right, "Group 2")

    plt.suptitle(rf"Wave Speed vs $\log_{{10}}(\lambda)$", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    

import os, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------- helpers to locate and load a run ----------
def _approx_eq(a, b, tol=1e-12):
    a = float(a); b = float(b)
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

def _scan_float_from_suffix(name: str, prefix: str):
    if not name.startswith(prefix):
        return None
    try:
        return float(name[len(prefix):])
    except Exception:
        return None

def _find_run_dir_plain(base_dir: Path, lam: float, m0: float):
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return None
    lam_dir = None
    for cand in base_dir.glob("lambda_*"):
        v = _scan_float_from_suffix(cand.name, "lambda_")
        if v is not None and _approx_eq(v, lam):
            lam_dir = cand
            break
    if lam_dir is None:
        return None
    for cand in lam_dir.glob("m0_*"):
        v = _scan_float_from_suffix(cand.name, "m0_")
        if v is not None and _approx_eq(v, m0):
            return cand
    return None

def _load_summary(run_dir: Path):
    s = run_dir / "summary.json"
    if not s.exists(): 
        return {}
    with open(s, "r") as f:
        return json.load(f)

def _load_snapshots(run_dir: Path):
    z = run_dir / "snapshots.npz"
    if not z.exists():
        return None
    data = np.load(z)
    return dict(x=data["x"], times=data["times"], N_arr=data["N_arr"], M_arr=data["M_arr"])

# ---------- main plotting function in your custom style ----------
def plot_u_m_style_grid_three_m0_from_runs(
    base_dir="speeds_plain_L200N20001",
    m0_rows=(0.05, 0.5, 1.0),
    lambda_cols=(0.1, 1.0, 10.0, 100.0),
    times=(60, 120, 180),           # physical times to sample (nearest in snapshots)
    arrow_len_frac=0.15,            # fraction of L for arrow length
    arrow_lw=2.5,
    arrow_start_frac=0.5,
    head_length=6, head_width=3,
    bottom_y=0.2,
    ylim=(-0.05, 1.05),             # raise to ( -0.05, 1.5 ) if you want extra headroom
    suptitle="Profiles across λ and $m_0$ (from saved runs)"
):
    base_dir = Path(base_dir)
    nrows, ncols = len(m0_rows), len(lambda_cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.8*nrows), sharex=True, sharey=True)

    # legend lines (use colors by time)
    cmap = plt.get_cmap("tab10")
    legend_lines, legend_labels = [], []

    for r, m0 in enumerate(m0_rows):
        for c, lam in enumerate(lambda_cols):
            ax = axes[r, c] if nrows > 1 else axes[c]
            run_dir = _find_run_dir_plain(base_dir, lam, m0)

            if run_dir is None:
                ax.set_title(rf"$\lambda = {lam:g}$" + "\n❌ missing", fontsize=12)
                ax.grid(True, ls=":", alpha=0.6)
                continue

            snap = _load_snapshots(run_dir)
            if snap is None:
                ax.set_title(rf"$\lambda = {lam:g}$" + "\n❌ no snapshots", fontsize=12)
                ax.grid(True, ls=":", alpha=0.6)
                continue

            meta = _load_summary(run_dir)
            c_fit = meta.get("wave_speed", None)

            x = snap["x"]; t = snap["times"]
            N_arr = snap["N_arr"]; M_arr = snap["M_arr"]
            L = float(x[-1] - x[0])

            # plot profiles at requested times
            for j, tphys in enumerate(times):
                idx = int(np.argmin(np.abs(t - tphys)))
                color = cmap(j)
                ln, = ax.plot(x, N_arr[idx], color='red', linewidth=2.2,
                              linestyle='-', label=f"$t={int(t[idx])}$")
                ax.plot(x, M_arr[idx], color='blue', linewidth=2.2,
                        linestyle='--', alpha=0.85)

                if r == 0 and c == 0:
                    legend_lines.append(ln)
                    legend_labels.append(f"$t={int(t[idx])}$")

            # arrows in same horizontal positions as your style
            arrow_len = arrow_len_frac * L
            ax.annotate('', xy=(x[0] + (arrow_start_frac*L + arrow_len)),  xytext=(x[0] + arrow_start_frac*L),
                        arrowprops=dict(arrowstyle=f'->,head_length={head_length},head_width={head_width}',
                                        color='red', lw=arrow_lw))
            ax.annotate('', xy=(x[0] + (arrow_start_frac*L + arrow_len)),  xytext=(x[0] + arrow_start_frac*L),
                        arrowprops=dict(arrowstyle=f'->,head_length={head_length},head_width={head_width}',
                                        color='blue', lw=arrow_lw))
            # position blue arrow at bottom_y by drawing a dummy, then moving via annotation coords is fine;
            # simpler: just add a small text indicator for m at bottom_y (keeps style minimal)
            # If you really want the blue arrow at a different y, use annotation with 'xycoords=("data","axes fraction")'.
            # Here we mimic your plot by placing them at fixed y-levels:
            ax.annotate('', xy=(x[0] + (arrow_start_frac*L + arrow_len), bottom_y),
                        xytext=(x[0] + arrow_start_frac*L, bottom_y),
                        xycoords=('data', 'data'),
                        arrowprops=dict(arrowstyle=f'->,head_length={head_length},head_width={head_width}',
                                        color='blue', lw=arrow_lw))

            # left-side text for m0 and c
            x_text = x[0] + 0.02 * L
            c_str = (str(c_fit)[:4] if (c_fit is not None and np.isfinite(c_fit)) else "—")
            ax.text(x_text, ylim[1]*0.92, rf"$\overline{{m}} = {m0}$", fontsize=13, ha='left')
            ax.text(x_text, ylim[1]*0.82, rf"$c = {c_str}$", fontsize=13, ha='left')

            # subplot title (lambda)
            ax.set_title(rf"$\lambda = {lam:g}$", fontsize=14)

            # axes cosmetics
            ax.set_xlim([x.min(), x.max()])
            ax.set_ylim(ylim)
            ax.grid(False)
            ax.tick_params(axis='both', labelsize=11)

            # y-label only on first column
            if c == 0:
                ax.set_ylabel(r"$u(x,t),\, m(x,t)$", fontsize=13)
            # x-label only on bottom row
            if r == nrows - 1:
                ax.set_xlabel(r"$x$", fontsize=13)

        # Row labels for m0 along the left margin
        axes[r, 0].text(-0.08, 0.5, rf"$m_0 = {m0}$",
                        transform=axes[r, 0].transAxes, rotation=90,
                        va='center', ha='right', fontsize=13)

    # common legend (times)
    fig.legend(legend_lines, legend_labels, loc='lower center',
               ncol=len(times), fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

    import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt

# ---------- helpers (match your saving scheme exactly) ----------

def _exact_run_dir(base_dir, lam, m0):
    """Use the *same* naming your runner used: lambda_{lam}/m0_{m0}."""
    d = os.path.join(base_dir, f"lambda_{lam}", f"m0_{m0}")
    return d if os.path.isdir(d) else None

def _fallback_run_dir(base_dir, lam, m0, tol=1e-12):
    """
    If exact string names don't exist (float repr quirks), parse numbers from
    folder names and choose the closest (lam,m0).
    """
    # find closest lambda folder
    lam_dirs = [d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("lambda_")]
    def _parse_lam(name):
        m = re.match(r"lambda_([0-9eE\.\-]+)$", name)
        return float(m.group(1)) if m else None
    if not lam_dirs:
        return None
    lam_pairs = [(d, _parse_lam(d)) for d in lam_dirs]
    lam_dir = min(lam_pairs, key=lambda p: abs((p[1] if p[1] is not None else 1e99) - lam))[0]
    lam_path = os.path.join(base_dir, lam_dir)

    # find closest m0 folder inside
    m0_dirs = [d for d in os.listdir(lam_path)
               if os.path.isdir(os.path.join(lam_path, d)) and d.startswith("m0_")]
    def _parse_m0(name):
        m = re.match(r"m0_([0-9eE\.\-]+)$", name)
        return float(m.group(1)) if m else None
    if not m0_dirs:
        return None
    m0_pairs = [(d, _parse_m0(d)) for d in m0_dirs]
    m0_dir = min(m0_pairs, key=lambda p: abs((p[1] if p[1] is not None else 1e99) - m0))[0]
    run_dir = os.path.join(lam_path, m0_dir)
    return run_dir

def _get_run_dir(base_dir, lam, m0):
    """Try exact path first, then tolerant fallback."""
    d = _exact_run_dir(base_dir, lam, m0)
    if d:
        return d
    return _fallback_run_dir(base_dir, lam, m0)

def _load_snapshots(run_dir):
    """
    Load x, times, N_arr, M_arr from snapshots.npz.
    Ensures arrays shape (Nt, Nx).
    """
    path = os.path.join(run_dir, "snapshots.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    x = z["x"]; times = z["times"]; N_arr = z["N_arr"]; M_arr = z["M_arr"]
    # ensure (Nt, Nx)
    if N_arr.shape[0] == x.size and N_arr.shape[-1] == times.size:
        N_arr = N_arr.T
    if M_arr.shape[0] == x.size and M_arr.shape[-1] == times.size:
        M_arr = M_arr.T
    return x, times, N_arr, M_arr

def _load_speed_label(run_dir):
    """Fetch wave_speed from summary.json if present."""
    s = os.path.join(run_dir, "summary.json")
    if not os.path.exists(s):
        return None
    try:
        with open(s, "r") as f:
            meta = json.load(f)
        return meta.get("wave_speed", None)
    except Exception:
        return None

# ---------- main grid plotter ----------

def plot_u_m_style_grid_from_nested(
    base_dir="speeds_plain_L200N20001",
    m0_rows=(0.05, 0.5, 1.0),
    lambda_cols=(0.1, 1.0, 10.0),
    times=(60, 120, 180),
    xlim=None,                 # e.g. (0, 200) or (0, 1000)
    ylim=(-0.05, 1.05),
    suptitle=None
):
    """
    Render a grid of u/m profiles.
      rows   -> m0 values
      cols   -> lambda values
      curves -> profiles at nearest available 'times' from snapshots.npz
    """
    cmap = plt.get_cmap("tab10")
    nrows, ncols = len(m0_rows), len(lambda_cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8*ncols, 3.4*nrows), squeeze=False, sharex=True, sharey=True)

    legend_lines, legend_labels = [], []

    for r, m0 in enumerate(m0_rows):
        for c, lam in enumerate(lambda_cols):
            ax = axes[r, c]
            run_dir = _get_run_dir(base_dir, lam, m0)

            if not run_dir:
                ax.set_title(rf"$\lambda={lam:g}$" + "\n❌ run not found", fontsize=12)
                ax.set_axis_off()
                continue

            snaps = _load_snapshots(run_dir)
            if snaps is None:
                ax.set_title(rf"$\lambda={lam:g}$" + "\n⚠ no snapshots.npz", fontsize=12)
                ax.set_axis_off()
                continue

            x, t_avail, N_arr, M_arr = snaps

            for j, tt in enumerate(times):
                tidx = int(np.argmin(np.abs(t_avail - tt)))
                color = cmap(j % 10)
                (ln_u,) = ax.plot(x, N_arr[tidx], color=color, linewidth=2.4, label=rf"$t={int(t_avail[tidx])}$")
                ax.plot(x, M_arr[tidx], linestyle="--", color=color, linewidth=2.0, alpha=0.75)

                if r == 0 and c == 0:
                    legend_lines.append(ln_u)
                    legend_labels.append(rf"$t={int(t_avail[tidx])}$")

            ax.grid(True, alpha=0.3)
            ax.set_title(rf"$\lambda = {lam:g}$", fontsize=12)
            if xlim is not None:
                ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            # annotate speed if available
            spd = _load_speed_label(run_dir)
            if spd is not None and np.isfinite(spd):
                x_left = x[0] if xlim is None else xlim[0]
                pad = 0.05 * ((ylim[1] - ylim[0]) if ylim else 1.0)
                ax.text(x_left, (ylim[1] - pad) if ylim else 0.95,
                        rf"$c={float(spd):.3f}$", fontsize=10)

            # row label on first column
            if c == 0:
                ax.set_ylabel(rf"$m_0 = {m0:g}$", fontsize=12)

    # x-labels on last row
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$x$", fontsize=12)

    # figure legend
    if legend_lines:
        fig.legend(legend_lines, legend_labels,
                   loc="lower center", ncol=max(3, len(times)),
                   fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))

    if suptitle:
        fig.suptitle(suptitle, fontsize=15, y=0.98)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.show()

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt

# ---------- helpers to match your folder naming ----------

def _closest_dir_by_number(parent, prefix, target):
    """
    From subdirs of `parent` that start with `prefix_`, pick the one whose
    numeric suffix is closest to `target`. Returns absolute path or None.
    """
    if not os.path.isdir(parent):
        return None
    cands = [d for d in os.listdir(parent)
             if os.path.isdir(os.path.join(parent, d)) and d.startswith(prefix + "_")]
    if not cands:
        return None

    def _num(d):
        m = re.match(rf"{re.escape(prefix)}_([0-9eE\.\-]+)$", d)
        return float(m.group(1)) if m else None

    pairs = [(d, _num(d)) for d in cands]
    best = min(pairs, key=lambda p: abs((p[1] if p[1] is not None else 1e99) - target))[0]
    return os.path.join(parent, best)

def _exact_dir(parent, prefix, val):
    d = os.path.join(parent, f"{prefix}_{val}")
    return d if os.path.isdir(d) else None

def _get_run_dir(base_dir, lam, m0):
    """Match exactly first; if that fails, pick closest numeric folders."""
    lam_dir = _exact_dir(base_dir, "lambda", lam)
    if lam_dir is None:
        lam_dir = _closest_dir_by_number(base_dir, "lambda", lam)
    if lam_dir is None:
        return None

    m0_dir = _exact_dir(lam_dir, "m0", m0)
    if m0_dir is None:
        m0_dir = _closest_dir_by_number(lam_dir, "m0", m0)
    return m0_dir

def _load_speed(run_dir, which="N"):
    """
    Load speed from summary.json. Defaults to N-front 'wave_speed'.
    For ECM you can pass which='M' to get 'm_wave_speed'.
    """
    if run_dir is None:
        return np.nan
    s = os.path.join(run_dir, "summary.json")
    if not os.path.exists(s):
        return np.nan
    try:
        with open(s, "r") as f:
            meta = json.load(f)
        if which.upper() == "M":
            return float(meta.get("m_wave_speed", np.nan))
        return float(meta.get("wave_speed", np.nan))
    except Exception:
        return np.nan

# ---------- heatmap builder/plotter ----------

def plot_speed_heatmap_nested(
    base_dir="speeds_plain_L200N20001",
    lambda_vals=(0.001, 0.01, 0.05, 1, 5, 10, 100, 1000),
    m0_vals=(0.0, 0.1, 0.5, 1.0),
    n_label=r"$m_0$",
    x_label=r"$\lambda$",
    title="Wave speed heatmap",
    which_speed="N",        # "N" for tumour front, "M" for ECM front
    annotate=False,         # write numbers inside cells
    cmap="viridis",
    vmin=None, vmax=None,   # manual color limits if you want
    log_lambda_ticks=False  # only affects tick labeling, not color scale
):
    """
    Build a (len(m0_vals) x len(lambda_vals)) grid of speeds and plot as heatmap.
    Missing runs are NaN (shown as blank).
    """
    # Assemble matrix (rows=m0, cols=lambda)
    H = np.full((len(m0_vals), len(lambda_vals)), np.nan, dtype=float)

    for i, m0 in enumerate(m0_vals):
        for j, lam in enumerate(lambda_vals):
            run_dir = _get_run_dir(base_dir, lam, m0)
            H[i, j] = _load_speed(run_dir, which=which_speed)

    # Plot
    fig, ax = plt.subplots(figsize=(1.1*len(lambda_vals)+2, 0.7*len(m0_vals)+2))
    im = ax.imshow(H, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    # Ticks/labels
    ax.set_yticks(np.arange(len(m0_vals)))
    ax.set_yticklabels([f"{m0:g}" for m0 in m0_vals])
    ax.set_ylabel(n_label, fontsize=12)

    ax.set_xticks(np.arange(len(lambda_vals)))
    if log_lambda_ticks:
        ax.set_xticklabels([rf"$10^{{{np.log10(lam):.0f}}}$" if lam>0 else "0" for lam in lambda_vals], rotation=0)
    else:
        ax.set_xticklabels([f"{lam:g}" for lam in lambda_vals], rotation=0)
    ax.set_xlabel(x_label, fontsize=12)

    ax.set_title(title, fontsize=14)

    # Gridlines to separate cells (optional)
    ax.set_xticks(np.arange(-.5, len(lambda_vals), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(m0_vals), 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.6, alpha=0.6)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Wave speed $c$", fontsize=12)

    # Optional annotations
    if annotate:
        for i in range(len(m0_vals)):
            for j in range(len(lambda_vals)):
                val = H[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=9, color='w')

    plt.tight_layout()
    plt.show()

    return H  # return the matrix in case you want to save/export it