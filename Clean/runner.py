"""
This script runs multiple simulations of the 1D tumour–ECM model PDE
(using the Tumour_ECM_1D class defined in Model.py) over a grid of
parameters. Simulations can be executed in parallel, and results
are saved in a structured directory format:

results/
    lambda_<val>/
        alpha_<val>/
            m0_<val>/
                u0_<val>/
                    summary.json
                    fronts.npz
                    snapshots.npz

Example usage from the command line:
    python runner.py --lambdas 0.01,0.1,1 --alphas 0.1,1,10 \
                     --m0s 0.1,0.5 --u0s 0.3,0.7 \
                     --outdir results --jobs 4

Notes:
- The --jobs option controls parallel runs. 
  For example: --jobs 1 runs sequentially, --jobs 4 runs on 4 cores,
  and --jobs -1 uses all available cores.
"""

import os, json, argparse
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed

from Model import Tumour_ECM_1D


# save / path helpers
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _atomic_write_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def _save_summary(run_dir: Path, meta: dict):
    with open(run_dir / "summary.json", "w") as f:
        json.dump(meta, f, indent=2)

def _save_fronts(run_dir: Path, t_fronts, x_fronts, name=None):
    fname = "fronts.npz" if not name else f"fronts_{name}.npz"
    np.savez_compressed(run_dir / fname,
                        t_fronts=np.asarray(t_fronts),
                        x_fronts=np.asarray(x_fronts))

def _save_snapshots_every_stride(run_dir: Path, model, stride=150):
    idx = np.unique(np.concatenate([
        np.arange(0, model.Nt, stride),
        np.array([model.Nt - 1])
    ]))
    np.savez_compressed(
        run_dir / "snapshots.npz",
        x=model.x,
        times=model.times[idx],
        N_arr=model.N_arr[idx, :],
        M_arr=model.M_arr[idx, :]
    )

def _fmt_val(v):
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer()):
        return f"{int(v)}"
    s = f"{v}"
    return s.rstrip('0').rstrip('.') if '.' in s else s


# single-run worker
def run_one(lam, alpha, m0, u0,
            base_dir="results",
            model_kwargs=None,
            overwrite=False,
            snapshot_stride=150):
    if model_kwargs is None:
        model_kwargs = {}
    try:
        local_kwargs = dict(model_kwargs)
        local_kwargs["n0"] = float(u0)

        run_dir = (Path(base_dir)
                   / f"lambda_{_fmt_val(lam)}"
                   / f"alpha_{_fmt_val(alpha)}"
                   / f"m0_{_fmt_val(m0)}"
                   / f"u0_{_fmt_val(u0)}")
        _ensure_dir(run_dir)

        if not overwrite and (run_dir / "summary.json").exists():
            return ("skipped", lam, alpha, m0, u0)

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

        model = Tumour_ECM_1D(k=lam, alpha=alpha, m0=m0, **local_kwargs)
        model.solve()

        c, _, r2 = model.estimate_wave_speed(threshold=0.5, band=(0.1, 0.9), spline_type='cubic', plot=False, target='N')
        if c is None or (isinstance(c, float) and np.isnan(c)):
            raise ValueError("wave speed could not be calculated")
        model.wave_speed = c

        t_fronts, x_fronts = model.track_wavefront_local_interpolation(threshold=0.5, band=(0.1, 0.9), spline_type='cubic', target='N')

        _save_summary(run_dir, dict(
            lambda_val=float(lam), alpha=float(alpha),
            m0=float(m0), u0=float(u0),
            wave_speed=float(c),
            r2=(float(r2) if r2 is not None else None),
            dt=model.dt, T=model.T, L=model.L, N=model.N,
            init_type=model.init_type,
            steepness=getattr(model, "steepness", None),
            perc=getattr(model, "perc", None),
            t_start=model.t_start, t_end=model.t_end,
            num_points=getattr(model, "num_points", None),
            saved_stride=int(snapshot_stride)
        ))
        _save_fronts(run_dir, t_fronts, x_fronts, name="N")
        _save_snapshots_every_stride(run_dir, model, stride=snapshot_stride)

        return ("done", lam, alpha, m0, u0, float(c), (float(r2) if r2 is not None else None))

    except Exception as e:
        return ("failed", lam, alpha, m0, u0, str(e))


# parallel grid
def run_grid(lambda_vals, alpha_vals, m0_vals, u0_vals,
             base_dir="results",
             model_kwargs=None,
             overwrite=False,
             snapshot_stride=150,
             n_jobs=-1, verbose=10):
    if model_kwargs is None:
        model_kwargs = {}

    tasks = [(lam, alpha, m0, u0)
             for lam in lambda_vals
             for alpha in alpha_vals
             for m0 in m0_vals
             for u0 in u0_vals]

    results = Parallel(n_jobs=n_jobs, verbose=verbose, backend="loky")(
        delayed(run_one)(
            lam, alpha, m0, u0,
            base_dir=base_dir,
            model_kwargs=model_kwargs,
            overwrite=overwrite,
            snapshot_stride=snapshot_stride
        ) for lam, alpha, m0, u0 in tasks
    )

    done, skipped, failed, low_r2 = [], [], [], []
    for r in results:
        tag = r[0]
        if tag == "done":
            _, lam, a_eff, m0_eff, u0_eff, c, r2 = r
            done.append({"lambda": lam, "alpha": a_eff, "m0": m0_eff, "u0": u0_eff, "c": c, "r2": r2})
            if (r2 is None) or (isinstance(r2, float) and (np.isnan(r2) or r2 < .999)):
                low_r2.append({"lambda": lam, "alpha": a_eff, "m0": m0_eff, "u0": u0_eff, "c": c, "r2": r2})
        elif tag == "skipped":
            _, lam, a_eff, m0_eff, u0_eff = r
            skipped.append({"lambda": lam, "alpha": a_eff, "m0": m0_eff, "u0": u0_eff})
        elif tag == "failed":
            _, lam, a_orig, m0_orig, u0_orig, msg = r
            failed.append({"lambda": lam, "alpha": a_orig, "m0": m0_orig, "u0": u0_orig, "error": msg})

    base = Path(base_dir); _ensure_dir(base)
    _atomic_write_json(base / "failed_runs.json", failed)
    _atomic_write_json(base / "low_r2_runs.json", low_r2)

    print(f"Done: {len(done)}, Skipped: {len(skipped)}, Failed: {len(failed)}, Low-R2: {len(low_r2)}")
    return {"done": done, "skipped": skipped, "failed": failed, "low_r2": low_r2}


# CLI
def _parse_floats_list(vals):
    out = []
    for v in vals:
        if "," in v:
            out.extend([float(x) for x in v.split(",") if x])
        else:
            out.append(float(v))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", nargs="+", required=True)
    ap.add_argument("--alphas",  nargs="+", required=True)
    ap.add_argument("--m0s",     nargs="+", required=True)
    ap.add_argument("--u0s",     nargs="+", required=True)  # u0 == n0
    ap.add_argument("--outdir",  type=str, default="results")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--stride", type=int, default=150)
    ap.add_argument("--L", type=float, default=200)
    ap.add_argument("--N", type=int,   default=20001)
    ap.add_argument("--T", type=float, default=400)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--init_type", type=str, default="tanh", choices=["tanh","step"])
    ap.add_argument("--steepness", type=float, default=0.85)
    ap.add_argument("--perc", type=float, default=0.4)
    ap.add_argument("--t_start", type=float, default=100)
    ap.add_argument("--t_end",   type=float, default=350)
    ap.add_argument("--num_points", type=int, default=250)
    ap.add_argument("--K", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--D", type=float, default=1.0)
    ap.add_argument("--Mmax", type=float, default=1.0)
    args = ap.parse_args()

    shared_kwargs = dict(
        L=args.L, N=args.N, T=args.T, dt=args.dt,
        init_type=args.init_type, steepness=args.steepness, perc=args.perc,
        t_start=args.t_start, t_end=args.t_end, num_points=args.num_points,
        K=args.K, rho=args.rho, D=args.D, Mmax=args.Mmax,
        n0=1.0
    )

    lambda_vals = _parse_floats_list(args.lambdas)
    alpha_vals  = _parse_floats_list(args.alphas)
    m0_vals     = _parse_floats_list(args.m0s)
    u0_vals     = _parse_floats_list(args.u0s)

    run_grid(lambda_vals, alpha_vals, m0_vals, u0_vals,
             base_dir=args.outdir,
             model_kwargs=shared_kwargs,
             overwrite=args.overwrite,
             snapshot_stride=args.stride,
             n_jobs=args.jobs, verbose=10)


