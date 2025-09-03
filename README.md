
# ECM–Tumour Invasion

This repository contains code for simulating and analysing the tumour–ECM invasion model.  

This is a **double-degenerate cross-diffusion model** with singular nonlinear diffusion and ECM regeneration:

\[
u_t = \partial_x \!\left(D\,m(1-m)\,\partial_x u\right) + \rho u(1-u/K),
\qquad
m_t = \alpha(1-m) - \lambda u m.
\]

Currently the code is implemented for **1D analysis**, but extensions to **2D simulations** will be added in future updates.

The solver is written in Python with support for parameter sweeps, parallel execution, and structured result storage.  

---

## Folder Structure

- **Clean/**  
  Actively maintained code:
  - `Model.py` – implementation of the `Tumour_ECM_1D` solver  
  - `runner.py` – command-line runner to launch parameter sweeps  
  - `requirements.txt` – Python dependencies  

- **Legacy/**  
  Older experiments, plots, and datasets retained for reference.  

The **Legacy** folder is only meant for viewing older plots and experiments.  
To run new simulations, only the **Clean** folder is required.

---

## Requirements

Install dependencies (preferably inside a virtual environment to avoid conflicts):

```bash
pip install -r Clean/requirements.txt

## Usage
You can run parameter sweeps for the model directly from the terminal.

Example:

    python Clean/runner.py \
        --lambdas 0.01,0.1,1 \
        --alphas 0.1,1,10 \
        --m0s 0.1,0.5 \
        --u0s 0.3,0.7 \
        --outdir results \
        --jobs 4

If you get an error with "python", try "python3" instead.

## Output

Results are stored in the folder specified by --outdir (default: results/) with the structure:

    results/
        lambda_<val>/
            alpha_<val>/
                m0_<val>/
                    u0_<val>/
                        summary.json
                        fronts_N.npz
                        snapshots.npz

- summary.json – metadata (parameters, wave speed estimate, R² fit, etc.)
- fronts_N.npz – tumour front locations over time (interpolated)
- snapshots.npz – tumour and ECM profiles at sampled times

## Notes

- Parallel runs: adjust --jobs to change the number of parallel workers (e.g. --jobs 8).
- Custom runs: import Tumour_ECM_1D from Model.py in a Python script or notebook to run individual simulations.
- Data size: results can be large — use separate --outdir folders to stay organised.
