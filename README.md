# ECM–Tumour Invasion

This repository contains code for simulating and analysing the tumour–ECM invasion model.  

This is a **double-degenerate cross-diffusion model** with singular nonlinear diffusion and ECM regeneration:

$$
\frac{\partial u}{\partial t} = \frac{partial }{partial x} \\left(m(1-m)\,\frac{partial u}{partial x}) + \,u\left(1-u\right),
\qquad
\frac{\partial m}{\partial t} = \alpha(1-m) - \lambda\,u m.
$$

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
