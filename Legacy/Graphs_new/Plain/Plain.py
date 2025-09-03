import os
import numpy as np
from scipy.sparse import eye, diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator, interp1d
from scipy.optimize import root_scalar
from scipy.stats import linregress
import matplotlib.pyplot as plt
from numba import njit

# ---------- Helpers ----------
def _latex_sci(val, pow10_threshold=100.0):
    """Format val for LaTeX titles; switch to scientific when |val| ≥ threshold."""
    if val == 0:
        return "0"
    a = abs(val)
    sign = "-" if val < 0 else ""
    if a < pow10_threshold:
        return f"{val:g}"
    exp = int(np.floor(np.log10(a)))
    mant = a / (10**exp)
    if np.isclose(mant, 1.0, rtol=1e-10, atol=1e-12):
        return rf"{sign}10^{{{exp}}}"
    return rf"{sign}{mant:.2g}\times 10^{{{exp}}}"

def _file_sci(val, pow10_threshold=100.0):
    """Compact number for filenames: 1000 -> '1e3', 2500 -> '2.5e3', 10 -> '10'."""
    if val == 0:
        return "0"
    a = abs(val)
    if a < pow10_threshold:
        return f"{int(val)}" if float(val).is_integer() else f"{val:g}"
    return f"{val:.0e}".replace("+0","").replace("+","").replace("e0","e")

def _m0_two_digits(m0):
    """m0 in [0,1] to two digits: 0.0->'00', 0.5->'05', 1.0->'10'."""
    return f"{int(round(m0*10)):02d}"

# ---------- Model kernels ----------
@njit
def f_numba(N, rho, K):
    # tumour growth: N_t = rho * N * (1 - N/K)
    return rho * N * (1 - N / K)

@njit
def build_laplacian_diagonals_avg(m, D, dx):
    """
    Variable-coefficient diffusion with edge-averaged M:
    (D * M*(1-M) * u_x)_x  with homogeneous Neumann BCs.
    Returns three diagonals (lower, center, upper) scaled by 1/dx^2.
    """
    N = len(m)
    lower = np.zeros(N)
    center = np.zeros(N)
    upper = np.zeros(N)

    for i in range(1, N - 1):
        ml = 0.5 * (m[i - 1] + m[i])
        mr = 0.5 * (m[i] + m[i + 1])
        Dl = max(1e-6, D * ml * (1 - ml))
        Dr = max(1e-6, D * mr * (1 - mr))
        lower[i] = Dl
        upper[i] = Dr
        center[i] = - (Dl + Dr)

    # Neumann BCs at x=0
    mr = 0.5 * (m[0] + m[1])
    Dr = max(1e-6, D * mr * (1 - mr))
    center[0] = -2 * Dr
    upper[0]  =  2 * Dr

    # Neumann BCs at x=L
    ml = 0.5 * (m[-2] + m[-1])
    Dl = max(1e-6, D * ml * (1 - ml))
    center[-1] = -2 * Dl
    lower[-1]  =  2 * Dl

    invdx2 = 1.0 / dx**2
    return invdx2 * lower, invdx2 * center, invdx2 * upper

# ---------- Main class ----------
class Dissertation_Plain_1D:
    def __init__(self, D=1.0, rho=1.0, K=1.0, k=1.0,
                 n0=1.0, m0=0.5, Mmax=1.0, perc=0.2,
                 L=1000.0, N=5001, T=1000.0, dt=0.1,
                 scheme="AB2AM2", init_type="step", steepness=0.1,
                 t_start=50.0, t_end=500.0, num_points=200):
        # PDE/ODE params
        self.D = D; self.rho = rho; self.K = K
        self.k = k; self.n0 = n0; self.m0 = m0
        self.Mmax = Mmax; self.perc = perc
        self.steepness = steepness

        # grid/time
        self.L = L; self.N = N; self.dx = L / (N - 1)
        self.x = np.linspace(0, L, N)
        self.T = T; self.dt = dt; self.Nt = int(T / dt)
        self.scheme = scheme.upper()
        self.init_type = init_type

        # storage
        self.times = np.linspace(0, T, self.Nt)
        self.N_arr = np.zeros((self.Nt, self.N))
        self.M_arr = np.zeros((self.Nt, self.N))
        self.wave_speed = None

        # wave speed tracking params
        self.t_start = t_start
        self.t_end = t_end
        self.num_points = num_points

    # ----- ICs -----
    def initial_condition(self):
        if self.init_type == "step":
            N0 = self.n0 * np.where(self.x < self.perc * self.L, 0.7, 0.0)
        elif self.init_type == "tanh":
            N0 = self.n0 * 0.5 * (1 - np.tanh(self.steepness * (self.x - self.perc * self.L)))
        else:
            raise ValueError("Unknown initial condition type.")
        M0 = self.m0 * self.Mmax * np.ones_like(self.x)
        return N0, M0

    # ----- Diffusion operator -----
    def update_laplacian(self, M):
        lower, center, upper = build_laplacian_diagonals_avg(M, self.D, self.dx)
        return diags([lower[1:], center, upper[:-1]], [-1, 0, 1], format="csr")

    # ----- Time stepping -----
    def solve(self):
        """
        N: AB2–AM2 (IMEX)
        M: implicit Euler exact form for dm/dt = -k*N*m with N taken at t^{n+1}:
           M^{n+1} = M^n / (1 + k*dt*N^{n+1})
        """
        # initial data
        N_prev, M_prev = self.initial_condition()
        f_prev = f_numba(N_prev, self.rho, self.K)
        L_prev = self.update_laplacian(M_prev)

        # first step for N: implicit Euler using M_prev
        A0 = (eye(self.N) - self.dt * L_prev)
        N_curr = spsolve(A0.tocsc(), N_prev + self.dt * f_prev)

        # first step for M: implicit Euler with N_curr
        denom = 1.0 + self.k * self.dt * np.maximum(N_curr, 0.0)
        M_curr = M_prev / denom
        np.clip(M_curr, 0.0, self.Mmax, out=M_curr)

        # store first two frames
        self.N_arr[0], self.M_arr[0] = N_prev, M_prev
        self.N_arr[1], self.M_arr[1] = N_curr, M_curr

        # main loop
        for i in range(2, self.Nt):
            # assemble with current M for N's diffusion
            L_curr = self.update_laplacian(M_curr)
            f_curr = f_numba(N_curr, self.rho, self.K)

            # AB2–AM2 for N
            rhs = (eye(self.N) + 0.5 * self.dt * L_prev) @ N_curr + self.dt * (1.5 * f_curr - 0.5 * f_prev)
            A = (eye(self.N) - 0.5 * self.dt * L_curr)
            N_next = spsolve(A.tocsc(), rhs)

            # enforce Neumann ends by copying interior neighbors
            N_next[0], N_next[-1] = N_next[1], N_next[-2]

            # implicit Euler for M with N_next
            denom = 1.0 + self.k * self.dt * np.maximum(N_next, 0.0)
            M_next = M_curr / denom
            np.clip(M_next, 0.0, self.Mmax, out=M_next)

            # store
            self.N_arr[i] = N_next
            self.M_arr[i] = M_next

            # roll
            N_prev, N_curr = N_curr, N_next
            M_prev, M_curr = M_curr, M_next
            f_prev = f_curr
            L_prev = L_curr

    # ----- Front tracking -----
    def track_wavefront_local_interpolation(self, threshold=0.5, band=(0.1, 0.9),
                                            spline_type='cubic', target='N'):
        x = self.x
        t_vec = self.times
        u_arr = self.N_arr if target.lower() == 'n' else self.M_arr
        t_list = np.linspace(self.t_start, self.t_end, self.num_points)
        x_fronts, t_fronts = [], []

        def get_spline(method, x, y):
            if method == 'cubic': return CubicSpline(x, y)
            if method == 'pchip': return PchipInterpolator(x, y)
            if method == 'akima': return Akima1DInterpolator(x, y)
            if method == 'linear': return interp1d(x, y, kind='linear', fill_value="extrapolate")
            raise ValueError(f"Unsupported spline_type: {method}")

        for t_target in t_list:
            idx = np.argmin(np.abs(t_vec - t_target))
            u = u_arr[idx]
            mask = (u > band[0]) & (u < band[1])
            if np.sum(mask) < 5:
                continue
            x_local, u_local = x[mask], u[mask]
            sort_idx = np.argsort(x_local)
            x_local, u_local = x_local[sort_idx], u_local[sort_idx]
            spline = get_spline(spline_type, x_local, u_local)
            crossing_idx = np.where(np.sign(u_local[:-1] - threshold) != np.sign(u_local[1:] - threshold))[0]
            if len(crossing_idx) == 0:
                continue
            i = crossing_idx[0]
            x_left, x_right = x_local[i], x_local[i + 1]
            try:
                sol = root_scalar(lambda xv: spline(xv) - threshold, bracket=[x_left, x_right])
                if sol.converged:
                    x_star = sol.root
                    x_fronts.append(x_star)
                    t_fronts.append(t_target)
            except Exception:
                continue

        return np.array(t_fronts), np.array(x_fronts)

    def estimate_wave_speed(self, threshold=0.5, band=(0.1, 0.9),
                            spline_type='cubic', plot=True, target='N'):
        t_fronts, x_fronts = self.track_wavefront_local_interpolation(threshold, band, spline_type, target)
        if len(t_fronts) < 2:
            print("❌ Not enough valid front points.")
            return None, None, None

        slope, intercept, r_value, _, _ = linregress(t_fronts, x_fronts)

        if plot:
            plt.figure(figsize=(8, 4))
            plt.plot(t_fronts, x_fronts, 'o', label='Front')
            plt.plot(t_fronts, slope * t_fronts + intercept, 'k--', label=f'Slope = {slope:.3f}')
            plt.xlabel("Time t")
            plt.ylabel("Wavefront x(t)")
            plt.title("Wave Speed via Linear Fit")
            plt.legend(); plt.grid(True); plt.tight_layout()
            plt.show()

        return slope, intercept, r_value**2

    # ----- Publication-ready plot with axis presets -----
    def plot_u_m_with_custom_style(self, 
                               t_points=[0, 100, 200, 300],
                               yticks_mode="basic",  # "basic" | "split" | "splitplus"
                               show_arrows=True,
                               show_speed_text=True,
                               print_speed=False,
                               ceil_speed=False,  # False | "up" | "down"
                               arrow_len=None, arrow_lw=2.5,
                               arrow_x_frac=0.7,       # <-- NEW: default 60% along domain
                               y_red=0.8,              # <-- NEW: default vertical pos for red arrow
                               y_blue=0.25,             # <-- NEW: default vertical pos for blue arrow
                               head_length=1.5, head_width=0.65,  # publication styling
                               save=False, folder="Plots_Plain", filename=None):
        """
        Plot u and m at given *times* (not indices) with publication settings.

        Parameters
        ----------
        t_points : list[float]
            Times (in the same units as self.times) to plot snapshots.
        yticks_mode : {"basic","split","splitplus"}
            "basic"     -> ticks [0, 0.5, 1.0],   ylim [0, 1.0]
            "split"     -> ticks [0:0.2:1.0],     ylim [0, 1.0]
            "splitplus" -> ticks [0:0.2:1.2],     ylim [0, 1.2]
        show_arrows : bool
            Draw direction arrows (red at y=y_red, blue at y=y_blue).
        arrow_x_frac : float
            Fractional x-position of arrows (0=left, 1=right). Default 0.6.
        y_red, y_blue : float
            Vertical positions of the arrows. Defaults: y_red=0.9, y_blue=0.6.
        show_speed_text : bool
            Annotate estimated wave speed c on the figure.
        print_speed : bool
            Print estimated speed c to console.
        ceil_speed : {False, "up", "down"}
            False -> format c with 3 significant figures.
            "up"  -> ceil c to 2 decimal places.
            "down"-> floor c to 2 decimal places.
        save : bool
            If True, save to `folder/filename` (PNG). If filename is None, auto-named.
        folder : str
            Output directory when save=True.
        filename : str | None
            Custom filename (no path). If None, auto-named.
        """
        import os, math
        import numpy as np
        import matplotlib.pyplot as plt

        # Data shortcuts
        x, N_arr, M_arr, t_vec = self.x, self.N_arr, self.M_arr, self.times

        # Map requested times -> nearest indices
        t_indices = [int(np.argmin(np.abs(t_vec - t))) for t in t_points]

        # Compute wave speed only if needed
        if (show_speed_text or print_speed) and getattr(self, "wave_speed", None) is None:
            self.wave_speed, _, _ = self.estimate_wave_speed(
                plot=False, target='N', threshold=0.5, band=(0.1, 0.9), spline_type='cubic'
            )
        if print_speed and (self.wave_speed is not None):
            print(f"[plot] Estimated wave speed c = {self.wave_speed:.6g}")

        # ---- speed string formatting ----
        c_str = "—"
        if self.wave_speed is not None:
            if ceil_speed == "down":
                c_str = f"{math.floor(self.wave_speed * 100) / 100:.2f}"
            elif ceil_speed == "up":
                c_str = f"{math.ceil(self.wave_speed * 100) / 100:.2f}"
            else:
                c_str = f"{self.wave_speed:.3g}"
        # ---------------------------------

        # Arrow geometry
        if arrow_len is None:
            arrow_len = 0.15 * self.L
        arrow_x_start = np.clip(arrow_x_frac * self.L, 0.0, self.L)
        arrow_x_end   = np.clip(arrow_x_start + arrow_len, 0.0, self.L)
        if arrow_x_end <= arrow_x_start:
            arrow_x_start = np.clip(self.L - arrow_len, 0.0, self.L)
            arrow_x_end   = self.L

        # Figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot profiles (dashed for t=0)
        for t, tidx in zip(t_points, t_indices):
            ls = '--' if np.isclose(t, 0.0) else '-'
            ax.plot(x, N_arr[tidx], color='red',  linestyle=ls, label=rf"$u(x,{int(t)})$")
            ax.plot(x, M_arr[tidx], color='blue', linestyle=ls, label=rf"$m(x,{int(t)})$")

        # Arrows
        if show_arrows:
            arrow_style_red  = dict(arrowstyle=f'->,head_length={head_length},head_width={head_width}',
                                    color='red',  lw=arrow_lw)
            arrow_style_blue = dict(arrowstyle=f'->,head_length={head_length},head_width={head_width}',
                                    color='blue', lw=arrow_lw)
            ax.annotate('', xy=(arrow_x_end, y_red),  xytext=(arrow_x_start, y_red),  arrowprops=arrow_style_red)
            ax.annotate('', xy=(arrow_x_end, y_blue), xytext=(arrow_x_start, y_blue), arrowprops=arrow_style_blue)

        # Corner text
        x_text = x[0] + 0.02 * self.L
        ax.text(x_text, 0.92, rf"$\overline{{m}} = {self.m0}$", fontsize=18, ha='left')
        if show_speed_text:
            ax.text(x_text, 0.82, rf"$c = {c_str}$", fontsize=18, ha='left')

        # Axes + title
        ax.set_xlabel(r"$x$", fontsize=18)
        ax.set_ylabel(r"$u(x,t),\, m(x,t)$", fontsize=18)
        ax.set_xlim([0, self.L])

        mode = str(yticks_mode).lower()
        if mode == "basic":
            ax.set_ylim([0, 1.05]); ax.set_yticks([0.0, 0.5, 1.0])
        elif mode == "split":
            ax.set_ylim([0, 1.05]); ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        elif mode == "splitplus":
            ax.set_ylim([0, 1.25]); ax.set_yticks(np.arange(0.0, 1.21, 0.2))
        else:
            ax.set_ylim([0, 1.05]); ax.set_yticks([0.0, 0.5, 1.0])

        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=16)
        lam_str = _latex_sci(self.k, pow10_threshold=100.0)
        ax.set_title(rf"$\lambda = {lam_str}$", fontsize=20)
        ax.grid(False)
        fig.tight_layout()

        # Save or show
        if save:
            os.makedirs(folder, exist_ok=True)
            if filename is None:
                m0_str = _m0_two_digits(self.m0)
                lam_file = _file_sci(self.k, 100.0)
                filename = f"plain_{m0_str}_lam{lam_file}.png"
            outpath = os.path.join(folder, filename)
            fig.savefig(outpath, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[plot] Figure saved to {outpath}")
        else:
            plt.show()