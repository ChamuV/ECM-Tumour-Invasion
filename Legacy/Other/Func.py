import numpy as np
from scipy.sparse import eye, diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator, interp1d
from scipy.optimize import root_scalar
from scipy.stats import linregress
import os
import matplotlib.pyplot as plt
from numba import njit

@njit
def f_numba(N, rho, K):
    return rho * N * (1 - N / K)

@njit
def g_numba(N, M, k, alpha):
    return alpha*(1-M)-k * M * N

@njit
def build_laplacian_diagonals_avg(m, D, dx):
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

    # Neumann BCs (reflecting): symmetric two-point stencil
    mr = 0.5 * (m[0] + m[1])
    Dr = max(1e-6, D * mr * (1 - mr))
    center[0] = -2 * Dr
    upper[0] = 2 * Dr

    ml = 0.5 * (m[-2] + m[-1])
    Dl = max(1e-6, D * ml * (1 - ml))
    center[-1] = -2 * Dl
    lower[-1] = 2 * Dl

    return (1.0 / dx**2) * lower, (1.0 / dx**2) * center, (1.0 / dx**2) * upper

class Dissertation_Func_1D:
    def __init__(self, D=1.0, rho=1.0, K=1.0, k=1.0,
                 n0=1.0, m0=0.5, Mmax=1.0, perc=0.2,
                 L=1000.0, N=5001, T=1000.0, dt=0.1,
                 scheme="AB2AM2", init_type="step", steepness =0.1, alpha = 0.1):
        self.D = D; self.rho = rho; self.K = K
        self.k = k; self.n0 = n0; self.m0 = m0
        self.steepness = steepness
        self.Mmax = Mmax; self.perc = perc
        self.L = L; self.N = N; self.dx = L / (N - 1)
        self.x = np.linspace(0, L, N)
        self.T = T; self.dt = dt; self.Nt = int(T / dt)
        self.scheme = scheme.upper()
        self.init_type = init_type
        self.alpha = alpha
        self.times = np.linspace(0, T, self.Nt)
        self.N_arr = np.zeros((self.Nt, self.N))
        self.M_arr = np.zeros((self.Nt, self.N))
        self.wave_speed = None

    def initial_condition(self):
        if self.init_type == "step":
            N0 = self.n0 * np.where(self.x < self.perc * self.L, 0.7, 0.0)
        elif self.init_type == "tanh":
            N0 = self.n0 * 0.5 * (1 - np.tanh(self.steepness * (self.x - self.perc * self.L)))
        else:
            raise ValueError("Unknown initial condition type.")
        M0 = self.m0 * self.Mmax * np.ones_like(self.x)
        return N0, M0

    def update_laplacian(self, M):
        lower, center, upper = build_laplacian_diagonals_avg(M, self.D, self.dx)
        return diags([lower[1:], center, upper[:-1]], [-1, 0, 1], format="csr")

    def solve(self):
        N0, M0 = self.initial_condition()
        N_prev, M_prev = N0.copy(), M0.copy()
        f_prev = f_numba(N_prev, self.rho, self.K)
        g_prev = g_numba(N_prev, M_prev, self.k, self.alpha)
        L_prev = self.update_laplacian(M_prev)

        N_curr = spsolve(eye(self.N) - self.dt * L_prev, N_prev + self.dt * f_prev)
        M_curr = M_prev + self.dt * g_prev

        self.N_arr[0], self.M_arr[0] = N_prev, M_prev
        self.N_arr[1], self.M_arr[1] = N_curr, M_curr

        for i in range(2, self.Nt):
            L_curr = self.update_laplacian(M_curr)
            f_curr = f_numba(N_curr, self.rho, self.K)
            g_curr = g_numba(N_curr, M_curr, self.k, self.alpha)

            rhs = (eye(self.N) + 0.5 * self.dt * L_prev) @ N_curr + self.dt * (1.5 * f_curr - 0.5 * f_prev)
            A = (eye(self.N) - 0.5 * self.dt * L_curr)
            N_next = spsolve(A.tocsc(), rhs)
            N_next[0], N_next[-1] = N_next[1], N_next[-2]

            M_next = M_curr + self.dt * g_curr
            self.N_arr[i] = N_next
            self.M_arr[i] = M_next

            N_prev, N_curr = N_curr, N_next
            M_prev, M_curr = M_curr, M_next
            f_prev, g_prev = f_curr, g_curr
            L_prev = L_curr

    def track_wavefront_local_interpolation(self, t_start=10.0, t_end=400.0, num_points=200, threshold=0.5, band=(0.1, 0.9), spline_type='cubic', target='N'):
        x = self.x
        t_vec = self.times
        u_arr = self.N_arr if target.lower() == 'n' else self.M_arr
        t_list = np.linspace(t_start, t_end, num_points)
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
            u_interp = spline(x_local)
            crossing_idx = np.where(np.sign(u_interp[:-1] - threshold) != np.sign(u_interp[1:] - threshold))[0]
            if len(crossing_idx) == 0:
                continue
            i = crossing_idx[0]
            x_left, x_right = x_local[i], x_local[i + 1]
            try:
                sol = root_scalar(lambda x_val: spline(x_val) - threshold, bracket=[x_left, x_right])
                if sol.converged:
                    x_star = sol.root
                    x_fronts.append(x_star)
                    t_fronts.append(t_target)
            except:
                continue

        return np.array(t_fronts), np.array(x_fronts)

    def estimate_wave_speed(self, t_start=50.0, t_end=500.0, num_points=200, threshold=0.5, band=(0.1, 0.9), spline_type='cubic', plot=True, target='N'):
        t_fronts, x_fronts = self.track_wavefront_local_interpolation(t_start, t_end, num_points, threshold, band, spline_type, target)
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

        print(f"Estimated speed = {slope:.4f}, R² = {r_value**2:.4f}")
        return slope, intercept, r_value**2

    def plot_u_m_with_style(self, t_indices=[50, 75, 100]):
        x = self.x
        N_arr = self.N_arr
        M_arr = self.M_arr
        t_vec = self.times
        # Compute wave speed if not available
        if not hasattr(self, 'wave_speed') or self.wave_speed is None or np.isnan(self.wave_speed):
            self.wave_speed, _, _ = self.estimate_wave_speed(plot=False)
        cmap = plt.colormaps['tab10']
        plt.figure(figsize=(6, 6))
        for i, tidx in enumerate(t_indices):
            u = N_arr[tidx]
            m = M_arr[tidx]
            t = t_vec[tidx]
            color = cmap(i % 10)
            plt.plot(x, u, color=color, linestyle='-', label=rf"$u(x, {int(t)})$")
            plt.plot(x, m, color=color, linestyle='--', label=rf"$m(x, {int(t)})$")
        # Annotate front and wave speed
        front_idx = np.argmax(N_arr[t_indices[-1]] < 0.5)
        arrow_x_end = x[front_idx]
        arrow_x_start = arrow_x_end - 120
        plt.annotate('', xy=(arrow_x_end, 0.9), xytext=(arrow_x_start, 0.9),
                     arrowprops=dict(arrowstyle='->', color='black', lw=2))
        plt.annotate('', xy=(arrow_x_end, 0.1), xytext=(arrow_x_start, 0.1),
                     arrowprops=dict(arrowstyle='->', color='black', lw=2))
        # Text annotations
        plt.text(40, 0.85, rf"$\overline{{m}} = {self.m0}$", fontsize=12)
        plt.text(40, 0.75, rf"$c = {self.wave_speed:.2f}$", fontsize=12)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$u(x,t),\, m(x,t)$")
        plt.title(rf"$\lambda = {self.k}$", fontsize=13)
        #plt.legend(loc="lower right", fontsize=10)
        plt.grid(False)
        plt.xlim([0, self.L])
        plt.ylim([-0.05, 1.05])
        plt.tight_layout()
        plt.show()

# model = Chloe_Plain_1D(init_type="tanh", perc=0.2, m0 = 0.4, steepness = 0.1, dt = 0.1, T = 800)
#model.solve()
#model.plot_u_m_with_style([500, 750, 1000])