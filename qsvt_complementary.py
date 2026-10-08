import numpy as np
import scipy
import matplotlib.pyplot as plt
import qsppack as qsp
import math
import time

import sys
sys.setrecursionlimit(100000)


NORM_LIMIT = 0.3

############################################################################## POLYNOMIAL TOOLS

def comp_sq(a_coef): #given a coef in laurent, return (1 - a^2) . 
    # the input polynomial should be symmetric, and either even or odd (with all zeroes removed).
    # note that the returned polynomial is even, i.e. has odd # of coefficients.
    comp_coef = -scipy.signal.fftconvolve(a_coef, a_coef, mode='full')
    comp_coef[len(a_coef) - 1] += 1
    return np.array(comp_coef)

def deriv(poly_coef): # derivative of even polynomial
    deriv_coef = []
    for i in range(1, len(poly_coef)):
        deriv_coef.append(poly_coef[i] * 2 * i)
    return np.array(deriv_coef)

############################################################################## FFT SAMPLING

def pad(coeffs, length):
    parity = 1 - len(coeffs) % 2
    coeffs_parity = np.zeros(2 * len(coeffs), dtype=coeffs.dtype)
    coeffs_parity[parity::2] = coeffs
    return np.pad(coeffs_parity, (0, length - len(coeffs_parity)))

def eval_g(h_coeffs, n):
    h_eval= np.fft.fft(pad(h_coeffs,n))
    dh_eval = np.fft.fft(pad(deriv(h_coeffs),n))
    raw_output = dh_eval/h_eval

    return np.concatenate([[raw_output[0]], raw_output[1:][::-1]]) # reverse, preserving first element

def eval_poly_laurent(coef, z):
    val = 0
    for i in range(len(coef)):
        val += coef[i] * (z ** i)
    return val

def poly_from_power_sums(power_sums, check=True):
    """
    O(d log d) conversion from power sums to polynomial coefficients,
    replacing the O(d^2) Newton-identity recurrence.

    Uses the identity E(x) = exp(L(x)), where E(x) = sum e_k x^k are the
    elementary symmetric polynomials and L(x) = sum (-1)^(k-1) p_k x^k / k,
    computed via Newton iteration on top of FFT-based polynomial multiplication.

    Returns coefficients lowest-degree first.

    Raises RuntimeError if float64 precision collapses (NaN/Inf) rather than
    silently returning garbage -- this happens when the true coefficients
    need more digits than a double can represent, which is a property of
    the input magnitude/n, not something more speed can fix. Pass
    check=False to skip this and get raw NaN/Inf instead.
    """
    def next_pow2(n):
        return 1 << max(n - 1, 0).bit_length()

    def pad(a, n):
        return a[:n] if len(a) >= n else np.pad(a, (0, n - len(a)))

    def mul(a, b, trunc):
        if len(a) == 0 or len(b) == 0:
            return np.zeros(trunc)
        n = next_pow2(len(a) + len(b) - 1)
        res = np.fft.irfft(np.fft.rfft(a, n) * np.fft.rfft(b, n), n)
        return pad(res[:len(a) + len(b) - 1], trunc)

    def inv(a, n):
        b = np.array([1.0 / a[0]])
        m = 1
        while m < n:
            m2 = min(2 * m, n)
            ab = mul(pad(a, m2), b, m2)
            t = -ab
            t[0] += 2.0
            b = mul(b, t, m2)
            m = m2
        return b[:n]

    def log(a, n):
        a = pad(a, n)
        da = a[1:] * np.arange(1, n) if n > 1 else np.zeros(0)
        ia = inv(a, max(n - 1, 1))
        prod = mul(da, ia, max(n - 1, 0))
        out = np.zeros(n)
        if n > 1:
            out[1:] = prod / np.arange(1, n)
        return out

    def exp(a, n):
        a = pad(a, n)
        b = np.array([1.0])
        m = 1
        while m < n:
            m2 = min(2 * m, n)
            diff = pad(a, m2) - log(b, m2)
            diff[0] += 1.0
            b = mul(b, diff, m2)
            m = m2
        return b[:n]

    
    print(f"Imaginary part, should be 0:\t{np.max(np.abs(power_sums.imag))}")
    p = np.real(power_sums)
    n = len(p)
    if n == 0:
        return np.array([1.0])

    k = np.arange(1, n + 1)
    L = np.zeros(n + 1)
    L[1:] = np.where(k % 2 == 1, 1.0, -1.0) * p / k

    with np.errstate(over="ignore", invalid="ignore"):
        e = exp(L, n + 1)

    if check and not np.all(np.isfinite(e)):
        raise RuntimeError(
            f"poly_from_power_sums: float64 precision collapsed at n={n}. "
            "This means the true coefficients need more precision than a "
            "double can hold -- not fixable by further speed optimization. "
            "Pass check=False to get raw NaN/Inf back instead."
        )

    coeffs = ((-1.0) ** np.arange(n + 1)) * e
    return coeffs[::-1]

_PEEL_BASE = 64  # switch to the naive loop below this many steps


def _poly_mul(a, b):
    if len(a) == 1:
        return a[0] * b
    if len(b) == 1:
        return b[0] * a
    if min(len(a), len(b)) < 500:
        return np.convolve(a, b)
    return scipy.signal.fftconvolve(a, b)


def _mat_mul(A, B):
    a00, a01, a10, a11 = A
    b00, b01, b10, b11 = B
    return (_poly_mul(a00, b00) + _poly_mul(a01, b10),
            _poly_mul(a00, b01) + _poly_mul(a01, b11),
            _poly_mul(a10, b00) + _poly_mul(a11, b10),
            _poly_mul(a10, b01) + _poly_mul(a11, b11))


def _apply_transfer(G, targ, supp, outlen):
    g00, g01, g10, g11 = G
    k = len(g00) - 1

    def corr(g, v):
        out = np.zeros(outlen, dtype=complex)
        gr = g[::-1]
        for par in (0, 1):
            vp = v[par::2]
            nm = len(range(par, outlen, 2))
            if len(g) == 1:
                seg = g[0] * vp[:nm]
            else:
                seg = _poly_mul(vp, gr)[k:k + nm]
            out[par:outlen:2] = seg[:nm]
        return out

    return (corr(g00, targ) + corr(g01, supp),
            corr(g10, targ) + corr(g11, supp))


def _peel(targ, supp):
    # Returns (reflection_coeffs, G, targ0) where G is the combined 2x2
    # transfer matrix (polynomial in w = z^-2) for all steps performed and
    # targ0 is the fully reduced scalar targ[0].
    L = len(targ)
    nsteps = (L - 1) // 2

    if nsteps <= _PEEL_BASE:
        G = (np.array([1.0 + 0j]), np.array([0.0 + 0j]),
             np.array([0.0 + 0j]), np.array([1.0 + 0j]))
        peeled = []
        for _ in range(nsteps):
            p = targ[-1] / supp[-1]
            peeled.append(p)
            f = np.sqrt(p)
            step = (np.array([0.5 / f, 0.5 / f]), np.array([-0.5 * f, 0.5 * f]),
                    np.array([-0.5 / f, 0.5 / f]), np.array([0.5 * f, 0.5 * f]))
            G = _mat_mul(step, G)
            targ, supp = (
                0.5 * ((targ[:-2] + targ[2:]) / f + f * (supp[2:] - supp[:-2])),
                0.5 * ((targ[2:] - targ[:-2]) / f + f * (supp[:-2] + supp[2:])),
            )
        return peeled, G, targ[0]

    K = nsteps // 2
    M = 2 * K + 1
    peeled1, G1, _ = _peel(targ[-M:], supp[-M:])
    targ2, supp2 = _apply_transfer(G1, targ, supp, L - 2 * K)
    peeled2, G2, targ0 = _peel(targ2, supp2)
    return peeled1 + peeled2, _mat_mul(G2, G1), targ0


def phases_from_polys(targ_poly, supp_poly):
    targ = np.asarray(targ_poly, dtype=complex)
    supp = np.asarray(supp_poly, dtype=complex)
    peeled, _, targ0 = _peel(targ, supp)
    peeled.append(targ0 ** 2)
    peeled.reverse()
    return 0.5 * np.angle(np.asarray(peeled))

########################################################################### MAIN FUNCTION

def poly_approx(a, n_s, parity, thres, known_degree = None):
    n_s = n_s - (n_s % 2) 
    func_val = a(np.cos(np.linspace(0, 2*np.pi, n_s, endpoint=False)))
    scale_factor = NORM_LIMIT /np.max(np.abs(func_val))
    func_val = func_val * scale_factor #/ np.max(np.abs(func_val)) * scale_factor
    func_coef = np.fft.fft(func_val)
    func_coef_abs = np.abs(func_coef[:n_s//2])

    # identify suitable d and truncate
    if known_degree == None:
        threshold = thres * np.max(func_coef_abs)
        d = np.where(func_coef_abs > threshold)[0][-1]
    else:
        d = int(known_degree)
    if (d % 2) != parity:
        d+=1

    freq = np.concatenate((
        np.arange(n_s // 2),
        np.arange(-n_s // 2, 0)
    ))
    #print(d)
    func_coef[np.abs(freq) > d] = 0
    #print(func_coef[100])
    func_coef /= n_s
    cheb_coef = np.zeros(d + 1)
    cheb_coef[0] = func_coef[0].real
    cheb_coef[1:] = 2 * func_coef[1:d+1].real
    #trig_approx = np.fft.ifft(func_coef)
    return cheb_coef, d, scale_factor
    

def solve(a, n_samples, parity, thres = 1e-12, known_deg = None):
    a_coef_full, deg, scale_factor = poly_approx(a, n_samples, parity, thres, known_deg)
    start_time = time.perf_counter() #start timer after polynomial approximation

    print(f"degree: \t{deg}, scale: \t{scale_factor}")
    print(f"Should be 0 due to parity:\t{max(np.abs(a_coef_full[parity+1::2]))}")
    a_coef = a_coef_full[parity::2]
    a_coef_laurent = qsp.nlfa.b_from_cheb(a_coef, parity)
    g_coef_laurent = comp_sq(a_coef_laurent)

    ###################################################################################################################CHANGE ABOVE


    ######################################################################################################
    eta = 0.05
    d = 2* deg
    n_points = max(math.ceil(d / eta), 4 * d + 2)
    output_points = eval_g(g_coef_laurent,n_points)
    power_sums = np.fft.ifft(output_points)

    print("power sums constructed")

    ######################################################################################################

    #INCREASING ORDER BTW
    e_coef_laurent = poly_from_power_sums(power_sums[2:d+2]) # e(z) * z^3
    e_1z_coef_laurent = e_coef_laurent[::-1] # e(1/z) * (1/z)^3

    print("characteristic polynomial constructed")



    pt = 1 #pt=1 allows us to cancel a lot of pt^d powers
    alpha = (eval_poly_laurent(g_coef_laurent, pt**2))/eval_poly_laurent(e_coef_laurent,pt) / eval_poly_laurent(e_1z_coef_laurent,pt)
    #alpha should have negligible imaginary part
    rt_alpha = np.real(alpha) ** 0.5

    c_coef = rt_alpha * (e_coef_laurent + e_1z_coef_laurent) * (0.5) # /2
    d_coef = rt_alpha * (e_coef_laurent - e_1z_coef_laurent) * (-0.5j) # /2i


    a_coef_laurent_full = np.zeros(2 * len(a_coef_laurent))
    a_coef_laurent_full[1::2] = a_coef_laurent
    a_coef_laurent_full = a_coef_laurent_full[1:]

    targ_coef_full = a_coef_laurent_full + 1j * c_coef

    print("target and supplementary polynomials constructed")

    gammas = phases_from_polys(targ_coef_full,1j*d_coef)

    out = {
        'targetPre': True,    # Compute the real part (Pre) of the matrix entry
        'parity': parity,     # Pass the function's parity (0 for even, 1 for odd)
        'typePhi': 'full'     # Your root-finding outputs the complete set of phases
    }

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print("done. time elapsed: " + str(elapsed_time) + " s")

    #From claude: copy xlist of Ying's paper
    N = n_samples - (n_samples % 2)                     # same sampling grid as poly_approx
    ts = 2 * np.pi * np.arange(N) / N # 0 to 2pi
    buf = np.zeros(N); buf[:len(a_coef_full)] = a_coef_full
    a_grid = np.fft.fft(buf).real                     # a(cos t_n) on the whole grid
    #L = 2 * deg + 2
    rng = np.random.default_rng(0)
    PK = np.sort(np.append(rng.integers(0, N, 20), np.argmax(np.abs(a_grid))))
    xlist = np.cos(ts[PK]) #0 to 2pi
    ###########################


    func = lambda x: qsp.utils.chebyshev_to_func(x, a_coef, parity, True)
    #targ_value = scale_factor * a(xlist) #original function
    cheb_value = func(xlist) #chebyshev approximation
    QSP_value = qsp.utils.get_entry(xlist, np.array(gammas), out) #phases approximation
    phase_err = np.linalg.norm(QSP_value - cheb_value, np.inf)/NORM_LIMIT #rescale error to fit norm 1
    #cheb_err = np.linalg.norm(targ_value - cheb_value, np.inf)/NORM_LIMIT
    print('The residual error is')
    print(phase_err)
    #print(cheb_err)

    #plt.plot(xlist, QSP_value - func_value)
    #plt.xlabel('$x$', fontsize=12)
    #plt.ylabel('$g(x,\\Phi^*)-f_\\mathrm{poly}(x)$', fontsize=12)
    #plt.show()

    info = {
        "time": elapsed_time,
        "phase_error": phase_err,
        #"cheb_error": cheb_err,
        "deg": deg,
        "scale": scale_factor,
        "gammas":gammas
    }
    return info

def plot_graphs(func, parity, input_vals, n_vals, var_name, title, known_degrees = None, graph_val = None, thres = 1e-12):
    print("Starting work on..." + title)
    if len(input_vals) != len(n_vals):
        print(len(input_vals))
        print(len(n_vals))
        raise Exception("Must have same number of inputs and sampling sizes")
    times=[]
    phase_errors = []
    degs = []
    for i in range(len(input_vals)):
        print("_" * 50)
        if known_degrees is None:
            info = solve(lambda x: func(x,input_vals[i]), n_vals[i], parity, thres) 
        else:
            info = solve(lambda x: func(x,input_vals[i]), n_vals[i], parity, thres, known_degrees[i]) 
        times.append(info["time"])
        phase_errors.append(info["phase_error"])
        degs.append(info["deg"])
        if i == 0:
            scale = info["scale"]
    fig, ((g_func, g_d),(g_t,g_pe)) = plt.subplots(2,2, figsize = (6,6))
    fig.suptitle(title)

    xlist = np.linspace(-1, 1, 1000)
    if graph_val is None:
        graph_val = input_vals[0]
    a_value = scale * func(xlist, graph_val)
    g_func.plot(xlist, a_value)

    g_func.set_xlabel('x', fontsize=12)
    g_func.set_ylabel('a(x)', fontsize=12)
    g_func.axhline(y=0, color='black', linestyle='-', linewidth=1)
    g_func.axvline(x=0, color='black', linestyle='-', linewidth=1)
    g_func.set_title("a)", loc='left', fontsize=12, fontweight='bold')

    g_d.plot(input_vals, degs,marker='o')
    g_d.set_xlabel(var_name, fontsize=12)
    g_d.set_ylabel('d', fontsize=12)
    g_d.set_title("b)", loc='left', fontsize=12, fontweight='bold')

    g_t.plot(input_vals, times,marker='o')
    g_t.set_xlabel(var_name, fontsize=12)
    g_t.set_ylabel('time (s)', fontsize=12)
    g_t.set_title("c)", loc='left', fontsize=12, fontweight='bold')

    g_pe.plot(input_vals, phase_errors,marker='o')
    g_pe.set_xlabel(var_name, fontsize=12)
    g_pe.set_ylabel('phase factor error', fontsize=12)
    g_pe.set_title("d)", loc='left', fontsize=12, fontweight='bold')


    plt.tight_layout()
    #plt.show()
    plt.savefig(title + ".pdf", format="pdf")

def ham_sim():
    taus = 1000 * np.arange(1, 6)
    hamiltonian_sim_re = lambda x, tau: np.cos(tau * x)
    plot_graphs(hamiltonian_sim_re, 0, taus, 40 * taus,'\u03c4', 'Hamiltonian simulation (Real)', graph_val=25)
    hamiltonian_sim_im = lambda x, tau: np.sin(tau * x)
    plot_graphs(hamiltonian_sim_im, 1, taus, 40 * taus,'\u03c4', 'Hamiltonian simulation (Imaginary)', graph_val=25)

def eig_fil():
    invdeltas = np.array([12.5, 25, 50, 100, 200]) # 100, 200
    t2_degs = 40 * invdeltas
    def eigen_filter(x, invdelta):
        delta = 1.0 / invdelta
        k = int(20 * invdelta)
        y  = -1 + 2 * (x**2 - delta**2) / (1 - delta**2)
        y0 = -1 - 2 * delta**2 / (1 - delta**2)      # normalization point (x = 0)
        u0 = np.arccosh(-y0)
        sign = (-1.0) ** k                           # T_k(y0) = (-1)^k cosh(k u0)

        out = np.empty_like(y)
        inside = np.abs(y) <= 1                      # |x| >= delta: T_k(y) = cos(k arccos y)
        out[inside] = np.cos(k * np.arccos(y[inside])) / (sign * np.cosh(k * u0))
        peak = ~inside                               # |x| < delta: y < -1, the (-1)^k signs cancel
        out[peak] = np.cosh(k * np.arccosh(-y[peak])) / np.cosh(k * u0)
        return out
    plot_graphs(eigen_filter, 0, invdeltas, (2000 * invdeltas).astype(int), "1/\u0394", 'Eigenstate filtering', t2_degs)

def mat_inv():
    kappas = np.array([16, 64, 256, 1024]) 
    matrix_inv = lambda x,kappa: 1/kappa*( (1-np.exp(-(5*kappa*x)**2))/x)
    plot_graphs(matrix_inv, 1, kappas, 1000 * kappas,'\u03BA', 'Matrix Inversion') 

def fer_dir():
    betas = 100 * np.array([1,2,4,8,16])
    fermi_dirac = lambda x, beta: -1 + 2/(1+np.exp(np.clip(beta*x, -709, 709)))#values too high break np.exp
    plot_graphs(fermi_dirac, 1, betas,200*betas,'\u03B2', 'Fermi-Dirac Operator') 

if __name__ == "__main__":
    ham_sim()
    eig_fil()
    mat_inv()
    fer_dir()
