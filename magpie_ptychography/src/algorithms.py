import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import random

def downsample(vh):
    if len(vh.shape) == 1:
        nh = int(np.sqrt(vh.shape[0]))
        vh = vh.reshape(nh, nh)
    nh = vh.shape[0]
    nw = vh.shape[1]
    vH = 0.25 * (
        vh[0:nh:2, 0:nw:2] +
        vh[0:nh:2, 1:nw:2] +
        vh[1:nh:2, 0:nw:2] +
        vh[1:nh:2, 1:nw:2]
    )
    return vH

def upsample(vH):
    if len(vH.shape) == 1:
        nH = int(np.sqrt(vH.shape[0]))
        vH = vH.reshape(nH, nH)
    vh = np.repeat(np.repeat(vH, 2, axis=0), 2, axis=1)
    return vh

def args_init(data):
    return np.zeros_like(data, dtype=float)

def rew(Q, z_k, d_k, args_pre):
    exit_wave = Q * z_k
    exit_wave_ft = fft2(exit_wave)
    mask = np.abs(exit_wave_ft) == 0
    args = np.angle(exit_wave_ft)
    args[mask] = args_pre[mask]
    rew = ifft2(np.sqrt(d_k) * np.exp(1j * args))
    return rew, args

def W_u(Q_sq, Q_H_sq):
    A = Q_H_sq
    B = downsample(Q_sq)
    return np.divide(A, B, where=(B != 0), out=np.ones_like(A))

def W_z(Q_sq):
    A = Q_sq
    B = upsample(downsample(Q_sq))
    return np.divide(A, B, where=(B != 0), out=np.zeros_like(A))

def W_rew(Q):
    Q_sq = np.abs(Q)**2
    A = Q_sq*upsample(downsample(Q))
    B = upsample(downsample(Q_sq))*Q
    return np.divide(A, B, where=(np.abs(B) != 0), out=np.zeros_like(A))

def magps(z_k, Q, rew, u):
    '''Recursively update z_k using probe Q and measurement rew_k.
    Base case: stop when downsampled Q has size 1x1.'''
    # Compute step for this level
    Q_sq = np.abs(Q) ** 2
    Q_sqmax = np.max(Q_sq)
    step = np.conj(Q) / (Q_sq + u)

    # Base case: if coarse region is 1x1, apply update and return
    if z_k.shape == (1, 1):
        return z_k + step * (rew - Q * z_k)

    # Compute coarse-level terms
    Q_H    = downsample(Q)
    Q_H_sq = np.abs(Q_H) ** 2
    z_H    = downsample(W_z(Q_sq) * z_k)
    rew_H  = downsample(W_rew(Q) * rew)
    u_H    = W_u(Q_sq, Q_H_sq) * downsample(u)

    z_H_hat = magps(z_H, Q_H, rew_H, u_H)

    # Propagate update up and apply fine-level correction
    z_k_tilde = z_k + upsample(z_H_hat - z_H)
    z_k_plus  = z_k_tilde + step * (rew - Q * z_k_tilde)
    return z_k_plus


def magpie_recursion(z_init, Q, data, pos_array, num_iter, misfit, gt, alpha=0.01, metric=False, tol=1e-4):
    z       = z_init.copy()
    z_recon = None
    args    = args_init(data)
    res     = []
    err     = []
    h, w    = Q.shape
    Q_sq    = np.abs(Q) ** 2
    Q_sqmax = np.max(Q_sq)
    u       = alpha * (Q_sqmax - Q_sq)
    arr     = list(range(len(pos_array)))

    tol_hit_iter = None
    if metric:
        f, _, g = misfit(z, Q, data, pos_array)
        res.append(f)
        err.append(np.linalg.norm(np.abs(z) - np.abs(gt)))

    for j in range(num_iter):
        random.shuffle(arr)

        for k in arr:
            x, y = pos_array[k]
            z_k  = z[x:x+h, y:y+w]
            rew_k, args[k] = rew(Q, z_k, data[k], args[k])

            z_k_plus = magps(z_k, Q, rew_k, u)
            z[x:x+h, y:y+w] = z_k_plus

        if metric:
            f, _, g = misfit(z, Q, data, pos_array)
            res.append(f)
            err.append(np.linalg.norm(np.abs(z) - np.abs(gt)))
            if tol_hit_iter is None and g < tol:
                tol_hit_iter = j + 1
                z_recon = z.copy()

    if z_recon is None:
        z_recon = z.copy()
    return z_recon, np.array(res), np.array(err), tol_hit_iter

def magpie_loop(z_init, Q, data, pos_array, num_iter, misfit, gt, alpha=0.01, metric=False, tol=1e-4, max_level=7):
    h, w   = Q.shape
    # Number of levels is max_level+1 (0 through max_level)
    levels = max_level + 1

    # 1) Build downsampled Qs
    Qs    = [None] * levels
    Qs[0] = Q.copy()
    for level in range(1, levels):
        Qs[level] = downsample(Qs[level-1])

    # 2) Precompute Q_sq and regularizaiton terms us
    Q_sq     = [np.abs(q)**2 for q in Qs]
    Q0_sqmax = np.max(Q_sq[0])

    us    = [None] * levels
    us[0] = alpha * (Q0_sqmax - Q_sq[0])
    for level in range(1, levels):
        us[level] = W_u(Q_sq[level-1], Q_sq[level]) * downsample(us[level-1])

    # 3) Compute step sizes
    steps = [
        np.conj(Qs[level]) / (Q_sq[level] + us[level])
        for level in range(levels)
    ]

    # 4) Precompute downsampling weights for z and rew
    Wz   = [W_z(Q_sq[level]) for level in range(levels-1)]
    Wrew = [W_rew(Qs[level]) for level in range(levels-1)]

    # 5) Initialize variables
    z    = z_init.copy()
    z_recon = None
    args = args_init(data)
    arr  = list(range(len(pos_array)))

    res = []
    err = []

    tol_hit_iter = None
    if metric:
        f, _, g = misfit(z, Q, data, pos_array)
        res.append(f)
        err.append(np.linalg.norm(np.abs(z) - np.abs(gt)))

    for j in range(num_iter):
        random.shuffle(arr)

        # Loop over scanning positions
        for k in arr:
            x, y = pos_array[k]
            z_k  = z[x:x+h, y:y+w]

            # Down-pass: build lists of z_patches and rew_values
            z_patches = [z_k]
            rew_vals = [None] * levels

            # Level 0 rew
            rew_vals[0], args[k] = rew(Q, z_k, data[k], args[k])

            # Levels 1…max_level: downsample
            for level in range(1, levels):
                z_down   = downsample(Wz[level-1] * z_patches[level-1])
                rew_down = downsample(Wrew[level-1] * rew_vals[level-1])
                z_patches.append(z_down)
                rew_vals[level] = rew_down

            # Up-pass: correction from coarsest level back to 0
            z_new  = z_patches[-1].copy()
            z_new += steps[-1] * (rew_vals[-1] - Qs[-1] * z_new)

            for level in range(levels-2, -1, -1):
                # upsample correction
                delta  = upsample(z_new - z_patches[level+1])
                z_new  = z_patches[level].copy() + delta
                z_new += steps[level] * (rew_vals[level] - Qs[level] * z_new)

            # Write updated patch back into z
            z[x:x+h, y:y+w] = z_new

        if metric:
            f, _, g = misfit(z, Q, data, pos_array)
            res.append(f)
            err.append(np.linalg.norm(np.abs(z) - np.abs(gt)))
            if tol_hit_iter is None and g < tol:
                tol_hit_iter = j + 1
                z_recon = z.copy()

    if z_recon is None:
        z_recon = z.copy()
    return z_recon, np.array(res), np.array(err), tol_hit_iter


def rpie(z_init, Q, data, pos_array, num_iter, misfit, gt, alpha=0.01, metric=False, tol=1e-4):
    z = z_init.copy()
    z_recon = None
    Q_abs   = np.abs(Q)
    Q_maxsq = np.max(Q_abs)**2
    Q_sq    = Q_abs**2
    step    = np.conj(Q) / ((1 - alpha)*Q_sq + alpha*Q_maxsq)

    m    = Q.shape[0]
    res  = []
    err  = []
    arr  = list(range(len(pos_array)))
    args = args_init(data)

    tol_hit_iter = None
    if metric:
            f, df, g = misfit(z, Q, data, pos_array)
            res.append(f)
            err.append(np.linalg.norm(np.abs(z) - np.abs(gt)))

    for j in range(num_iter):
        random.shuffle(arr)

        for k in arr:
            x, y = pos_array[k]
            z_k  = z[x:x+m, y:y+m]
            rew_k, args[k] = rew(Q, z_k, data[k], args[k])
            z[x:x+m, y:y+m] += step * (rew_k - Q * z_k)

        if metric:
            f, df, g = misfit(z, Q, data, pos_array)
            res.append(f)
            err.append(np.linalg.norm(np.abs(z) - np.abs(gt)))
            if tol_hit_iter is None and g < tol:
                tol_hit_iter = j + 1
                z_recon = z.copy()

    if z_recon is None:
        z_recon = z.copy()
    return z_recon, np.array(res), np.array(err), tol_hit_iter


def line_s(p, x, f, g, sfun, alpha=1.0, rho=0.8, c=1e-4, max_iter=20, eval_counters=None, eval_increment = 1):
    iter_count = 0
    while iter_count < max_iter:
        x_new = x + alpha * p
        f_new, g_new, g2 = sfun(x_new)
        if eval_counters is not None:
            eval_counters['func'] += eval_increment
        if f_new <= f + c * alpha * np.dot(g, p):
            eval_counters['grad'] += eval_increment
            return x_new, f_new, g_new, g2, alpha
        alpha *= rho
        iter_count += 1
    return x_new, f_new, g_new, g2, alpha

def lbfgs(x0, Q, data, pos_array, maxiter, misfit, gt, m=5, tol=1e-4):

    def sfun(z):
        return misfit(z.view(np.complex128), Q, data, pos_array)

    object_shape = x0.shape
    x = x0.flatten().view(float)
    x_recon = None
    f, g, g2 = sfun(x)
    eval_counters = {'func': 1, 'grad': 1}
    eval_history = {'func': [1], 'grad': [1], 'f': [f], 'g2': [g2],
                    'error': [np.linalg.norm(np.abs(x.view(np.complex128).reshape(object_shape))-np.abs(gt))]}

    n = len(x)
    S = np.zeros((n, m))
    Y = np.zeros((n, m))
    rho_list = np.zeros(m)
    h0 = 1.0
    counter = 1
    tol_hit_iter = None

    for k in range(maxiter):
        q = g
        alpha_list = np.zeros(m)

        for i in range(min(k, m)):
            j = (k - i - 1) % m
            rho_list[j] = 1.0 / np.dot(Y[:, j], S[:, j])
            alpha_list[j] = rho_list[j] * np.dot(S[:, j], q)
            q = q - alpha_list[j] * Y[:, j]

        r = h0 * q

        for i in range(min(k, m)):
            j = (k - min(k, m) + i) % m
            beta = rho_list[j] * np.dot(Y[:, j], r)
            r = r + S[:, j] * (alpha_list[j] - beta)

        p = -r

        x_new, f_new, g_new, g2, alpha = line_s(p, x, f, g, sfun, max_iter=100, eval_counters=eval_counters)

        s = x_new - x
        y = g_new - g

        if k < m:
            S[:, k] = s
            Y[:, k] = y
        else:
            S = np.roll(S, -1, axis=1)
            Y = np.roll(Y, -1, axis=1)
            S[:, -1] = s
            Y[:, -1] = y

        h0 = np.dot(s, y) / np.dot(y, y)

        x = x_new
        f = f_new
        g = g_new

        if g2 < tol:
            if counter > 10:
                if tol_hit_iter is None:
                    tol_hit_iter = k + 1
                    x_recon = x.copy()
            else:
                counter += 1

        eval_history['func'].append(eval_counters['func'])
        eval_history['grad'].append(eval_counters['grad'])
        eval_history['f'].append(f)
        eval_history['g2'].append(g2)
        eval_history['error'].append(np.linalg.norm(np.abs(x.view(np.complex128).reshape(object_shape))-np.abs(gt)))

    if x_recon is None:
        x_recon = x.copy()
    return x_recon, eval_history, tol_hit_iter
