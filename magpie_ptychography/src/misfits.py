import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def feasibility_distance_misfit(z, Q, data, pos_array):
    m = data.shape[1]
    n = int(np.sqrt(z.size))
    N = data.shape[0]

    f   = 0
    g   = 0
    dfz = np.zeros(2 * n * n, dtype=float)

    z = z.reshape(n, n)
    Q = Q.reshape(m, m)

    for k in range(N):
        x, y = pos_array[k]
        z_k  = z[x:x+m, y:y+m]
        ew_k = Q * z_k

        # Fourier and inverse Fourier operations
        fft_ew_k = fft2(ew_k)
        rew_k    = ifft2(np.sqrt(data[k]) * np.exp(1j * np.angle(fft_ew_k)))
        res_k    = ew_k - rew_k  

        # Calculating the residual and the gradient
        f += 0.5 * np.linalg.norm(res_k.flatten(), 2)**2

        dfz_temp = np.conj(Q) * res_k
        g       += np.linalg.norm(dfz_temp.flatten(), 2) / N / m
        dfz_k    = np.zeros((n, n), dtype=complex)
        dfz_k[x:x+m, y:y+m] = dfz_temp
        
        dfz += dfz_k.flatten().view(float)

    return f, dfz, g