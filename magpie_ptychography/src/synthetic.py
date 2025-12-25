import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def generate_intensity_measurements(z, Q, positions, noise_level):
    measurements = []
    for position in positions:
        x, y = position
        z_k = z[x:x+Q.shape[0], y:y+Q.shape[1]]
        exit_wave = Q * z_k
        d_k = np.abs(fft2(exit_wave))**2
        
        if noise_level > 0:
            # Scale the intensity to control noise level
            scaled_d_k = d_k / noise_level
            
            # Add Poisson noise
            noisy_scaled_d_k = np.random.poisson(scaled_d_k)
            # Scale back the noisy data to the original range
            d_k_noisy = noisy_scaled_d_k * noise_level
        else:
            # No noise added, return the original data
            d_k_noisy = d_k
        
        measurements.append(d_k_noisy)
    return np.array(measurements)


def generate_scanning_positions(object_shape, n_probe, shift):
    return [(x, y) for x in range(0, object_shape[0] - n_probe + 1, shift)
                    for y in range(0, object_shape[1] - n_probe + 1, shift)]