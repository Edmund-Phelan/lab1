import numpy as np


def descriptors(x, y):
    """DFT of z[n] = x[n] + i*y[n]. Returns coefficients and integer frequencies, largest amplitude first."""
    z = x + 1j * y
    Z = np.fft.fft(z)
    freqs = np.fft.fftfreq(len(z), d=1 / len(z))  # integer frequencies ..., -1, 0, 1, ...
    order = np.argsort(-np.abs(Z))
    return Z[order], freqs[order]


def reconstruct(Z, freqs, K, N):
    """z'[n] = 1/N * sum_{u<K} F[u] exp(2*pi*i*w[u]*n/N), using the top K harmonics."""
    n = np.arange(N)
    return (Z[:K, None] * np.exp(2j * np.pi * freqs[:K, None] * n / N)).sum(axis=0) / N


if __name__ == '__main__':
    # Self-check: all harmonics give back the input exactly, and it matches numpy's ifft
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    x, y = 3 * np.cos(t) + np.cos(5 * t), 2 * np.sin(t)
    Z, f = descriptors(x, y)
    assert np.allclose(reconstruct(Z, f, len(Z), 64), x + 1j * y)
    print('fourier ok')
