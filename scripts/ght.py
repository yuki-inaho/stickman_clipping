# https://github.com/jonbarron/hist_thresh/blob/master/experiments.ipynb

import numpy as np

csum = lambda z: np.cumsum(z)[:-1]
dsum = lambda z: np.cumsum(z[::-1])[-2::-1]
argmax = lambda x, f: np.mean(x[:-1][f == np.max(f)])  # Use the mean for ties.
clip = lambda z: np.maximum(1e-30, z)


def preliminaries(n, x):
    """Some math that is shared across multiple algorithms."""
    assert np.all(n >= 0)
    x = np.arange(len(n), dtype=n.dtype) if x is None else x
    assert np.all(x[1:] >= x[:-1])
    w0 = clip(csum(n))
    w1 = clip(dsum(n))
    p0 = w0 / (w0 + w1)
    p1 = w1 / (w0 + w1)
    mu0 = csum(n * x) / w0
    mu1 = dsum(n * x) / w1
    d0 = csum(n * x**2) - w0 * mu0**2
    d1 = dsum(n * x**2) - w1 * mu1**2
    return x, w0, w1, p0, p1, mu0, mu1, d0, d1


def GHT(n, x=None, nu=0, tau=0, kappa=0, omega=0.5):
    assert nu >= 0
    assert tau >= 0
    assert kappa >= 0
    assert omega >= 0 and omega <= 1
    x, w0, w1, p0, p1, _, _, d0, d1 = preliminaries(n, x)
    v0 = clip((p0 * nu * tau**2 + d0) / (p0 * nu + w0))
    v1 = clip((p1 * nu * tau**2 + d1) / (p1 * nu + w1))
    f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa * omega) * np.log(w0)
    f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
    return argmax(x, f0 + f1), f0 + f1


# Implementations of the existing algorithms that are special cases of GHT().


def Otsu(n, x=None):
    """Otsu's method."""
    x, w0, w1, _, _, mu0, mu1, _, _ = preliminaries(n, x)
    o = w0 * w1 * (mu0 - mu1) ** 2
    return argmax(x, o), o


def Otsu_equivalent(n, x=None):
    """Equivalent to Otsu's method."""
    x, _, _, _, _, _, _, d0, d1 = preliminaries(n, x)
    o = np.sum(n) * np.sum(n * x**2) - np.sum(n * x) ** 2 - np.sum(n) * (d0 + d1)
    return argmax(x, o), o


def MET(n, x=None):
    """Minimum Error Thresholding."""
    x, w0, w1, _, _, _, _, d0, d1 = preliminaries(n, x)
    ell = 1 + w0 * np.log(clip(d0 / w0)) + w1 * np.log(clip(d1 / w1)) - 2 * (w0 * np.log(clip(w0)) + w1 * np.log(clip(w1)))
    return argmax(x, -ell), ell  # argmin()


def wprctile(n, x=None, omega=0.5):
    """Weighted percentile, with weighted median as default."""
    assert omega >= 0 and omega <= 1
    x, _, _, p0, p1, _, _, _, _ = preliminaries(n, x)
    h = -omega * np.log(clip(p0)) - (1.0 - omega) * np.log(clip(p1))
    return argmax(x, -h), h  # argmin()


# An additional equivalent implementation of GHT that uses an explicit for-loop
# over splits of the histogram instead of the cumulative sum approach used by
# the paper. This additional implementation is intended to allow for easier
# comparisons with similar implementations of MET or Otsu's method, and to
# allow existing implementations of MET or Otsu's method to be easily
# generalized into implementations of GHT.


def GHT_forloop(n, x=None, nu=0, tau=0, kappa=0, omega=0.5):
    """An implementation of GHT() written using for loops."""
    assert np.all(n >= 0)
    x = np.arange(len(n), dtype=n.dtype) if x is None else x
    assert np.all(x[1:] >= x[:-1])
    assert nu >= 0
    assert tau >= 0
    assert kappa >= 0
    assert omega >= 0 and omega <= 1

    n_sum = np.sum(n)
    nx_sum = np.sum(n * x)
    nxx_sum = np.sum(n * x**2)

    max_score, n_c, nx_c, nxx_c = -np.inf, 0, 0, 0
    for i in range(len(n) - 1):
        n_c += n[i]
        nx_c += n[i] * x[i]
        nxx_c += n[i] * x[i] ** 2
        w0 = clip(n_c)
        w1 = clip(n_sum - n_c)
        p0 = w0 / n_sum
        p1 = w1 / n_sum
        d0 = np.maximum(0, nxx_c - nx_c**2 / w0)
        d1 = np.maximum(0, (nxx_sum - nxx_c) - (nx_sum - nx_c) ** 2 / w1)
        v0 = clip((p0 * nu * tau**2 + d0) / (p0 * nu + w0))
        v1 = clip((p1 * nu * tau**2 + d1) / (p1 * nu + w1))
        f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa * omega) * np.log(w0)
        f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
        score = f0 + f1

        # Argmax where the mean() is used for ties.
        if score > max_score:
            max_score, t_numer, t_denom = score, 0, 0
        if score == max_score:
            t_numer += x[i]
            t_denom += 1
    return t_numer / t_denom
