"""SPEC-Σ.2 [1.1]: native sigma_m and a family-wise controlled futility gate.

Baseline = factor-L4-P-O (16 logged seeds, T2 v1; seed-0 records reproduced bitwise by the
current tree). Rule: prune at S0 or S1 iff some row has z_m = mean_m / (sigma_m / sqrt n) <= c,
c = Bonferroni floor over 10 rows x 2 gates (union bound). beta_FWER = P(prune | all Delta_m = 0)
(worst case of H1: Delta_m >= 0). Verified (a) Gaussian with sigma-hat ~ sigma*sqrt(chi2_15/15),
(b) joint bootstrap of the centred native seed vectors (keeps cross-row correlation, tails).
Final S2 test untouched (gates only prune) -> alpha_final <= ungated alpha."""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "factor-L4-P-O"
BETA = .02
C_FW = -3.7328                   # t_15 quantile at BETA/20 (10 rows x 2 gates; sigma from 16 seeds)
T15 = 2.131449546


def native():
    d = json.load(open(os.path.join(HERE, "..", "results", "t2v1_factorO.json")))
    return {r: np.array(v["effect"]) for r, v in d["summary"][BASE].items()}


def gate_prune(x, sig, c):
    """x (sims, rows, 16) effects; prune iff any row z <= c at S0 (seed 0) or S1 (seeds 0..3)."""
    z0 = x[:, :, 0] / sig
    z1 = x[:, :, :4].mean(2) / (sig / 2.)
    return ((z0 <= c) | (z1 <= c)).any(1)


def main(n_sim=200000, seed=0):
    rng = np.random.default_rng(seed)
    E = native()
    rows = sorted(E)
    sig = np.array([E[r].std(ddof=1) for r in rows])
    R = len(rows)
    # (a) Gaussian, independent rows (worst case for the union), sigma-hat error
    x = rng.normal(0., 1., (n_sim, R, 16)) * sig[None, :, None]
    shat = sig * np.sqrt(rng.chisquare(15, (n_sim, R)) / 15)
    fw_a = gate_prune(x, shat, C_FW).mean()
    fw_a_known = gate_prune(x, sig, C_FW).mean()
    m, sd = x.mean(2), x.std(2, ddof=1)
    alpha = ((m - T15 * sd / 4.) > 0).mean()                 # per-row, ungated = gated bound
    fw_old = gate_prune_old(x, sig).mean()
    # (b) joint bootstrap of centred native seed vectors (same seeds across rows)
    X = np.stack([E[r] - E[r].mean() for r in rows])        # (R, 16)
    idx = rng.integers(0, 16, (n_sim, 16))
    xb = np.transpose(X[:, idx], (1, 0, 2))                  # (sims, R, 16)
    fw_b = gate_prune(xb, sig, C_FW).mean()
    out = dict(base=BASE, rows=rows, sigma=dict(zip(rows, map(float, np.round(sig, 4)))),
               c_fw=round(C_FW, 3), beta_fwer_gauss_sigmahat=float(fw_a),
               beta_fwer_gauss_known=float(fw_a_known), beta_fwer_bootstrap=float(fw_b),
               beta_fwer_old_conjunctive=float(fw_old), alpha_final_row=float(alpha), n_sim=n_sim)
    return out


def gate_prune_old(x, sig):
    z0 = x[:, :, 0] / sig
    z1 = x[:, :, :4].mean(2) / (sig / 2.)
    return ((z0 <= -2.0) | (z1 <= -1.5)).any(1)


if __name__ == "__main__":
    print(json.dumps(main(), indent=0))
