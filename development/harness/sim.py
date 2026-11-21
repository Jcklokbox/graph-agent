"""Choice of the futility thresholds c_s by simulation (SPEC-Σ [1]).

Per metric the per-seed effect ~ N(mu, sigma); the gate uses the fixed sigma (z = mean /
(sigma/sqrt n)); the final test (S2, n=16) is the harness verdict: two-sided 95 % t-CI above 0
(one-sided alpha_nom = .025). beta_loss(c) = P(pruned at S0 or S1 | mu = mu80), mu80 = the
effect with final power .8 without gating; trial-level for K independent metrics.
null alpha: P(final yes | mu = 0) with gating, must be <= alpha_nom + eps."""
import numpy as np

T15 = 2.131449546      # t_.975, df 15


def simulate(c0, c1, mu, n_sim=200000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(mu, 1., (n_sim, 16))
    g0 = x[:, 0] > c0
    g1 = x[:, :4].mean(1) * 2. > c1
    m, sd = x.mean(1), x.std(1, ddof=1)
    final = (m - T15 * sd / 4.) > 0
    return dict(pruned=float(1 - (g0 & g1).mean()), final=float((g0 & g1 & final).mean()),
                final_nogate=float(final.mean()))


def mu80(n_sim=200000):
    lo, hi = 0., 2.
    for _ in range(30):
        mid = (lo + hi) / 2
        if simulate(-99, -99, mid, n_sim)["final"] < .8:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


if __name__ == "__main__":
    from harness.beacon import cli

    with cli():
        K, n_sim = 10, 200000
        m80 = mu80(50000)
        print(f"mu80 = {m80:.3f} sigma (final power .8, n=16, no gate)")
        print("c0 c1 | beta_loss(1 metric) beta_loss(K=10) | prune|mu=0 | alpha_null  | prune|mu=-0.5")
        for c0, c1 in [(-1., -.5), (-1.5, -1.), (-2., -1.), (-2., -1.5), (-2.5, -1.5), (-2.5, -2.)]:
            b = simulate(c0, c1, m80, n_sim)["pruned"]
            z0 = simulate(c0, c1, 0., n_sim)
            neg = simulate(c0, c1, -.5, n_sim)["pruned"]
            print(f"{c0:+.1f} {c1:+.1f} | {b:.4f} {1 - (1 - b) ** K:.4f} | {z0['pruned']:.3f} | "
                  f"{z0['final']:.5f} (nogate {z0['final_nogate']:.5f}) | {neg:.3f}")
        eps = 3 * (.025 * .975 / n_sim) ** .5
        print(f"eps (3 se at alpha .025, n_sim {n_sim}) = {eps:.4f}")
