"""
AI-DDS · tLNP module — targeted lipid-nanoparticle formulation optimisation
===========================================================================

Scope: the *carrier*. This module optimises LNP composition and process for a
given RNA payload. It does not design payloads — no CAR constructs, no circRNA
sequences, no nucleotide optimisation. Those belong to the payload-design
platform (CircuLab / CARFORGE) and, per the programme's own research proposal,
are deliberately out of scope for formulation documents.

Why this is the natural next step for Bio-NN: the four-component LNP system is
the same class of problem as the NAL1276 liposome already benchmarked — a
narrow multi-CQA feasible region governed by non-additive lipid interactions —
and it reuses the same manufacturing assets (lipid handling, microfluidic /
T-junction mixing, TFF, sterile fill).

IMPORTANT — no surrogate ground truth is shipped here, deliberately.
The NAL1276 benchmark used a mechanistic surrogate to validate loop *mechanics*.
For a real programme, inventing a surrogate would produce numbers that look like
evidence and are not. This module therefore exposes `propose()` / `ingest()`:
you feed it measured batches, it proposes the next ones. Benchmark numbers
appear only once measured data exists.

Design space and specification defaults follow published LNP practice
(Onpattro-class and mRNA-vaccine-class formulations); every value is a
parameter the programme's CMC lead is expected to overwrite.
"""
import numpy as np

from ai_dds_engine import _gp  # same Matern 5/2 GP surrogate as the liposome loop

# ───────────────── design space ─────────────────
# Lipid molar percentages are constrained to sum to 100 and are therefore
# parameterised on the 3-simplex (ionizable, helper, cholesterol, PEG-lipid).
LNP_VARS = [
    ("ionizable_mol_pct", 30.0, 60.0),   # e.g. MC3 / ALC-0315 / SM-102 class
    ("helper_mol_pct",     5.0, 20.0),   # DSPC / DOPE
    ("peg_lipid_mol_pct",  0.5,  3.0),   # DMG-PEG2000 / ALC-0159
    ("np_ratio",           3.0, 12.0),   # ionizable-N : RNA-phosphate
    ("mix_pH",             3.5,  5.5),   # aqueous-phase acetate/citrate pH
    ("frr",                1.0,  5.0),   # flow-rate ratio aqueous:organic
    ("tfr_mL_min",         2.0, 20.0),   # total flow rate (microfluidic mixing)
]
LNP_NAMES = [v[0] for v in LNP_VARS]
LNP_LO = np.array([v[1] for v in LNP_VARS])
LNP_HI = np.array([v[2] for v in LNP_VARS])


def cholesterol_mol_pct(x):
    """Closure of the lipid simplex: cholesterol is the balance to 100 mol%."""
    x = np.atleast_2d(np.asarray(x, float))
    return 100.0 - x[:, 0] - x[:, 1] - x[:, 2]


def valid(x, chol_min=20.0, chol_max=50.0):
    """A formulation is physically meaningful only if the balance lipid is sane."""
    c = cholesterol_mol_pct(x)
    return (c >= chol_min) & (c <= chol_max)


# ───────────────── quality attributes ─────────────────
# (lower bound, upper bound) — None means unbounded on that side.
LNP_SPEC = {
    "EE_pct":          (90.0, None),   # RNA encapsulation efficiency
    "size_nm":         (60.0, 100.0),  # Z-average hydrodynamic diameter
    "PDI":             (None, 0.20),   # polydispersity index
    "zeta_mV":         (-10.0, 10.0),  # near-neutral at physiological pH
    "rna_integrity":   (0.80, None),   # intact RNA fraction after formulation
    "apparent_pKa":    (6.2, 6.8),     # endosomal-escape window
}
LNP_RESPONSES = list(LNP_SPEC.keys())

# Desirability normalisation ranges (worst plausible -> best plausible).
LNP_D_RANGE = {
    "EE_pct":        (50.0, 99.5),
    "size_nm":       (40.0, 200.0),
    "PDI":           (0.02, 0.45),
    "zeta_mV":       (-35.0, 35.0),
    "rna_integrity": (0.40, 1.00),
    "apparent_pKa":  (5.0, 8.0),
}


def lnp_desirability(Y, per_response=False):
    """Derringer-Suich geometric mean over the LNP CQAs (same convention as the
    liposome loop, so scores are comparable across platforms)."""
    Y = np.atleast_2d(np.asarray(Y, float))
    ds = np.zeros_like(Y, float)
    for j, r in enumerate(LNP_RESPONSES):
        lo, hi = LNP_SPEC[r]
        a, b = LNP_D_RANGE[r]
        y = Y[:, j]
        if lo is not None and hi is not None:
            mid = 0.5 * (lo + hi)
            d = 1.0 - np.abs(y - mid) / max(mid - a, 1e-9)
        elif lo is not None:
            d = (y - a) / (lo - a)
        else:
            d = (b - y) / (b - hi)
        ds[:, j] = np.clip(d, 1e-3, 1.0)
    return ds if per_response else np.exp(np.log(ds).mean(axis=1))


def lnp_passes(Y):
    Y = np.atleast_2d(np.asarray(Y, float))
    ok = np.ones(len(Y), bool)
    for j, r in enumerate(LNP_RESPONSES):
        lo, hi = LNP_SPEC[r]
        if lo is not None:
            ok &= Y[:, j] >= lo
        if hi is not None:
            ok &= Y[:, j] <= hi
    return ok


# ───────────────── closed loop over measured batches ─────────────────
class TLNPLoop:
    """Active-learning loop driven by *measured* batches only.

    ingest(X, Y)  X: (n, 7) formulation/process matrix in LNP_NAMES order
                  Y: (n, 6) measured CQAs in LNP_RESPONSES order (NaN = not measured)
    propose(k)    -> k next formulations, with the frozen predictions that the
                     prospective-error KPI will later be settled against.
    """

    def __init__(self, seed=0):
        self.X = np.empty((0, len(LNP_VARS)))
        self.Y = np.empty((0, len(LNP_RESPONSES)))
        self.rng = np.random.default_rng(seed)
        self.frozen = []          # [(X_proposed, Y_predicted)] — the audit trail

    def ingest(self, X, Y):
        X = np.atleast_2d(np.asarray(X, float))
        Y = np.atleast_2d(np.asarray(Y, float))
        if X.shape[1] != len(LNP_VARS) or Y.shape[1] != len(LNP_RESPONSES):
            raise ValueError(f"expected X[:, {len(LNP_VARS)}] and Y[:, {len(LNP_RESPONSES)}], "
                             f"got {X.shape} and {Y.shape}")
        self.X, self.Y = np.vstack([self.X, X]), np.vstack([self.Y, Y])
        return len(self.X)

    def _fit(self):
        Xn = (self.X - LNP_LO) / (LNP_HI - LNP_LO)
        gps = {}
        for j, r in enumerate(LNP_RESPONSES):
            m = ~np.isnan(self.Y[:, j])
            if m.sum() >= 4:                       # a GP needs a few points to mean anything
                t = self.Y[m, j]
                if r in ("size_nm", "PDI"):        # right-skewed, positive -> log space
                    t = np.log(np.maximum(t, 1e-6))
                gps[r] = (_gp().fit(Xn[m], t), r in ("size_nm", "PDI"))
        return gps

    def predict(self, X):
        """Posterior mean per CQA; NaN where that CQA has too few observations."""
        gps = self._fit()
        Xn = (np.atleast_2d(np.asarray(X, float)) - LNP_LO) / (LNP_HI - LNP_LO)
        out = np.full((len(Xn), len(LNP_RESPONSES)), np.nan)
        for j, r in enumerate(LNP_RESPONSES):
            if r in gps:
                gp, logsp = gps[r]
                mu, sd = gp.predict(Xn, return_std=True)
                out[:, j] = np.exp(mu + 0.5 * sd ** 2) if logsp else mu
        return out

    def propose(self, k=4, n_cand=4000, n_draw=64):
        """Composite expected improvement over the measured set; respects the
        lipid-simplex constraint. Freezes predictions for later error settlement."""
        if len(self.X) < 4:
            cand = LNP_LO + self.rng.random((n_cand, len(LNP_VARS))) * (LNP_HI - LNP_LO)
            cand = cand[valid(cand)][:k]
            self.frozen.append((cand, np.full((len(cand), len(LNP_RESPONSES)), np.nan)))
            return cand, None                      # space-filling start
        gps = self._fit()
        cand = LNP_LO + self.rng.random((n_cand, len(LNP_VARS))) * (LNP_HI - LNP_LO)
        cand = cand[valid(cand)]
        mus, sds = [], []
        for r in LNP_RESPONSES:
            if r in gps:
                gp, logsp = gps[r]
                mu, sd = gp.predict((cand - LNP_LO) / (LNP_HI - LNP_LO), return_std=True)
                if logsp:
                    mu, sd = np.exp(mu + 0.5 * sd ** 2), np.exp(mu) * sd
            else:                                   # unobserved CQA -> neutral prior
                a, b = LNP_D_RANGE[r]
                mu = np.full(len(cand), 0.5 * (a + b)); sd = np.full(len(cand), (b - a) / 4)
            mus.append(mu); sds.append(sd)
        MU, SD = np.column_stack(mus), np.column_stack(sds)
        obs = lnp_desirability(np.nan_to_num(self.Y, nan=0.0))
        d_best = float(obs.max()) if len(obs) else 0.0
        draws = MU[None] + SD[None] * self.rng.standard_normal((n_draw, *MU.shape))
        d_draw = np.array([lnp_desirability(draws[s]) for s in range(n_draw)])
        ei = np.maximum(d_draw - d_best, 0.0).mean(axis=0)
        pick = np.argsort(-ei)[:k]
        self.frozen.append((cand[pick], MU[pick]))
        return cand[pick], MU[pick]

    def settle(self, observed):
        """Per-CQA prospective MAPE against the most recent frozen predictions."""
        Xp, Yp = self.frozen[-1]
        observed = np.atleast_2d(np.asarray(observed, float))
        m = ~np.isnan(observed) & ~np.isnan(Yp)
        if m.sum() == 0:
            return {}
        ape = np.abs(Yp - observed) / np.maximum(np.abs(observed), 1e-9) * 100.0
        return {r: float(np.nanmean(np.where(m[:, j], ape[:, j], np.nan)))
                for j, r in enumerate(LNP_RESPONSES) if m[:, j].any()}
