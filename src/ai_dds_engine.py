"""
AI-DDS Engine — 闭环处方优化参考实现
=====================================
产品案例: NAL1276 罗哌卡因长效脂质体注射液 (Bio-NN 平台)

本文件实现三件事:
  1. 处方设计空间 + 机理代理模型 ("湿实验替身", 用于闭环机制验证)
  2. AI-Formulation / AI-Release / AI-PK-PD 三个一期核心模块
  3. 三臂同条件对照基准: AI闭环 vs 经典DOE-RSM vs 人工OFAT

重要声明: ground_truth() 是带噪声的机理代理模型, 不是 Galaxy 实测数据。
它的作用是在真实湿实验数据接入前, 验证闭环控制逻辑与度量口径。
四项 KPI 的计算代码与真实数据接入后完全一致 — 只需把 ground_truth()
替换为 LIMS 实测接口。
"""
import numpy as np
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# ───────────────────────── 1. 处方设计空间 ─────────────────────────
# Bio-NN 多囊脂质体: 6 个处方/工艺变量
VARS = [
    ("dl_ratio",   0.05, 0.50),   # 药脂比 (w/w)
    ("chol_pct",   0.0,  45.0),   # 胆固醇 mol%
    ("peg_pct",    0.0,   8.0),   # DSPE-PEG2000 mol%
    ("grad_mM",  100.0, 500.0),   # 硫酸铵梯度浓度 mM
    ("size_nm",   80.0, 400.0),   # 挤出后平均粒径 nm
    ("dspc_frac",  0.0,   1.0),   # DSPC 占磷脂比例 (相变温度代理)
]
NAMES = [v[0] for v in VARS]
LO = np.array([v[1] for v in VARS])
HI = np.array([v[2] for v in VARS])

# 目标产品概况 (TPP) — 示意性规格, 需由 Galaxy 法规/临床确认
SPEC = {
    "EE_pct":      (85.0, None),   # 包封率 ≥ 85%
    "burst_pct":   (None, 10.0),   # 0-2h 突释 ≤ 10%
    "t50_h":       (60.0, 84.0),   # 半释时间 60-84 h
    "F168_pct":    (85.0, None),   # 7 天累积释放 ≥ 85%
    "leak_pct":    (None, 10.0),   # 25°C/30d 渗漏 ≤ 10%
    "Cmax_ug_mL":  (None,  2.0),   # 血浆峰浓度 ≤ 2.0 µg/mL (全身安全)
}
RESPONSES = list(SPEC.keys())

DOSE_MG, V_L, CL_L_H = 400.0, 47.0, 26.4          # 罗哌卡因处置参数 (文献值)
K_EL = CL_L_H / V_L
COST_PER_RUN_USD = 3500.0                          # 占位值, 待 Galaxy 实际成本替换


def _norm(x):
    return (np.asarray(x, float) - LO) / (HI - LO)


# ──────────────── 2. 机理代理模型 (湿实验替身) ────────────────
def release_params(x):
    """处方 -> 释放曲线参数 (无噪声真值)。

    交互项依据脂质体制剂学既有机理设定, 非为偏袒某一优化臂而构造:
      * 胆固醇 x DSPC: 胆固醇消除凝胶-液晶相变, 二者对膜刚性的贡献饱和叠加
        (soft-OR), 而非独立相乘。
      * 药脂比 x 铵盐梯度: 主动载药的捕获容量有限, 包封率由"梯度容量/载药量"
        之比决定 — 典型的饱和型交互。
      * PEG x 粒径: 空间位阻稳定对小囊泡更有效, PEG 的抗渗漏作用随粒径增大衰减。
      * 胆固醇 >~40 mol%: 相分离/胆固醇微晶析出, 包封率转为下降 (非单调)。
    """
    dl, chol, peg, grad, size, dspc = _norm(x).T
    dl_r = np.asarray(x, float).T[0]
    chol_pct = np.asarray(x, float).T[1]

    rigidity = chol + dspc - chol * dspc                      # 饱和叠加的膜刚性
    capacity = (0.25 + grad) / (0.25 + grad + 1.8 * dl_r)      # 梯度容量 / 载药量
    chol_pen = 1.0 - 0.9 * np.clip((chol_pct - 40.0) / 5.0, 0, 1) ** 2   # 相分离惩罚

    ee = (100.0 * (0.30 + 0.72 * capacity)
          * (0.90 + 0.20 * chol - 0.08 * chol ** 2)
          * chol_pen * (1 - 0.08 * peg) * (0.95 + 0.06 * size))
    ee = np.clip(ee, 1.0, 99.5)

    burst = (0.30 * np.exp(-2.6 * rigidity)
             * (1 + 1.1 * dl_r) * (1 - 0.35 * size) * (1 + 0.5 * peg))
    burst = np.clip(burst, 0.003, 0.60)

    t50 = 24.0 * np.exp(1.55 * rigidity + 0.40 * size - 0.60 * dl + 0.15 * grad)
    beta = 0.85 + 0.50 * rigidity
    leak = (30.0 * np.exp(-2.4 * rigidity)
            * (1 + 0.9 * dl) * (1 - 0.30 * peg * (1 - size)))
    return ee, burst, t50, beta, np.clip(leak, 0.05, 95.0)


def release_curve(t_h, burst, t50, beta):
    """AI-Release: 突释 + Weibull 持续释放的双相累积释放曲线 F(t), 0-1。"""
    t = np.atleast_1d(t_h).astype(float)
    tau = t50 / (np.log(2.0) ** (1.0 / beta))          # 使持续相在 t50 释放 50%
    f_b = burst * (1 - np.exp(-t / 0.35))
    f_s = (1 - burst) * (1 - np.exp(-((t / tau) ** beta)))
    return np.clip(f_b + f_s, 0.0, 1.0)


def pk_profile(t_h, burst, t50, beta, dose_mg=DOSE_MG):
    """AI-PK/PD: IVIVC Level A + 一室处置, 卷积求血浆浓度 (µg/mL)。"""
    dt = t_h[1] - t_h[0]
    F = release_curve(t_h, burst, t50, beta)
    rate = np.gradient(F, dt) * dose_mg                 # mg/h 入血速率
    imp = np.exp(-K_EL * t_h) / V_L                     # 单位脉冲响应 (1/L)
    return np.convolve(rate, imp)[:len(t_h)] * dt       # mg/L == µg/mL


T_GRID = np.arange(0.0, 336.0 + 0.25, 0.25)


def ground_truth(X, rng=None, noise=0.05):
    """代理湿实验: 处方矩阵 -> 6 个响应 (含测定噪声)。"""
    X = np.atleast_2d(np.asarray(X, float))
    ee, burst, t50, beta, leak = release_params(X)
    out = np.zeros((len(X), len(RESPONSES)))
    for i in range(len(X)):
        F = release_curve(T_GRID, burst[i], t50[i], beta[i])
        C = pk_profile(T_GRID, burst[i], t50[i], beta[i])
        out[i] = [ee[i], burst[i] * 100.0, t50[i],
                  float(np.interp(168.0, T_GRID, F) * 100.0),
                  leak[i], float(C.max())]
    if rng is not None and noise > 0:
        out *= rng.lognormal(0.0, noise, out.shape)
    return out


# ──────────────── 3. 合格判定与 Derringer-Suich 合意度 ────────────────
def passes(Y):
    Y = np.atleast_2d(Y)
    ok = np.ones(len(Y), bool)
    for j, r in enumerate(RESPONSES):
        lo, hi = SPEC[r]
        if lo is not None:
            ok &= Y[:, j] >= lo
        if hi is not None:
            ok &= Y[:, j] <= hi
    return ok


_D_RANGE = {"EE_pct": (60.0, 95.0), "burst_pct": (3.0, 30.0), "t50_h": (20.0, 150.0),
            "F168_pct": (50.0, 99.0), "leak_pct": (2.0, 35.0), "Cmax_ug_mL": (0.3, 4.0)}


def desirability(Y, per_response=False):
    """各响应 d∈[0,1] 的几何平均; 双边规格用目标区间三角形式。"""
    Y = np.atleast_2d(Y)
    ds = np.zeros_like(Y, float)
    for j, r in enumerate(RESPONSES):
        lo, hi = SPEC[r]
        a, b = _D_RANGE[r]
        y = Y[:, j]
        if lo is not None and hi is not None:                  # 目标区间 (t50)
            mid = 0.5 * (lo + hi)
            d = 1.0 - np.abs(y - mid) / (mid - a)
        elif lo is not None:                                   # 越大越好
            d = (y - a) / (lo - a)
        else:                                                  # 越小越好
            d = (b - y) / (b - hi)
        ds[:, j] = np.clip(d, 1e-3, 1.0)
    return ds if per_response else np.exp(np.log(ds).mean(axis=1))


# ──────────────── 4. 三臂优化器 ────────────────
def _gp():
    return GaussianProcessRegressor(
        kernel=ConstantKernel(1.0, (1e-2, 1e2)) * Matern(np.ones(6), (1e-2, 1e2), nu=2.5)
               + WhiteKernel(1e-3, (1e-6, 1e-1)),
        normalize_y=True, n_restarts_optimizer=3, random_state=0)


LOG_RESP = {"t50_h", "burst_pct", "leak_pct", "Cmax_ug_mL"}      # 正值右偏, 对数域建模


def _fit_surrogates(Xn, Y):
    """AI-Release / AI-PK-PD 代理模型: 每个 CQA 一个独立 GP。"""
    gps = []
    for j, r in enumerate(RESPONSES):
        t = np.log(np.maximum(Y[:, j], 1e-6)) if r in LOG_RESP else Y[:, j]
        gps.append(_gp().fit(Xn, t))
    return gps


def _predict_responses(gps, Cn, n_draw=0, rng=None):
    """返回 (均值矩阵, 后验抽样) — 抽样用于复合期望改进。"""
    mus, sds = [], []
    for r, gp in zip(RESPONSES, gps):
        m, s = gp.predict(Cn, return_std=True)
        if r in LOG_RESP:
            m, s = np.exp(m + 0.5 * s ** 2), np.exp(m) * s      # 对数正态矩
        mus.append(m); sds.append(s)
    MU, SD = np.column_stack(mus), np.column_stack(sds)
    if n_draw and rng is not None:
        draws = MU[None] + SD[None] * rng.standard_normal((n_draw, *MU.shape))
        return MU, np.maximum(draws, 1e-6)
    return MU, None


def arm_ai(seed, n_init=12, n_round=7, batch=4, n_cand=3000, n_draw=64):
    """AI 闭环: LHS 起始 -> 每 CQA 独立 GP -> 复合期望改进批量采点 -> 主动学习。

    与架构一致: AI-Release / AI-PK-PD 分别预测各 CQA, AI-Formulation 在
    其复合合意度上做多目标寻优 (composite Bayesian optimization)。
    """
    rng = np.random.default_rng(seed)
    Xn = qmc.LatinHypercube(d=6, seed=seed).random(n_init)
    X = LO + Xn * (HI - LO)
    Y = ground_truth(X, rng)
    pred_log = []

    for _ in range(n_round):
        gps = _fit_surrogates(Xn, Y)
        d_best = desirability(Y).max()
        cand = qmc.LatinHypercube(d=6, seed=int(rng.integers(1e6))).random(n_cand)
        Xn_aug, Y_aug = Xn.copy(), Y.copy()
        props, pred_mu = [], []
        for _b in range(batch):
            MU, draws = _predict_responses(gps, cand, n_draw, rng)
            d_draw = np.array([desirability(draws[s]) for s in range(n_draw)])
            ei = np.maximum(d_draw - d_best, 0.0).mean(axis=0)     # 复合期望改进
            k = int(np.argmax(ei))
            props.append(cand[k]); pred_mu.append(MU[k])
            # constant-liar: 以后验均值占位, 保证批内多样性
            Xn_aug = np.vstack([Xn_aug, cand[k]]); Y_aug = np.vstack([Y_aug, MU[k]])
            gps = _fit_surrogates(Xn_aug, Y_aug)
            cand = np.delete(cand, k, axis=0)
        Pn = np.array(props)
        Px = LO + Pn * (HI - LO)
        Py = ground_truth(Px, rng)
        for p, o in zip(pred_mu, Py):                              # 逐 CQA 前瞻性误差
            pred_log.append((np.asarray(p, float), np.asarray(o, float)))
        X, Y = np.vstack([X, Px]), np.vstack([Y, Py])
        Xn = np.vstack([Xn, Pn])
    return X, Y, pred_log


def arm_doe(seed, n_screen=30, n_confirm=10):
    """经典 DOE: 中心复合设计 -> 二次响应面 -> 合意度寻优 -> 确认批。"""
    rng = np.random.default_rng(seed + 10000)
    # 2^(6-2) 部分析因 + 轴向点 + 中心点
    fr = np.array([[int(b) for b in format(i, "04b")] for i in range(16)], float)
    gen = np.column_stack([fr, (fr[:, 0] + fr[:, 1] + fr[:, 2]) % 2, (fr[:, 1] + fr[:, 2] + fr[:, 3]) % 2])
    design = [gen * 2 - 1]
    ax = np.zeros((12, 6))
    for j in range(6):
        ax[2 * j, j], ax[2 * j + 1, j] = -1.0, 1.0
    design += [ax, np.zeros((max(0, n_screen - 28), 6))]
    Dn = (np.vstack(design)[:n_screen] + 1.0) / 2.0
    X = LO + Dn * (HI - LO)
    Y = ground_truth(X, rng)

    poly = PolynomialFeatures(2, include_bias=False)
    A = poly.fit_transform(Dn)
    models = [Ridge(alpha=1.0).fit(A, Y[:, j]) for j in range(Y.shape[1])]
    cand = qmc.LatinHypercube(d=6, seed=seed + 7).random(4000)
    Yhat = np.column_stack([m.predict(poly.transform(cand)) for m in models])
    order = np.argsort(-desirability(Yhat))[:n_confirm]
    Cn = cand[order]
    Cx = LO + Cn * (HI - LO)
    Cy = ground_truth(Cx, rng)
    pred_log = [(Yhat[o].astype(float), Cy[i].astype(float)) for i, o in enumerate(order)]
    return np.vstack([X, Cx]), np.vstack([Y, Cy]), pred_log


def arm_ofat(seed, levels=6):
    """人工 OFAT: 自中心点起逐因素扫描, 保留每轮最优。"""
    rng = np.random.default_rng(seed + 20000)
    cur = np.full(6, 0.5)
    X, Y = [], []
    for j in range(6):
        grid = np.linspace(0.0, 1.0, levels)
        blk = np.tile(cur, (levels, 1)); blk[:, j] = grid
        bx = LO + blk * (HI - LO)
        by = ground_truth(bx, rng)
        X.append(bx); Y.append(by)
        cur = blk[int(np.argmax(desirability(by)))]
    ref = np.clip(cur + rng.normal(0, 0.05, (4, 6)), 0, 1)
    rx = LO + ref * (HI - LO)
    X.append(rx); Y.append(ground_truth(rx, rng))
    return np.vstack(X), np.vstack(Y), []


# ──────────────── 5. KPI ────────────────
def kpis(X, Y, pred_log):
    ok = passes(Y)
    d = desirability(Y)
    first = int(np.argmax(ok)) + 1 if ok.any() else None
    out = {"n_runs": len(Y), "n_pass": int(ok.sum()),
           "hit_rate_pct": 100.0 * ok.mean(),
           "runs_to_first_pass": first,
           "best_desirability": float(d.max()),
           "cost_per_viable_usd": (len(Y) * COST_PER_RUN_USD / ok.sum()) if ok.any() else np.nan}
    if pred_log:
        P = np.array([a for a, _ in pred_log]); O = np.array([b for _, b in pred_log])
        ape = np.abs(P - O) / np.maximum(np.abs(O), 1e-6) * 100.0
        for j, r in enumerate(RESPONSES):
            out[f"mape_{r}"] = float(np.mean(ape[:, j]))
        out["pred_mape_pct"] = float(np.mean(ape))
    else:
        for r in RESPONSES:
            out[f"mape_{r}"] = np.nan
        out["pred_mape_pct"] = np.nan
    return out
