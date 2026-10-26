# AI-DDS Engine

**A domain foundation model and closed-loop optimiser for pharmaceutical drug-delivery systems (DDS).**

Most "AI for drug discovery" optimises the *molecule*. Drug delivery optimises
everything else — excipient ratios, carrier architecture, process parameters,
release kinetics. This repository is the engine for that second problem: a
formulation-native transformer (**DDS-Former**), a teacher→student distillation
pipeline that turns patent prose into structured supervision, and a closed-loop
optimiser that proposes the next wet-lab experiment and scores its own
prospective error.

> **Status.** Corpus construction and the closed-loop optimiser are implemented
> and benchmarked. Foundation-model pre-training is in progress; weights and the
> `DDS-Bench` leaderboard are not yet published. Every number below states
> whether it came from real measured data or from a mechanistic surrogate.

---

## Why a domain model rather than a general one

Three properties of formulation data defeat general-purpose models:

1. **A formulation is an unordered set of triples**, not a sequence of tokens or
   a graph of atoms: `(excipient, functional role, amount)`, conditioned on a
   dosage form and an API, under process parameters. Permutation invariance
   across components but binding between an amount and *its own* excipient is a
   structural requirement, not a modelling preference.
2. **Labels are scarce and locked in prose.** The quantitative
   composition→performance pairs exist almost exclusively inside patent
   examples and ASCII tables. This is what the distillation stage exists to
   unlock.
3. **The feasible region is tiny and interaction-dominated.** In the benchmark
   below, formulations meeting all six specifications occupy **0.035%** of the
   design space, and the controlling effects are non-additive (cholesterol
   abolishes the phospholipid phase transition; ammonium-sulfate gradient
   loading saturates against drug load). This is precisely where one-factor-at-
   a-time optimisation fails and where sample-efficient search pays.

---

## Architecture

```
          +----------------------------------------------------------+
 corpus   |  A61K9 patent mining (HUPD / USPTO full text)             |
          |  + NAL Pharmaceuticals patent family (16 publications)    |
          +---------------------------+------------------------------+
                                      v
          +----------------------------------------------------------+
 teacher  |  distill_extract.py - large model extracts structured     |
          |  (composition, process, performance) records from prose   |
          +---------------------------+------------------------------+
                                      v
          +----------------------------------------------------------+
 student  |  dds_former.py - DDS-Former encoder                       |
          |  pre-train: masked component modelling (MCM)              |
          |  fine-tune: per-CQA heads with presence masking           |
          |  distil:    soft teacher targets + hard labels            |
          +---------------------------+------------------------------+
                                      v
          +----------------------------------------------------------+
 loop     |  ai_dds_engine.py - composite desirability over six CQAs, |
          |  composite expected improvement, batch active learning,   |
          |  prospective error settled on every proposal              |
          +----------------------------------------------------------+
```

### The six modules

| Module | What it predicts | Status |
|---|---|---|
| **AI-Formulation** | excipient selection, ratios, multi-objective optimum | implemented |
| **AI-Release** | in-vitro release / dissolution curves, degradation kinetics | implemented |
| **AI-PK/PD** | IVIVC, plasma exposure, dose and interval | implemented |
| AI-Permeation | skin, mucosal and tissue flux | planned |
| AI-Manufacturing | process parameters, CQAs, batch consistency | planned |
| AI-Experimental Agent | autonomous experiment design, active learning | partial (loop implemented) |

---

## Results

### 1. Release model trained on real patent data

Trained on the formulation/dissolution series in **US9114073B2** (oral dissolving
film, 6 formulations x 2 administration sites, n = 14 measured points):

```
ln t_dissolution = 3.342 + 1.000*ln(f_alginate) + 0.024*ln(MW_PVP/1e5) - 0.240*sublingual
```

| Metric | Value |
|---|---|
| Leave-one-out MAPE | **17.8 %** |
| log-R2 | **0.881** |
| RMSE | 2.29 min |

The coefficient on `ln(f_alginate)` is ~1.00 — dissolution time is first-order
in the alginate mass fraction of the polymer blend, which matches the patent's
own qualitative claim quantitatively.

*Limitation: n = 14, and the two PVP-molecular-weight examples also change the
alginate viscosity grade, so the MW term is confounded and must not be
interpreted alone.*

### 2. Closed loop vs. classical DOE vs. manual OFAT

Case: **NAL1276 ropivacaine long-acting liposome**. Six formulation/process
variables, six specifications, 5 % log-normal assay noise, 40-experiment budget,
20 random seeds, identical surrogate wet-lab for all three arms.

| Metric | **AI closed loop** | Classical DOE-RSM | Manual OFAT |
|---|---|---|---|
| Spec-passing hit rate | **5.50 %** [3.54, 7.46] | 0 % | 1.12 % [0.32, 1.93] |
| Seeds succeeding within budget | **17/20** | 0/20 | 7/20 |
| Experiments to first passing formulation (median) | **25** | not reached | 23 |
| Prospective prediction MAPE | **7.17 %** | 33.44 % | no model |
| Cost per viable formulation | **US$63.6k** | none produced | US$311k |

Per-CQA prospective MAPE:

| | Encapsulation | Burst | t50 | 7-day release | Leakage | Plasma Cmax |
|---|---|---|---|---|---|---|
| **AI closed loop** | **8.44** | **7.36** | **6.41** | 5.93 | **7.26** | **7.60** |
| Classical DOE-RSM | 14.19 | 63.20 | 11.99 | **5.06** | 42.60 | 63.62 |

The quadratic response surface fails on the right-skewed CQAs (burst, leakage,
Cmax) because it fits exponential responses in linear space; the Gaussian-process
surrogates model them in log space. On the near-symmetric 7-day release the two
are equivalent — the gap is a modelling-space choice, not a blanket "AI beats
statistics" claim.

> **Read this before quoting the table.** The wet lab in this benchmark is a
> *mechanistic surrogate with noise*, not Galaxy Bio measurements. Its
> interaction terms are set from established liposome formulation science, not
> tuned to favour any arm. It validates the loop's control logic and the KPI
> definitions; it is **not** a prediction of NAL1276's real performance. The KPI
> code is unchanged when real data arrives — only `ground_truth()` is swapped
> for the LIMS interface.

Honest caveats: manual OFAT's desirability trajectory *leads* early before
stalling — high desirability is not spec compliance — and the AI arm itself
missed in 3/20 seeds.

---

## The four auditable metrics

Every claim is defined so a third party can re-measure it against traditional
DOE or manual formulation under matched conditions.

| Metric | Definition |
|---|---|
| Prediction error | Per-CQA prospective MAPE. **Predictions are frozen when the work order is issued**, before the experiment runs. |
| Prospective hit rate | Fraction of model-proposed formulations meeting every specification. |
| Cycle-time reduction | Experiments to the first passing formulation, plus within-budget success rate. |
| Cost per formulation | Total experiments x unit cost / passing formulations. |

Prediction freezing is what makes this auditable rather than a backtest.

---

## Repository layout

```
src/   ai_dds_engine.py    closed-loop engine + three-arm benchmark
       dds_former.py       DDS-Former architecture, MCM, distillation losses
       distill_extract.py  teacher extraction pipeline
       mine_hupd.py        A61K9 corpus mining from HUPD
data/  patent formulation + performance corpora, Bio-FX dataset, data inventory
results/ KPI summaries, per-run benchmark results, figures
docs/  full design specification (Chinese)
```

Reproduce the benchmark:

```bash
pip install -r requirements.txt
python -c "
import sys; sys.path.insert(0,'src')
import ai_dds_engine as E
X, Y, pl = E.arm_ai(seed=13)
print(E.kpis(X, Y, pl))"
```

## Data provenance

All patent text in `data/` is extracted from **published patent documents**
(USPTO/WIPO full text, and the Harvard USPTO Patent Dataset, CC-BY-SA-4.0).
No proprietary or unpublished formulation data is included in this repository.

## License

Code: MIT. Derived patent-text corpora inherit the licence of their source
(HUPD: CC-BY-SA-4.0).

---

## 中文摘要

本仓库是面向**药物递送制剂(DDS)**的领域基础模型与闭环优化引擎。

通用药物 AI 优化的是分子,DDS 优化的是分子之外的一切——辅料配比、载体结构、
工艺参数、释放动力学。本工程针对后者:一个以处方为原生表示的 Transformer
(**DDS-Former**)、一条把专利正文蒸馏为结构化监督信号的教师-学生管线,以及一个
会提出下一批实验并对自己的前瞻性误差即时结算的闭环优化器。

**已完成**:专利语料构建、三个一期模块(AI-Formulation / AI-Release / AI-PK-PD)、
闭环三臂对照基准。**进行中**:基础模型预训练与 `DDS-Bench` 公开基准。

核心结果:

- Bio-FX 口溶膜溶解时间模型(真实专利数据,n=14):留一交叉验证 **MAPE 17.8%**,
  log-R² 0.881;`ln(f_海藻酸盐)` 系数 ≈ 1.00,与专利定性描述定量吻合。
- NAL1276 脂质体闭环基准(同预算 40 批、20 种子、同条件对照):AI 闭环命中率
  **5.50%**、17/20 种子成功、前瞻 MAPE **7.17%**、单处方成本 **US$63.6k**;
  经典 DOE-RSM 在预算内 0/20 命中;人工 OFAT 命中率 1.12%、成本 US$311k。

**必须同时说明**:上述闭环基准的"湿实验"是带噪声的机理代理模型,交互项依据脂质体
制剂学既有机理设定,不是 Galaxy Bio 实测数据;它验证闭环控制逻辑与指标口径,不构成
对 NAL1276 实际性能的预测。真实数据接入后 KPI 计算代码无需改动。
