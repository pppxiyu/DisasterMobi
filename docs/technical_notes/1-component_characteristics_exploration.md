# Technical Notes


## Part 1 — `component_characteristics/func/`: Origin × Destination Functionality Cross-Tabs

### 1.1 Functional labels — data source

- **EPA Smart Location Database (SLD) v3**, queried from the public ArcGIS
  REST service (`geodata.epa.gov/.../SmartLocationDatabase/MapServer`, layer 7)
  with paginated `GEOID20 LIKE '<state+county>%'` queries (page size 2000).
- **Spatial unit: census block group.** Every SLD variable is reported per
  census block group. No spatial re-aggregation or crosswalk is involved.
- **Provenance and vintages.** The pipeline touches a single table (SLD), but
  SLD itself is an EPA *compilation* that pre-merges several federal sources
  onto one block-group grid:
  - the employment fields (`totemp`, `e8_*`) come from **LEHD LODES Workplace
    Area Characteristics (WAC), 2017** — Census Bureau administrative job
    counts (state UI wage records, near-census of formal employment;
    self-employment excluded).  *Workplace* Area Characteristics means jobs
    are counted **at the census block where the job is located**, not where
    workers live
  - the population/housing fields (`totpop`, `counthu`, `hh`) come from the
    **Census/ACS, 2018**.
  - Note the mismatch of vintages (2017 jobs, 2018 housing) is inherited from SLD.

### 1.2 Functional classification (per block group, computed on the fly)

Raw category scores:

| Category | Source fields | NAICS content and examples |
|---|---|---|
| residential | `counthu × RESIDENTIAL_WEIGHT` (=1.0) | housing units (proxy; no "residential employment" exists) |
| commercial | `e8_ret + e8_off + e8_svc` | the broad commerce-and-office bundle. Retail trade 44-45 (supermarkets, malls, car dealers, gas stations, clothing stores); office industries 51-53, 55 (TV/internet/publishing companies, banks, insurers, real-estate agencies, corporate headquarters); service industries 54, 56, 81 (law/accounting/consulting/engineering firms, temp agencies, building cleaning and landscaping, car repair shops, hair and nail salons, dry cleaners, religious and civic organisations). Note that "service" here is professional/support/personal services — NOT food service |
| leisure | `e8_ent` | broader than the literal word "entertainment": arts, entertainment and recreation 71 (theatres, museums, stadiums, casinos, gyms, golf courses, marinas) **plus accommodation and food services 72 (hotels, motels, restaurants, bars, cafés, fast food)**. In practice restaurant and lodging jobs dominate this tier, so a high-`leisure` block group is typically a dining/hotel/tourism cluster, not just an amusement district |
| industrial | `e8_ind` | agriculture–construction–manufacturing–wholesale–transport 11, 21, 22, 23, 31-33, 42, 48-49 (farms, mines, utilities, construction firms, factories, wholesalers, trucking and warehousing) |
| health | `e8_hlth` | health care and social assistance 62 (hospitals, clinics, doctors' offices, nursing homes, social-assistance facilities) |
| public | `e8_ed + e8_pub` | education 61 (schools, universities) and public administration 92 (government offices, police, courts) |

**TF-IWF reweighting** (`LANDUSE_WEIGHTING='tf_iwf'`, `IWF_SCALE=1.0`). Plain
shares label ~75–80% of block groups "residential" because housing mass is
ubiquitous. The fix mirrors TF-IDF from text mining (documents = block groups,
words = the six categories) and works in three steps:

1. **Local share (TF)** — "how residential / commercial / … is this place,
   seen from the inside":

$$
\text{local share} \;=\;
\frac{\text{this category's mass in the block group}}
     {\text{the block group's total mass (housing + jobs)}}
$$

2. **Rarity weight (IWF)** — a category that is everywhere (residential) gets
   a small weight, a scarce and concentrated one (leisure, industrial) gets a
   large one. The exponent β below is the config knob `IWF_SCALE`: β = 1.0 uses
   the weight as is; larger β pushes labels further toward rare categories:

$$
\text{rarity weight} \;=\;
\left[\;\ln\!\left(
\frac{\text{total mass of all categories, region-wide}}
     {\text{this category's mass, region-wide}}
\right)\right]^{\beta}
$$

3. **Combine and rescale** — the labelling threshold is applied to these
   reweighted shares:

$$
\text{reweighted share} \;=\;
\frac{\text{local share} \times \text{rarity weight}}
     {\text{sum of the six (local share} \times \text{rarity weight) products}}
$$


**Labelling rule.** `dominant_category` = the argmax category if its
reweighted share > `LANDUSE_DOMINANT_THRESHOLD = 0.4`, else **Mix**;
**Unknown** when a block group has neither housing nor jobs (zero in both
cities currently). Hard labels (one category per block group) follow the
paper's EULUC convention; a soft alternative (distributing each flow by the
outer product of endpoint share vectors, `M = Sᵀ F S`) is documented in
`space_function.py` but not used.



## Part 2 — The Two Correlation Analyses

Both analyses correlate **per-component characteristics** across one city's
components (each component is one observation; BR n = 20, FM n = 25; both
cities are analysed **separately**, never pooled). The shared feature table
(one row per component) is built in memory at every run and consumed directly
by the plots; it is not written to disk.


### 2.1 Functional characteristics (computed from the Part 1 cross-tab `M`)

**`share_from_<cat>`** (6 columns) = row sums of the renormalised matrix —
  the fraction of the component's flow **departing from** each function
  (outflow side). The six values sum to 1.
**`share_to_<cat>`** (6 columns) = column sums — the fraction **arriving
  at** each function (inflow side). Sum to 1.

- Mix and Unknown rows/columns are **dropped** and the
  remaining 6×6 matrix is renormalised to sum 1, so the features describe
  substantive function types only. A component whose kept-category mass is
  zero would get NaN features (none currently).
- **Compositional caveat.** Because each share block sums to 1, raising one
  category's share mechanically lowers the others; some negative correlations
  are partly built-in.


### 2.2 Temporal characteristics (computed from the CLEAN NORMAL segment only)

Input: `W[:104]` — the first 13 days × 8 slots of the
temporal factor; buffer and disaster columns are never used here.

| Metric | Type | Definition | Notes |
|---|---|---|---|
| `weekday_ratio` | continuous | (mean daily total over the 9 normal weekdays) ÷ (mean daily total over the 4 normal weekend days) | > 1 = weekday-dominated (commute-like); < 1 = weekend-dominated (leisure-like). NaN if the weekend mean is zero. |
| `peak_slot` | categorical | the within-day profile's argmax slot, labelled '6-8h' … '20-22h', or 'weekend' | Numeric code = slot index 0–7, weekend = 8 so it sorts after all slots. |
| `peak_period` | categorical | the day-period band with the highest **width-corrected** intensity — mean `p(s)` per slot inside the band, so a wider band gains no mechanical advantage — or 'weekend' | Bands (hour bounds, resolution-independent): morning_peak 06–10, midday 10–16, evening_peak 16–20, night 20–22. |

  


### 2.3 Resilience characteristics (computed from the DISASTER segment)

For each component,
`r(d) = daily_total(d) / baseline(day-type of d)` over the 15 disaster days
(`d = 0` is landfall day). Daily totals = sum of the 8 slot values per day.

- **Weekday and weekend baseline**: weekday baseline = mean daily total over the
  9 normal weekdays; weekend baseline = mean over the 4 normal weekend days.
  Buffer days enter **neither** the baseline **nor** the curve;
- The curve is smoothed with a **3-day centred rolling mean**
  (`min_periods=1`);
- A zero baseline yields NaN (not inf); such a component gets NaN in **all**
  resilience metrics. Interpretation caveat: small baselines inflate `r` for
  disaster-emergent components (their denominators are tiny). **This problem is not obvious upon checking**
- **Metrics — all defined so HIGHER = WORSE.**

| Metric | Definition | Notes |
|---|---|---|
| `drop_depth` | `1 − min(r)` | 1 = total stop. **Negative = rose above baseline** |
| `early_collapse` | `(15−1) − argmin(r)` = `14 − lowest_day` | How soon the curve bottomed out: 14 = lowest on landfall day, 0 = lowest on the final day. |
| `recovery_day` | `1 + (last day with r < 0.9)`; 0 if never below | Days until `r` reaches the threshold **and stays there**. The threshold 0.9 (parameter `recovery_threshold`) absorbs ±10% noise around normal. **Value 15 (= window length) means not recovered within the window**. |
| `recovery_deficit` | `1 − mean(r over the last 3 disaster days)` | Shortfall still left at the window end. 0 = fully recovered; **negative = overshoot** above normal. |
| `cum_loss` | `Σ_d max(0, 1−r(d))` | Resilience-triangle area in day-equivalents. **Above-baseline excess is clipped to 0, never credited against losses**. A component that never dips gets 0. |


### 2.4 Correlation computation (for some analysis)

- **Spearman rank correlation** (`scipy.stats.spearmanr`), computed pair by
  pair with pairwise NaN dropping; a pair with fewer than 3 valid components
  returns NaN. Ties (e.g. several components censored at `recovery_day = 15`)
  are handled with average ranks.
- **Significance stars**: two-sided p < 0.05 (\*) and p < 0.01 (\*\*),
  **uncorrected for multiple comparisons**. The resilience block alone has
  5 × 13 = 65 cells, so ~3 single stars are expected by chance at the 5%
  level; treat isolated \* as suggestive and look for coherent patterns
  (e.g. from/to of the same category agreeing) — \*\* is the stronger
  evidence tier. Rough significance bars: |ρ| ≳ 0.44 (n = 20) and ≳ 0.40
  (n = 25) for p < 0.05.



## Part 3 — Context-aware


### 3.1 Gram-matching

> Wang et al., *Understanding Urban Dynamics via Context-Aware Tensor Factorization with Neighboring Regularization*, IEEE TKDE 2020. The model itself is a 3D Tucker factorization; what matters here is the **form of its context-aware term (Gram-matching)**.

**What they do.** The context-aware term is the two terms in the paper's Eq. 17:

$$\alpha\,\|W - OO^\top\|_F^2 \;+\; \beta\,\|W - DD^\top\|_F^2$$

- $W\in\mathbb{R}^{M\times M}$: zone–zone POI cosine similarity ($M$ = number of zones; $w_{pq}=\frac{u_p\cdot u_q}{\|u_p\|\|u_q\|}$, $u_p$ = zone $p$'s POI-category proportion vector) — an **external** similarity matrix.
- $O\in\mathbb{R}^{M\times I}$, $D\in\mathbb{R}^{M\times J}$: the origin / destination projection matrices (**rows = zones, columns = that mode's $I/J$ components**). $\alpha,\beta$ are weights.
- **Meaning.** $OO^\top$ contracts away the component axis, giving an $M\times M$ matrix whose $(p,q)$ entry $=o_p\cdot o_q$ is the inner product of two zones' loading vectors (their latent-space similarity), so it has **the same shape as $W$ and can be subtracted from it**. The term pulls the zone similarity *learned from trajectories* ($OO^\top$) toward the zone similarity *computed from POI* ($W$). This is **Gram-matching** (directly fitting a similarity matrix), not regression ($D$ is the destination-side analogue).

**Scale caveat.** $OO^\top$'s entries $o_p\cdot o_q=\|o_p\|\,\|o_q\|\cos\theta_{pq}$ scale with the norms (not confined to $[0,1]$), while the target $W$ is a cosine in $[0,1]$. So fitting $OO^\top\to W$ does two jobs at once: align directions ($\cos\theta_{pq}\to W_{pq}$) **and** drive the norms toward 1 (the diagonal $(\,\|o_p\|^2-1)^2$ is the purest such pressure). Since $\|o_p\|$ would otherwise carry zone $p$'s flow volume (even when spatial components are normalized across spatial units), this mismatch fights reconstruction and distorts $O$. It is harmless **for the reference paper** only because the paper reads the *direction* of $O$ — communities are $\arg\max_i O[x,i]$ — never the magnitude.

### 3.2 Shared-factor

> Chen et al., *A Context-Aware Nonnegative Matrix Factorization Framework for Traffic Accident Risk Estimation via Heterogeneous Data*, IEEE MIPR 2018. This one does context-aware on a **2D matrix factorization** (not a tensor). **Notation here follows the paper's own** — in particular its $H$ is a **temporal** factor, not this pipeline's $H$ (the spatial factor).

**Base factorization.** Accident-risk matrix $X\in\mathbb{R}^{n\times m}$ (region × time; $X_{r,t}$ = number of accidents in region $r$ at slot $t$), $X\approx RH$: $R\in\mathbb{R}^{n\times l}$ is the region factor, $H\in\mathbb{R}^{l\times m}$ the time factor. The context data are **also factorized**, sharing the **same region factor $R$**. Fitting the dense, complete $Y,Z$ constrains $R$.

- Geographic features $Y\in\mathbb{R}^{n\times c}$: a **region × category** matrix. Columns are ~12 POI categories (food, bank, hospital, shopping, education, …) plus road-intersection count and road length. **Entries are counts**: $Y[r,c]$ = how many of category $c$ are in region $r$ (number of those POIs / intersections / road length).
- Human density $Z\in\mathbb{R}^{n\times q}$: a **region × fine-time-slot** matrix (slots of e.g. 15 min, $q$ of them). **Entries are human density**: $Z[r,t]$ = region $r$'s GPS-log density at slot $t$ (pedestrians + drivers).
- $G\in\mathbb{R}^{l\times c}$ = category factor, $U\in\mathbb{R}^{l\times q}$ = fine-time factor — each **translates** the same region embedding $R[r,:]$ into category space / human-time space.

**Context-aware part of the objective (excerpt of Eq. 4):**

$$\tfrac{\lambda_1}{2}\,\|Y-RG\|_F^2\;+\;\tfrac{\lambda_2}{2}\,\|Z-RU\|_F^2$$

All factors $\ge0$; the objective also carries an L2 term $\tfrac{\lambda_3}{2}(\|R\|_F^2+\|H\|_F^2+\|G\|_F^2+\|U\|_F^2)$.

**Scale caveat.**
- **How it works in the reference.** In $R$, a location's vector (its values across the components) carries scale; $Y$ and $Z$ carry scale too, being raw counts. The shared $R$, mapped through $G$/$U$, reconstructs $Y$/$Z$, and the scales can be made to match.
- **Why $G,U$ shape only the component-distribution, never a per-location scale.** A location's predicted feature is $Y[r,c]=\langle R[r,:],\,G[:,c]\rangle$ — an inner product with the **same** column $G[:,c]$ for every location ($U$ likewise for $Z$). So $G,U$ are single global maps; they cannot rescale location by location. The only lever left to explain a location's $Y$/$Z$ is **where it sits across the components** (the distribution of $R[r,:]$). That is what makes the method work — context enters through the component-distribution, not through a per-location rescaling.
- **What it therefore requires.**  Because $G,U$ can't rescale per location, the **per-location magnitudes of $R$ and of $Y$/$Z$ must co-vary** — otherwise $R$ cannot reconstruct $Y$/$Z$. In the paper this holds: $X$ (accidents) and $Y$ (POI counts) are both ∝ a region's activity, and $R[r,:]$ encodes exactly that activity, so they line up.
- **Why this pipeline is at risk.** Here the shared factor (our $H$) carries **flow** magnitude. If $Y$/$Z$ were endpoint POI counts, those scale with endpoint **mass**, which need not be ∝ flow. 


### 3.3 在本管线用这两种方法，需要的调整



前提：本管线 $H$ 的列携带 flow scale，且 gauge 搬不走（见 §3.1 / §3.2 的 Scale caveat）。下面按方法列出要做的调整。

**通用（两种都要）——换 solver。** sklearn 的 NMF（现在 `solver='cd'`，[decomposition.py:84](utils_pattern_analysis/decomposition.py:84) / [:123](utils_pattern_analysis/decomposition.py:123)）只做纯重构 $\|X-W_tH\|$，塞不进任何 context 项（$\mathrm{tr}(HLH^\top)$ 或 $\|Y-RG\|$）。所以得离开它、自己写非负乘法更新 / 投影梯度 solver。

**要用 Gram（§3.1），两步：**

- **造边-边相似度 $S$。** Gram 是拿 $H^\top H$ 去拟合一个目标 $S=S^{orig}\otimes S^{dest}$（$n_{OD}\times n_{OD}$）。它**稠密、百万级**，且**不能简单稀疏化**——没填的格子被当成"相似度 0 = 该正交"会压坏（除非额外 mask）。这是 Gram 在这儿最重的负担。
- **处理 norm 压平**（§3.1 caveat：Gram 把 $\|H[:,j]\|$ 压向 1、抹平 flow）。三选一：context 项里把 $H$ 列归一化（→ 余弦匹配）/ 只匹配非对角（跳过对角的 $\|H[:,j]\|^2\!\to\!1$）/ 或——若下游只读方向（argmax 类、scale-invariant、不读列模长）——像论文一样直接忽略。

**要用 CMF（§3.2），两步：**

- **造 $Y$ 让它与 flow 同向。** flow $\sim m_o m_d/$距离 是**乘积/距离**型，所以用**外积 ÷ 距离**：$Y[j,:]=\mathrm{vec}(p_o\otimes p_d)/\text{dist}^\beta$（$p_o,p_d$ = 两端**原始** POI 计数向量）——大小 $\propto m_o m_d/$距离 $\sim$ flow、方向 = "起点类 × 终点类"的功能类型（即 per-边、原始计数、再除距离的 O×D 交叉表）。别拼接/相加（那是和 $m_o+m_d$、对不上），也别 share / 平均（把质量除掉 = volume-free，最差）。
- **兜残留张力**（§3.2 caveat）。引力也只**近似**，actual flow 偏离引力处仍有 mismatch：用软 $\lambda$ 容忍、或在 context 项里把共享因子列归一化。

**或者绕开 → §3.4 Laplacian（图正则 / GNMF，Cai et al. 2011, *Graph Regularized NMF*）。** 不拟合整张相似度，只加一项 $\lambda\,\mathrm{tr}(HL^{OD}H^\top)=\tfrac{\lambda}{2}\sum_{p,q}S^{OD}_{pq}\|H[:,p]-H[:,q]\|_2^2$——让功能相似（$S^{OD}_{pq}$ 大）的两条边 loading 相近（$L^{OD}=D_{\deg}-S^{OD}$ 是图拉普拉斯）。它用**稀疏 top-κ 图**（不用稠密 $S$）、**不要求 co-scale**、对角恒 0 **不锚绝对尺度**——把上面 Gram 的"稠密 + 压平"、CMF 的"co-scale"全省掉。详见 §3.4。**GNMF**（Cai et al. 2011, *Graph Regularized NMF*）
