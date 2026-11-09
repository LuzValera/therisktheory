# Topic 4: Free-Parameter Credibility, Claim-Reserving Models, and Statistical Diagnostics

## Mathematical Concepts Implemented

This module implements the non-parametric empirical Bayes credibility framework and credibility-based loss reserving models within the `therisktheory.credibility` package:

1. `free_parameter_credibility`: Computes the classical Bühlmann (1967) and Bühlmann-Straub (1970) non-parametric credibility estimators for longitudinal risk portfolios without specifying parametric prior distributions.
2. `credibility_reserve_estimate`: Synthesizes actuarial claim reserving with credibility theory, formalizing the Bornhuetter-Ferguson (1972) and Cape Cod bridges as credibility-weighted compromises between empirical link-ratio development and apriori pricing expectations.
3. `reserve_model_diagnostics`: Evaluates statistical goodness-of-fit, portfolio balance conditions, shrinkage metrics, and theoretical mean squared prediction error reductions.

---

## Model Definitions & Mathematical Formulations

### 1. Non-Parametric Empirical Bayes Credibility (Bühlmann-Straub Model)

Consider a portfolio consisting of $I \ge 2$ risk contracts (or cohorts). Each risk $i \in \{1, \dots, I\}$ is observed over $T_i \ge 2$ operational periods (e.g., years). For period $t \in \{1, \dots, T_i\}$, we record:
- Loss experience (loss ratio or pure claim amount per exposure unit): $X_{it}$
- Known natural weight or exposure volume: $P_{it} > 0$

Each risk contract possesses an unobservable latent risk parameter $\Theta_i$, assumed to be independent and identically distributed draws from an unknown structure distribution $U(\theta)$.

#### Conditional Moment Assumptions
For given risk parameter $\Theta_i$:
1. Conditional hypothetical mean:
   $$\mathbb{E}[X_{it} \mid \Theta_i] = m(\Theta_i)$$
2. Conditional process variance:
   $$\text{Var}(X_{it} \mid \Theta_i) = \frac{s^2(\Theta_i)}{P_{it}}$$
3. Conditional independence: Conditionally on $\Theta_i$, the annual observations $X_{i1}, \dots, X_{i T_i}$ are mutually independent.

#### Structural Parameters
The collective distribution is characterized by three fundamental structural parameters:
- **Collective mean:**
  $$\mu = \mathbb{E}[m(\Theta_i)]$$
- **Expected process variance (within-risk variance):**
  $$s^2 = \mathbb{E}[s^2(\Theta_i)] = \mathbb{E}[P_{it} \text{Var}(X_{it} \mid \Theta_i)]$$
- **Variance of hypothetical means (between-risk variance):**
  $$a = \text{Var}(m(\Theta_i))$$

---

### 2. Unbiased Estimation of Structural Parameters (Free-Parameter Framework)

In free-parameter credibility, structural parameters are estimated directly from sample data without imposing parametric families (such as Gamma-Poisson or Normal-Normal).

#### Notation & Aggregations
- Total exposure for risk $i$:
  $$P_{i\bullet} = \sum_{t=1}^{T_i} P_{it}$$
- Total portfolio exposure:
  $$P_{\bullet\bullet} = \sum_{i=1}^I P_{i\bullet} = \sum_{i=1}^I \sum_{t=1}^{T_i} P_{it}$$
- Individual exposure-weighted sample mean:
  $$\bar{X}_i = \frac{1}{P_{i\bullet}} \sum_{t=1}^{T_i} P_{it} X_{it}$$
- Overall portfolio exposure-weighted sample mean:
  $$\bar{X}_{\bullet\bullet} = \frac{1}{P_{\bullet\bullet}} \sum_{i=1}^I P_{i\bullet} \bar{X}_i = \frac{1}{P_{\bullet\bullet}} \sum_{i=1}^I \sum_{t=1}^{T_i} P_{it} X_{it}$$

#### Estimators
1. **Unbiased Estimator of Process Variance $\widehat{s}^2$:**
   $$\widehat{s}^2 = \frac{1}{\sum_{i=1}^I (T_i - 1)} \sum_{i=1}^I \sum_{t=1}^{T_i} P_{it} (X_{it} - \bar{X}_i)^2$$
   This estimator is strictly unbiased: $\mathbb{E}[\widehat{s}^2] = s^2$.

2. **Unbiased Estimator of Between-Risk Variance $\widehat{a}$:**
   Let the Bühlmann-Straub exposure normalization coefficient be:
   $$c^* = P_{\bullet\bullet} - \frac{\sum_{i=1}^I P_{i\bullet}^2}{P_{\bullet\bullet}} = \frac{1}{P_{\bullet\bullet}} \sum_{i < j} P_{i\bullet} P_{j\bullet} > 0$$
   The between-risk sum of squares is:
   $$\text{SSB} = \sum_{i=1}^I P_{i\bullet} (\bar{X}_i - \bar{X}_{\bullet\bullet})^2$$
   The raw unbiased estimator is:
   $$\widehat{a}_{\text{raw}} = \frac{1}{c^*} \left[ \text{SSB} - (I - 1) \widehat{s}^2 \right]$$

3. **Boundary Condition and Truncation of $\widehat{a}$:**
   Because $a = \text{Var}(m(\Theta)) \ge 0$ is a variance, negative estimates are mathematically outside the parameter space. In homogeneous portfolios where between-risk differences are smaller than sampling noise, $\widehat{a}_{\text{raw}}$ may be negative. Standard actuarial practice (Bühlmann & Straub 1970; Kaas et al. 2008; Klugman et al. 2019) mandates truncation:
   $$\widehat{a} = \max\left(0.0, \widehat{a}_{\text{raw}}\right)$$
   When $\widehat{a} = 0$, the between-risk heterogeneity is negligible; the credibility factor collapses to $Z_i = 0$ for all contracts, and all risks are charged the collective mean $\mu$.

4. **Bühlmann Parameter Ratio & Credibility Weights:**
   $$k = \begin{cases} \frac{\widehat{s}^2}{\widehat{a}} & \text{if } \widehat{a} > 0 \\ +\infty & \text{if } \widehat{a} = 0 \end{cases}$$
   $$Z_i = \frac{P_{i\bullet}}{P_{i\bullet} + k} = \frac{\widehat{a} P_{i\bullet}}{\widehat{a} P_{i\bullet} + \widehat{s}^2} \in [0.0, 1.0]$$

5. **Credibility-Weighted Grand Mean & Premiums:**
   $$\widehat{\mu}_Z = \frac{\sum_{i=1}^I Z_i \bar{X}_i}{\sum_{i=1}^I Z_i} \quad \left(\text{or } \bar{X}_{\bullet\bullet} \text{ if } \sum Z_i = 0\right)$$
   $$\widehat{P}_i = Z_i \bar{X}_i + (1 - Z_i) \widehat{\mu}_Z$$

---

### 3. Actuarial Claim Reserving & Credibility Bridges

In general insurance claim reserving, estimating ultimate claims and Incurred But Not Reported (IBNR) reserves can be formulated as a credibility weighting problem between empirical loss development and pricing expectations (Bornhuetter & Ferguson 1972; Mack 2000).

For an accident cohort $j$:
- Cumulative reported claims to date: $C_j \ge 0$
- Apriori expected ultimate claims: $M_j > 0$ (e.g., $M_j = \text{Premium}_j \times \text{Expected Loss Ratio}$)
- Cumulative development factor ($CDF$): $f_j \ge 1.0$
- Emergence proportion (percent reported): $p_j = \frac{1}{f_j} \in (0, 1]$

#### The Credibility Reserving Formula
$$\widehat{U}_{\text{cred}, j} = Z_j \widehat{U}_{\text{CL}, j} + (1 - Z_j) M_j = Z_j (C_j \cdot f_j) + (1 - Z_j) M_j$$

#### Structural Special Cases
1. **Full Credibility ($Z_j = 1.0$):**
   Pure Chain-Ladder (Development) method: $\widehat{U} = C_j f_j$, Reserve $= C_j (f_j - 1)$.
2. **Bornhuetter-Ferguson Bridge ($Z_j = p_j = 1/f_j$):**
   Setting credibility equal to emergence proportion yields:
   $$\widehat{U}_{\text{BF}} = \frac{1}{f_j} (C_j f_j) + \left(1 - \frac{1}{f_j}\right) M_j = C_j + (1 - p_j) M_j$$
   $$\widehat{R}_{\text{BF}} = \widehat{U}_{\text{BF}} - C_j = (1 - p_j) M_j$$
   The Bornhuetter-Ferguson reserve is the expected unreported loss, unaffected by random fluctuations in historical reported claims.
3. **No Credibility ($Z_j = 0.0$):**
   Pure pricing expectation: $\widehat{U} = M_j$, Reserve $= \max(0, M_j - C_j)$.

---

### 4. Statistical Model Diagnostics

The `reserve_model_diagnostics` function computes:
1. **Portfolio Balance / Imbalance:**
   $$\text{Imbalance} = \sum_{i=1}^I P_{i\bullet} (\widehat{P}_i - \bar{X}_i)$$
   Under the credibility-weighted mean $\widehat{\mu}_Z$, the total predicted losses replicate total observed claims: $\text{Imbalance} = 0$.
2. **Theoretical Mean Squared Prediction Error:**
   In the Bühlmann model, the mean squared error of the linear predictor is:
   $$\text{MSE}(Z_i) = \mathbb{E}[(\widehat{P}_i - m(\Theta_i))^2] = (1 - Z_i) a$$
   The relative variance reduction compared to charging everyone the collective mean $a$ is exactly $Z_i \times 100\%$.
3. **Goodness of Fit:** Weighted Mean Absolute Error (MAE) and Weighted Root Mean Squared Error (RMSE).

---

## Domain & Parameter Bounds

- Loss matrix: $I \ge 2$ risks, $T \ge 2$ periods.
- Natural weights / exposures: $P_{it} \ge 0$, with $\sum_t P_{it} > 0$ for each risk.
- Development factors: $f \ge 1.0$.
- Credibility weights: $Z \in [0.0, 1.0]$.
- Reported claims: $C \ge 0$, expected claims $M > 0$.

---

## Bibliographic References

- **Bühlmann, H. (1967).** *Experience Rating and Credibility*. ASTIN Bulletin, 4(3), 199–207. https://doi.org/10.2143/AST.4.3.5630444
- **Bühlmann, H., & Straub, E. (1970).** *Glaubwürdigkeit für Schandensätze*. Bulletin of the Association of Swiss Actuaries, 70, 111–133.
- **Kaas, R., Goovaerts, M., Dhaene, J., & Denuit, M. (2008).** *Modern Actuarial Risk Theory: Using R* (2nd ed.). Springer, Chapter 8 (Credibility Theory, Sections 8.2–8.4). https://doi.org/10.1007/978-3-540-70998-5
- **Klugman, S. A., Panjer, H. H., & Willmot, G. E. (2019).** *Loss Models: From Data to Decisions* (5th ed.). Wiley, Chapters 16–18.
- **Bornhuetter, R. L., & Ferguson, R. E. (1972).** *The Actuary and IBNR*. Proceedings of the Casualty Actuarial Society, 59(112), 181–195.
- **Mack, T. (2000).** *Credibility Theory for Loss Reserving*. ASTIN Bulletin, 30(2), 333–347. https://doi.org/10.2143/AST.30.2.504639
