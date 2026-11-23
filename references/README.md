# References

Record bibliographic sources used to validate mathematical implementations here.

Do not upload copyrighted books or papers unless redistribution is permitted.

Prefer bibliographic metadata and stable links/DOIs where available.

## Sources

### Kaas, Goovaerts, Dhaene and Denuit (2008)

Kaas, R., Goovaerts, M., Dhaene, J. and Denuit, M. (2008). *Modern Actuarial Risk Theory:
Using R*, 2nd ed. Springer. ISBN 978-3-540-70992-3, e-ISBN 978-3-540-70998-5.
DOI: [10.1007/978-3-540-70998-5](https://doi.org/10.1007/978-3-540-70998-5)
(metadata checked against Crossref).

| Used for | Location |
|---|---|
| Collective model and independence assumptions | Section 3.1, eq. (3.1), p. 41 |
| `expected_aggregate_loss`: `E[S] = E[N] E[X]` | Section 3.2, eqs. (3.2)–(3.3), pp. 42–43 |
| `pure_premium`: net premium / equivalence principle | Section 5.3, principle (a), p. 119 |
| `expected_value_premium`: `(1 + θ) E[S]` | Section 5.3, principle (b), p. 119 |
| Properties of the expected value principle | Section 5.3.1, pp. 120–121; Table 5.1, p. 122 |
| Lundberg fundamental equation and adjustment coefficient under proportional and excess-of-loss reinsurance | Chapter 4, Section 4.3 (eq. 4.13) & Section 4.6 (eq. 4.45) |
| Cramér-Lundberg exact ruin probability and Lundberg upper bounds | Chapter 4, Section 4.4 & Section 4.5 |

Modules:
- `therisktheory.solvency_pricing.premiums` (Topic 1)
- `therisktheory.collective_risk.reinsurance_effects` (Topic 5)

### Bowers, Newton L.; Gerber, Hans U.; Hickman, James C.; Jones, Donald A.; Nesbitt, Cecil J. (1997)

Bowers, N. L., Gerber, H. U., Hickman, J. C., Jones, D. A., and Nesbitt, C. J. (1997). *Actuarial Mathematics* (2nd ed.). Society of Actuaries.

| Used for | Location |
|---|---|
| Compound Poisson collective risk process with reinsurance and retained claim moments | Chapter 14, Sections 14.1–14.4 |
| Ruin tolerance initial capital relief under target ruin probability | Chapter 14, Section 14.6 |

Module: `therisktheory.collective_risk.reinsurance_effects` (Topic 5)

### Rolski, Tomasz; Schmidli, Hanspeter; Schmidt, Volker; Teugels, Jozef (1999)

Rolski, T., Schmidli, H., Schmidt, V., and Teugels, J. (1999). *Stochastic Processes for Insurance and Finance*. John Wiley & Sons. DOI: [10.1002/9780470317044](https://doi.org/10.1002/9780470317044).

| Used for | Location |
|---|---|
| Martingale approaches to Lundberg inequalities and optimal retention | Chapter 5, Sections 5.1–5.3 |

Module: `therisktheory.collective_risk.reinsurance_effects` (Topic 5)

### De Finetti, Bruno (1940)

De Finetti, B. (1940). *Il risco nel giuoco e nelle assicurazioni*. IV Congresso dell'Unione Matematica Italiana.

| Used for | Location |
|---|---|
| Comparative benchmarking between De Finetti safety index and Lundberg collective risk retention | Sections 1–3 |

Module: `therisktheory.collective_risk.reinsurance_effects` (Topic 5)
