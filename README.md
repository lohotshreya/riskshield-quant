# riskshield-quant

An institutional-grade asset allocation engine that discards Markowitz mean-variance optimization flaws, optimizing capital strictly through the mathematical lens of **Hierarchical Risk Parity (HRP)** combined with extreme **Macroeconomic Stress-Testing Scenario Simulators**.

## Core Value Proposition

Standard Modern Portfolio Theory (MPT) relies heavily on expected return tracking metrics which are notoriously unstable. A minor decimal change shifts structural outputs massively. 
**RiskShield.quant** removes returns from structural clustering completely, running on the following pipeline:
1. Translates correlation profiles into physical distance matrices ($d_{i,j} = \sqrt{\frac{1 - \rho_{i,j}}{2}}$).
2. Builds hierarchical clustering graphs via Scipy Linkage.
3. Distributes portfolio capital down branches via recursive bisection inverse variance metrics.

---

## Installation & Setup

1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/lohotshreya/riskshield-quant.git](https://github.com/lohotshreya/riskshield-quant.git)
   cd riskshield-quant
