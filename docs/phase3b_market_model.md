# Phase 3B: Two-Stage Stochastic BESS Optimization — Market Model

## 1. Settlement Logic

The optimizer manages three distinct power flows:
1.  **$x_t$ (Stage 1 Decision)**: Day-Ahead Market (DAM) commitment. This is the scheduled sell (positive) or buy (negative) quantity cleared in the DAM. This must be the **same for all scenarios**.
2.  **$y_{s,t}$ (Stage 2 Decision)**: Real-Time Market (RTM) physical dispatch. This is the actual physical net injection from the BESS in scenario $s$.
3.  **$d_{s,t} = y_{s,t} - x_t$**: Deviation. The mismatch between the physical RTM dispatch and the DAM schedule is settled at the RTM price.

### Revenue Formula
The total revenue for a single scenario $s$ is the sum of DAM settlement and RTM deviation settlement:

$$R_s = \sum_{t=1}^{24} \left( p_{DAM,s,t} \cdot x_t + p_{RTM,s,t} \cdot (y_{s,t} - x_t) \right)$$

This simplifies to:
$$R_s = \sum_{t=1}^{24} \left( p_{RTM,s,t} \cdot y_{s,t} + (p_{DAM,s,t} - p_{RTM,s,t}) \cdot x_t \right)$$

This reveals that the optimizer chooses $x_t$ to exploit the spread between DAM and RTM expected prices.

## 2. Optimization Formulation (LP)

### Objective
Maximize the expected revenue across all $N$ scenarios, adjusted for degradation and deviation penalties:

$$\max \frac{1}{N} \sum_{s=1}^N R_s - c_{deg} \sum_{s,t} (y_{charge,s,t} + y_{discharge,s,t}) - \lambda_{dev} \sum_{s,t} |d_{s,t}|$$

### Constraints
For each scenario $s \in \{1 \dots N\}$ and hour $t \in \{1 \dots 24\}$:

1.  **Power Balance**:
    $y_{s,t} = y_{discharge,s,t} - y_{charge,s,t}$
2.  **Power Limits**:
    $0 \le y_{charge,s,t} \le P_{max}$
    $0 \le y_{discharge,s,t} \le P_{max}$
    $-P_{max} \le x_t \le P_{max}$
3.  **SOC Dynamics**:
    $soc_{s,t} = soc_{s,t-1} + \eta_c \cdot y_{charge,s,t} - \frac{1}{\eta_d} y_{discharge,s,t}$
    $E_{min} \le soc_{s,t} \le E_{max}$
4.  **Terminal Constraint**:
    $soc_{s,24} \ge soc_{s,0}$ (Ensures battery is not depleted at end-of-day)
5.  **Deviation Bound**:
    $|y_{s,t} - x_t| \le dev_{max}$ (Ensures physically implementable deviations)

## 3. Physical Parameters (Derived from Sample Trace)
- **$P_{max} = 50$ MW**
- **$E_{max} = 180$ MWh**
- **$E_{min} = 20$ MWh**
- **$\eta = 0.9487$** (Charge/Discharge efficiency, 90% round-trip)
