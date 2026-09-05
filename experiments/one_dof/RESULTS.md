# One-DoF Experiment: Preliminary Results

## What the experiment tests

The visual state of the one-joint arm is

$$
y(q)=(\cos q,\sin q).
$$

Training data come from one continuous trajectory containing only positive actions. Models use $(y_t,u_t)$ to predict $e_t=y_{t+1}-y_t$, and are then evaluated on negative actions that never occur in training and on one-step target control. The default local action radius is $\rho=0.2$ rad. Every reported value aggregates five random seeds.

## True structure diagnostics

Reverse, scaling, and additivity relations depart progressively from the first-order approximation as the action radius increases. At $\rho=0.2$:

| Relation | Normalized violation error |
|---|---:|
| $E_y(u)+E_y(-u)\approx0$ | 0.01215 |
| $E_y(2u)-2E_y(u)\approx0$ | 0.00305 |
| $E_y(u)-2E_y(u/2)\approx0$ | 0.00076 |

The first-order relation is therefore strong but not exact at this scale. The principal missing component in the reverse relation is the sign-invariant second-order curvature of circular motion.

## Learning results

With 1,000 positive-action transitions:

| Model | Positive-action NMSE | Unseen negative-action NMSE | Mean unseen negative-target error | Unseen negative-target success |
|---|---:|---:|---:|---:|
| Free | $2.07\times10^{-4}$ | 0.02511 | 0.001468 | 47.5% |
| Hard first-order | $2.18\times10^{-4}$ | 0.02636 | 0.001428 | 45.9% |
| First + second order | $1.77\times10^{-6}$ | $1.88\times10^{-4}$ | 0.000958 | 61.0% |

The hard first-order model does not clearly outperform the free network: both retain approximately 2.5%–2.6% NMSE on unseen negative actions as more data are added. In contrast, the second-order correction reduces negative-action NMSE by approximately 134× relative to the free model and 141× relative to the hard first-order model.

The prediction gain transfers only partially to control. Mean negative-target error falls by 34.7% relative to the free model and 32.9% relative to the first-order model. In this noiseless problem, the major difference is already visible with 50 transitions because the model form closely matches the true Taylor structure.

## Interpretation

The result does not support the simple hypothesis that imposing reverse odd symmetry alone is sufficient. The free network already receives a continuous action input and carries its own smooth interpolation bias. Moreover, a hard first-order model omits the even second-order term in

$$
E_y(u)=J(y)u+O(u^2).
$$

Negating the positive-action effect therefore cannot exactly recover the negative-action effect. Allowing $H(y)u^2$ lets positive-action data identify this curvature. The useful prior in this setting is better described as local low-order action-effect structure than as one isolated symmetry rule.

The smaller control gain also shows that improving the complete visual-effect prediction does not necessarily improve action selection by the same factor. Future experiments must separate errors that alter the selected action from errors that are irrelevant to the control objective.

## Scope

This experiment is a mechanism check rather than evidence of real-robot sample efficiency. It has one degree of freedom, noiseless handcrafted observations, no contact or actuator dynamics, and a corrected model closely matched to the true expansion. Positive-only training is a deliberate action-distribution gap used to test counterfactual generalization.


