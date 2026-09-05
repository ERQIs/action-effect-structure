# Two-DoF Experiment: Preliminary Results

## Main finding

In two action dimensions, the local reverse relation remains approximately first-order odd:

```math
E_y(-u)\approx-E_y(u),
```

and the effects of the two joints are approximately additive to first order. At second order, however, a cross term $u_1u_2$ appears. Axis-only actions make this term identically zero, so no number of additional axis-only samples can identify it. The experiment demonstrates both the value and the identifiability boundary of the candidate structure.

## Why a cross term appears

The second link is oriented by $q_1+q_2$. When both joints move, its angular increment is $u_1+u_2$, whose quadratic expansion contains

```math
(u_1+u_2)^2=u_1^2+2u_1u_2+u_2^2.
```

Separate observations of $(u_1,0)$ and $(0,u_2)$ can constrain the square terms but provide no supervision for $u_1u_2$. A small number of joint actions activates this missing direction.

## Setup

- System: one planar two-link arm.
- Observation: the two-dimensional elbow and endpoint coordinates, giving four dimensions.
- Learned map: $(y,u)\mapsto y^+-y$.
- Every sample belongs to one continuous trajectory; no reversed trajectory is replayed after a reset.
- `axis_only`: positive actions of one joint at a time.
- `axis_plus_10pct_joint`: approximately 90% positive axis actions and 10% simultaneous positive joint actions.
- Models: free, first-order $J(y)u$, and a corrected model containing all three quadratic monomials.
- Action radius: 0.2 rad; reported values aggregate three random seeds.
- `train_size` counts all collected transitions; 80% are used for fitting and 20% for early stopping, with no extra uncounted validation interaction.

## Prediction results

Mean NMSE after 2,000 collected transitions is shown below; lower is better.

| Interaction condition | Model | Negative axis | Positive joint | Negative joint | Mixed-sign joint |
|---|---:|---:|---:|---:|---:|
| Axis only | Free | 0.02680 | 0.001702 | 0.03355 | 0.01013 |
| Axis only | First order | 0.02593 | **0.001389** | 0.03154 | 0.009875 |
| Axis only | First + second order | **0.002891** | 0.001953 | **0.006313** | **0.005760** |
| 10% joint | Free | 0.02746 | 0.001067 | 0.03019 | 0.01236 |
| 10% joint | First order | 0.02645 | 0.001106 | 0.03245 | 0.009848 |
| 10% joint | First + second order | **0.003379** | **0.0001024** | **0.002982** | **0.001720** |

The most informative comparison holds the corrected model and total interaction budget fixed while changing the action composition:

- Positive joint actions: $1.953\times10^{-3}$ to $1.024\times10^{-4}$, a **19.1×** reduction.
- Negative joint actions: a **2.12×** reduction.
- Mixed-sign joint actions: a **3.35×** reduction.
- Negative axis actions: no improvement beyond seed-level variation.

This selective gain matches the cross-term explanation: joint experience supplies information about joint directions rather than improving every query. At 2,000 transitions, the $12\times12$ state-grid coverage is 98.8% for axis-only trajectories and 98.1% for trajectories with joint actions, making broader state coverage an unlikely explanation.

With axis-only data, the corrected model is slightly worse than the pure first-order model on positive joint actions. The first-order model assumes additivity directly, while the corrected model learns $u_1^2$ and $u_2^2$ without observing the missing $u_1u_2$ term. An incomplete higher-order correction can therefore be worse than a clean first-order approximation.

## One-step control

After 2,000 transitions, mean visual error for negative-axis targets is:

| Interaction condition | Free | First order | First + second order |
|---|---:|---:|---:|
| Axis only | 0.01349 | 0.01308 | **0.00561** |
| 10% joint | 0.01392 | 0.01340 | **0.00587** |

For mixed-sign joint targets, the corrected model trained with joint actions reaches 0.00761 mean error, compared with 0.00901 for the first-order model. Control gains are smaller than prediction gains because the candidate actions lie on a finite grid whose approximately 0.007 discretization error creates a visible floor.

## Interpretation and scope

The one-dimensional structure extends naturally to $E_y(-u)\approx-E_y(u)$ and to first-order composition through $J(y)u$. Its boundary is equally explicit: higher-order interactions require action directions that activate their monomials. One transition can constrain untried actions only within the approximation order and identifiable subspace supported by the collected inputs.

The observations remain noiseless coordinates, the dynamics are purely kinematic, the model sizes are close but not yet exactly matched, and only three seeds are used. The 10% joint-action fraction is a diagnostic choice rather than an optimized minimum. These results support a mechanism hypothesis, not a real-robot sample-efficiency claim.

## Output files

- `results/structure_diagnostics.png`: true reverse, scaling, and two-joint additivity violations across action scales.
- `results/prediction_axis_only.png`: learning curves with positive axis actions only.
- `results/prediction_axis_plus_10pct_joint.png`: learning curves after adding a small fraction of joint actions.
- `results/control_axis_only.png` and `results/control_axis_plus_10pct_joint.png`: one-step control errors.
- `results/metrics_per_seed.csv` and `results/metrics_aggregate.csv`: complete numerical results.


