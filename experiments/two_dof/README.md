# Two-DoF Local Visual Action-Effect Experiment

This is the minimal two-dimensional extension of the one-joint experiment. It uses one planar two-link arm, not two independent robots.

## Environment and learning problem

The joint state is $q=(q_1,q_2)$. The observation contains the planar elbow and endpoint coordinates:

```math
y(q)=(x_{\mathrm{elbow}},y_{\mathrm{elbow}},x_{\mathrm{tip}},y_{\mathrm{tip}})\in\mathbb R^4.
```

An action is a short-horizon joint displacement, equivalently an integrated velocity command over a fixed interval:

```math
u=(u_1,u_2),\qquad q^+=q+u.
```

The learned forward map remains

```math
(y_t,u_t)\longmapsto e_t=y_{t+1}-y_t.
```

## Interaction conditions

- `axis_only`: each step selects one joint and applies a positive action.
- `axis_plus_10pct_joint`: approximately 10% of the same continuous trajectory contains simultaneous positive actions of both joints.

Tests query in-distribution positive axis actions, unseen negative axis actions, positive and negative joint actions, and mixed-sign joint actions. The second condition is not mirrored data augmentation: it collects a small number of genuinely informative joint actions without increasing the total transition budget.

## Models

The free and first-order models match the one-DoF experiment. The corrected model is

```math
\hat E(y,u)=J_\theta(y)u
+H_{11,\theta}(y)u_1^2
+H_{12,\theta}(y)u_1u_2
+H_{22,\theta}(y)u_2^2.
```

Axis-only actions always have $u_1u_2=0$, so they cannot identify $H_{12}$ regardless of how many such transitions are collected.

## Run

```bash
python experiment.py --quick --output-dir results_quick
python experiment.py --output-dir results
```

`train_size` is the total number of physically collected transitions. Twenty percent is used for early stopping; no uncounted validation interactions are introduced. See [RESULTS.md](RESULTS.md) for the full results.


