# One-DoF Local Visual Action-Effect Experiment

This experiment asks whether a model that observes only positive actions of a one-joint arm can predict unseen negative actions at the same visual states and use those predictions for one-step target control.

## Environment and learning problem

The arm endpoint moves on the unit circle:

```math
y(q)=(\cos q,\sin q),
\qquad
q^+=q+u.
```

The training data form one continuous trajectory and contain only $u>0$. The input is $(y_t,u_t)$ and the supervised target is

```math
e_t=y_{t+1}-y_t.
```

Negative actions are queried only at test time and are never added to training.

## Models

- `free`: $\hat E(y,u)=u\,m_\theta(y,u)$, with no enforced action relation beyond $E(y,0)=0$.
- `first_order`: $\hat E(y,u)=J_\theta(y)u$.
- `corrected`: $\hat E(y,u)=J_\theta(y)u+H_\theta(y)u^2$.

All models minimize action-effect mean squared error. The term $H_\theta(y)u^2$ is the one-dimensional instance of the proposal's general correction $r_\theta(y,u)$.

## Evaluation

The script evaluates true reverse, scaling, and additivity violations; in-distribution positive prediction; unseen negative prediction; doubled action magnitudes; positive and negative one-step target control; and learning curves as the number of collected transitions changes.

## Run

```bash
python experiment.py --quick --output-dir results_quick
python experiment.py --output-dir results
```

See [RESULTS.md](RESULTS.md) for the reported full-run results and their interpretation.


