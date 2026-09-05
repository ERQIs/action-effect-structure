# Learning Visual Action–Effect Structure from Limited Physical Interaction

> Research Positioning Paper · 5 September 2026  
> GitHub: https://github.com/ERQIs/action-effect-structure

## Abstract

When a robot must adapt to a new body, viewpoint, or changing physical system, it needs interaction-derived knowledge of how its actions change its observations. Real interaction is costly, while a continuous action space cannot be sampled exhaustively. The central issue is therefore not only how many transitions are observed, but also how much each transition constrains the effects of other actions.

This proposal asks: **Under limited physical interaction, which local action–effect relations can be identified, reused, and corrected? Can a learning system exploit these relations to reduce the number of real transitions required to reach a specified level of prediction and control performance?**

As a first candidate hypothesis, the proposal examines whether short-horizon, small-action effects can be decomposed into a shared state-dependent first-order component and learnable higher-order deviations:

```math
E_y(u)=J(y)u+R_y(u).
```

This hypothesis assigns part of the relation among actions to model structure while learning its state-dependent coefficients and deviations from experience. Preliminary experiments show that a purely first-order relation is insufficient at finite action magnitudes. With a second-order correction, prediction error on unseen negative actions falls by approximately 134× in a one-DoF arm. In a two-DoF arm, axis-only actions cannot identify the $u_1u_2$ cross term; adding approximately 10% joint actions within the same interaction budget reduces prediction error on positive joint actions by approximately 19.1×. The next stage will test whether this advantage persists with learned visual representations, multiple viewpoints and embodiments, realistic dynamics, and continual interaction on physical robots.

---

## 1. Scientific Problem Definition

### 1.1 Overall problem

Let an embodied system derive a control representation from observation history and available internal state:

```math
y_t=\phi(o_{t-k:t},s_t).
```

After executing a continuous action $u_t$, the system receives a new observation. Local goal-directed control selects an action from the current representation $y_t$ and goal $g$ so that the next observation moves toward the goal. For a new robot, an uncalibrated viewpoint, a compliant body, or a changed actuation system, this mapping must be identified and continually updated through interaction.

A continuous action space cannot be enumerated. If a learner treats each action–effect pair mainly as an isolated sample, it needs dense coverage of the Cartesian product of states and actions. If action effects share local structure, one observed transition may also narrow the possible outcomes of a family of unexecuted actions.

The scientific question is:

> **Under limited physical interaction, which local action–effect relations can be identified, reused, and corrected? Can a learning system exploit these relations to reduce the number of real transitions required to reach a specified level of prediction and control performance?**

### 1.2 Why this problem matters

One important route toward embodied intelligence is continual learning through acting in the physical world, observing the outcome, and revising behavior. Such interaction cannot be parallelized without limit: an action consumes physical time, produces one realized outcome, and may introduce wear, safety risk, or irreversible environmental change. Changes in the body, payload, camera, or environment also require continued adaptation from new interaction. The utilization of real transitions therefore directly limits the speed and scope of continual learning.

Local action–effect knowledge is a low-level capability shared by many tasks. Whether the objective is to approach a visual target, adjust a gripper, manipulate an object, or change viewpoint, the agent must estimate how candidate actions will move its current observation. Independent trial and error for every state and action scales poorly with state, action, and viewpoint dimensions. Reliable constraints among action effects could allow the same physical experience to support more prediction and control decisions.

This work studies one concrete source of efficiency: extracting reusable relations among actions from limited causal experience. It connects three broader aims: adaptation to unknown or changing embodiment, reusable sensorimotor knowledge from continuous experience, and a measurable structure–experience trade-off.

### 1.3 Three dimensions of the question

The following dimensions jointly address the overall problem without assuming that any particular relation holds:

1. **The relation:** Which stable relations exist among the observation consequences of different actions? Should they be expressed in pixels, learned visual features, or another control-relevant coordinate system?
2. **Learning value:** When a candidate relation is introduced as a structural bias, can a learner predict unexecuted actions and achieve target control more accurately under matched model capacity and physical interaction budget? How many real transitions are required to reach a fixed performance level?
3. **Scope:** Which relations transfer across states, viewpoints, tasks, and embodiments? How are they altered by environmental motion, dynamics, contact, and observation changes?

The first dimension determines what knowledge can be reused, the second whether it improves experience efficiency, and the third how broadly it generalizes. A local differential relation is introduced below as the first physical hypothesis to test.

---

## 2. Research Positioning and Literature Review

| Research line | Established foundation | Relevance to this proposal |
|---|---|---|
| Physical interaction and autonomous exploration | Guided Policy Search repeatedly fits local models from robot rollouts; Goal Babbling organizes autonomous motor exploration around goals [4,7] | Robots can learn directly from physical interaction, and the organization of experience affects learning efficiency |
| Visual servoing | The interaction matrix or image Jacobian locally relates camera or joint velocity to visual-feature velocity [1] | A local action–observation operator is a mature and control-relevant object |
| Uncalibrated visual control | A visual Jacobian can be estimated online for closed-loop control without known camera and robot parameters [2] | Local mappings can be identified from interaction without complete manual calibration |
| Learned visual Jacobians | Neural Jacobian Fields learns how actuator commands differentially move visible surface points from real robot video and controls rigid, compliant, and soft robots [3] | Demonstrates that visual-conditioned Jacobians are learnable and can serve as a common control interface |
| Data informativity and system identification | Parameter identifiability depends on whether inputs sufficiently excite the relevant directions; sample count cannot replace missing directional information [5] | Interaction efficiency depends on action-direction coverage as well as data volume |
| Equivariant and structured learning | When known structure matches the task, equivariant representations share statistical strength across related state–action pairs and improve data efficiency [6] | Architectural constraints can convert a physical relation into statistical sharing |

Together, these lines establish that visual–action mappings can be learned from physical interaction, local differential relations are useful representations, and both input coverage and structural bias shape how experience is used. Neural Jacobian Fields is the closest technical anchor: it learns first-order visuomotor Jacobians on diverse physical robots from two to three hours of multiview random-action data and identifies higher-order transients and dynamics as open directions [3].

The broader questions remain: which other relations among action effects can be learned; whether such relations reduce the physical interaction required for fixed prediction and control performance under partial action coverage; whether they survive first- or third-person visual representation learning; and whether different embodiments share structure more general than a particular Jacobian. This proposal uses the local differential relation as a first experimental entry point and then expands the observation space, embodiment, and dynamics.

---

## 3. Physical / Structural Prior

### 3.1 Source of the intuition

Consider a robot at a fixed local operating state. If a small positive joint command moves the gripper slightly upward in the image, a similar negative command will often move it in the opposite direction. Doubling a sufficiently small command often approximately doubles the short-term displacement. When two joints make small motions together, the resulting visual change often contains a composition of their individual effects.

These observations follow from the local response of a continuous body over a short interval. They do not specify a complete control law, but suggest that the consequences of different actions may share local coefficients. A physical action may therefore reveal more than its own realized outcome: it may constrain the directions, scales, or compositions of neighboring actions.

The first candidate physical prior is: **At suitable temporal, action, and representation scales, visual action effects contain a shared component expressible by a low-order local model, while systematic deviations from that component remain learnable from experience.**

### 3.2 Formalization

Let $y$ be the current control representation and $u$ a small control perturbation around the current operating point over a short interval. After subtracting predictable zero-action drift, define the action effect as

```math
E_y(u)
=
\mathbb E[y_{t+1}-y_t\mid y_t=y,u_t=u]
-
\mathbb E[y_{t+1}-y_t\mid y_t=y,u_t=0].
```

If the local dynamics, actuation interface, and observation representation are sufficiently smooth in $u$, then

```math
E_y(u)
=
J(y)u
+
\frac12\mathcal H(y)[u,u]
+
O(\|u\|^3).
```

The first-order term implies three reusable approximate relations:

```math
E_y(-u)\approx-E_y(u),
\qquad
E_y(cu)\approx cE_y(u),
\qquad
E_y(u+v)\approx E_y(u)+E_y(v).
```

Second- and higher-order terms describe systematic deviations as action magnitude, state, and interaction mode change. Nonsmooth events such as contact switching and occlusion mark boundaries of the local structure or its representation.

### 3.3 Structure–experience trade-off

An unconstrained model estimates the full function $(y,u)\mapsto E$ from data. A structured model lets actions near the same state share $J(y)$ and a small number of higher-order coefficients. Every transition constrains these shared quantities, and predictions for other actions are composed from the same coefficients.

The structure also determines which experience is informative. An action direction provides one projected constraint; multiple independent directions jointly identify a Jacobian; higher-order interactions require actions that activate the corresponding monomials. For example, a two-dimensional second-order model contains

```math
H_{12}(y)u_1u_2.
```

This term is always zero under $(u_1,0)$ and $(0,u_2)$. At least some joint actions are therefore required to identify it. Architecture reuses the candidate relation, while experience estimates its coefficients and supplies directions absent from existing data.

### 3.4 Working hypotheses

- **H1 — Local structure:** At sufficiently small action and temporal scales, $J(y)u$ explains the dominant change in action effect, while the residual grows in a measurable way with action scale.
- **H2 — Experience efficiency:** Introducing the candidate relation as a structural bias allows a learner to reach the same unseen-action prediction and local-control error with fewer transitions.
- **H3 — Representation and embodiment:** A suitable visual representation preserves this local relation. Its coefficients vary with body and viewpoint, while more general properties—continuity, low-order structure, sparsity, or compositionality—may transfer across embodiments.

---

## 4. Mathematical Modeling and Evaluation

### 4.1 Model families

The first stage constructs capacity- and compute-matched comparisons among three model families:

```math
\begin{aligned}
\text{Free:}\quad
&\hat E(y,u)=m_\theta(y,u)-m_\theta(y,0),\\
\text{First-order:}\quad
&\hat E(y,u)=J_\theta(y)u,\\
\text{Correctable:}\quad
&\hat E(y,u)=J_\theta(y)u+r_\theta(y,u).
\end{aligned}
```

The minimal two-dimensional instance uses a quadratic correction:

```math
r_\theta(y,u)
=H_{11,\theta}(y)u_1^2
+H_{12,\theta}(y)u_1u_2
+H_{22,\theta}(y)u_2^2.
```

The training objective is

```math
\mathcal L
=
\mathbb E\|\hat E(y,u)-E_y(u)\|^2
+\lambda\,\mathbb E\|r_\theta(y,u)\|^2.
```

A small $\lambda$ encourages the model to use the shared low-order component first and introduce corrections when required by data.

### 4.2 Identifiability

Write the two-dimensional quadratic action basis as

```math
\psi(u)
=
[u_1,u_2,u_1^2,u_1u_2,u_2^2]^\top,
\qquad
\hat E(y,u)=C_\theta(y)\psi(u).
```

Locally collected actions form a design matrix $\Psi=[\psi(u_1),\ldots,\psi(u_N)]^\top$. Its rank and condition number describe which coefficients are constrained by experience and provide direct criteria for later active action selection.

### 4.3 From effect prediction to control

Given current representation $y$ and local goal $g$, the controller selects

```math
u^*(y,g)
=
\arg\min_{u\in\mathcal U_\rho}
d\big(y+\hat E(y,u),g\big)
+\beta\|u\|^2,
```

then acts, observes again, and updates in closed loop.

Primary measurements include effect NMSE on held-out action directions and magnitudes, one-step and closed-loop goal error, calibration under structural violation, and the number of real interactions needed to reach control error $\epsilon$:

```math
N_\epsilon
=
\min\{N:\operatorname{ControlError}(N)\le\epsilon\}.
```

---

## 5. Computation and Implementation

The current PyTorch implementation learns from a single continuous interaction trajectory. Each training input is the current visual state and executed action $(y_t,u_t)$; the supervised target is the short-horizon visual effect $y_{t+1}-y_t$. Training covers only part of the action space. Reverse, scaled, and joint actions are held out to test how far one physical experience constrains unexecuted actions. A forward effect model supports action search for target control and is updated after each new observation.

The computational pipeline is

```math
\text{continuous interaction}
\rightarrow
\text{local effect learning}
\rightarrow
\text{held-out action prediction}
\rightarrow
\text{goal-conditioned action selection}
\rightarrow
\text{residual / identifiability analysis}.
```

Code, complete per-seed results, and reproduction instructions are available in the project repository: [github.com/ERQIs/action-effect-structure](https://github.com/ERQIs/action-effect-structure). Implementation details appear in Appendix A.

---

## 6. Preliminary Experiments and Results

### 6.1 Main results

The minimal experiments produce three findings:

1. **An appropriate structural bias can improve effect prediction for unseen actions.** In the one-DoF experiment, a first- plus second-order model trained only on positive actions reaches approximately 134× lower NMSE on unseen negative actions than a similarly sized free model.
2. **The first-order relation is a useful starting point, while finite actions require learnable systematic deviations.** The hard first-order model performs similarly to the free model; learning even-order curvature sharply improves reverse-action prediction.
3. **The directional composition of experience determines which structure can be learned.** In two action dimensions, additional axis-only data cannot reveal the $u_1u_2$ interaction. Adding approximately 10% joint actions under the same model and total interaction budget reduces positive-joint-action NMSE by approximately 19.1×.

![Preliminary results](assets/preliminary_results_summary.png)

*Figure 1. Mean prediction NMSE on a logarithmic scale. One-DoF values use five random seeds; two-DoF values use three. Complete learning curves and per-seed results are linked in Appendix A.*

| Key test | Baseline | First + second-order model | Result |
|---|---:|---:|---:|
| 1 DoF, 1,000 positive actions, unseen negative-action NMSE | Free: 0.02511 | 0.0001875 | Approximately 134× lower |
| 2 DoF, 2,000 axis-only actions, negative-axis NMSE | First order: 0.02593 | 0.002891 | Approximately 9.0× lower |
| 2 DoF, same structured model, positive-joint NMSE | Axis only: 0.001953 | With 10% joint: 0.0001024 | Approximately 19.1× lower |

Control results show the same tendency at a smaller scale. Mean error on one-DoF negative targets decreases from 0.001468 for the free model to 0.000958 for the corrected model. In the two-DoF axis-only setting, negative-axis target error falls from 0.01349 to 0.00561.

### 6.2 Experimental design

The one-DoF environment maps joint state $q$ to a visual location on a circle, $y=(\cos q,\sin q)$. Its training trajectory contains only positive actions; negative actions are held out. The two-DoF environment uses a planar two-link arm whose visual state contains elbow and endpoint coordinates. Training first contains only positive single-joint actions, then adds approximately 10% simultaneous positive joint actions while holding the total interaction budget fixed. Both experiments learn $(y_t,u_t)\mapsto y_{t+1}-y_t$, compare free, first-order, and first- plus second-order models, and use the learned effect model for one-step target control.

Full training procedures, model sizes, data splits, and per-seed results are given in Appendix A and the linked experiment directories.

### 6.3 Current interpretation

The experiments provide a sequence of evidence:

1. **First-order structure is present but insufficient.** The hard first-order and free models perform similarly on unseen negative actions in one dimension; even-order curvature is the dominant systematic error at finite magnitude.
2. **A first- plus second-order model can reuse one-sided experience.** Positive-action data identify the first-order component shared across signs and the second-order component that does not change sign.
3. **Action-direction coverage determines higher-order identifiability.** In two dimensions, axis-only samples provide no information about $u_1u_2$. A small number of joint actions selectively improves joint directions while leaving negative-axis prediction almost unchanged.

The evidence shifts the research from one reverse-action symmetry to a broader question: **Which low-order action structures transfer to unexecuted actions, and which small set of directionally complementary interactions identifies their deviations?** The current experiments establish the mechanism and measurement procedure in the simplest setting. They suggest that suitable action relations can reduce the interaction required to predict unexecuted actions, while showing that input coverage and structural form are equally important.

---

## 7. Next Research Plan

The long-term objective is to make embodied learning systems better able to identify, test, reuse, and correct action–observation effect structure, so that limited experience constrains unexecuted actions and supports new goal-directed behavior. The specific algorithmic form of this capability remains open; the current first- plus higher-order model is only its smallest candidate instance.

The research will progress through the formation of this capability: from testing the minimal reuse mechanism to identifying unknown relations, actively testing them, and composing them into goal-directed intervention.

### Phase 1: Confirm that structural reuse reduces interaction

- Match model capacity, optimization budget, and training data between free and structured models; increase seeds and standardize data splits.
- Vary transition count, action radius, directional coverage, noise, and degree of structural violation.
- Use $N_\epsilon$ as the primary measure to distinguish higher accuracy at a fixed data size from fewer real interactions at matched performance.

This phase tests the basic causal chain: whether the algorithm reuses a candidate action relation and whether that reuse produces experience efficiency.

### Phase 2: Discover controllable change in visual observations

- Replace known state coordinates with first- and third-person RGB observations containing the robot body, camera motion, tools, external objects, and distractor motion.
- Without labels for robot pixels or objects, learn to separate changes that are consistently associated with action from exogenous changes.
- Compare pixels, keypoints, optical flow, and learned features to determine where action relations are most identifiable and reusable under limited interaction.

This phase moves from predefined control coordinates to potential control interfaces in observation and tests whether they can be found through a small number of interventions.

### Phase 3: Actively test and correct action–effect hypotheses

- Consider candidate relations including local linearity, reversal, scaling, composition, sparse coupling, delayed response, and piecewise modes.
- Select diagnostic actions that distinguish current hypotheses according to uncertainty, complementing random exploration with targeted tests.
- Create settings in which a controllable interface briefly appears, leaves the field of view, or changes its response; test whether stopping, reversal, and small probes can confirm, recover, or reject a relation with fewer transitions.
- Use residuals and uncertainty to estimate the scope of a relation and to correct or switch it when the structure fails.

This phase expands the prior from a fixed formula to the broader ability to find and test relations, and measures the experience saved by active identification.

### Phase 4: Compose local relations into goal-directed intervention

- Use confirmed action–effect relations to choose actions from desired observation changes and update them continuously in closed loop.
- Progress from recovering a controllable interface to approaching a visual target, controlling a gripper, establishing contact, and using a tool to affect a distant object.
- Test whether multiple local relations compose into longer control chains and reduce the additional interaction required for each new goal.

This phase connects local effect learning to goal-directed intervention and tests whether structural knowledge lowers the cost of acquiring new control capabilities.

### Phase 5: Validate across viewpoints, bodies, and real physical systems

- Compare fixed and wrist-mounted cameras, serial and bimanual arms, wheeled robots, dexterous hands, and compliant or underactuated mechanisms.
- Introduce inertia, friction, hysteresis, payload change, contact, sliding, collision, occlusion, and autonomous environmental motion to determine which relations persist and which must be relearned.
- Learn from continuous real-robot interaction that cannot be reset arbitrarily; vary camera, tool, payload, and actuation response to measure online adaptation.
- Test whether the ability to identify and reuse structure acquired in one viewpoint, task, or embodiment reduces the startup interaction cost in another.

All phases use a common comparison principle: match model capacity and compute, plot performance against the number of real transitions, and separately ablate structural representation, active testing, and relation reuse. The ultimate objective is a reproducible relationship among the structure-learning capability encoded by an algorithm, the physical experience it requires, and the intervention capability it acquires.

---

## References

[1] F. Chaumette and S. Hutchinson. “Visual Servo Control, Part I: Basic Approaches.” *IEEE Robotics & Automation Magazine*, 2006. https://doi.org/10.1109/MRA.2006.250573

[2] J. M. Sebastián et al. “Uncalibrated Visual Servoing Using the Fundamental Matrix.” *Robotics and Autonomous Systems*, 2009. https://doi.org/10.1016/j.robot.2008.04.002

[3] S. L. Li et al. “Controlling Diverse Robots by Inferring Jacobian Fields with Deep Networks.” *Nature*, 2025. https://doi.org/10.1038/s41586-025-09170-0

[4] S. Levine, N. Wagener, and P. Abbeel. “Learning Contact-Rich Manipulation Skills with Guided Policy Search.” *ICRA*, 2015. https://arxiv.org/abs/1501.05611

[5] H. J. van Waarde, J. Eising, H. L. Trentelman, and M. K. Camlibel. “Data Informativity: A New Perspective on Data-Driven Analysis and Control.” *IEEE Transactions on Automatic Control*, 2020. https://arxiv.org/abs/1908.00468

[6] A. K. Mondal, V. Jain, K. Siddiqi, and S. Ravanbakhsh. “EqR: Equivariant Representations for Data-Efficient Reinforcement Learning.” *ICML*, 2022. https://proceedings.mlr.press/v162/mondal22a.html

[7] M. Rolf, J. J. Steil, and M. Gienger. “Goal Babbling Permits Direct Learning of Inverse Kinematics.” *IEEE Transactions on Autonomous Mental Development*, 2010. https://doi.org/10.1109/TAMD.2010.2062511

---

## Appendix A. Preliminary Experiment Details

### A.1 One-DoF experiment

- State and observation: $q\in[-\pi,\pi)$, $y=(\cos q,\sin q)$.
- Action: $q^+=q+u$; training contains only $u>0$ with local radius $\rho=0.2$ rad.
- Data sizes: 50, 100, 200, 500, and 1,000; five random seeds.
- Parameter counts: Free 4,546; First-order 4,482; Correctable 4,612.
- Complete results: [experiments/one_dof/RESULTS.md](experiments/one_dof/RESULTS.md)

### A.2 Two-DoF experiment

- Robot: a planar two-link arm with unit link lengths.
- Observation: two-dimensional elbow and endpoint coordinates, for four dimensions total.
- Action: $u=(u_1,u_2)$.
- Data sizes: 200, 500, 1,000, and 2,000; three random seeds.
- “axis_only”: every transition applies a positive action to one joint.
- “axis_plus_10pct_joint”: approximately 10% simultaneous positive joint actions; all samples still come from one continuous trajectory.
- Current parameter counts: Free 4,868; First-order 5,000; Correctable 5,780. A matched run will use widths 70, 69, and 64, giving 5,744, 5,735, and 5,780 parameters.
- At 2,000 transitions, coverage of a $12\times12$ state grid is 98.8% and 98.1% for the two interaction conditions.
- Complete results: [experiments/two_dof/RESULTS.md](experiments/two_dof/RESULTS.md)

### A.3 Control and evaluation

Prediction is evaluated with mean squared error and normalized mean squared error. One-step control minimizes predicted goal error over a local action set. The current two-dimensional experiment uses a finite action grid, so effect prediction provides the primary structural evidence and control error serves as an initial downstream measurement.

## Appendix B. The two-joint cross term

The second link has orientation $q_1+q_2$. When both joints execute small actions, its angular increment is $u_1+u_2$. The second-order expansion contains

```math
(u_1+u_2)^2=u_1^2+2u_1u_2+u_2^2.
```

The two axis directions are sufficient to constrain both columns of the first-order Jacobian but cannot reveal the second-order cross coefficient. This example connects local physical structure to input informativity and provides a minimal setting for actively selecting joint actions.

## Appendix C. Repository Map

- One-DoF code and results: [experiments/one_dof/](experiments/one_dof/)
- Two-DoF code and results: [experiments/two_dof/](experiments/two_dof/)
- Main-figure generation script: [assets/make_preliminary_figure.py](assets/make_preliminary_figure.py)


