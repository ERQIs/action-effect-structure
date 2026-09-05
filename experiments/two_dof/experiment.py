from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def wrap_angles(q: np.ndarray) -> np.ndarray:
    return (np.asarray(q) + np.pi) % (2.0 * np.pi) - np.pi


def observe(q: np.ndarray) -> np.ndarray:
    """Elbow and end-effector positions of a unit-length two-link arm."""
    q_array = np.asarray(q, dtype=np.float32)
    q1 = q_array[..., 0]
    q12 = q_array[..., 0] + q_array[..., 1]
    elbow_x = np.cos(q1)
    elbow_y = np.sin(q1)
    tip_x = elbow_x + np.cos(q12)
    tip_y = elbow_y + np.sin(q12)
    return np.stack((elbow_x, elbow_y, tip_x, tip_y), axis=-1).astype(np.float32)


def true_effect(q: np.ndarray, u: np.ndarray) -> np.ndarray:
    return observe(wrap_angles(q + u)) - observe(q)


@dataclass
class Config:
    rho: float = 0.2
    min_action_fraction: float = 0.5
    train_sizes: tuple[int, ...] = (200, 500, 1000, 2000)
    seeds: tuple[int, ...] = (0, 1, 2)
    diagonal_fractions: tuple[float, ...] = (0.0, 0.1)
    epochs: int = 900
    validation_interval: int = 20
    early_stop_checks: int = 15
    hidden_dim: int = 64
    learning_rate: float = 2e-3
    weight_decay: float = 1e-6
    residual_penalty: float = 1e-4
    test_state_count: int = 96
    test_action_count: int = 8
    controller_state_count: int = 48
    controller_grid_size: int = 31
    controller_action_penalty: float = 1e-5
    controller_success_threshold: float = 0.02


def condition_name(diagonal_fraction: float) -> str:
    if diagonal_fraction == 0.0:
        return "axis_only"
    return f"axis_plus_{int(round(100 * diagonal_fraction))}pct_joint"


def generate_positive_trajectory(
    count: int,
    rho: float,
    min_action_fraction: float,
    diagonal_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """One continuous trajectory containing only positive joint velocities."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(-np.pi, np.pi, size=2).astype(np.float32)
    q_values = np.empty((count, 2), dtype=np.float32)
    actions = np.zeros((count, 2), dtype=np.float32)
    action_types = np.empty(count, dtype=np.int64)

    for index in range(count):
        magnitude = float(rng.uniform(min_action_fraction * rho, rho))
        if rng.random() < diagonal_fraction:
            u = np.asarray([magnitude, magnitude], dtype=np.float32) / np.sqrt(2.0)
            action_types[index] = 2
        else:
            joint = int(rng.integers(0, 2))
            u = np.zeros(2, dtype=np.float32)
            u[joint] = magnitude
            action_types[index] = joint
        q_values[index] = q
        actions[index] = u
        q = wrap_angles(q + u).astype(np.float32)

    return {
        "q": q_values,
        "y": observe(q_values),
        "u": actions,
        "effect": true_effect(q_values, actions),
        "action_type": action_types,
    }


DIRECTIONS: dict[str, np.ndarray] = {
    "positive_axis": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    "negative_axis": np.asarray([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float32),
    "positive_joint": np.asarray([[1.0, 1.0]], dtype=np.float32) / np.sqrt(2.0),
    "negative_joint": np.asarray([[-1.0, -1.0]], dtype=np.float32) / np.sqrt(2.0),
    "mixed_joint": np.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
    / np.sqrt(2.0),
}


def evenly_spaced_rows(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return values[indices]


def generate_query_set(
    q_states: np.ndarray,
    rho: float,
    action_count: int,
    category: str,
) -> dict[str, np.ndarray]:
    magnitudes = np.linspace(0.35 * rho, rho, action_count, dtype=np.float32)
    actions = np.concatenate(
        [magnitudes[:, None] * direction[None, :] for direction in DIRECTIONS[category]],
        axis=0,
    )
    q = np.repeat(q_states, len(actions), axis=0)
    u = np.tile(actions, (len(q_states), 1))
    return {"q": q, "y": observe(q), "u": u, "effect": true_effect(q, u)}


def split_collected_data(
    trajectory: dict[str, np.ndarray], train_size: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Use every fifth collected transition for validation; no hidden interactions."""
    indices = np.arange(train_size)
    validation_mask = indices % 5 == 0
    fit_mask = ~validation_mask
    train = {
        key: value[:train_size][fit_mask]
        for key, value in trajectory.items()
        if key != "action_type"
    }
    validation = {
        key: value[:train_size][validation_mask]
        for key, value in trajectory.items()
        if key != "action_type"
    }
    return train, validation


def make_mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class EffectModel(nn.Module):
    model_name: str

    def residual(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(y)


class FreeEffectModel(EffectModel):
    """Flexible in u, with only the exact zero-action anchor imposed."""

    model_name = "free"

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = make_mlp(6, 4, hidden_dim)

    def forward(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        zero_u = torch.zeros_like(u)
        at_u = self.net(torch.cat((y, u), dim=-1))
        at_zero = self.net(torch.cat((y, zero_u), dim=-1))
        return at_u - at_zero


class FirstOrderEffectModel(EffectModel):
    model_name = "first_order"

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.jacobian_net = make_mlp(4, 8, hidden_dim)

    def forward(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        jacobian = self.jacobian_net(y).reshape(-1, 4, 2)
        return torch.einsum("bij,bj->bi", jacobian, u)


class CorrectedEffectModel(EffectModel):
    """First order plus all degree-two monomials in the two actions."""

    model_name = "corrected"

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = make_mlp(4, 20, hidden_dim)

    def components(
        self, y: torch.Tensor, u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.net(y)
        jacobian = output[:, :8].reshape(-1, 4, 2)
        hessian_terms = output[:, 8:].reshape(-1, 4, 3)
        quadratic_basis = torch.stack(
            (u[:, 0].square(), u[:, 0] * u[:, 1], u[:, 1].square()), dim=-1
        )
        first_order = torch.einsum("bij,bj->bi", jacobian, u)
        second_order = torch.einsum("bik,bk->bi", hessian_terms, quadratic_basis)
        return first_order, second_order

    def residual(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.components(y, u)[1]

    def forward(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        first_order, second_order = self.components(y, u)
        return first_order + second_order


def make_models(config: Config) -> list[EffectModel]:
    return [
        FreeEffectModel(config.hidden_dim),
        FirstOrderEffectModel(config.hidden_dim),
        CorrectedEffectModel(config.hidden_dim),
    ]


def to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32)


def prediction_mse(model: EffectModel, data: dict[str, np.ndarray]) -> float:
    model.eval()
    with torch.no_grad():
        prediction = model(to_tensor(data["y"]), to_tensor(data["u"]))
        target = to_tensor(data["effect"])
        return float(torch.mean((prediction - target).square()).item())


def train_model(
    model: EffectModel,
    train_data: dict[str, np.ndarray],
    validation_data: dict[str, np.ndarray],
    config: Config,
) -> tuple[EffectModel, dict[str, float]]:
    y = to_tensor(train_data["y"])
    u = to_tensor(train_data["u"])
    target = to_tensor(train_data["effect"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_validation = math.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale_checks = 0

    for epoch in range(config.epochs):
        model.train()
        prediction = model(y, u)
        prediction_loss = torch.mean((prediction - target).square())
        residual_loss = torch.mean(model.residual(y, u).square())
        loss = prediction_loss + config.residual_penalty * residual_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % config.validation_interval == 0 or epoch == config.epochs - 1:
            validation_loss = prediction_mse(model, validation_data)
            if validation_loss < best_validation * (1.0 - 1e-5):
                best_validation = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale_checks = 0
            else:
                stale_checks += 1
                if stale_checks >= config.early_stop_checks:
                    break

    model.load_state_dict(best_state)
    return model, {"best_validation_mse": best_validation, "best_epoch": best_epoch}


def evaluate_prediction(
    model: EffectModel, data: dict[str, np.ndarray]
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        prediction = model(to_tensor(data["y"]), to_tensor(data["u"])).cpu().numpy()
    target = data["effect"]
    squared_error = np.square(prediction - target)
    return {
        "mse": float(np.mean(squared_error)),
        "nmse": float(np.sum(squared_error) / (np.sum(np.square(target)) + 1e-12)),
        "mean_vector_error": float(np.mean(np.linalg.norm(prediction - target, axis=1))),
    }


def controller_candidates(config: Config) -> np.ndarray:
    axis = np.linspace(
        -config.rho, config.rho, config.controller_grid_size, dtype=np.float32
    )
    u1, u2 = np.meshgrid(axis, axis, indexing="ij")
    actions = np.stack((u1.reshape(-1), u2.reshape(-1)), axis=-1)
    return actions[np.linalg.norm(actions, axis=1) <= config.rho + 1e-7]


def evaluate_one_step_control(
    model: EffectModel,
    q_states: np.ndarray,
    config: Config,
    category: str,
) -> dict[str, float]:
    q_states = evenly_spaced_rows(q_states, config.controller_state_count)
    candidates = controller_candidates(config)
    magnitudes = np.linspace(0.35 * config.rho, 0.9 * config.rho, 4, dtype=np.float32)
    goal_actions = np.concatenate(
        [magnitudes[:, None] * direction[None, :] for direction in DIRECTIONS[category]],
        axis=0,
    )
    goal_errors: list[float] = []
    action_errors: list[float] = []

    model.eval()
    with torch.no_grad():
        for q in q_states:
            current_y = observe(q)
            candidate_y = np.repeat(current_y[None, :], len(candidates), axis=0)
            predicted_effect = model(
                to_tensor(candidate_y), to_tensor(candidates)
            ).cpu().numpy()
            predicted_next = candidate_y + predicted_effect
            for goal_action in goal_actions:
                goal = observe(wrap_angles(q + goal_action))
                costs = np.sum(np.square(predicted_next - goal[None, :]), axis=1)
                costs += config.controller_action_penalty * np.sum(
                    np.square(candidates), axis=1
                )
                selected = candidates[int(np.argmin(costs))]
                true_next = observe(wrap_angles(q + selected))
                goal_errors.append(float(np.linalg.norm(true_next - goal)))
                action_errors.append(float(np.linalg.norm(selected - goal_action)))

    goal_error_array = np.asarray(goal_errors)
    return {
        "mean_goal_error": float(np.mean(goal_error_array)),
        "median_goal_error": float(np.median(goal_error_array)),
        "action_error": float(np.mean(action_errors)),
        "success_rate": float(
            np.mean(goal_error_array < config.controller_success_threshold)
        ),
    }


def normalized_relation_error(numerator: np.ndarray, *terms: np.ndarray) -> float:
    denominator = sum(float(np.sum(np.square(term))) for term in terms)
    return float(np.sum(np.square(numerator)) / (denominator + 1e-12))


def structure_diagnostics(rhos: list[float], seed: int = 123) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    q = rng.uniform(-np.pi, np.pi, size=(8192, 2)).astype(np.float32)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=len(q)).astype(np.float32)
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    rows: list[dict[str, float]] = []
    for rho in rhos:
        u = rho * directions
        positive = true_effect(q, u)
        negative = true_effect(q, -u)
        half = true_effect(q, 0.5 * u)

        axis_1 = np.zeros_like(u)
        axis_2 = np.zeros_like(u)
        axis_1[:, 0] = rho / np.sqrt(2.0)
        axis_2[:, 1] = rho / np.sqrt(2.0)
        effect_1 = true_effect(q, axis_1)
        effect_2 = true_effect(q, axis_2)
        effect_joint = true_effect(q, axis_1 + axis_2)

        rows.append(
            {
                "rho": rho,
                "odd_error": normalized_relation_error(
                    positive + negative, positive, negative
                ),
                "scale_error": normalized_relation_error(
                    positive - 2.0 * half, positive, 2.0 * half
                ),
                "cross_additivity_error": normalized_relation_error(
                    effect_joint - effect_1 - effect_2,
                    effect_joint,
                    effect_1,
                    effect_2,
                ),
            }
        )
    return rows


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


PREDICTION_CATEGORIES = list(DIRECTIONS.keys())
CONTROL_CATEGORIES = ["negative_axis", "positive_joint", "mixed_joint"]


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metric_names = [f"{category}_nmse" for category in PREDICTION_CATEGORIES]
    metric_names += [f"{category}_control_error" for category in CONTROL_CATEGORIES]
    metric_names += [f"{category}_control_success" for category in CONTROL_CATEGORIES]
    groups: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["condition"]), int(row["train_size"]), str(row["model"]))
        groups.setdefault(key, []).append(row)

    aggregates: list[dict[str, object]] = []
    for (condition, train_size, model_name), group in sorted(groups.items()):
        aggregate: dict[str, object] = {
            "condition": condition,
            "train_size": train_size,
            "model": model_name,
            "parameter_count": int(group[0]["parameter_count"]),
            "seed_count": len(group),
            "fit_transition_count": int(group[0]["fit_transition_count"]),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in group])
            aggregate[f"{metric}_mean"] = float(np.mean(values))
            aggregate[f"{metric}_std"] = float(np.std(values))
        aggregates.append(aggregate)
    return aggregates


MODEL_LABELS = {
    "free": "Free",
    "first_order": "First-order",
    "corrected": "First + second order",
}

MODEL_COLORS = {
    "free": "#777777",
    "first_order": "#2b6cb0",
    "corrected": "#c05621",
}

CATEGORY_LABELS = {
    "positive_axis": "Seen positive axes",
    "negative_axis": "Unseen negative axes",
    "positive_joint": "Positive joint action",
    "negative_joint": "Negative joint action",
    "mixed_joint": "Mixed-sign joint action",
}


def plot_structure_diagnostics(rows: list[dict[str, float]], output_dir: Path) -> None:
    rhos = np.asarray([row["rho"] for row in rows])
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for metric, label in [
        ("odd_error", "reverse relation"),
        ("scale_error", "2x scaling"),
        ("cross_additivity_error", "two-joint additivity"),
    ]:
        ax.plot(rhos, [row[metric] for row in rows], marker="o", label=label)
    ax.set_xlabel("Action radius rho (radians)")
    ax.set_ylabel("Normalized relation error")
    ax.set_title("Local relations become less exact as actions grow")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "structure_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_prediction_curves(
    aggregates: list[dict[str, object]], output_dir: Path, condition: str
) -> None:
    condition_rows = [row for row in aggregates if row["condition"] == condition]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    axes_flat = axes.reshape(-1)
    for category_index, category in enumerate(PREDICTION_CATEGORIES):
        ax = axes_flat[category_index]
        for model_name in MODEL_LABELS:
            model_rows = [row for row in condition_rows if row["model"] == model_name]
            x = np.asarray([int(row["train_size"]) for row in model_rows])
            mean = np.asarray(
                [float(row[f"{category}_nmse_mean"]) for row in model_rows]
            )
            std = np.asarray(
                [float(row[f"{category}_nmse_std"]) for row in model_rows]
            )
            ax.plot(
                x,
                mean,
                marker="o",
                label=MODEL_LABELS[model_name],
                color=MODEL_COLORS[model_name],
            )
            ax.fill_between(
                x,
                np.maximum(mean - std, 1e-9),
                mean + std,
                color=MODEL_COLORS[model_name],
                alpha=0.16,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Collected transitions")
        ax.set_ylabel("Prediction NMSE")
        ax.set_title(CATEGORY_LABELS[category])
        ax.grid(alpha=0.25)
    axes_flat[-1].axis("off")
    axes_flat[1].legend()
    fig.suptitle(condition.replace("_", " "), y=1.0)
    fig.tight_layout()
    fig.savefig(output_dir / f"prediction_{condition}.png", dpi=180)
    plt.close(fig)


def plot_control_curves(
    aggregates: list[dict[str, object]], output_dir: Path, condition: str
) -> None:
    condition_rows = [row for row in aggregates if row["condition"] == condition]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, category in zip(axes, CONTROL_CATEGORIES):
        for model_name in MODEL_LABELS:
            model_rows = [row for row in condition_rows if row["model"] == model_name]
            x = np.asarray([int(row["train_size"]) for row in model_rows])
            mean = np.asarray(
                [float(row[f"{category}_control_error_mean"]) for row in model_rows]
            )
            std = np.asarray(
                [float(row[f"{category}_control_error_std"]) for row in model_rows]
            )
            ax.plot(
                x,
                mean,
                marker="o",
                label=MODEL_LABELS[model_name],
                color=MODEL_COLORS[model_name],
            )
            ax.fill_between(
                x,
                np.maximum(mean - std, 1e-9),
                mean + std,
                color=MODEL_COLORS[model_name],
                alpha=0.16,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Collected transitions")
        ax.set_ylabel("Mean one-step goal error")
        ax.set_title(CATEGORY_LABELS[category])
        ax.grid(alpha=0.25)
    axes[-1].legend()
    fig.suptitle(condition.replace("_", " "), y=1.0)
    fig.tight_layout()
    fig.savefig(output_dir / f"control_{condition}.png", dpi=180)
    plt.close(fig)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def state_bin_coverage(q: np.ndarray, bins_per_joint: int = 12) -> float:
    scaled = (q + np.pi) / (2.0 * np.pi) * bins_per_joint
    bins = np.floor(scaled).astype(int) % bins_per_joint
    occupied = np.unique(bins[:, 0] * bins_per_joint + bins[:, 1])
    return float(len(occupied) / (bins_per_joint**2))


def run_experiment(config: Config, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    diagnostics = structure_diagnostics([0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8])
    plot_structure_diagnostics(diagnostics, output_dir)

    rows: list[dict[str, object]] = []
    maximum_train_size = max(config.train_sizes)
    for condition_index, diagonal_fraction in enumerate(config.diagonal_fractions):
        condition = condition_name(diagonal_fraction)
        for seed in config.seeds:
            trajectory = generate_positive_trajectory(
                maximum_train_size,
                config.rho,
                config.min_action_fraction,
                diagonal_fraction,
                seed + 100_000 * condition_index,
            )
            for train_size in config.train_sizes:
                train_data, validation_data = split_collected_data(trajectory, train_size)
                q_states = evenly_spaced_rows(
                    trajectory["q"][:train_size], config.test_state_count
                )
                queries = {
                    category: generate_query_set(
                        q_states, config.rho, config.test_action_count, category
                    )
                    for category in PREDICTION_CATEGORIES
                }
                for model_index in range(3):
                    set_seed(
                        condition_index * 1_000_000
                        + seed * 10_000
                        + train_size * 10
                        + model_index
                    )
                    model = make_models(config)[model_index]
                    model, training_info = train_model(
                        model, train_data, validation_data, config
                    )
                    predictions = {
                        category: evaluate_prediction(model, data)
                        for category, data in queries.items()
                    }
                    controls = {
                        category: evaluate_one_step_control(
                            model, q_states, config, category
                        )
                        for category in CONTROL_CATEGORIES
                    }
                    prefix_types = trajectory["action_type"][:train_size]
                    row: dict[str, object] = {
                        "condition": condition,
                        "diagonal_fraction": diagonal_fraction,
                        "seed": seed,
                        "train_size": train_size,
                        "fit_transition_count": len(train_data["q"]),
                        "model": model.model_name,
                        "parameter_count": parameter_count(model),
                        "q_bin_coverage": state_bin_coverage(
                            trajectory["q"][:train_size]
                        ),
                        "observed_joint_action_count": int(np.sum(prefix_types == 2)),
                        "best_epoch": int(training_info["best_epoch"]),
                        "validation_mse": training_info["best_validation_mse"],
                    }
                    for category in PREDICTION_CATEGORIES:
                        row[f"{category}_nmse"] = predictions[category]["nmse"]
                    for category in CONTROL_CATEGORIES:
                        row[f"{category}_control_error"] = controls[category][
                            "mean_goal_error"
                        ]
                        row[f"{category}_control_success"] = controls[category][
                            "success_rate"
                        ]
                        row[f"{category}_action_error"] = controls[category][
                            "action_error"
                        ]
                    rows.append(row)
                    print(
                        f"condition={condition:24s} seed={seed} N={train_size:4d} "
                        f"model={model.model_name:11s} "
                        f"neg_axis={row['negative_axis_nmse']:.4g} "
                        f"positive_joint={row['positive_joint_nmse']:.4g} "
                        f"mixed={row['mixed_joint_nmse']:.4g}"
                    )

    aggregates = aggregate_rows(rows)
    for diagonal_fraction in config.diagonal_fractions:
        condition = condition_name(diagonal_fraction)
        plot_prediction_curves(aggregates, output_dir, condition)
        plot_control_curves(aggregates, output_dir, condition)
    write_csv(rows, output_dir / "metrics_per_seed.csv")
    write_csv(aggregates, output_dir / "metrics_aggregate.csv")
    write_csv(
        [dict(row) for row in diagnostics], output_dir / "structure_diagnostics.csv"
    )
    result: dict[str, object] = {
        "config": asdict(config),
        "structure_diagnostics": diagnostics,
        "metrics_per_seed": rows,
        "metrics_aggregate": aggregates,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    return result


def parse_integer_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-joint local visual action-effect structure experiment."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--train-sizes", default="200,500,1000,2000")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--diagonal-fractions", default="0,0.1")
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        rho=args.rho,
        train_sizes=parse_integer_tuple(args.train_sizes),
        seeds=parse_integer_tuple(args.seeds),
        diagonal_fractions=parse_float_tuple(args.diagonal_fractions),
        epochs=args.epochs,
    )
    if args.quick:
        config.train_sizes = (200, 1000)
        config.seeds = (0, 1)
        config.epochs = min(config.epochs, 450)
        config.test_state_count = 48
        config.controller_state_count = 24

    result = run_experiment(config, args.output_dir)
    largest_size = max(config.train_sizes)
    print("\nAggregate prediction NMSE at the largest collection size:")
    for row in result["metrics_aggregate"]:
        if row["train_size"] == largest_size:
            print(
                f"{str(row['condition']):24s} {MODEL_LABELS[str(row['model'])]:20s} "
                f"negative-axis={row['negative_axis_nmse_mean']:.4g}, "
                f"positive-joint={row['positive_joint_nmse_mean']:.4g}, "
                f"mixed={row['mixed_joint_nmse_mean']:.4g}"
            )
    print(f"\nSaved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

