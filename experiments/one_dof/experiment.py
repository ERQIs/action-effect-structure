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


def wrap_angle(q: np.ndarray | float) -> np.ndarray:
    return (np.asarray(q) + np.pi) % (2.0 * np.pi) - np.pi


def observe(q: np.ndarray | float) -> np.ndarray:
    q_array = np.asarray(q, dtype=np.float32)
    return np.stack((np.cos(q_array), np.sin(q_array)), axis=-1).astype(np.float32)


def true_effect(q: np.ndarray, u: np.ndarray) -> np.ndarray:
    return observe(wrap_angle(q + u)) - observe(q)


@dataclass
class Config:
    rho: float = 0.2
    min_action_fraction: float = 0.5
    train_sizes: tuple[int, ...] = (50, 100, 200, 500, 1000)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 1200
    validation_interval: int = 25
    hidden_dim: int = 64
    learning_rate: float = 2e-3
    weight_decay: float = 1e-6
    residual_penalty: float = 1e-4
    validation_size: int = 1024
    test_q_count: int = 128
    test_action_count: int = 12
    controller_grid_size: int = 2001
    controller_action_penalty: float = 1e-5
    controller_success_threshold: float = 0.001


def generate_positive_trajectory(
    count: int, rho: float, min_action_fraction: float, seed: int
) -> dict[str, np.ndarray]:
    """Collect one continuous trajectory containing positive actions only."""
    rng = np.random.default_rng(seed)
    q = float(rng.uniform(-np.pi, np.pi))
    q_values = np.empty(count, dtype=np.float32)
    actions = np.empty(count, dtype=np.float32)

    for index in range(count):
        u = float(rng.uniform(min_action_fraction * rho, rho))
        q_values[index] = q
        actions[index] = u
        q = float(wrap_angle(q + u))

    y = observe(q_values)
    effects = true_effect(q_values, actions)
    return {
        "q": q_values,
        "y": y,
        "u": actions[:, None],
        "effect": effects,
    }


def generate_query_set(
    rho: float,
    q_count: int,
    action_count: int,
    category: str,
) -> dict[str, np.ndarray]:
    q_grid = np.linspace(-np.pi, np.pi, q_count, endpoint=False, dtype=np.float32)
    base_magnitudes = np.linspace(0.5 * rho, rho, action_count, dtype=np.float32)

    if category == "positive":
        action_values = base_magnitudes
    elif category == "negative":
        action_values = -base_magnitudes
    elif category == "scaled_positive":
        action_values = 2.0 * base_magnitudes
    elif category == "scaled_negative":
        action_values = -2.0 * base_magnitudes
    else:
        raise ValueError(f"Unknown query category: {category}")

    q, u = np.meshgrid(q_grid, action_values, indexing="ij")
    q_flat = q.reshape(-1)
    u_flat = u.reshape(-1)
    return {
        "q": q_flat,
        "y": observe(q_flat),
        "u": u_flat[:, None].astype(np.float32),
        "effect": true_effect(q_flat, u_flat),
    }


def generate_positive_validation(config: Config, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    q = rng.uniform(-np.pi, np.pi, size=config.validation_size).astype(np.float32)
    u = rng.uniform(
        config.min_action_fraction * config.rho,
        config.rho,
        size=config.validation_size,
    ).astype(np.float32)
    return {
        "q": q,
        "y": observe(q),
        "u": u[:, None],
        "effect": true_effect(q, u),
    }


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
    """Flexible in action, while respecting the known zero-effect anchor E(y, 0)=0."""

    model_name = "free"

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = make_mlp(3, 2, hidden_dim)

    def forward(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        slope = self.net(torch.cat((y, u), dim=-1))
        return u * slope


class FirstOrderEffectModel(EffectModel):
    model_name = "first_order"

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.jacobian_net = make_mlp(2, 2, hidden_dim)

    def forward(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.jacobian_net(y) * u


class CorrectedEffectModel(EffectModel):
    """A first-order term plus the smallest useful Taylor-style correction."""

    model_name = "corrected"

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = make_mlp(2, 4, hidden_dim)

    def components(
        self, y: torch.Tensor, u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.net(y)
        jacobian, curvature = output[:, :2], output[:, 2:]
        first_order = jacobian * u
        second_order = curvature * u.square()
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
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_validation = math.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

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
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch

    model.load_state_dict(best_state)
    return model, {
        "best_validation_mse": best_validation,
        "best_epoch": float(best_epoch),
    }


def evaluate_prediction(
    model: EffectModel, data: dict[str, np.ndarray]
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        prediction = model(to_tensor(data["y"]), to_tensor(data["u"])).cpu().numpy()
    target = data["effect"]
    squared_error = np.square(prediction - target)
    mse = float(np.mean(squared_error))
    nmse = float(np.sum(squared_error) / (np.sum(np.square(target)) + 1e-12))
    vector_error = np.linalg.norm(prediction - target, axis=1)
    return {
        "mse": mse,
        "nmse": nmse,
        "mean_vector_error": float(np.mean(vector_error)),
    }


def evaluate_one_step_control(
    model: EffectModel,
    config: Config,
    direction: str,
) -> dict[str, float]:
    q_values = np.linspace(
        -np.pi, np.pi, config.test_q_count, endpoint=False, dtype=np.float32
    )
    magnitudes = np.linspace(
        0.25 * config.rho,
        config.rho,
        config.test_action_count,
        dtype=np.float32,
    )
    goal_actions = magnitudes if direction == "positive" else -magnitudes
    candidate_actions = np.linspace(
        -config.rho,
        config.rho,
        config.controller_grid_size,
        dtype=np.float32,
    )

    goal_errors: list[float] = []
    action_errors: list[float] = []
    model.eval()
    with torch.no_grad():
        for q in q_values:
            current_y = observe(q)
            candidate_y = np.repeat(current_y[None, :], len(candidate_actions), axis=0)
            candidate_u = candidate_actions[:, None]
            predicted_effect = model(to_tensor(candidate_y), to_tensor(candidate_u)).cpu().numpy()
            predicted_next = candidate_y + predicted_effect

            for goal_action in goal_actions:
                goal = observe(wrap_angle(q + goal_action))
                costs = np.sum(np.square(predicted_next - goal[None, :]), axis=1)
                costs += config.controller_action_penalty * np.square(candidate_actions)
                selected_action = float(candidate_actions[int(np.argmin(costs))])
                true_next = observe(wrap_angle(q + selected_action))
                goal_errors.append(float(np.linalg.norm(true_next - goal)))
                action_errors.append(abs(selected_action - float(goal_action)))

    goal_error_array = np.asarray(goal_errors)
    return {
        "mean_goal_error": float(np.mean(goal_error_array)),
        "median_goal_error": float(np.median(goal_error_array)),
        "action_mae": float(np.mean(action_errors)),
        "success_rate": float(
            np.mean(goal_error_array < config.controller_success_threshold)
        ),
    }


def normalized_relation_error(numerator: np.ndarray, *effects: np.ndarray) -> float:
    denominator = sum(float(np.sum(np.square(effect))) for effect in effects)
    return float(np.sum(np.square(numerator)) / (denominator + 1e-12))


def structure_diagnostics(rhos: list[float]) -> list[dict[str, float]]:
    q = np.linspace(-np.pi, np.pi, 256, endpoint=False, dtype=np.float32)
    rows: list[dict[str, float]] = []
    for rho in rhos:
        u = np.linspace(0.1 * rho, rho, 64, dtype=np.float32)
        q_grid, u_grid = np.meshgrid(q, u, indexing="ij")
        q_flat = q_grid.reshape(-1)
        u_flat = u_grid.reshape(-1)
        effect_positive = true_effect(q_flat, u_flat)
        effect_negative = true_effect(q_flat, -u_flat)
        effect_double = true_effect(q_flat, 2.0 * u_flat)

        half_u = 0.5 * u_flat
        effect_half = true_effect(q_flat, half_u)
        effect_add = true_effect(q_flat, half_u + half_u)

        rows.append(
            {
                "rho": rho,
                "odd_error": normalized_relation_error(
                    effect_positive + effect_negative,
                    effect_positive,
                    effect_negative,
                ),
                "scale_error": normalized_relation_error(
                    effect_double - 2.0 * effect_positive,
                    effect_double,
                    2.0 * effect_positive,
                ),
                "additivity_error": normalized_relation_error(
                    effect_add - 2.0 * effect_half,
                    effect_add,
                    2.0 * effect_half,
                ),
            }
        )
    return rows


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def aggregate_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, object]]:
    metric_names = [
        "positive_nmse",
        "negative_nmse",
        "scaled_positive_nmse",
        "scaled_negative_nmse",
        "positive_control_error",
        "negative_control_error",
        "positive_control_success",
        "negative_control_success",
    ]
    groups: dict[tuple[int, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = (int(row["train_size"]), str(row["model"]))
        groups.setdefault(key, []).append(row)

    aggregates: list[dict[str, object]] = []
    for (train_size, model_name), group in sorted(groups.items()):
        aggregate: dict[str, object] = {
            "train_size": train_size,
            "model": model_name,
            "parameter_count": int(group[0]["parameter_count"]),
            "seed_count": len(group),
        }
        for metric_name in metric_names:
            values = np.asarray([float(row[metric_name]) for row in group])
            aggregate[f"{metric_name}_mean"] = float(np.mean(values))
            aggregate[f"{metric_name}_std"] = float(np.std(values))
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


def plot_structure_diagnostics(rows: list[dict[str, float]], output_dir: Path) -> None:
    rhos = np.asarray([row["rho"] for row in rows])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for metric, label in [
        ("odd_error", "reverse relation"),
        ("scale_error", "2x scaling"),
        ("additivity_error", "additivity"),
    ]:
        ax.plot(rhos, [row[metric] for row in rows], marker="o", label=label)
    ax.set_xlabel("Local action radius rho (radians)")
    ax.set_ylabel("Normalized relation error")
    ax.set_title("The first-order structure weakens with action scale")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "structure_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_learning_curves(aggregates: list[dict[str, object]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for model_name in MODEL_LABELS:
        model_rows = [row for row in aggregates if row["model"] == model_name]
        x = np.asarray([int(row["train_size"]) for row in model_rows])
        for ax, metric, title in [
            (axes[0], "positive_nmse", "Seen positive actions"),
            (axes[1], "negative_nmse", "Unseen negative actions"),
        ]:
            mean = np.asarray([float(row[f"{metric}_mean"]) for row in model_rows])
            std = np.asarray([float(row[f"{metric}_std"]) for row in model_rows])
            ax.plot(
                x,
                mean,
                marker="o",
                label=MODEL_LABELS[model_name],
                color=MODEL_COLORS[model_name],
            )
            ax.fill_between(
                x,
                np.maximum(mean - std, 1e-8),
                mean + std,
                color=MODEL_COLORS[model_name],
                alpha=0.16,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Real training transitions")
            ax.set_ylabel("Prediction NMSE")
            ax.set_title(title)
            ax.grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_learning_curves.png", dpi=180)
    plt.close(fig)


def plot_control_curves(aggregates: list[dict[str, object]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for model_name in MODEL_LABELS:
        model_rows = [row for row in aggregates if row["model"] == model_name]
        x = np.asarray([int(row["train_size"]) for row in model_rows])
        for ax, metric, title in [
            (axes[0], "positive_control_success", "Positive-direction goals"),
            (axes[1], "negative_control_success", "Unseen negative-direction goals"),
        ]:
            mean = np.asarray([float(row[f"{metric}_mean"]) for row in model_rows])
            std = np.asarray([float(row[f"{metric}_std"]) for row in model_rows])
            ax.plot(
                x,
                mean,
                marker="o",
                label=MODEL_LABELS[model_name],
                color=MODEL_COLORS[model_name],
            )
            ax.fill_between(
                x,
                np.clip(mean - std, 0.0, 1.0),
                np.clip(mean + std, 0.0, 1.0),
                color=MODEL_COLORS[model_name],
                alpha=0.16,
            )
            ax.set_xscale("log")
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("Real training transitions")
            ax.set_ylabel("One-step control success rate")
            ax.set_title(title)
            ax.grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "control_learning_curves.png", dpi=180)
    plt.close(fig)


def plot_control_error_curves(
    aggregates: list[dict[str, object]], output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for model_name in MODEL_LABELS:
        model_rows = [row for row in aggregates if row["model"] == model_name]
        x = np.asarray([int(row["train_size"]) for row in model_rows])
        for ax, metric, title in [
            (axes[0], "positive_control_error", "Positive-direction goals"),
            (axes[1], "negative_control_error", "Unseen negative-direction goals"),
        ]:
            mean = np.asarray([float(row[f"{metric}_mean"]) for row in model_rows])
            std = np.asarray([float(row[f"{metric}_std"]) for row in model_rows])
            ax.plot(
                x,
                mean,
                marker="o",
                label=MODEL_LABELS[model_name],
                color=MODEL_COLORS[model_name],
            )
            ax.fill_between(
                x,
                np.maximum(mean - std, 1e-8),
                mean + std,
                color=MODEL_COLORS[model_name],
                alpha=0.16,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Real training transitions")
            ax.set_ylabel("Mean one-step goal error")
            ax.set_title(title)
            ax.grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "control_error_learning_curves.png", dpi=180)
    plt.close(fig)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(config: Config, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    diagnostic_rhos = [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    diagnostics = structure_diagnostics(diagnostic_rhos)
    plot_structure_diagnostics(diagnostics, output_dir)

    queries = {
        category: generate_query_set(
            config.rho,
            config.test_q_count,
            config.test_action_count,
            category,
        )
        for category in [
            "positive",
            "negative",
            "scaled_positive",
            "scaled_negative",
        ]
    }

    rows: list[dict[str, float | int | str]] = []
    maximum_train_size = max(config.train_sizes)
    for seed in config.seeds:
        full_trajectory = generate_positive_trajectory(
            maximum_train_size,
            config.rho,
            config.min_action_fraction,
            seed,
        )
        validation_data = generate_positive_validation(config, seed + 10_000)

        for train_size in config.train_sizes:
            train_data = {
                key: value[:train_size] for key, value in full_trajectory.items()
            }
            for model_index, model in enumerate(make_models(config)):
                set_seed(seed * 1000 + train_size * 10 + model_index)
                # Recreate after seeding so initialization is exactly reproducible.
                model = make_models(config)[model_index]
                model, training_info = train_model(
                    model, train_data, validation_data, config
                )

                prediction_results = {
                    category: evaluate_prediction(model, query_data)
                    for category, query_data in queries.items()
                }
                positive_control = evaluate_one_step_control(model, config, "positive")
                negative_control = evaluate_one_step_control(model, config, "negative")
                q_coverage = float(
                    len(np.unique(np.floor((train_data["q"] + np.pi) / (2 * np.pi) * 36)))
                    / 36.0
                )

                row: dict[str, float | int | str] = {
                    "seed": seed,
                    "train_size": train_size,
                    "model": model.model_name,
                    "parameter_count": parameter_count(model),
                    "q_bin_coverage": q_coverage,
                    "best_epoch": int(training_info["best_epoch"]),
                    "validation_mse": training_info["best_validation_mse"],
                    "positive_nmse": prediction_results["positive"]["nmse"],
                    "negative_nmse": prediction_results["negative"]["nmse"],
                    "scaled_positive_nmse": prediction_results["scaled_positive"]["nmse"],
                    "scaled_negative_nmse": prediction_results["scaled_negative"]["nmse"],
                    "positive_control_error": positive_control["mean_goal_error"],
                    "negative_control_error": negative_control["mean_goal_error"],
                    "positive_control_success": positive_control["success_rate"],
                    "negative_control_success": negative_control["success_rate"],
                    "positive_action_mae": positive_control["action_mae"],
                    "negative_action_mae": negative_control["action_mae"],
                }
                rows.append(row)
                print(
                    f"seed={seed} N={train_size:4d} model={model.model_name:11s} "
                    f"positive_nmse={row['positive_nmse']:.4g} "
                    f"negative_nmse={row['negative_nmse']:.4g}"
                )

    aggregates = aggregate_rows(rows)
    plot_learning_curves(aggregates, output_dir)
    plot_control_curves(aggregates, output_dir)
    plot_control_error_curves(aggregates, output_dir)
    write_csv([dict(row) for row in rows], output_dir / "metrics_per_seed.csv")
    write_csv(aggregates, output_dir / "metrics_aggregate.csv")
    write_csv([dict(row) for row in diagnostics], output_dir / "structure_diagnostics.csv")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal local visual action-effect structure experiment."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--train-sizes", default="50,100,200,500,1000")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a two-seed, two-data-size smoke experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        rho=args.rho,
        train_sizes=parse_integer_tuple(args.train_sizes),
        seeds=parse_integer_tuple(args.seeds),
        epochs=args.epochs,
    )
    if args.quick:
        config.train_sizes = (100, 500)
        config.seeds = (0, 1)
        config.epochs = min(config.epochs, 500)

    result = run_experiment(config, args.output_dir)
    largest_size = max(config.train_sizes)
    print("\nAggregate results at the largest training size:")
    for row in result["metrics_aggregate"]:
        if row["train_size"] == largest_size:
            print(
                f"{MODEL_LABELS[str(row['model'])]:20s} "
                f"negative NMSE={row['negative_nmse_mean']:.4g}, "
                f"negative control success={row['negative_control_success_mean']:.3f}"
            )
    print(f"\nSaved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

