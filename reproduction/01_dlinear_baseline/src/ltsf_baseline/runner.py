"""CPU training and evaluation runner for the leakage-safe ETTm1 baseline."""

import json
import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ltsf_baseline.baselines import seasonal_naive
from ltsf_baseline.data import PreparedETTm1, SplitBoundaries, prepare_ettm1
from ltsf_baseline.dlinear import DLinear
from ltsf_baseline.metrics import original_scale_metrics


def run_experiment(
    config: Mapping[str, object], project_root: str | Path, run_name: str
) -> dict[str, float | dict[str, float]]:
    """Train or evaluate one named CPU run and write immutable local artifacts."""

    config = dict(config)
    _validate_run_name(run_name)
    _set_seed(_require_int(config, "seed"))
    input_length = _require_int(config, "input_length")
    prediction_length = _require_int(config, "prediction_length")
    model_config = _require_mapping(config, "model")
    training_config = _require_mapping(config, "training")
    prepared = _prepare_data(config, input_length, prediction_length)
    batch_size = _positive_int(training_config, "batch_size")
    model_name = model_config.get("name")
    if model_name not in {"seasonal_naive", "dlinear"}:
        raise ValueError(f"unsupported model name: {model_name!r}")
    run_dir = _create_run_directory(Path(project_root), run_name)

    if model_name == "seasonal_naive":
        prediction, target = _evaluate_seasonal_naive(
            prepared.test,
            batch_size,
            prediction_length,
            _positive_int(model_config, "period"),
        )
        history: dict[str, object] = {"model": model_name, "epochs": []}
    elif model_name == "dlinear":
        prediction, target, history, checkpoint = _train_and_evaluate_dlinear(
            prepared,
            input_length,
            prediction_length,
            model_config,
            training_config,
            batch_size,
            _require_int(config, "seed"),
        )
        torch.save(checkpoint, run_dir / "best.pt")
    metrics = original_scale_metrics(prediction, target, prepared.scaler, prepared.columns)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "history.json", history)
    _write_json(run_dir / "metrics.json", metrics)
    np.savez_compressed(
        run_dir / "predictions.npz",
        prediction=prediction,
        target=target,
        columns=np.asarray(prepared.columns),
    )
    return metrics


def _prepare_data(
    config: Mapping[str, object], input_length: int, prediction_length: int
) -> PreparedETTm1:
    dataset_path = config.get("dataset_path")
    if not isinstance(dataset_path, str) or not dataset_path:
        raise ValueError("config dataset_path must be a non-empty string")

    boundaries_config = config.get("split_boundaries")
    boundaries = None
    if boundaries_config is not None:
        values = _as_mapping(boundaries_config, "split_boundaries")
        boundaries = SplitBoundaries(
            train_end=_positive_int(values, "train_end"),
            validation_end=_positive_int(values, "validation_end"),
            test_end=_positive_int(values, "test_end"),
        )
    return prepare_ettm1(Path(dataset_path), input_length, prediction_length, boundaries)


def _train_and_evaluate_dlinear(
    prepared: PreparedETTm1,
    input_length: int,
    prediction_length: int,
    model_config: Mapping[str, object],
    training_config: Mapping[str, object],
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], dict[str, object]]:
    model = DLinear(
        input_length=input_length,
        prediction_length=prediction_length,
        channels=len(prepared.columns),
        individual=bool(model_config.get("individual", False)),
        kernel_size=_positive_int(model_config, "moving_average_window"),
    )
    learning_rate = _positive_float(training_config, "learning_rate")
    max_epochs = _positive_int(training_config, "max_epochs")
    patience = _positive_int(training_config, "patience")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    train_loader = _loader(prepared.train, batch_size, shuffle=True, seed=seed)
    validation_loader = _loader(prepared.validation, batch_size, shuffle=False, seed=seed)

    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    waiting_epochs = 0
    epochs: list[dict[str, float | int]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for history, target in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(history), target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))

        validation_mse = _normalized_mse(model, validation_loader, loss_fn)
        train_mse = float(np.mean(train_losses))
        epochs.append(
            {"epoch": epoch, "train_mse": train_mse, "validation_mse": validation_mse}
        )
        if validation_mse < best_validation:
            best_validation = validation_mse
            best_epoch = epoch
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            waiting_epochs = 0
        else:
            waiting_epochs += 1
            if waiting_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("DLinear did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    prediction, target = _collect_predictions(model, _loader(prepared.test, batch_size, False, seed))
    history = {"model": "dlinear", "epochs": epochs, "best_epoch": best_epoch}
    checkpoint: dict[str, object] = {
        "model_state": best_state,
        "epoch": best_epoch,
        "validation_mse": best_validation,
    }
    return prediction, target, history, checkpoint


def _evaluate_seasonal_naive(
    dataset, batch_size: int, prediction_length: int, period: int
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for history, target in _loader(dataset, batch_size, shuffle=False, seed=0):
        predictions.append(seasonal_naive(history, prediction_length, period).numpy())
        targets.append(target.numpy())
    return _concatenate_batches(predictions, "seasonal naive prediction"), _concatenate_batches(
        targets, "seasonal naive target"
    )


def _normalized_mse(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    total_squared_error = 0.0
    total_elements = 0
    with torch.no_grad():
        for history, target in loader:
            batch_elements = target.numel()
            total_squared_error += float(loss_fn(model(history), target)) * batch_elements
            total_elements += batch_elements
    if total_elements == 0:
        raise RuntimeError("validation DataLoader is empty")
    return total_squared_error / total_elements


def _collect_predictions(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for history, target in loader:
            predictions.append(model(history).numpy())
            targets.append(target.numpy())
    return _concatenate_batches(predictions, "DLinear prediction"), _concatenate_batches(
        targets, "DLinear target"
    )


def _concatenate_batches(batches: list[np.ndarray], name: str) -> np.ndarray:
    if not batches:
        raise RuntimeError(f"{name} DataLoader is empty")
    return np.concatenate(batches, axis=0)


def _loader(dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    if len(dataset) == 0:
        raise RuntimeError("forecasting dataset is empty")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _create_run_directory(project_root: Path, run_name: str) -> Path:
    run_parent = project_root / "runs"
    run_parent.mkdir(parents=True, exist_ok=True)
    run_dir = run_parent / run_name
    try:
        run_dir.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"run directory already exists: {run_dir}") from error
    return run_dir


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _validate_run_name(run_name: str) -> None:
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run name must be one non-empty directory name")


def _require_mapping(config: Mapping[str, object], key: str) -> Mapping[str, object]:
    return _as_mapping(config.get(key), key)


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"config {name} must be an object")
    return value


def _require_int(config: Mapping[str, object], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"config {key} must be an integer")
    return value


def _positive_int(config: Mapping[str, object], key: str) -> int:
    value = _require_int(config, key)
    if value <= 0:
        raise ValueError(f"config {key} must be positive")
    return value


def _positive_float(config: Mapping[str, object], key: str) -> float:
    value = config.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"config {key} must be a positive number")
    return float(value)
