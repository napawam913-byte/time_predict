import json
from pathlib import Path


def test_ettm1_first_run_config_is_leakage_safe():
    config_path = Path(__file__).parents[1] / "configs" / "ettm1_l96_h96.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    project_root = Path(__file__).resolve().parents[3]
    dataset_path = (config_path.parent / config["dataset_path"]).resolve()
    assert dataset_path == project_root / "DataSet" / "ETTm1" / "ETTm1.csv"
    assert config["input_length"] == 96
    assert config["prediction_length"] == 96
    assert config["seed"] == 2026
    assert config["model"]["name"] == "dlinear"
    assert config["model"]["individual"] is False
    assert config["training"]["batch_size"] == 32
    assert config["training"]["learning_rate"] == 1e-3
    assert config["training"]["max_epochs"] == 20
    assert config["training"]["patience"] == 3
