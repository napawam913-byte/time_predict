import json

from ltsf_baseline.cli import load_config


def test_load_config_resolves_dataset_path_from_config_directory(tmp_path):
    config_directory = tmp_path / "configs"
    config_directory.mkdir()
    data_path = tmp_path / "DataSet" / "ETTm1" / "ETTm1.csv"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("date,OT\n2024-01-01,1\n", encoding="utf-8")
    config_path = config_directory / "run.json"
    config_path.write_text(
        json.dumps({"dataset_path": "../DataSet/ETTm1/ETTm1.csv"}), encoding="utf-8"
    )

    config = load_config(config_path)

    assert config["dataset_path"] == str(data_path.resolve())
