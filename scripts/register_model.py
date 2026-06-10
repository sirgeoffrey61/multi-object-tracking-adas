"""Register the best KITTI YOLOv8 training run in the MLflow Model Registry."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_map50_from_results(results_csv: Path) -> float | None:
    if not results_csv.is_file():
        return None
    lines = results_csv.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return None
    header = lines[0].split(",")
    try:
        map_idx = header.index("metrics/mAP50(B)")
    except ValueError:
        return None
    last_row = lines[-1].split(",")
    return float(last_row[map_idx])


def find_best_run(client: MlflowClient, experiment_name: str):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.final_mAP50 DESC"],
        max_results=1,
    )
    if runs:
        return runs[0]

    fallback = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    return fallback[0] if fallback else None


def extract_map50(run) -> float | None:
    metrics = run.data.metrics
    for key in ("final_mAP50", "metrics_mAP50_B", "val_metrics_mAP50_B"):
        if key in metrics:
            return float(metrics[key])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Register best model in MLflow Model Registry")
    parser.add_argument("--experiment", type=str, default="kitti-yolov8")
    parser.add_argument("--model-name", type=str, default="kitti-yolov8n-adas")
    parser.add_argument("--stage", type=str, default="Staging")
    args = parser.parse_args()

    root = project_root()
    mlruns_dir = root / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = mlruns_dir.as_uri()
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    local_weights = root / "runs" / "train" / "kitti_yolov8n" / "weights" / "best.pt"
    results_csv = root / "runs" / "train" / "kitti_yolov8n" / "results.csv"

    best_run = find_best_run(client, args.experiment)
    map50 = None
    model_uri = None

    if best_run is not None:
        map50 = extract_map50(best_run)
        artifacts = [a.path for a in client.list_artifacts(best_run.info.run_id)]
        if "best.pt" in artifacts:
            model_uri = f"runs:/{best_run.info.run_id}/best.pt"
        elif local_weights.is_file():
            model_uri = local_weights.as_uri()

    if model_uri is None and local_weights.is_file():
        model_uri = local_weights.as_uri()
        map50 = map50 or read_map50_from_results(results_csv)

    if model_uri is None:
        raise FileNotFoundError(
            "No model to register. Train with scripts/train.py or ensure best.pt exists."
        )

    if map50 is None:
        map50 = read_map50_from_results(results_csv)

    result = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    client.transition_model_version_stage(
        name=args.model_name,
        version=result.version,
        stage=args.stage,
        archive_existing_versions=False,
    )

    print("\n=== MLflow Model Registry ===")
    print(f"Model name:    {args.model_name}")
    print(f"Version:       {result.version}")
    print(f"Stage:         {args.stage}")
    if map50 is not None:
        print(f"mAP50:         {map50:.4f}")
    else:
        print("mAP50:         (not logged — check training metrics)")
    print(f"Source URI:    {model_uri}")


if __name__ == "__main__":
    main()
