"""Fine-tune YOLOv8 on KITTI-derived dataset with MLflow tracking."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Import torch before ultralytics to avoid Windows c10.dll load conflicts.
import torch  # noqa: F401

# PyTorch 2.6+ defaults weights_only=True; allow trusted Ultralytics checkpoints.
try:
    from ultralytics.nn.tasks import DetectionModel

    torch.serialization.add_safe_globals([DetectionModel])
except Exception:
    pass

import mlflow
from mlflow.tracking import MlflowClient
import yaml
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

# Disable Ultralytics built-in MLflow (uses bare Windows paths that break MLflow).
SETTINGS.update(mlflow=False)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def write_kitti_yaml(dataset_root: Path, yaml_path: Path) -> Path:
    """Create Ultralytics dataset config for KITTI YOLO layout."""
    config = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": ["Car", "Pedestrian", "Cyclist"],
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on KITTI YOLO dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--experiment", type=str, default="kitti-yolov8")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers (0 avoids Windows multiprocessing memory issues)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from runs/train/kitti_yolov8n/weights/last.pt",
    )
    args = parser.parse_args()

    root = project_root()
    dataset_root = root / "data" / "kitti_yolo"
    yaml_path = dataset_root / "kitti.yaml"
    save_dir = root / "runs" / "train" / "kitti_yolov8n"
    last_ckpt = save_dir / "weights" / "last.pt"
    model_weights = root / "models" / "yolov8n.pt"

    if args.resume:
        if not last_ckpt.is_file():
            raise FileNotFoundError(f"No checkpoint to resume from: {last_ckpt}")
        weights = str(last_ckpt)
        args_yaml = save_dir / "args.yaml"
        if args_yaml.is_file():
            with open(args_yaml, encoding="utf-8") as f:
                saved = yaml.safe_load(f)
            args.epochs = saved.get("epochs", args.epochs)
            args.batch = saved.get("batch", args.batch)
            args.imgsz = saved.get("imgsz", args.imgsz)
    else:
        weights = str(model_weights) if model_weights.is_file() else args.model

    if not (dataset_root / "images" / "train").is_dir():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_root}. Run scripts/kitti_to_yolo.py first."
        )

    write_kitti_yaml(dataset_root, yaml_path)

    mlruns_dir = root / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = mlruns_dir.as_uri()
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)
    stem = "yolov8n" if args.resume else Path(weights).stem
    run_name = f"kitti_{stem}_e{args.epochs}" + ("_resume" if args.resume else "")

    client = MlflowClient()
    results = None

    with mlflow.start_run(run_name=run_name) as active_run:
        run_id = active_run.info.run_id

        def on_fit_epoch_end(trainer) -> None:
            """Log epoch metrics via run_id (stable across Ultralytics callbacks)."""
            step = trainer.epoch + 1
            try:
                for key, value in (trainer.metrics or {}).items():
                    if isinstance(value, (int, float)):
                        safe = key.replace("(", "_").replace(")", "").replace("/", "_")
                        client.log_metric(run_id, safe, float(value), step=step)
                if trainer.validator and hasattr(trainer.validator, "metrics"):
                    val_metrics = trainer.validator.metrics
                    if val_metrics and hasattr(val_metrics, "results_dict"):
                        for k, v in val_metrics.results_dict.items():
                            if isinstance(v, (int, float)):
                                safe = f"val_{k}".replace("(", "_").replace(")", "").replace("/", "_")
                                client.log_metric(run_id, safe, float(v), step=step)
            except Exception as exc:
                print(f"Warning: MLflow epoch log skipped at step {step}: {exc}")

        mlflow.log_params(
            {
                "model": weights,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": args.device,
                "data": str(yaml_path),
                "resume": args.resume,
            }
        )

        model = YOLO(weights)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        results = model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(root / "runs" / "train"),
            name="kitti_yolov8n",
            exist_ok=True,
            resume=args.resume,
        )

        metrics = results.results_dict if hasattr(results, "results_dict") else {}
        map50 = metrics.get("metrics/mAP50(B)", metrics.get("metrics/mAP50", None))
        map5095 = metrics.get("metrics/mAP50-95(B)", metrics.get("metrics/mAP50-95", None))

        if map50 is not None:
            mlflow.log_metric("final_mAP50", float(map50))
        if map5095 is not None:
            mlflow.log_metric("final_mAP50_95", float(map5095))

        save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else save_dir
        best_weights = save_dir / "weights" / "best.pt"
        if best_weights.is_file():
            mlflow.log_artifact(str(best_weights))

    print("\n=== Training complete ===")
    print(f"Run directory: {save_dir}")
    if results and hasattr(results, "results_dict"):
        metrics = results.results_dict
        map50 = metrics.get("metrics/mAP50(B)", metrics.get("metrics/mAP50", None))
        map5095 = metrics.get("metrics/mAP50-95(B)", metrics.get("metrics/mAP50-95", None))
        if map50 is not None:
            print(f"mAP50:     {float(map50):.4f}")
        if map5095 is not None:
            print(f"mAP50-95:  {float(map5095):.4f}")
    else:
        print("mAP50 / mAP50-95: see runs/train/kitti_yolov8n/results.csv")
    print(f"MLflow experiment: {args.experiment}")
    print(f"MLflow tracking URI: {mlruns_dir.as_uri()}")


if __name__ == "__main__":
    main()
