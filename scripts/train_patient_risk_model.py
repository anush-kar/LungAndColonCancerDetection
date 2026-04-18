import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk_model import train_reduced_risk_model


def main() -> None:
    project_root = PROJECT_ROOT

    csv_path = project_root / "data" / "cancer patient data sets.csv"
    artifact_path = project_root / "models" / "patient_risk_selected_features.pkl"

    artifact = train_reduced_risk_model(csv_path=csv_path, artifact_path=artifact_path)

    print("Training complete.")
    print("Saved artifact:", artifact_path)
    print("Available models:", artifact["available_models"])
    print("Default model:", artifact["default_model_name"])
    print("Class labels:", artifact["class_labels"])
    print("Metrics:", artifact["metrics"]["selected_model"])


if __name__ == "__main__":
    main()
