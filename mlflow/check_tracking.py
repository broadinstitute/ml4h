import base64
import configparser
import os
import pickle
import random
import tempfile
from pathlib import Path

import mlflow

PONY = """
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
""".strip().replace(
    "\n", ""
)


def load_env_values() -> None:
    config = configparser.ConfigParser(interpolation=None)
    config.sections()
    config.read("/path/to/.env")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config["google"]["GOOGLE_APPLICATION_CREDENTIALS"]

    os.environ["MLFLOW_TRACKING_USERNAME"] = config["mlflow"]["MLFLOW_TRACKING_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = config["mlflow"]["MLFLOW_TRACKING_PASSWORD"]
    os.environ["MLFLOW_TRACKING_URI"] = config["mlflow"]["MLFLOW_TRACKING_URI"]


def main() -> None:
    print("STARTING")
    load_env_values()
    with mlflow.start_run(run_name="rc_remote_test"):
        mlflow.log_param("a", random.choice([1, 2, 3, 5, 7]))
        for epoch in range(10):
            mlflow.log_metric("m", 2 * epoch * epoch + random.random() - 0.5, step=epoch)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            print("seems like we get here and then cant log the artifact")
            model_file_path = temp_dir_path / "model.pickle"
            with open(model_file_path, "wb") as f:
                pickle.dump({"rc_remote_test": "grzybowa"}, f)
            mlflow.log_artifact(model_file_path)

            image_file_path = temp_dir_path / "pony.png"
            with open(image_file_path, "wb") as f:
                f.write(base64.b64decode(PONY))
            mlflow.log_artifact(image_file_path)
    print("DONE")


if __name__ == "__main__":
    main()
