# test_model.py
import pytest
import numpy as np
import pandas as pd
import mlflow
import os
from mlflow.tracking import MlflowClient

# MLflow setup
os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'
mlflow.set_tracking_uri("http://mlflow.ml.brain.cs.ait.ac.th/")

MODEL_NAME = "st125981_best_logistic_regression"


def load_model_from_mlflow():
    """Load the FULL PIPELINE model from MLflow registry."""
    client = MlflowClient()
    try:
        model_uri = f"models:/{MODEL_NAME}/Staging"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Loaded model from Staging stage")
        return model
    except Exception:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            pytest.skip(f"No model versions found for '{MODEL_NAME}'")
        latest_version = max([int(v.version) for v in versions])
        model_uri = f"models:/{MODEL_NAME}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Loaded model version {latest_version}")
        return model


# Test 1: Model accepts expected input (numeric)
def test_model_accepts_expected_input():
    model = load_model_from_mlflow()

    # Get model’s expected number of features
    try:
        num_features = model._model_impl.sklearn_model.W.shape[0]
    except AttributeError:
        num_features = 40  # fallback based on your notebook

    # Create random numeric input (model expects encoded features)
    sample_input = pd.DataFrame(
        np.random.rand(1, num_features),
        columns=[f"feature_{i}" for i in range(num_features)]
    )

    prediction = model.predict(sample_input)
    assert prediction is not None
    assert len(prediction) == 1
    print(f"Model accepted numeric input and returned: {prediction}")


# Test 2: Model output shape
def test_model_output_shape():
    model = load_model_from_mlflow()
    num_features = 40

    test_input = pd.DataFrame(
        np.random.rand(3, num_features),
        columns=[f"feature_{i}" for i in range(num_features)]
    )

    predictions = model.predict(test_input)
    predictions = np.array(predictions)

    assert predictions.shape == (3,)
    assert all(int(p) in [0, 1, 2, 3] for p in predictions)
    print(f"Model output shape and labels OK: {predictions}")


# Test 3: Model registration
def test_model_registered_in_mlflow():
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    assert len(versions) > 0, f"No versions found for model '{MODEL_NAME}'"
    staging_versions = [v for v in versions if v.current_stage == "Staging"]

    print(f"Model '{MODEL_NAME}' has {len(versions)} version(s).")
    if not staging_versions:
        print("No versions in Staging stage — please promote one in MLflow.")
