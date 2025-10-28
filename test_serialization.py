#!/usr/bin/env python3
"""
Test script to verify that the transformer model can be serialized and deserialized properly.
"""

import tempfile
import os
import numpy as np
import keras
from ml4h.models.transformer_blocks_embedding import build_embedding_transformer

def test_model_serialization():
    """Test that the model can be saved and loaded without errors."""

    # Model parameters
    INPUT_NUMERIC_COLS = ['feature1', 'feature2', 'feature3']
    REGRESSION_TARGETS = ['target1']
    BINARY_TARGETS = ['target2']
    MAX_LEN = 10
    EMB_DIM = 32
    TOKEN_HIDDEN = 64
    TRANSFORMER_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 2
    DROPOUT = 0.1
    view2id = {'view1': 1, 'view2': 2}

    print("Building model...")
    model = build_embedding_transformer(
        INPUT_NUMERIC_COLS=INPUT_NUMERIC_COLS,
        REGRESSION_TARGETS=REGRESSION_TARGETS,
        BINARY_TARGETS=BINARY_TARGETS,
        MAX_LEN=MAX_LEN,
        EMB_DIM=EMB_DIM,
        TOKEN_HIDDEN=TOKEN_HIDDEN,
        TRANSFORMER_DIM=TRANSFORMER_DIM,
        NUM_HEADS=NUM_HEADS,
        NUM_LAYERS=NUM_LAYERS,
        DROPOUT=DROPOUT,
        view2id=view2id,
    )

    # Create dummy data for testing
    batch_size = 2
    dummy_input = {
        'view': np.random.randint(0, len(view2id) + 1, (batch_size, MAX_LEN)),
        'num': np.random.random((batch_size, MAX_LEN, len(INPUT_NUMERIC_COLS))),
        'mask': np.ones((batch_size, MAX_LEN), dtype=bool)
    }

    dummy_output = {
        'target1': np.random.random((batch_size, 1)),
        'target2': np.random.randint(0, 2, (batch_size, 1))
    }

    print("Testing forward pass...")
    # Test forward pass
    predictions = model.predict(dummy_input, verbose=0)
    print(f"Forward pass successful. Output shapes: {[(k, v.shape) for k, v in predictions.items()]}")

    # Test serialization
    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
        model_path = tmp_file.name

    try:
        print(f"Saving model to {model_path}...")
        model.save(model_path)
        print("Model saved successfully!")

        print("Loading model...")
        loaded_model = keras.models.load_model(model_path)
        print("Model loaded successfully!")

        # Test that loaded model works
        print("Testing loaded model forward pass...")
        loaded_predictions = loaded_model.predict(dummy_input, verbose=0)
        print(f"Loaded model forward pass successful. Output shapes: {[(k, v.shape) for k, v in loaded_predictions.items()]}")

        # Compare predictions (should be identical)
        for key in predictions:
            diff = np.abs(predictions[key] - loaded_predictions[key]).max()
            print(f"Max difference for {key}: {diff:.2e}")
            assert diff < 1e-6, f"Predictions differ too much for {key}"

        print("✅ Serialization test passed!")

    finally:
        # Clean up
        if os.path.exists(model_path):
            os.unlink(model_path)

def test_model_without_view2id():
    """Test model serialization without view2id parameter."""

    # Model parameters
    INPUT_NUMERIC_COLS = ['feature1', 'feature2', 'feature3']
    REGRESSION_TARGETS = ['target1']
    BINARY_TARGETS = ['target2']
    MAX_LEN = 10
    EMB_DIM = 32
    TOKEN_HIDDEN = 64
    TRANSFORMER_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 2
    DROPOUT = 0.1
    view2id = None

    print("\nBuilding model without view2id...")
    model = build_embedding_transformer(
        INPUT_NUMERIC_COLS=INPUT_NUMERIC_COLS,
        REGRESSION_TARGETS=REGRESSION_TARGETS,
        BINARY_TARGETS=BINARY_TARGETS,
        MAX_LEN=MAX_LEN,
        EMB_DIM=EMB_DIM,
        TOKEN_HIDDEN=TOKEN_HIDDEN,
        TRANSFORMER_DIM=TRANSFORMER_DIM,
        NUM_HEADS=NUM_HEADS,
        NUM_LAYERS=NUM_LAYERS,
        DROPOUT=DROPOUT,
        view2id=view2id,
    )

    # Create dummy data for testing
    batch_size = 2
    dummy_input = {
        'num': np.random.random((batch_size, MAX_LEN, len(INPUT_NUMERIC_COLS))),
        'mask': np.ones((batch_size, MAX_LEN), dtype=bool)
    }

    print("Testing forward pass...")
    predictions = model.predict(dummy_input, verbose=0)
    print(f"Forward pass successful. Output shapes: {[(k, v.shape) for k, v in predictions.items()]}")

    # Test serialization
    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
        model_path = tmp_file.name

    try:
        print(f"Saving model to {model_path}...")
        model.save(model_path)
        print("Model saved successfully!")

        print("Loading model...")
        loaded_model = keras.models.load_model(model_path)
        print("Model loaded successfully!")

        print("Testing loaded model forward pass...")
        loaded_predictions = loaded_model.predict(dummy_input, verbose=0)
        print(f"Loaded model forward pass successful. Output shapes: {[(k, v.shape) for k, v in loaded_predictions.items()]}")

        print("✅ Serialization test without view2id passed!")

    finally:
        # Clean up
        if os.path.exists(model_path):
            os.unlink(model_path)

if __name__ == "__main__":
    test_model_serialization()
    test_model_without_view2id()
    print("\n🎉 All serialization tests passed!")