#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
SageMaker Inference Flask Application

This Flask app handles inference requests for SAM 3 with pre-computed embeddings.

Expected request format:
POST /invocations
{
    "embeddings": <base64 encoded npz file>,
    "text_prompt": "person in red shirt",
    "visualize": false,
    "alpha": 0.5
}

Or with S3 path:
{
    "embeddings_s3_path": "s3://bucket/path/to/embeddings.npz",
    "text_prompt": "person",
    "visualize": false
}
"""

import base64
import io
import json
import os
import traceback
from pathlib import Path
from typing import Dict, Any

import boto3
import numpy as np
import torch
from flask import Flask, request, jsonify
from PIL import Image

# Import SAM3 and inference functions
import sys
sys.path.insert(0, '/opt/ml/code')

from inference_with_embeddings import (
    predict_with_text_prompt,
    load_image_embeddings,
    embeddings_to_inference_state
)
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# Initialize Flask app
app = Flask(__name__)

# Global model variable (loaded once on startup)
MODEL = None
DEVICE = None


def load_model():
    """Load SAM3 model on server startup."""
    global MODEL, DEVICE

    print("Loading SAM 3 model...")

    # Determine device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Load model
    MODEL = build_sam3_image_model()
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()

    print("Model loaded successfully!")


def download_from_s3(s3_path: str, local_path: str) -> str:
    """Download file from S3."""
    # Parse S3 path
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}")

    parts = s3_path[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    # Download
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)

    return local_path


def decode_embeddings(embeddings_b64: str, temp_dir: str = "/tmp") -> str:
    """Decode base64 embeddings and save to temp file."""
    # Decode base64
    embeddings_bytes = base64.b64decode(embeddings_b64)

    # Save to temp file
    temp_path = os.path.join(temp_dir, "embeddings.npz")
    with open(temp_path, "wb") as f:
        f.write(embeddings_bytes)

    return temp_path


def encode_response(predictions: Dict) -> Dict[str, Any]:
    """Encode numpy arrays in predictions to base64 for JSON response."""
    response = {}

    for key, value in predictions.items():
        if isinstance(value, np.ndarray):
            # Convert to bytes and base64 encode
            buffer = io.BytesIO()
            np.save(buffer, value)
            buffer.seek(0)
            response[key] = base64.b64encode(buffer.read()).decode('utf-8')
            response[f"{key}_shape"] = value.shape
            response[f"{key}_dtype"] = str(value.dtype)
        else:
            response[key] = value

    return response


@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint."""
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 503

    return jsonify({"status": "healthy"}), 200


@app.route("/invocations", methods=["POST"])
def invoke():
    """
    Main inference endpoint.

    Expected JSON payload:
    {
        "embeddings": "<base64 encoded npz>",  # Option 1: inline embeddings
        "embeddings_s3_path": "s3://...",      # Option 2: S3 path
        "text_prompt": "person",
        "visualize": false,
        "alpha": 0.5
    }
    """
    try:
        # Parse request
        if request.content_type == "application/json":
            data = request.get_json()
        else:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        # Validate required fields
        if "text_prompt" not in data:
            return jsonify({"error": "Missing required field: text_prompt"}), 400

        if "embeddings" not in data and "embeddings_s3_path" not in data:
            return jsonify({
                "error": "Missing embeddings. Provide either 'embeddings' (base64) or 'embeddings_s3_path'"
            }), 400

        text_prompt = data["text_prompt"]
        visualize = data.get("visualize", False)
        alpha = data.get("alpha", 0.5)

        # Get embeddings path
        if "embeddings_s3_path" in data:
            # Download from S3
            s3_path = data["embeddings_s3_path"]
            print(f"Downloading embeddings from S3: {s3_path}")
            embeddings_path = download_from_s3(s3_path, "/tmp/embeddings.npz")
        else:
            # Decode base64 embeddings
            print("Decoding inline embeddings...")
            embeddings_b64 = data["embeddings"]
            embeddings_path = decode_embeddings(embeddings_b64)

        print(f"Running inference with prompt: '{text_prompt}'")

        # Load embeddings
        embeddings = load_image_embeddings(embeddings_path)

        # Convert to inference state
        inference_state = embeddings_to_inference_state(embeddings, DEVICE)

        # Get processor
        processor = Sam3Processor(MODEL)

        # Run prediction
        with torch.no_grad():
            output = processor.set_text_prompt(
                state=inference_state,
                prompt=text_prompt
            )

        # Prepare response
        predictions = {
            "masks": output["masks"].cpu().numpy(),
            "boxes": output["boxes"].cpu().numpy(),
            "scores": output["scores"].cpu().numpy(),
            "text_prompt": text_prompt,
            "image_size": embeddings["image_size"],
            "num_detections": len(output["scores"]),
        }

        print(f"Found {predictions['num_detections']} detections")

        # Encode response (convert numpy to base64)
        response = encode_response(predictions)

        # Clean up temp file
        if os.path.exists(embeddings_path):
            os.remove(embeddings_path)

        return jsonify(response), 200

    except Exception as e:
        print(f"Error during inference: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# Load model on startup
with app.app_context():
    load_model()


# For local testing
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
