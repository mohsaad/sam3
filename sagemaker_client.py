#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
SAM 3 SageMaker Client

This script provides a convenient client for invoking the SAM 3 SageMaker endpoint.

Usage:
    python sagemaker_client.py --endpoint sam3-inference --embeddings image_emb.npz --text "person"
    python sagemaker_client.py --endpoint sam3-inference --embeddings-s3 s3://bucket/emb.npz --text "car"
"""

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Dict, Optional

import boto3
import numpy as np


class SAM3SageMakerClient:
    """Client for SAM 3 SageMaker endpoint."""

    def __init__(
        self,
        endpoint_name: str,
        region_name: Optional[str] = None
    ):
        """
        Initialize SageMaker client.

        Args:
            endpoint_name: Name of the SageMaker endpoint
            region_name: AWS region (uses default if not specified)
        """
        self.endpoint_name = endpoint_name
        self.region_name = region_name or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        # Initialize boto3 client
        self.client = boto3.client(
            "sagemaker-runtime",
            region_name=self.region_name
        )

        print(f"Initialized SageMaker client for endpoint: {endpoint_name}")
        print(f"Region: {self.region_name}")

    def predict_with_embeddings_file(
        self,
        embeddings_path: str,
        text_prompt: str,
        visualize: bool = False,
        alpha: float = 0.5
    ) -> Dict:
        """
        Run inference with embeddings file.

        Args:
            embeddings_path: Path to embeddings file (.npz)
            text_prompt: Text prompt for segmentation
            visualize: Whether to generate visualization
            alpha: Transparency for visualization

        Returns:
            Dictionary with predictions
        """
        print(f"\nLoading embeddings from: {embeddings_path}")

        # Read and encode embeddings
        with open(embeddings_path, "rb") as f:
            embeddings_bytes = f.read()

        embeddings_b64 = base64.b64encode(embeddings_bytes).decode('utf-8')

        print(f"Embeddings size: {len(embeddings_bytes) / 1024 / 1024:.2f} MB")
        print(f"Running inference with prompt: '{text_prompt}'")

        # Prepare payload
        payload = {
            "embeddings": embeddings_b64,
            "text_prompt": text_prompt,
            "visualize": visualize,
            "alpha": alpha
        }

        # Invoke endpoint
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload)
        )

        # Parse response
        result = json.loads(response["Body"].read())

        # Decode numpy arrays from base64
        predictions = self._decode_predictions(result)

        print(f"✓ Inference completed successfully!")
        print(f"  Found {predictions['num_detections']} detections")
        if predictions['num_detections'] > 0:
            print(f"  Scores: {predictions['scores']}")

        return predictions

    def predict_with_s3_embeddings(
        self,
        embeddings_s3_path: str,
        text_prompt: str,
        visualize: bool = False,
        alpha: float = 0.5
    ) -> Dict:
        """
        Run inference with embeddings stored in S3.

        Args:
            embeddings_s3_path: S3 path to embeddings (s3://bucket/key)
            text_prompt: Text prompt for segmentation
            visualize: Whether to generate visualization
            alpha: Transparency for visualization

        Returns:
            Dictionary with predictions
        """
        print(f"\nUsing embeddings from S3: {embeddings_s3_path}")
        print(f"Running inference with prompt: '{text_prompt}'")

        # Prepare payload with S3 path
        payload = {
            "embeddings_s3_path": embeddings_s3_path,
            "text_prompt": text_prompt,
            "visualize": visualize,
            "alpha": alpha
        }

        # Invoke endpoint
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload)
        )

        # Parse response
        result = json.loads(response["Body"].read())

        # Decode numpy arrays from base64
        predictions = self._decode_predictions(result)

        print(f"✓ Inference completed successfully!")
        print(f"  Found {predictions['num_detections']} detections")
        if predictions['num_detections'] > 0:
            print(f"  Scores: {predictions['scores']}")

        return predictions

    def _decode_predictions(self, result: Dict) -> Dict:
        """Decode base64-encoded numpy arrays in predictions."""
        predictions = {}

        for key, value in result.items():
            if isinstance(value, str) and f"{key}_shape" in result:
                # This is a base64-encoded numpy array
                array_bytes = base64.b64decode(value)
                array = np.load(base64.BytesIO(array_bytes), allow_pickle=True)
                predictions[key] = array
            else:
                predictions[key] = value

        return predictions

    def save_predictions(self, predictions: Dict, output_path: str):
        """Save predictions to file."""
        output_path = Path(output_path)

        if output_path.suffix == ".npz":
            # Save as numpy
            save_dict = {}
            for key, value in predictions.items():
                if not key.endswith("_shape") and not key.endswith("_dtype"):
                    save_dict[key] = value

            np.savez_compressed(output_path, **save_dict)
            print(f"\n✓ Predictions saved to: {output_path}")
        elif output_path.suffix == ".json":
            # Save as JSON (convert arrays to lists)
            save_dict = {}
            for key, value in predictions.items():
                if isinstance(value, np.ndarray):
                    save_dict[key] = value.tolist()
                else:
                    save_dict[key] = value

            with open(output_path, "w") as f:
                json.dump(save_dict, f, indent=2)

            print(f"\n✓ Predictions saved to: {output_path}")
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3 SageMaker Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference with local embeddings file
  python sagemaker_client.py \\
    --endpoint sam3-inference \\
    --embeddings image_emb.npz \\
    --text "person" \\
    --output predictions.npz

  # Inference with S3 embeddings
  python sagemaker_client.py \\
    --endpoint sam3-inference \\
    --embeddings-s3 s3://my-bucket/embeddings/image_emb.npz \\
    --text "car" \\
    --output predictions.npz

  # Multiple prompts on same embeddings
  python sagemaker_client.py --endpoint sam3-inference --embeddings img.npz --text "person" --output person.npz
  python sagemaker_client.py --endpoint sam3-inference --embeddings img.npz --text "car" --output car.npz
        """
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        required=True,
        help="SageMaker endpoint name"
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        help="Path to local embeddings file (.npz)"
    )
    parser.add_argument(
        "--embeddings-s3",
        type=str,
        help="S3 path to embeddings (s3://bucket/key)"
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text prompt for segmentation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.npz",
        help="Output file path (.npz or .json)"
    )
    parser.add_argument(
        "--region",
        type=str,
        help="AWS region (uses default if not specified)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Transparency for visualization (0-1)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.embeddings and not args.embeddings_s3:
        parser.error("Either --embeddings or --embeddings-s3 must be provided")

    if args.embeddings and args.embeddings_s3:
        parser.error("Provide only one of --embeddings or --embeddings-s3")

    if args.embeddings and not os.path.exists(args.embeddings):
        parser.error(f"Embeddings file not found: {args.embeddings}")

    # Initialize client
    client = SAM3SageMakerClient(
        endpoint_name=args.endpoint,
        region_name=args.region
    )

    # Run inference
    try:
        if args.embeddings:
            predictions = client.predict_with_embeddings_file(
                embeddings_path=args.embeddings,
                text_prompt=args.text,
                visualize=args.visualize,
                alpha=args.alpha
            )
        else:
            predictions = client.predict_with_s3_embeddings(
                embeddings_s3_path=args.embeddings_s3,
                text_prompt=args.text,
                visualize=args.visualize,
                alpha=args.alpha
            )

        # Save predictions
        client.save_predictions(predictions, args.output)

        print("\n" + "="*60)
        print("Inference completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
