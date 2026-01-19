#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
SAM 3 Image Embedding Extraction Script

This script extracts visual embeddings from images using the SAM 3 model.
Visual embeddings: Multi-scale 256-dimensional feature pyramids from images

The extracted embeddings can be used later at inference time with text prompts.

Usage:
    python extract_embeddings.py --image path/to/image.jpg --output image_embeddings.npz
    python extract_embeddings.py --image-dir /path/to/images/ --output-dir ./embeddings/
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def extract_image_embeddings(
    model: torch.nn.Module,
    image_path: str,
    device: str = "cuda",
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Extract visual embeddings from an image.

    Args:
        model: SAM3 model
        image_path: Path to input image
        device: Device to run on
        verbose: Print extraction details

    Returns:
        Dictionary containing multi-scale visual embeddings and metadata
    """
    if verbose:
        print(f"Extracting image embeddings from: {image_path}")

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    if verbose:
        print(f"  Image size: {image.size}")

    # Get processor
    processor = Sam3Processor(model)

    # Set image and extract features
    with torch.no_grad():
        inference_state = processor.set_image(image)

        # Extract multi-scale visual features from the state
        vision_features = inference_state["vision_features"]  # Multi-scale features
        vision_pos_enc = inference_state["vision_pos_enc"]    # Position encodings

    embeddings = {
        "vision_features": [feat.cpu().numpy() for feat in vision_features],
        "vision_pos_enc": [pos.cpu().numpy() for pos in vision_pos_enc],
        "image_size": np.array(image.size),  # (width, height)
        "image_path": str(image_path),
    }

    if verbose:
        print(f"  Number of feature scales: {len(embeddings['vision_features'])}")
        for i, feat in enumerate(embeddings['vision_features']):
            print(f"    Scale {i}: {feat.shape}")

    return embeddings


def process_image_directory(
    model: torch.nn.Module,
    image_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    image_extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
) -> int:
    """
    Process all images in a directory and save embeddings.

    Args:
        model: SAM3 model
        image_dir: Directory containing images
        output_dir: Directory to save embeddings
        device: Device to run on
        image_extensions: List of valid image extensions

    Returns:
        Number of images processed
    """
    # Find all images
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"No images found in {image_dir}")
        return 0

    print(f"Found {len(image_files)} images in {image_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Extract embeddings
            embeddings = extract_image_embeddings(
                model, image_path, device, verbose=False
            )

            # Save with same name but .npz extension
            output_path = output_dir / f"{image_path.stem}_embeddings.npz"
            save_embeddings(embeddings, output_path, verbose=False)

        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            continue

    return len(image_files)


def save_embeddings(embeddings: Dict, output_path: str, verbose: bool = True):
    """Save embeddings to a file."""
    output_path = Path(output_path)

    # Convert any remaining tensors to numpy
    save_dict = {}
    for key, value in embeddings.items():
        if isinstance(value, torch.Tensor):
            save_dict[key] = value.cpu().numpy()
        elif isinstance(value, list):
            # Handle list of tensors
            if value and isinstance(value[0], torch.Tensor):
                save_dict[key] = np.array([v.cpu().numpy() for v in value], dtype=object)
            else:
                save_dict[key] = value
        else:
            save_dict[key] = value

    # Save based on extension
    if output_path.suffix == ".npz":
        np.savez_compressed(output_path, **save_dict)
    elif output_path.suffix == ".pt":
        torch.save(embeddings, output_path)
        output_path = output_path  # Keep the .pt extension
    else:
        # Default to npz
        output_path = output_path.with_suffix(".npz")
        np.savez_compressed(output_path, **save_dict)

    if verbose:
        print(f"\nEmbeddings saved to: {output_path}")
        # Print file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Extract image embeddings from SAM 3 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract embeddings from a single image
  python extract_embeddings.py --image photo.jpg --output embeddings.npz

  # Process all images in a directory
  python extract_embeddings.py --image-dir ./images/ --output-dir ./embeddings/

  # Save as PyTorch file
  python extract_embeddings.py --image photo.jpg --output embeddings.pt

  # Use CPU instead of GPU
  python extract_embeddings.py --image photo.jpg --output embeddings.npz --device cpu
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=str,
        help="Path to a single input image"
    )
    input_group.add_argument(
        "--image-dir",
        type=str,
        help="Path to directory containing images (batch processing)"
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for single image (.npz or .pt)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for batch processing"
    )

    # Model options
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: download from HuggingFace)"
    )

    args = parser.parse_args()

    # Validate inputs
    if args.image and not args.output:
        parser.error("--output is required when using --image")
    if args.image_dir and not args.output_dir:
        parser.error("--output-dir is required when using --image-dir")
    if args.image and not os.path.exists(args.image):
        parser.error(f"Image file not found: {args.image}")
    if args.image_dir and not os.path.isdir(args.image_dir):
        parser.error(f"Image directory not found: {args.image_dir}")

    # Load model
    print("="*60)
    print("Loading SAM 3 model...")
    print("="*60)

    try:
        model = build_sam3_image_model(checkpoint_path=args.checkpoint)
        model = model.to(args.device)
        model.eval()
        print(f"Model loaded successfully on {args.device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nNote: You need to authenticate with HuggingFace to download checkpoints.")
        print("Run: huggingface-cli login")
        return 1

    print("="*60)

    # Extract embeddings
    try:
        if args.image:
            # Single image mode
            embeddings = extract_image_embeddings(
                model, args.image, args.device, verbose=True
            )
            save_embeddings(embeddings, args.output, verbose=True)

            print("="*60)
            print("Image embedding extraction completed successfully!")
            print("="*60)

        else:
            # Batch directory mode
            num_processed = process_image_directory(
                model,
                Path(args.image_dir),
                Path(args.output_dir),
                args.device
            )

            print("="*60)
            print(f"Batch processing completed! Processed {num_processed} images.")
            print(f"Embeddings saved to: {args.output_dir}")
            print("="*60)

    except Exception as e:
        print(f"\nError during embedding extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
