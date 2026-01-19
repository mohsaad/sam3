#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
SAM 3 Embedding Extraction Script

This script extracts text and visual embeddings from the SAM 3 model.
Text embeddings: 256-dimensional vectors from natural language prompts
Visual embeddings: Multi-scale 256-dimensional feature pyramids from images

Usage:
    python extract_embeddings.py --image path/to/image.jpg --text "your prompt" --output embeddings.npz
    python extract_embeddings.py --text "person" --output text_embeddings.npz
    python extract_embeddings.py --image path/to/image.jpg --output visual_embeddings.npz
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def extract_text_embeddings(
    model: torch.nn.Module,
    text_prompt: str,
    device: str = "cuda"
) -> Dict[str, np.ndarray]:
    """
    Extract text embeddings from a text prompt.

    Args:
        model: SAM3 model
        text_prompt: Text description (e.g., "person in red shirt")
        device: Device to run on

    Returns:
        Dictionary containing text embeddings
    """
    print(f"Extracting text embeddings for prompt: '{text_prompt}'")

    # Get the text encoder from the backbone
    text_encoder = model.backbone.text

    # Tokenize and encode the text
    with torch.no_grad():
        text_output = text_encoder(text_prompt)
        text_tokens = text_output["text_tokens"]  # Shape: [batch, seq_len, 256]
        text_pooled = text_output["text_pooled"]  # Shape: [batch, 256]

    embeddings = {
        "text_tokens": text_tokens.cpu().numpy(),  # Token-level embeddings
        "text_pooled": text_pooled.cpu().numpy(),  # Pooled sentence embedding
        "text_prompt": text_prompt,
    }

    print(f"  Text tokens shape: {embeddings['text_tokens'].shape}")
    print(f"  Text pooled shape: {embeddings['text_pooled'].shape}")

    return embeddings


def extract_visual_embeddings(
    model: torch.nn.Module,
    image_path: str,
    device: str = "cuda"
) -> Dict[str, np.ndarray]:
    """
    Extract visual embeddings from an image.

    Args:
        model: SAM3 model
        image_path: Path to input image
        device: Device to run on

    Returns:
        Dictionary containing multi-scale visual embeddings
    """
    print(f"Extracting visual embeddings from: {image_path}")

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
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
        "image_size": image.size,
    }

    print(f"  Number of feature scales: {len(embeddings['vision_features'])}")
    for i, feat in enumerate(embeddings['vision_features']):
        print(f"  Scale {i} shape: {feat.shape}")

    return embeddings


def extract_combined_embeddings(
    model: torch.nn.Module,
    image_path: str,
    text_prompt: str,
    device: str = "cuda"
) -> Dict[str, np.ndarray]:
    """
    Extract combined text and visual embeddings after fusion.

    Args:
        model: SAM3 model
        image_path: Path to input image
        text_prompt: Text description
        device: Device to run on

    Returns:
        Dictionary containing all embeddings (text, visual, and fused)
    """
    print(f"Extracting combined embeddings:")
    print(f"  Image: {image_path}")
    print(f"  Text: '{text_prompt}'")

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Get processor
    processor = Sam3Processor(model)

    # Process image with text prompt
    with torch.no_grad():
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

    embeddings = {
        "text_prompt": text_prompt,
        "image_size": image.size,
        "vision_features": [feat.cpu().numpy() for feat in inference_state["vision_features"]],
        "masks": output["masks"].cpu().numpy(),
        "boxes": output["boxes"].cpu().numpy(),
        "scores": output["scores"].cpu().numpy(),
    }

    print(f"  Found {len(output['scores'])} detections")
    print(f"  Masks shape: {embeddings['masks'].shape}")
    print(f"  Boxes shape: {embeddings['boxes'].shape}")
    print(f"  Scores shape: {embeddings['scores'].shape}")

    return embeddings


def save_embeddings(embeddings: Dict, output_path: str):
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
        print(f"\nEmbeddings saved to: {output_path}")
    elif output_path.suffix == ".pt":
        torch.save(embeddings, output_path)
        print(f"\nEmbeddings saved to: {output_path}")
    else:
        # Default to npz
        output_path = output_path.with_suffix(".npz")
        np.savez_compressed(output_path, **save_dict)
        print(f"\nEmbeddings saved to: {output_path}")

    # Print file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from SAM 3 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract text embeddings only
  python extract_embeddings.py --text "person" --output text_emb.npz

  # Extract visual embeddings only
  python extract_embeddings.py --image photo.jpg --output visual_emb.npz

  # Extract combined embeddings (text + visual + detections)
  python extract_embeddings.py --image photo.jpg --text "person" --output combined_emb.npz

  # Save as PyTorch file
  python extract_embeddings.py --image photo.jpg --text "person" --output embeddings.pt
        """
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text prompt for embedding extraction"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (.npz or .pt)"
    )
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
    if not args.image and not args.text:
        parser.error("At least one of --image or --text must be provided")

    if args.image and not os.path.exists(args.image):
        parser.error(f"Image file not found: {args.image}")

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

    # Extract embeddings based on provided inputs
    try:
        if args.image and args.text:
            # Extract combined embeddings
            embeddings = extract_combined_embeddings(
                model, args.image, args.text, args.device
            )
        elif args.text:
            # Extract text embeddings only
            embeddings = extract_text_embeddings(
                model, args.text, args.device
            )
        elif args.image:
            # Extract visual embeddings only
            embeddings = extract_visual_embeddings(
                model, args.image, args.device
            )

        # Save embeddings
        save_embeddings(embeddings, args.output)

        print("="*60)
        print("Embedding extraction completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\nError during embedding extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
