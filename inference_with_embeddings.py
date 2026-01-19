#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
SAM 3 Inference with Pre-computed Embeddings

This script performs fast inference using pre-computed image embeddings and text prompts.
It loads saved embeddings and generates segmentation masks based on text descriptions.

Usage:
    python inference_with_embeddings.py --embeddings image_emb.npz --text "person" --output masks.npz
    python inference_with_embeddings.py --embeddings image_emb.npz --text "person" --visualize --output result.png
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def load_image_embeddings(embeddings_path: str) -> Dict:
    """
    Load pre-computed image embeddings from a file.

    Args:
        embeddings_path: Path to saved embeddings (.npz or .pt)

    Returns:
        Dictionary containing embeddings and metadata
    """
    embeddings_path = Path(embeddings_path)

    if embeddings_path.suffix == ".npz":
        data = np.load(embeddings_path, allow_pickle=True)
        return {
            "vision_features": data["vision_features"],
            "vision_pos_enc": data["vision_pos_enc"],
            "image_size": tuple(data["image_size"]),
            "image_path": str(data["image_path"]) if "image_path" in data else None,
        }
    elif embeddings_path.suffix == ".pt":
        return torch.load(embeddings_path)
    else:
        raise ValueError(f"Unsupported file format: {embeddings_path.suffix}")


def embeddings_to_inference_state(
    embeddings: Dict,
    device: str = "cuda"
) -> Dict:
    """
    Convert saved embeddings to inference state for SAM3 processor.

    Args:
        embeddings: Dictionary with vision_features, vision_pos_enc, image_size
        device: Device to move tensors to

    Returns:
        Inference state dictionary ready for SAM3 processor
    """
    # Convert numpy arrays to torch tensors and move to device
    vision_features = [
        torch.from_numpy(feat).to(device)
        for feat in embeddings["vision_features"]
    ]
    vision_pos_enc = [
        torch.from_numpy(pos).to(device)
        for pos in embeddings["vision_pos_enc"]
    ]

    return {
        "vision_features": vision_features,
        "vision_pos_enc": vision_pos_enc,
        "image_size": embeddings["image_size"],
    }


def predict_with_text_prompt(
    model: torch.nn.Module,
    embeddings_path: str,
    text_prompt: str,
    device: str = "cuda",
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Generate segmentation masks from pre-computed embeddings and text prompt.

    Args:
        model: SAM3 model
        embeddings_path: Path to saved image embeddings
        text_prompt: Text description for segmentation (e.g., "person", "red car")
        device: Device to run on
        verbose: Print prediction details

    Returns:
        Dictionary containing masks, boxes, and scores
    """
    if verbose:
        print(f"Loading embeddings from: {embeddings_path}")

    # Load embeddings
    embeddings = load_image_embeddings(embeddings_path)

    if verbose:
        print(f"  Image size: {embeddings['image_size']}")
        if embeddings["image_path"]:
            print(f"  Original image: {embeddings['image_path']}")
        print(f"\nGenerating masks for prompt: '{text_prompt}'")

    # Convert to inference state
    inference_state = embeddings_to_inference_state(embeddings, device)

    # Get processor and run prediction
    processor = Sam3Processor(model)

    with torch.no_grad():
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt
        )

    # Convert to numpy
    result = {
        "masks": output["masks"].cpu().numpy(),
        "boxes": output["boxes"].cpu().numpy(),
        "scores": output["scores"].cpu().numpy(),
        "text_prompt": text_prompt,
        "image_size": embeddings["image_size"],
    }

    if verbose:
        print(f"\nResults:")
        print(f"  Found {len(result['scores'])} detections")
        print(f"  Masks shape: {result['masks'].shape}")
        print(f"  Boxes shape: {result['boxes'].shape}")
        print(f"  Scores: {result['scores']}")

    return result


def visualize_predictions(
    predictions: Dict,
    original_image_path: Optional[str] = None,
    output_path: str = "visualization.png",
    show_boxes: bool = True,
    show_labels: bool = True,
    alpha: float = 0.5
):
    """
    Visualize segmentation predictions on the original image.

    Args:
        predictions: Dictionary with masks, boxes, scores from predict_with_text_prompt
        original_image_path: Path to original image (if available)
        output_path: Where to save visualization
        show_boxes: Draw bounding boxes
        show_labels: Show scores as labels
        alpha: Transparency of mask overlay (0-1)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib import cm
    except ImportError:
        print("Error: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        return

    # Load original image if provided
    if original_image_path and os.path.exists(original_image_path):
        image = Image.open(original_image_path).convert("RGB")
    else:
        # Create blank canvas with the right size
        width, height = predictions["image_size"]
        image = Image.new("RGB", (width, height), color="white")

    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    # Get colormap for masks
    colors = cm.get_cmap("tab10", len(predictions["masks"]))

    # Draw each mask
    masks = predictions["masks"]
    boxes = predictions["boxes"]
    scores = predictions["scores"]

    for idx, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        color = colors(idx)[:3]

        # Draw mask
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask > 0] = (*color, alpha)
        ax.imshow(colored_mask)

        if show_boxes:
            # Draw bounding box (box format: [x1, y1, x2, y2])
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)

        if show_labels:
            # Add score label
            x1, y1 = box[:2]
            ax.text(
                x1, y1 - 5,
                f"{score:.2f}",
                color="white",
                fontsize=10,
                bbox=dict(facecolor=color, alpha=0.8, boxstyle="round,pad=0.3")
            )

    # Add title with text prompt
    ax.set_title(f"Segmentation: '{predictions['text_prompt']}'", fontsize=14, pad=10)
    ax.axis("off")

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def save_predictions(predictions: Dict, output_path: str):
    """Save prediction results to a file."""
    output_path = Path(output_path)

    if output_path.suffix == ".npz":
        np.savez_compressed(output_path, **predictions)
    elif output_path.suffix == ".pt":
        torch.save(predictions, output_path)
    else:
        # Default to npz
        output_path = output_path.with_suffix(".npz")
        np.savez_compressed(output_path, **predictions)

    print(f"Predictions saved to: {output_path}")

    # Print file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Perform inference with pre-computed SAM 3 image embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference with text prompt
  python inference_with_embeddings.py \\
    --embeddings image_emb.npz \\
    --text "person" \\
    --output predictions.npz

  # Inference with visualization
  python inference_with_embeddings.py \\
    --embeddings image_emb.npz \\
    --text "person in red shirt" \\
    --visualize \\
    --original-image photo.jpg \\
    --output result.png

  # Multiple prompts on the same embeddings
  python inference_with_embeddings.py --embeddings image_emb.npz --text "person" --output person.npz
  python inference_with_embeddings.py --embeddings image_emb.npz --text "car" --output car.npz
  python inference_with_embeddings.py --embeddings image_emb.npz --text "tree" --output tree.npz
        """
    )

    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to pre-computed image embeddings (.npz or .pt)"
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text prompt for segmentation (e.g., 'person', 'red car')"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for predictions (.npz, .pt) or visualization (.png, .jpg)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization instead of saving raw predictions"
    )
    parser.add_argument(
        "--original-image",
        type=str,
        help="Path to original image for visualization (optional)"
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
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Mask transparency for visualization (0-1, default: 0.5)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.embeddings):
        parser.error(f"Embeddings file not found: {args.embeddings}")

    if args.visualize and args.original_image and not os.path.exists(args.original_image):
        parser.error(f"Original image not found: {args.original_image}")

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

    # Run inference
    try:
        predictions = predict_with_text_prompt(
            model=model,
            embeddings_path=args.embeddings,
            text_prompt=args.text,
            device=args.device,
            verbose=True
        )

        print("="*60)

        if args.visualize:
            # Generate visualization
            visualize_predictions(
                predictions=predictions,
                original_image_path=args.original_image,
                output_path=args.output,
                alpha=args.alpha
            )
        else:
            # Save raw predictions
            save_predictions(predictions, args.output)

        print("="*60)
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
