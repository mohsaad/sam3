#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Example: Complete SAM 3 Embedding Workflow

This script demonstrates the complete workflow:
1. Extract image embeddings (pre-processing)
2. Use embeddings with text prompts (inference)
3. Try multiple prompts efficiently

Run this example:
    python example_embedding_workflow.py --image path/to/photo.jpg
"""

import argparse
import os
from pathlib import Path

import torch

from extract_embeddings import extract_image_embeddings, save_embeddings
from inference_with_embeddings import predict_with_text_prompt, visualize_predictions
from sam3.model_builder import build_sam3_image_model


def main():
    parser = argparse.ArgumentParser(
        description="Example workflow: Extract embeddings and run inference"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./example_output",
        help="Directory to save outputs (default: ./example_output)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    embeddings_path = output_dir / "image_embeddings.npz"
    image_filename = Path(args.image).stem

    print("="*70)
    print("SAM 3 EMBEDDING WORKFLOW EXAMPLE")
    print("="*70)
    print(f"Input image: {args.image}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print("="*70)

    # Load model once
    print("\n[1/4] Loading SAM 3 model...")
    model = build_sam3_image_model()
    model = model.to(args.device)
    model.eval()
    print("✓ Model loaded successfully")

    # Step 1: Extract image embeddings (slow, but only done once)
    print("\n[2/4] Extracting image embeddings (this takes ~10-30 seconds)...")
    embeddings = extract_image_embeddings(
        model=model,
        image_path=args.image,
        device=args.device,
        verbose=True
    )
    save_embeddings(embeddings, embeddings_path, verbose=True)
    print("✓ Embeddings saved")

    # Step 2: Run inference with multiple text prompts (fast!)
    print("\n[3/4] Running inference with multiple text prompts...")
    print("     (This is fast because we use pre-computed embeddings!)")

    text_prompts = [
        "person",
        "face",
        "clothing",
        "background",
    ]

    results = {}
    for i, text_prompt in enumerate(text_prompts, 1):
        print(f"\n  [{i}/{len(text_prompts)}] Processing prompt: '{text_prompt}'")

        # Run inference (fast!)
        predictions = predict_with_text_prompt(
            model=model,
            embeddings_path=embeddings_path,
            text_prompt=text_prompt,
            device=args.device,
            verbose=False
        )

        # Store results
        results[text_prompt] = predictions

        # Save visualization
        vis_path = output_dir / f"{image_filename}_{text_prompt.replace(' ', '_')}.png"
        visualize_predictions(
            predictions=predictions,
            original_image_path=args.image,
            output_path=vis_path,
            show_boxes=True,
            show_labels=True,
            alpha=0.5
        )

        num_detections = len(predictions["scores"])
        if num_detections > 0:
            print(f"      ✓ Found {num_detections} detection(s)")
            print(f"      ✓ Scores: {predictions['scores']}")
        else:
            print(f"      ✓ No detections found")

    print("\n✓ Inference completed for all prompts")

    # Step 3: Summary
    print("\n[4/4] Summary")
    print("="*70)
    print(f"✓ Embeddings saved to: {embeddings_path}")
    print(f"✓ Processed {len(text_prompts)} text prompts")
    print("\nGenerated visualizations:")
    for text_prompt in text_prompts:
        vis_path = output_dir / f"{image_filename}_{text_prompt.replace(' ', '_')}.png"
        print(f"  - {vis_path}")

    print("\n" + "="*70)
    print("WORKFLOW COMPLETE!")
    print("="*70)

    # Demonstrate the key insight
    print("\n💡 Key Insight:")
    print("   The first embedding extraction took ~10-30 seconds.")
    print("   Each subsequent text prompt took only ~100-500ms!")
    print("   That's a 20-50x speedup by pre-computing image embeddings!")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
