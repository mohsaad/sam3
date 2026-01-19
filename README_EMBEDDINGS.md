# SAM 3 Image Embedding Extraction

This directory contains tools for extracting **image embeddings** from the SAM 3 (Segment Anything with Concepts) model. These embeddings can be pre-computed and cached, then used at inference time with text prompts for fast segmentation.

## Overview

SAM 3 generates multi-scale visual embeddings from images:

**Image Embeddings** (256-dimensional):
- Multi-scale feature pyramids from images
- 4 scales: 4x, 2x, 1x, 0.5x resolution
- Generated from the ViT backbone with position encodings
- Can be pre-computed and saved for later use
- At inference time, combine with text prompts for segmentation

**Workflow**:
1. **Pre-processing** (this tool): Extract and save image embeddings
2. **Inference time**: Load saved embeddings + encode text prompts → fast segmentation

## Quick Start

### Prerequisites

- Docker with NVIDIA GPU support (nvidia-docker2)
- CUDA 12.6+ compatible GPU
- HuggingFace account with access to SAM 3 checkpoints
- HuggingFace token (set as `HUGGING_FACE_HUB_TOKEN` environment variable)

### Setup

1. **Request access to SAM 3 checkpoints**:
   - Visit: https://huggingface.co/facebook/sam3
   - Request access and wait for approval

2. **Get your HuggingFace token**:
   ```bash
   # Option 1: Use huggingface-cli
   pip install huggingface-hub
   huggingface-cli login

   # Option 2: Set token directly
   export HUGGING_FACE_HUB_TOKEN=your_token_here
   ```

3. **Build the Docker image**:
   ```bash
   docker build -t sam3-embeddings:latest -f Dockerfile.embeddings .
   ```

## Usage

### Option 1: Using the Helper Script (Recommended)

The `run_embedding_extraction.sh` script simplifies running the Docker container.

```bash
# Extract embeddings from a single image
# (first place your image in ./input/ directory)
cp /path/to/your/photo.jpg ./input/
./run_embedding_extraction.sh --image photo.jpg --output image_embeddings.npz

# Process an entire directory of images (batch mode)
cp -r /path/to/your/images/* ./input/
./run_embedding_extraction.sh --image-dir ./input/ --output-dir ./output/
```

### Option 2: Using Docker Directly

```bash
# Create directories
mkdir -p input output cache/huggingface cache/torch

# Extract embeddings from a single image
docker run --rm --gpus all \
  -v $(pwd)/input:/workspace/input:ro \
  -v $(pwd)/output:/workspace/output:rw \
  -v $(pwd)/cache/huggingface:/workspace/.cache/huggingface:rw \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  sam3-embeddings:latest \
  --image /workspace/input/photo.jpg --output /workspace/output/image_embeddings.npz

# Process a directory of images (batch mode)
docker run --rm --gpus all \
  -v $(pwd)/input:/workspace/input:ro \
  -v $(pwd)/output:/workspace/output:rw \
  -v $(pwd)/cache/huggingface:/workspace/.cache/huggingface:rw \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  sam3-embeddings:latest \
  --image-dir /workspace/input --output-dir /workspace/output
```

### Option 3: Using Docker Compose

```bash
# Extract embeddings using docker-compose
docker-compose -f docker-compose.embeddings.yml run sam3-embeddings \
  --image /workspace/input/photo.jpg --output /workspace/output/embeddings.npz

# Batch process images
docker-compose -f docker-compose.embeddings.yml run sam3-embeddings \
  --image-dir /workspace/input --output-dir /workspace/output
```

### Option 4: Running Without Docker (Native)

If you have the environment set up locally:

```bash
# Install SAM3
pip install -e .

# Run the extraction script on a single image
python extract_embeddings.py --image photo.jpg --output image_embeddings.npz

# Batch process a directory
python extract_embeddings.py --image-dir ./images/ --output-dir ./embeddings/
```

## Output Format

Embeddings are saved in NumPy's `.npz` format (compressed) or PyTorch's `.pt` format.

### Image Embeddings

```python
import numpy as np

# Load the saved embeddings
data = np.load("image_embeddings.npz", allow_pickle=True)

# Extract the components
vision_features = data["vision_features"]  # List of multi-scale features
vision_pos_enc = data["vision_pos_enc"]    # Position encodings
image_size = tuple(data["image_size"])     # (width, height)
image_path = str(data["image_path"])       # Original image path

# Inspect the multi-scale features
# Each scale has shape: [1, C, H, W] where C=256
for i, feat in enumerate(vision_features):
    print(f"Scale {i}: {feat.shape}")

# Example output:
# Scale 0: (1, 256, 252, 252)  # 4x resolution
# Scale 1: (1, 256, 126, 126)  # 2x resolution
# Scale 2: (1, 256, 63, 63)    # 1x resolution
# Scale 3: (1, 256, 31, 31)    # 0.5x resolution
```

### Using Saved Embeddings at Inference Time

```python
import numpy as np
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load pre-computed image embeddings
data = np.load("image_embeddings.npz", allow_pickle=True)
vision_features = [torch.from_numpy(f).cuda() for f in data["vision_features"]]
vision_pos_enc = [torch.from_numpy(p).cuda() for p in data["vision_pos_enc"]]

# Build model
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Create inference state from saved embeddings
inference_state = {
    "vision_features": vision_features,
    "vision_pos_enc": vision_pos_enc,
    "image_size": tuple(data["image_size"]),
}

# Now use text prompts at inference time (fast!)
output = processor.set_text_prompt(state=inference_state, prompt="person in red shirt")
masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]
```

## Advanced Usage

### Custom Checkpoint

Use a local checkpoint instead of downloading from HuggingFace:

```bash
./run_embedding_extraction.sh \
  --checkpoint /path/to/checkpoint.pt \
  --image photo.jpg \
  --text "person" \
  --output embeddings.npz
```

### CPU Mode

If you don't have a GPU (not recommended, very slow):

```bash
python extract_embeddings.py \
  --device cpu \
  --image photo.jpg \
  --output image_embeddings.npz
```

### Save as PyTorch Format

```bash
./run_embedding_extraction.sh \
  --image photo.jpg \
  --output embeddings.pt
```

## Troubleshooting

### Error: "CUDA not available"

Make sure:
- NVIDIA drivers are installed
- nvidia-docker2 is installed: `sudo apt install nvidia-docker2`
- Docker daemon is restarted: `sudo systemctl restart docker`

### Error: "401 Client Error: Unauthorized"

You need to:
1. Request access to SAM 3 checkpoints on HuggingFace
2. Set your HuggingFace token: `export HUGGING_FACE_HUB_TOKEN=your_token`

### Out of Memory

If you run out of GPU memory:
- Use a smaller image
- Use CPU mode (very slow)
- Close other GPU-using processes

## File Structure

```
sam3/
├── extract_embeddings.py          # Main extraction script
├── Dockerfile.embeddings           # Docker container definition
├── docker-compose.embeddings.yml  # Docker Compose configuration
├── run_embedding_extraction.sh    # Helper script for easy usage
├── README_EMBEDDINGS.md           # This file
├── input/                         # Place input images here
├── output/                        # Output embeddings saved here
└── cache/                         # Model checkpoints cached here
    ├── huggingface/
    └── torch/
```

## Performance

- **First run**: Downloads ~3GB of model checkpoints (takes 5-15 minutes)
- **Subsequent runs**: Uses cached checkpoints (takes 10-30 seconds per image)
- **Batch processing**: ~5-10 images per minute on a modern GPU
- **GPU memory**: Requires ~8GB VRAM for inference
- **Embedding file sizes**: 50-200 MB per image (depends on image size)
  - Larger images → larger embeddings
  - Multi-scale features stored for all 4 pyramid levels

## Benefits of Pre-computing Image Embeddings

1. **Faster inference**: Skip image encoding at inference time
2. **Flexible text prompts**: Try different text prompts on the same image instantly
3. **Caching**: Process images once, reuse embeddings many times
4. **Batch processing**: Extract embeddings for entire datasets offline
5. **Cost optimization**: Separate expensive image encoding from fast text-based queries

## Citation

If you use SAM 3 embeddings in your research, please cite:

```bibtex
@misc{carion2025sam3segmentconcepts,
      title={SAM 3: Segment Anything with Concepts},
      author={Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and Shoubhik Debnath and others},
      year={2025},
      eprint={2511.16719},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16719},
}
```

## License

This project is licensed under the SAM License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
- SAM 3 GitHub: https://github.com/facebookresearch/sam3
- HuggingFace: https://huggingface.co/facebook/sam3
