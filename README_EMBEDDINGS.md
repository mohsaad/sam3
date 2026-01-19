# SAM 3 Embedding Extraction

This directory contains tools for extracting embeddings from the SAM 3 (Segment Anything with Concepts) model. You can extract text embeddings, visual embeddings, or combined embeddings with detection results.

## Overview

SAM 3 generates two types of embeddings:

1. **Text Embeddings** (256-dimensional):
   - Token-level embeddings: One vector per word/token
   - Pooled sentence embedding: Single vector representing the entire prompt
   - Generated from the TextTransformer encoder

2. **Visual Embeddings** (256-dimensional):
   - Multi-scale feature pyramids from images
   - 4 scales: 4x, 2x, 1x, 0.5x resolution
   - Generated from the ViT backbone with position encodings

3. **Combined Embeddings**:
   - Fused text and visual features
   - Includes detection results (masks, boxes, scores)

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
# Extract text embeddings only
./run_embedding_extraction.sh --text "person in red shirt" --output text_embeddings.npz

# Extract visual embeddings only
# (first place your image in ./input/ directory)
cp /path/to/your/photo.jpg ./input/
./run_embedding_extraction.sh --image photo.jpg --output visual_embeddings.npz

# Extract combined embeddings (text + visual + detections)
./run_embedding_extraction.sh --image photo.jpg --text "person" --output combined_embeddings.npz
```

### Option 2: Using Docker Directly

```bash
# Create directories
mkdir -p input output cache/huggingface cache/torch

# Extract text embeddings
docker run --rm --gpus all \
  -v $(pwd)/input:/workspace/input:ro \
  -v $(pwd)/output:/workspace/output:rw \
  -v $(pwd)/cache/huggingface:/workspace/.cache/huggingface:rw \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  sam3-embeddings:latest \
  --text "person" --output /workspace/output/text_emb.npz

# Extract visual embeddings
docker run --rm --gpus all \
  -v $(pwd)/input:/workspace/input:ro \
  -v $(pwd)/output:/workspace/output:rw \
  -v $(pwd)/cache/huggingface:/workspace/.cache/huggingface:rw \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  sam3-embeddings:latest \
  --image /workspace/input/photo.jpg --output /workspace/output/visual_emb.npz

# Extract combined embeddings
docker run --rm --gpus all \
  -v $(pwd)/input:/workspace/input:ro \
  -v $(pwd)/output:/workspace/output:rw \
  -v $(pwd)/cache/huggingface:/workspace/.cache/huggingface:rw \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  sam3-embeddings:latest \
  --image /workspace/input/photo.jpg --text "person" --output /workspace/output/combined_emb.npz
```

### Option 3: Using Docker Compose

```bash
# Extract embeddings using docker-compose
docker-compose -f docker-compose.embeddings.yml run sam3-embeddings \
  --image /workspace/input/photo.jpg --text "person" --output /workspace/output/embeddings.npz
```

### Option 4: Running Without Docker (Native)

If you have the environment set up locally:

```bash
# Install SAM3
pip install -e .

# Run the extraction script
python extract_embeddings.py --text "person" --output text_embeddings.npz
python extract_embeddings.py --image photo.jpg --output visual_embeddings.npz
python extract_embeddings.py --image photo.jpg --text "person" --output combined_embeddings.npz
```

## Output Format

Embeddings are saved in NumPy's `.npz` format (compressed) or PyTorch's `.pt` format.

### Text Embeddings Only

```python
import numpy as np

data = np.load("text_embeddings.npz", allow_pickle=True)
text_tokens = data["text_tokens"]  # Shape: [1, seq_len, 256]
text_pooled = data["text_pooled"]  # Shape: [1, 256]
text_prompt = str(data["text_prompt"])
```

### Visual Embeddings Only

```python
import numpy as np

data = np.load("visual_embeddings.npz", allow_pickle=True)
vision_features = data["vision_features"]  # List of multi-scale features
vision_pos_enc = data["vision_pos_enc"]    # Position encodings
image_size = tuple(data["image_size"])      # (width, height)

# Each scale has shape: [1, H, W, 256]
for i, feat in enumerate(vision_features):
    print(f"Scale {i}: {feat.shape}")
```

### Combined Embeddings

```python
import numpy as np

data = np.load("combined_embeddings.npz", allow_pickle=True)
masks = data["masks"]      # Shape: [num_detections, H, W]
boxes = data["boxes"]      # Shape: [num_detections, 4]
scores = data["scores"]    # Shape: [num_detections]
vision_features = data["vision_features"]  # Multi-scale features
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
  --text "person" \
  --output text_embeddings.npz
```

### Save as PyTorch Format

```bash
./run_embedding_extraction.sh \
  --image photo.jpg \
  --text "person" \
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
- **GPU memory**: Requires ~8GB VRAM for inference
- **Embedding file sizes**:
  - Text only: <1 MB
  - Visual only: 50-200 MB (depends on image size)
  - Combined: 50-200 MB

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
