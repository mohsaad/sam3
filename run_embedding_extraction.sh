#!/bin/bash

# SAM 3 Embedding Extraction Helper Script
# This script simplifies running the embedding extraction container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored message
print_message() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_message "$RED" "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    print_message "$YELLOW" "Warning: NVIDIA Docker runtime may not be properly configured."
    print_message "$YELLOW" "Please ensure nvidia-docker2 is installed and configured."
fi

# Create necessary directories
mkdir -p input output cache/huggingface cache/torch

# Build the Docker image if it doesn't exist
if ! docker image inspect sam3-embeddings:latest &> /dev/null; then
    print_message "$YELLOW" "Building Docker image (this may take a few minutes)..."
    docker build -t sam3-embeddings:latest -f Dockerfile.embeddings .
    print_message "$GREEN" "Docker image built successfully!"
fi

# Check if HuggingFace token is set
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    print_message "$YELLOW" "Warning: HUGGING_FACE_HUB_TOKEN environment variable is not set."
    print_message "$YELLOW" "You need to authenticate with HuggingFace to download model checkpoints."
    print_message "$YELLOW" "Set the token with: export HUGGING_FACE_HUB_TOKEN=your_token_here"
    print_message "$YELLOW" "Or run: huggingface-cli login (and restart this script)"
    echo
fi

# Display usage if no arguments provided
if [ $# -eq 0 ]; then
    print_message "$YELLOW" "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --image PATH      Path to input image (relative to ./input/ directory)"
    echo "  --text TEXT       Text prompt for embedding extraction"
    echo "  --output PATH     Output file path (relative to ./output/ directory)"
    echo
    echo "Examples:"
    echo "  # Extract text embeddings only"
    echo "  $0 --text \"person\" --output text_emb.npz"
    echo
    echo "  # Extract visual embeddings only (place image in ./input/ first)"
    echo "  $0 --image photo.jpg --output visual_emb.npz"
    echo
    echo "  # Extract combined embeddings"
    echo "  $0 --image photo.jpg --text \"person\" --output combined_emb.npz"
    echo
    exit 0
fi

# Parse arguments and convert paths
DOCKER_ARGS=()
for arg in "$@"; do
    case $arg in
        --image)
            DOCKER_ARGS+=("--image")
            shift
            # Convert relative path to container path
            DOCKER_ARGS+=("/workspace/input/$1")
            shift
            ;;
        --output)
            DOCKER_ARGS+=("--output")
            shift
            # Convert relative path to container path
            DOCKER_ARGS+=("/workspace/output/$1")
            shift
            ;;
        *)
            DOCKER_ARGS+=("$arg")
            shift
            ;;
    esac
done

# Run the container
print_message "$GREEN" "Running SAM 3 embedding extraction..."
echo

docker run --rm \
    --gpus all \
    -v "$(pwd)/input:/workspace/input:ro" \
    -v "$(pwd)/output:/workspace/output:rw" \
    -v "$(pwd)/cache/huggingface:/workspace/.cache/huggingface:rw" \
    -v "$(pwd)/cache/torch:/workspace/.cache/torch:rw" \
    -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
    sam3-embeddings:latest \
    "${DOCKER_ARGS[@]}"

echo
print_message "$GREEN" "Extraction completed! Check the ./output/ directory for results."
