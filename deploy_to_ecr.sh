#!/bin/bash

# SAM 3 ECR Deployment Script
# This script builds Docker images and pushes them to Amazon ECR

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_message() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Print section header
print_header() {
    echo
    print_message "$BLUE" "======================================================================"
    print_message "$BLUE" "$1"
    print_message "$BLUE" "======================================================================"
    echo
}

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    print_message "$RED" "Error: AWS CLI is not installed."
    print_message "$YELLOW" "Install with: pip install awscli"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_message "$RED" "Error: Docker is not installed."
    exit 1
fi

# Parse arguments
IMAGE_TYPE="both"  # Default: build both images
AWS_REGION=""
AWS_ACCOUNT_ID=""
REPOSITORY_PREFIX="sam3"
TAG="latest"

while [[ $# -gt 0 ]]; do
    case $1 in
        --image-type)
            IMAGE_TYPE="$2"
            shift 2
            ;;
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        --account-id)
            AWS_ACCOUNT_ID="$2"
            shift 2
            ;;
        --repository-prefix)
            REPOSITORY_PREFIX="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --image-type TYPE          Which image to build: 'extraction', 'sagemaker', or 'both' (default: both)"
            echo "  --region REGION            AWS region (e.g., us-east-1)"
            echo "  --account-id ID            AWS account ID"
            echo "  --repository-prefix PREFIX Repository prefix (default: sam3)"
            echo "  --tag TAG                  Docker image tag (default: latest)"
            echo
            echo "Examples:"
            echo "  # Build and push both images"
            echo "  $0 --region us-east-1 --account-id 123456789012"
            echo
            echo "  # Build and push only SageMaker image"
            echo "  $0 --image-type sagemaker --region us-east-1 --account-id 123456789012"
            echo
            echo "  # Build with custom tag"
            echo "  $0 --region us-east-1 --account-id 123456789012 --tag v1.0.0"
            echo
            exit 0
            ;;
        *)
            print_message "$RED" "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get AWS region if not provided
if [ -z "$AWS_REGION" ]; then
    AWS_REGION=$(aws configure get region)
    if [ -z "$AWS_REGION" ]; then
        print_message "$RED" "Error: AWS region not specified and not configured."
        print_message "$YELLOW" "Use --region or configure with: aws configure"
        exit 1
    fi
    print_message "$YELLOW" "Using AWS region from config: $AWS_REGION"
fi

# Get AWS account ID if not provided
if [ -z "$AWS_ACCOUNT_ID" ]; then
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    if [ -z "$AWS_ACCOUNT_ID" ]; then
        print_message "$RED" "Error: Could not determine AWS account ID."
        print_message "$YELLOW" "Use --account-id or ensure AWS credentials are configured."
        exit 1
    fi
    print_message "$YELLOW" "Using AWS account ID: $AWS_ACCOUNT_ID"
fi

# ECR registry URL
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Repository names
EXTRACTION_REPO="${REPOSITORY_PREFIX}-extraction"
SAGEMAKER_REPO="${REPOSITORY_PREFIX}-inference"

# Image URIs
EXTRACTION_IMAGE="${ECR_REGISTRY}/${EXTRACTION_REPO}:${TAG}"
SAGEMAKER_IMAGE="${ECR_REGISTRY}/${SAGEMAKER_REPO}:${TAG}"

print_header "SAM 3 ECR Deployment"

print_message "$GREEN" "Configuration:"
echo "  AWS Region:         $AWS_REGION"
echo "  AWS Account ID:     $AWS_ACCOUNT_ID"
echo "  Repository Prefix:  $REPOSITORY_PREFIX"
echo "  Image Tag:          $TAG"
echo "  Image Type:         $IMAGE_TYPE"
echo

# Function to create ECR repository if it doesn't exist
create_ecr_repository() {
    local repo_name=$1

    print_message "$YELLOW" "Checking if ECR repository exists: $repo_name"

    if aws ecr describe-repositories --repository-names "$repo_name" --region "$AWS_REGION" &> /dev/null; then
        print_message "$GREEN" "✓ Repository exists: $repo_name"
    else
        print_message "$YELLOW" "Creating ECR repository: $repo_name"
        aws ecr create-repository \
            --repository-name "$repo_name" \
            --region "$AWS_REGION" \
            --image-scanning-configuration scanOnPush=true \
            --encryption-configuration encryptionType=AES256
        print_message "$GREEN" "✓ Repository created: $repo_name"
    fi
}

# Function to build and push image
build_and_push() {
    local dockerfile=$1
    local image_uri=$2
    local repo_name=$3
    local image_name=$4

    print_header "Building $image_name Image"

    # Create repository
    create_ecr_repository "$repo_name"

    # Build image
    print_message "$YELLOW" "Building Docker image..."
    docker build -t "$image_uri" -f "$dockerfile" .

    if [ $? -ne 0 ]; then
        print_message "$RED" "✗ Docker build failed"
        exit 1
    fi
    print_message "$GREEN" "✓ Docker image built successfully"

    # Login to ECR
    print_message "$YELLOW" "Logging in to ECR..."
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin "$ECR_REGISTRY"

    if [ $? -ne 0 ]; then
        print_message "$RED" "✗ ECR login failed"
        exit 1
    fi
    print_message "$GREEN" "✓ Logged in to ECR"

    # Push image
    print_message "$YELLOW" "Pushing image to ECR..."
    docker push "$image_uri"

    if [ $? -ne 0 ]; then
        print_message "$RED" "✗ Docker push failed"
        exit 1
    fi
    print_message "$GREEN" "✓ Image pushed successfully to: $image_uri"
}

# Build and push images based on image type
case $IMAGE_TYPE in
    extraction)
        build_and_push "Dockerfile.embeddings" "$EXTRACTION_IMAGE" "$EXTRACTION_REPO" "Extraction"
        ;;
    sagemaker)
        build_and_push "Dockerfile.sagemaker" "$SAGEMAKER_IMAGE" "$SAGEMAKER_REPO" "SageMaker"
        ;;
    both)
        build_and_push "Dockerfile.embeddings" "$EXTRACTION_IMAGE" "$EXTRACTION_REPO" "Extraction"
        build_and_push "Dockerfile.sagemaker" "$SAGEMAKER_IMAGE" "$SAGEMAKER_REPO" "SageMaker"
        ;;
    *)
        print_message "$RED" "Error: Invalid image type: $IMAGE_TYPE"
        print_message "$YELLOW" "Valid options: extraction, sagemaker, both"
        exit 1
        ;;
esac

# Summary
print_header "Deployment Complete!"

print_message "$GREEN" "Images successfully pushed to ECR:"
echo

if [[ "$IMAGE_TYPE" == "extraction" || "$IMAGE_TYPE" == "both" ]]; then
    echo "  Extraction Image:"
    echo "    $EXTRACTION_IMAGE"
    echo
fi

if [[ "$IMAGE_TYPE" == "sagemaker" || "$IMAGE_TYPE" == "both" ]]; then
    echo "  SageMaker Image:"
    echo "    $SAGEMAKER_IMAGE"
    echo
fi

print_message "$BLUE" "Next steps:"
if [[ "$IMAGE_TYPE" == "extraction" || "$IMAGE_TYPE" == "both" ]]; then
    echo "  1. Use extraction image with ECS/Batch:"
    echo "     docker run --gpus all $EXTRACTION_IMAGE --image photo.jpg --output embeddings.npz"
    echo
fi

if [[ "$IMAGE_TYPE" == "sagemaker" || "$IMAGE_TYPE" == "both" ]]; then
    echo "  2. Deploy SageMaker endpoint:"
    echo "     ./deploy_to_sagemaker.sh --region $AWS_REGION --image $SAGEMAKER_IMAGE"
    echo
fi

print_message "$GREEN" "Deployment completed successfully!"
