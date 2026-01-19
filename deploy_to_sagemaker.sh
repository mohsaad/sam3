#!/bin/bash

# SAM 3 SageMaker Deployment Script
# This script deploys the SAM 3 inference container to a SageMaker endpoint

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

# Parse arguments
AWS_REGION=""
IMAGE_URI=""
ENDPOINT_NAME="sam3-inference"
INSTANCE_TYPE="ml.g4dn.xlarge"
INSTANCE_COUNT=1
EXECUTION_ROLE=""
MODEL_NAME=""
ENDPOINT_CONFIG_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        --image)
            IMAGE_URI="$2"
            shift 2
            ;;
        --endpoint-name)
            ENDPOINT_NAME="$2"
            shift 2
            ;;
        --instance-type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        --instance-count)
            INSTANCE_COUNT="$2"
            shift 2
            ;;
        --execution-role)
            EXECUTION_ROLE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --region REGION              AWS region (e.g., us-east-1)"
            echo "  --image IMAGE_URI            ECR image URI for inference"
            echo "  --endpoint-name NAME         SageMaker endpoint name (default: sam3-inference)"
            echo "  --instance-type TYPE         Instance type (default: ml.g4dn.xlarge)"
            echo "  --instance-count COUNT       Number of instances (default: 1)"
            echo "  --execution-role ARN         SageMaker execution role ARN (optional, will create if not provided)"
            echo
            echo "Examples:"
            echo "  # Deploy with default settings"
            echo "  $0 --region us-east-1 --image 123456789012.dkr.ecr.us-east-1.amazonaws.com/sam3-inference:latest"
            echo
            echo "  # Deploy with custom endpoint name and instance type"
            echo "  $0 --region us-east-1 --image <IMAGE_URI> --endpoint-name my-sam3 --instance-type ml.g4dn.2xlarge"
            echo
            echo "Available GPU instance types:"
            echo "  - ml.g4dn.xlarge   (1 GPU, 16 GB GPU memory)  # Recommended"
            echo "  - ml.g4dn.2xlarge  (1 GPU, 32 GB GPU memory)"
            echo "  - ml.g4dn.4xlarge  (1 GPU, 64 GB GPU memory)"
            echo "  - ml.p3.2xlarge    (1 GPU, 16 GB GPU memory)"
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

# Validate required arguments
if [ -z "$AWS_REGION" ]; then
    AWS_REGION=$(aws configure get region)
    if [ -z "$AWS_REGION" ]; then
        print_message "$RED" "Error: AWS region not specified and not configured."
        exit 1
    fi
    print_message "$YELLOW" "Using AWS region from config: $AWS_REGION"
fi

if [ -z "$IMAGE_URI" ]; then
    print_message "$RED" "Error: --image is required"
    echo "Use --help for usage information"
    exit 1
fi

# Generate unique names with timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
MODEL_NAME="${ENDPOINT_NAME}-model-${TIMESTAMP}"
ENDPOINT_CONFIG_NAME="${ENDPOINT_NAME}-config-${TIMESTAMP}"

print_header "SAM 3 SageMaker Deployment"

print_message "$GREEN" "Configuration:"
echo "  AWS Region:        $AWS_REGION"
echo "  Image URI:         $IMAGE_URI"
echo "  Endpoint Name:     $ENDPOINT_NAME"
echo "  Model Name:        $MODEL_NAME"
echo "  Config Name:       $ENDPOINT_CONFIG_NAME"
echo "  Instance Type:     $INSTANCE_TYPE"
echo "  Instance Count:    $INSTANCE_COUNT"
echo

# Get or create execution role
if [ -z "$EXECUTION_ROLE" ]; then
    print_message "$YELLOW" "No execution role provided, checking for default..."

    ROLE_NAME="SageMakerExecutionRole-SAM3"
    EXECUTION_ROLE=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text 2>/dev/null || echo "")

    if [ -z "$EXECUTION_ROLE" ]; then
        print_message "$YELLOW" "Creating SageMaker execution role..."

        # Create trust policy
        cat > /tmp/trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

        # Create role
        EXECUTION_ROLE=$(aws iam create-role \
            --role-name "$ROLE_NAME" \
            --assume-role-policy-document file:///tmp/trust-policy.json \
            --query 'Role.Arn' \
            --output text)

        # Attach policies
        aws iam attach-role-policy \
            --role-name "$ROLE_NAME" \
            --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

        aws iam attach-role-policy \
            --role-name "$ROLE_NAME" \
            --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

        print_message "$GREEN" "✓ Created execution role: $EXECUTION_ROLE"

        # Wait for role to propagate
        print_message "$YELLOW" "Waiting for role to propagate (10 seconds)..."
        sleep 10
    else
        print_message "$GREEN" "✓ Using existing execution role: $EXECUTION_ROLE"
    fi
fi

echo "  Execution Role:    $EXECUTION_ROLE"
echo

# Create SageMaker model
print_header "Creating SageMaker Model"

print_message "$YELLOW" "Creating model: $MODEL_NAME"

aws sagemaker create-model \
    --model-name "$MODEL_NAME" \
    --primary-container "{
        \"Image\": \"$IMAGE_URI\",
        \"Mode\": \"SingleModel\"
    }" \
    --execution-role-arn "$EXECUTION_ROLE" \
    --region "$AWS_REGION"

if [ $? -ne 0 ]; then
    print_message "$RED" "✗ Failed to create model"
    exit 1
fi
print_message "$GREEN" "✓ Model created successfully"

# Create endpoint configuration
print_header "Creating Endpoint Configuration"

print_message "$YELLOW" "Creating endpoint configuration: $ENDPOINT_CONFIG_NAME"

aws sagemaker create-endpoint-config \
    --endpoint-config-name "$ENDPOINT_CONFIG_NAME" \
    --production-variants "[
        {
            \"VariantName\": \"AllTraffic\",
            \"ModelName\": \"$MODEL_NAME\",
            \"InitialInstanceCount\": $INSTANCE_COUNT,
            \"InstanceType\": \"$INSTANCE_TYPE\",
            \"InitialVariantWeight\": 1.0
        }
    ]" \
    --region "$AWS_REGION"

if [ $? -ne 0 ]; then
    print_message "$RED" "✗ Failed to create endpoint configuration"
    exit 1
fi
print_message "$GREEN" "✓ Endpoint configuration created successfully"

# Check if endpoint already exists
print_header "Deploying Endpoint"

ENDPOINT_EXISTS=$(aws sagemaker describe-endpoint \
    --endpoint-name "$ENDPOINT_NAME" \
    --region "$AWS_REGION" 2>/dev/null || echo "")

if [ -n "$ENDPOINT_EXISTS" ]; then
    print_message "$YELLOW" "Endpoint already exists, updating..."

    aws sagemaker update-endpoint \
        --endpoint-name "$ENDPOINT_NAME" \
        --endpoint-config-name "$ENDPOINT_CONFIG_NAME" \
        --region "$AWS_REGION"

    ACTION="update"
else
    print_message "$YELLOW" "Creating new endpoint..."

    aws sagemaker create-endpoint \
        --endpoint-name "$ENDPOINT_NAME" \
        --endpoint-config-name "$ENDPOINT_CONFIG_NAME" \
        --region "$AWS_REGION"

    ACTION="creation"
fi

if [ $? -ne 0 ]; then
    print_message "$RED" "✗ Failed to $ACTION endpoint"
    exit 1
fi

# Wait for endpoint to be in service
print_message "$YELLOW" "Waiting for endpoint to be in service (this may take 5-10 minutes)..."

while true; do
    STATUS=$(aws sagemaker describe-endpoint \
        --endpoint-name "$ENDPOINT_NAME" \
        --region "$AWS_REGION" \
        --query 'EndpointStatus' \
        --output text)

    echo -ne "\r  Current status: $STATUS    "

    if [ "$STATUS" == "InService" ]; then
        echo
        print_message "$GREEN" "✓ Endpoint is in service!"
        break
    elif [ "$STATUS" == "Failed" ]; then
        echo
        print_message "$RED" "✗ Endpoint deployment failed"

        # Get failure reason
        FAILURE_REASON=$(aws sagemaker describe-endpoint \
            --endpoint-name "$ENDPOINT_NAME" \
            --region "$AWS_REGION" \
            --query 'FailureReason' \
            --output text)

        print_message "$RED" "Failure reason: $FAILURE_REASON"
        exit 1
    fi

    sleep 15
done

# Get endpoint URL
ENDPOINT_URL="https://runtime.sagemaker.$AWS_REGION.amazonaws.com/endpoints/$ENDPOINT_NAME/invocations"

# Summary
print_header "Deployment Complete!"

print_message "$GREEN" "SageMaker endpoint deployed successfully!"
echo
echo "  Endpoint Name:     $ENDPOINT_NAME"
echo "  Endpoint URL:      $ENDPOINT_URL"
echo "  Model Name:        $MODEL_NAME"
echo "  Instance Type:     $INSTANCE_TYPE"
echo "  Instance Count:    $INSTANCE_COUNT"
echo

print_message "$BLUE" "Test the endpoint:"
echo
echo "  Python:"
cat <<'EOF'
  import boto3
  import json
  import base64
  import numpy as np

  # Load embeddings
  with open("image_embeddings.npz", "rb") as f:
      embeddings_b64 = base64.b64encode(f.read()).decode('utf-8')

  # Prepare request
  payload = {
      "embeddings": embeddings_b64,
      "text_prompt": "person"
  }

  # Invoke endpoint
  client = boto3.client("sagemaker-runtime", region_name="REGION")
  response = client.invoke_endpoint(
      EndpointName="ENDPOINT_NAME",
      ContentType="application/json",
      Body=json.dumps(payload)
  )

  # Parse response
  result = json.loads(response["Body"].read())
  print(f"Found {result['num_detections']} detections")
EOF
echo
echo "  Replace REGION with: $AWS_REGION"
echo "  Replace ENDPOINT_NAME with: $ENDPOINT_NAME"
echo

print_message "$BLUE" "Manage the endpoint:"
echo "  # View endpoint status"
echo "  aws sagemaker describe-endpoint --endpoint-name $ENDPOINT_NAME --region $AWS_REGION"
echo
echo "  # View CloudWatch logs"
echo "  aws logs tail /aws/sagemaker/Endpoints/$ENDPOINT_NAME --follow --region $AWS_REGION"
echo
echo "  # Delete endpoint (to stop charges)"
echo "  aws sagemaker delete-endpoint --endpoint-name $ENDPOINT_NAME --region $AWS_REGION"
echo

print_message "$YELLOW" "Note: Remember to delete the endpoint when not in use to avoid charges!"
print_message "$GREEN" "Deployment completed successfully!"
