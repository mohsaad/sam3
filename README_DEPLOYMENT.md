:# SAM 3 AWS Deployment Guide

This guide covers deploying SAM 3 to AWS using Docker containers, Amazon ECR, and Amazon SageMaker.

## Overview

The deployment consists of two main components:

1. **Extraction Container** (`Dockerfile.embeddings`): For batch extraction of image embeddings
   - Can be run on AWS Batch, ECS, or EC2 instances
   - Extracts and saves image embeddings for later inference

2. **SageMaker Inference Container** (`Dockerfile.sagemaker`): For real-time inference
   - Deployed as a SageMaker endpoint
   - Accepts pre-computed embeddings + text prompts
   - Returns segmentation masks, boxes, and scores

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SAM 3 AWS Deployment                    │
└─────────────────────────────────────────────────────────────┘

Phase 1: Batch Embedding Extraction
┌──────────┐      ┌──────────┐      ┌──────────┐
│ Images   │ ───> │ AWS Batch│ ───> │ S3       │
│ (S3)     │      │ (ECR)    │      │ Embeddings│
└──────────┘      └──────────┘      └──────────┘

Phase 2: Real-time Inference
┌──────────┐      ┌──────────┐      ┌──────────┐
│Embeddings│ ───> │SageMaker │ ───> │ Results  │
│+ Text    │      │ Endpoint │      │ (JSON)   │
└──────────┘      └──────────┘      └──────────┘
```

## Prerequisites

### AWS Account Setup

1. **AWS CLI installed and configured**:
   ```bash
   pip install awscli
   aws configure
   ```

2. **Required AWS permissions**:
   - ECR: `CreateRepository`, `PushImage`
   - SageMaker: `CreateModel`, `CreateEndpoint`, `InvokeEndpoint`
   - IAM: `CreateRole`, `AttachRolePolicy`
   - S3: `GetObject`, `PutObject`

3. **Docker installed**:
   ```bash
   # Verify Docker is running
   docker --version
   ```

4. **NVIDIA Docker (for local testing)**:
   ```bash
   # Optional, for GPU testing locally
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

## Step 1: Build and Push to ECR

### Option 1: Using the Deployment Script (Recommended)

```bash
# Make scripts executable
chmod +x deploy_to_ecr.sh deploy_to_sagemaker.sh

# Build and push both images
./deploy_to_ecr.sh \
  --region us-east-1 \
  --account-id 123456789012

# Or build only the SageMaker image
./deploy_to_ecr.sh \
  --image-type sagemaker \
  --region us-east-1 \
  --account-id 123456789012

# Custom tag
./deploy_to_ecr.sh \
  --region us-east-1 \
  --tag v1.0.0
```

### Option 2: Manual Deployment

```bash
# Set variables
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=123456789012
export ECR_REGISTRY=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $ECR_REGISTRY

# Create repositories
aws ecr create-repository --repository-name sam3-extraction --region $AWS_REGION
aws ecr create-repository --repository-name sam3-inference --region $AWS_REGION

# Build images
docker build -t sam3-extraction:latest -f Dockerfile.embeddings .
docker build -t sam3-inference:latest -f Dockerfile.sagemaker .

# Tag images
docker tag sam3-extraction:latest $ECR_REGISTRY/sam3-extraction:latest
docker tag sam3-inference:latest $ECR_REGISTRY/sam3-inference:latest

# Push images
docker push $ECR_REGISTRY/sam3-extraction:latest
docker push $ECR_REGISTRY/sam3-inference:latest
```

## Step 2: Deploy SageMaker Endpoint

### Option 1: Using the Deployment Script (Recommended)

```bash
# Deploy with default settings (ml.g4dn.xlarge)
./deploy_to_sagemaker.sh \
  --region us-east-1 \
  --image 123456789012.dkr.ecr.us-east-1.amazonaws.com/sam3-inference:latest

# Deploy with custom instance type
./deploy_to_sagemaker.sh \
  --region us-east-1 \
  --image <IMAGE_URI> \
  --endpoint-name my-sam3-endpoint \
  --instance-type ml.g4dn.2xlarge \
  --instance-count 2
```

The script will:
1. Create or reuse a SageMaker execution role
2. Create a SageMaker model
3. Create an endpoint configuration
4. Deploy the endpoint (takes 5-10 minutes)
5. Wait for the endpoint to be ready

### Option 2: Manual Deployment

See the detailed steps in the `deploy_to_sagemaker.sh` script.

## Step 3: Extract Embeddings (Batch Processing)

### Using AWS Batch

1. **Create a Batch job definition**:
   ```bash
   aws batch register-job-definition \
     --job-definition-name sam3-extraction \
     --type container \
     --container-properties '{
       "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sam3-extraction:latest",
       "vcpus": 4,
       "memory": 16384,
       "resourceRequirements": [
         {"type": "GPU", "value": "1"}
       ],
       "command": [
         "--image", "s3://my-bucket/images/photo.jpg",
         "--output", "/tmp/embeddings.npz"
       ],
       "environment": [
         {"name": "HUGGING_FACE_HUB_TOKEN", "value": "your_token"}
       ]
     }'
   ```

2. **Submit jobs**:
   ```bash
   aws batch submit-job \
     --job-name extract-embeddings \
     --job-queue my-gpu-queue \
     --job-definition sam3-extraction
   ```

### Using EC2 or ECS

```bash
# Pull image
docker pull 123456789012.dkr.ecr.us-east-1.amazonaws.com/sam3-extraction:latest

# Run extraction
docker run --gpus all \
  -v ~/.aws:/root/.aws:ro \
  -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
  123456789012.dkr.ecr.us-east-1.amazonaws.com/sam3-extraction:latest \
  --image /data/photo.jpg \
  --output /data/embeddings.npz

# Upload to S3
aws s3 cp embeddings.npz s3://my-bucket/embeddings/
```

## Step 4: Invoke SageMaker Endpoint

### Using the Python Client (Recommended)

```bash
# Install dependencies
pip install boto3 numpy

# Make client executable
chmod +x sagemaker_client.py

# Invoke with local embeddings
python sagemaker_client.py \
  --endpoint sam3-inference \
  --embeddings image_embeddings.npz \
  --text "person" \
  --output predictions.npz

# Invoke with S3 embeddings (more efficient)
python sagemaker_client.py \
  --endpoint sam3-inference \
  --embeddings-s3 s3://my-bucket/embeddings/image_emb.npz \
  --text "person in red shirt" \
  --output predictions.npz
```

### Using Boto3 Directly

```python
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
client = boto3.client("sagemaker-runtime", region_name="us-east-1")
response = client.invoke_endpoint(
    EndpointName="sam3-inference",
    ContentType="application/json",
    Body=json.dumps(payload)
)

# Parse response
result = json.loads(response["Body"].read())
print(f"Found {result['num_detections']} detections")

# Decode masks
masks_b64 = result["masks"]
masks_bytes = base64.b64decode(masks_b64)
import io
masks = np.load(io.BytesIO(masks_bytes))
```

### Using S3 Embeddings (Recommended for Production)

```python
import boto3
import json

payload = {
    "embeddings_s3_path": "s3://my-bucket/embeddings/image_emb.npz",
    "text_prompt": "person"
}

client = boto3.client("sagemaker-runtime", region_name="us-east-1")
response = client.invoke_endpoint(
    EndpointName="sam3-inference",
    ContentType="application/json",
    Body=json.dumps(payload)
)

result = json.loads(response["Body"].read())
```

## Cost Optimization

### Instance Type Selection

| Instance Type   | GPU Memory | vCPUs | Price/Hour* | Use Case |
|----------------|------------|-------|-------------|----------|
| ml.g4dn.xlarge | 16 GB      | 4     | ~$0.736     | Development, low traffic |
| ml.g4dn.2xlarge| 32 GB      | 8     | ~$0.936     | Production, medium traffic |
| ml.g4dn.4xlarge| 64 GB      | 16    | ~$1.505     | High throughput |
| ml.p3.2xlarge  | 16 GB      | 8     | ~$3.825     | Maximum performance |

*Prices are approximate and vary by region

### Cost Saving Tips

1. **Use S3 for embeddings**: Avoid sending large embeddings in API calls
2. **Batch processing**: Extract embeddings offline using AWS Batch
3. **Auto-scaling**: Configure endpoint auto-scaling based on traffic
4. **Spot instances**: Use Spot instances for batch extraction (70% savings)
5. **Delete unused endpoints**: Stop endpoints when not in use

```bash
# Delete endpoint to stop charges
aws sagemaker delete-endpoint \
  --endpoint-name sam3-inference \
  --region us-east-1
```

## Monitoring and Logging

### CloudWatch Metrics

Monitor endpoint performance:
```bash
# View endpoint metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name ModelLatency \
  --dimensions Name=EndpointName,Value=sam3-inference \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 3600 \
  --statistics Average
```

### CloudWatch Logs

View endpoint logs:
```bash
# Tail logs in real-time
aws logs tail /aws/sagemaker/Endpoints/sam3-inference --follow

# Filter errors
aws logs filter-log-events \
  --log-group-name /aws/sagemaker/Endpoints/sam3-inference \
  --filter-pattern "ERROR"
```

## Troubleshooting

### Endpoint Deployment Failed

**Symptom**: Endpoint status is "Failed"

**Solution**:
```bash
# Check failure reason
aws sagemaker describe-endpoint \
  --endpoint-name sam3-inference \
  --query 'FailureReason' \
  --output text

# Check CloudWatch logs
aws logs tail /aws/sagemaker/Endpoints/sam3-inference
```

Common issues:
- Insufficient IAM permissions
- Invalid Docker image
- Out of memory (try larger instance)
- Model loading timeout (increase timeout)

### Inference Timeout

**Symptom**: Request times out after 60 seconds

**Solution**:
- Increase SageMaker timeout (max 15 minutes)
- Use smaller embeddings
- Use S3 for embeddings instead of inline base64

### Out of Memory

**Symptom**: Container crashes or OOM errors

**Solution**:
- Use larger instance type
- Reduce batch size
- Optimize model loading

### Invalid Credentials

**Symptom**: "403 Forbidden" or "Invalid credentials"

**Solution**:
```bash
# Check AWS credentials
aws sts get-caller-identity

# Verify IAM role permissions
aws iam get-role --role-name SageMakerExecutionRole-SAM3
```

## Security Best Practices

1. **Use IAM roles**: Don't hardcode AWS credentials
2. **Encrypt at rest**: Enable S3 bucket encryption
3. **Encrypt in transit**: Use HTTPS for API calls
4. **VPC configuration**: Deploy endpoints in VPC for added security
5. **Resource tagging**: Tag resources for cost tracking
6. **Least privilege**: Grant minimal required permissions

## Example: Complete Production Workflow

```bash
# 1. Deploy infrastructure
./deploy_to_ecr.sh --region us-east-1
./deploy_to_sagemaker.sh --region us-east-1 --image <IMAGE_URI>

# 2. Extract embeddings (batch)
aws batch submit-job \
  --job-name extract-all \
  --job-queue gpu-queue \
  --job-definition sam3-extraction \
  --array-properties size=1000

# 3. Invoke endpoint (real-time)
python sagemaker_client.py \
  --endpoint sam3-inference \
  --embeddings-s3 s3://bucket/embeddings/img.npz \
  --text "person" \
  --output predictions.npz

# 4. Clean up when done
aws sagemaker delete-endpoint --endpoint-name sam3-inference
```

## Additional Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Amazon ECR User Guide](https://docs.aws.amazon.com/ecr/)
- [AWS Batch Documentation](https://docs.aws.amazon.com/batch/)
- [SAM 3 Paper](https://arxiv.org/abs/2511.16719)

## Support

For issues:
- AWS Support: https://console.aws.amazon.com/support/
- SAM 3 GitHub: https://github.com/facebookresearch/sam3/issues
