# Deployment Guide

## Overview

This guide covers production deployment of the DeepSeek RAG application across different environments and platforms.

## Prerequisites

### System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ recommended (16GB+ for production)
- **Storage**: 50GB+ free space
- **Network**: Reliable internet connection for model downloads

### Software Requirements

- **Docker**: 20.10+ with Docker Compose
- **Kubernetes**: 1.20+ (for K8s deployment)
- **Git**: For source code management
- **Make**: For build automation

### Access Requirements

- **Container Registry**: Access to push/pull images
- **Ollama Access**: Ability to download DeepSeek R1 model
- **Kubernetes Cluster**: Admin access for K8s deployment

## Deployment Options

### 1. Local Development

Quick setup for development and testing:

```bash
# Clone repository
git clone https://github.com/danieleschmidt/DeepseekOllamaRag.git
cd DeepseekOllamaRag

# Setup development environment
make quick-setup

# Start development server
make dev
```

**Access**: http://localhost:8501

### 2. Docker Compose (Local Production)

Production-like environment on a single machine:

```bash
# Build and deploy
./scripts/deploy.sh local

# Or step by step
make build
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f deepseek-rag
```

**Services**:
- Application: http://localhost:8501
- Ollama: http://localhost:11434
- Redis: localhost:6379
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### 3. Kubernetes (Production)

Full production deployment with scaling and monitoring:

```bash
# Build and deploy to Kubernetes
./scripts/deploy.sh k8s -e prod -v v1.0.0

# Or step by step
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/persistent-volumes.yaml
kubectl apply -f k8s/deployments.yaml
kubectl apply -f k8s/services.yaml
kubectl apply -f k8s/ingress.yaml
```

## Environment Configuration

### Development Environment

```bash
# .env.dev
DEBUG=true
LLM_MODEL=deepseek-r1:1.5b
OLLAMA_BASE_URL=http://localhost:11434
MAX_FILE_SIZE_MB=25
SESSION_TIMEOUT_MINUTES=60
ENABLE_MONITORING=false
```

### Staging Environment

```bash
# .env.staging
DEBUG=false
LLM_MODEL=deepseek-r1:1.5b
OLLAMA_BASE_URL=http://ollama-service:11434
MAX_FILE_SIZE_MB=50
SESSION_TIMEOUT_MINUTES=30
ENABLE_MONITORING=true
ENABLE_RATE_LIMITING=true
RATE_LIMIT_REQUESTS=50
```

### Production Environment

```bash
# .env.prod
DEBUG=false
LLM_MODEL=deepseek-r1:1.5b
OLLAMA_BASE_URL=http://ollama-service:11434
MAX_FILE_SIZE_MB=50
SESSION_TIMEOUT_MINUTES=30
ENABLE_MONITORING=true
ENABLE_RATE_LIMITING=true
RATE_LIMIT_REQUESTS=30
CACHE_TTL_SECONDS=3600
MEMORY_CACHE_SIZE=100
DISK_CACHE_SIZE_MB=500
```

## Docker Deployment

### Building Images

```bash
# Build all images
make build-docker

# Build specific target
docker build --target production -t deepseek-rag:prod .
docker build --target development -t deepseek-rag:dev .
```

### Docker Compose Configuration

The `docker-compose.yml` includes:

- **deepseek-rag**: Main application
- **ollama**: LLM inference engine
- **redis**: Caching layer
- **nginx**: Reverse proxy (optional)
- **prometheus**: Metrics collection
- **grafana**: Monitoring dashboards

### Volume Management

Persistent volumes for:

```yaml
volumes:
  - ./logs:/app/logs          # Application logs
  - ./uploads:/app/uploads    # Uploaded files
  - ./cache:/app/cache        # Application cache
  - ollama-data:/root/.ollama # Ollama models
  - redis-data:/data          # Redis persistence
```

### Health Checks

All services include health checks:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## Kubernetes Deployment

### Cluster Setup

Minimum cluster requirements:

```yaml
# Resource requirements
CPU: 4 cores total
Memory: 8GB total
Storage: 100GB total
Nodes: 2+ (for HA)
```

### Namespace and Resources

```bash
# Create namespace with resource quotas
kubectl apply -f k8s/namespace.yaml

# Verify resource quotas
kubectl describe quota deepseek-rag-quota -n deepseek-rag
kubectl describe limits deepseek-rag-limits -n deepseek-rag
```

### Storage Configuration

#### For Local/On-Premise Clusters

```bash
# Create host directories
sudo mkdir -p /data/deepseek-rag/{logs,cache,uploads}
sudo mkdir -p /data/{ollama,redis}
sudo chown -R 1000:1000 /data/

# Apply persistent volumes
kubectl apply -f k8s/persistent-volumes.yaml
```

#### For Cloud Clusters

Update `persistent-volumes.yaml` with appropriate storage classes:

```yaml
# AWS EBS
storageClassName: gp2

# Google Cloud Persistent Disk
storageClassName: standard

# Azure Disk
storageClassName: managed-premium
```

### Application Deployment

```bash
# Deploy in order
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/persistent-volumes.yaml
kubectl apply -f k8s/deployments.yaml
kubectl apply -f k8s/services.yaml

# Wait for rollout
kubectl rollout status deployment/deepseek-rag-app -n deepseek-rag
kubectl rollout status deployment/ollama -n deepseek-rag
kubectl rollout status deployment/redis -n deepseek-rag
```

### Ingress and Load Balancing

```bash
# Apply ingress configuration
kubectl apply -f k8s/ingress.yaml

# Get external IP
kubectl get services -n deepseek-rag
kubectl get ingress -n deepseek-rag
```

### Scaling

```bash
# Scale application
kubectl scale deployment deepseek-rag-app --replicas=3 -n deepseek-rag

# Auto-scaling (optional)
kubectl autoscale deployment deepseek-rag-app \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n deepseek-rag
```

## SSL/TLS Configuration

### Using cert-manager (Kubernetes)

```bash
# Install cert-manager
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### Manual SSL (Docker Compose)

```bash
# Generate self-signed certificate
mkdir -p nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/private.key \
  -out nginx/ssl/certificate.crt
```

## Monitoring Setup

### Prometheus Configuration

```bash
# Create monitoring namespace
kubectl create namespace monitoring

# Deploy Prometheus
kubectl apply -f monitoring/prometheus.yml

# Port forward to access
kubectl port-forward svc/prometheus 9090:9090 -n monitoring
```

### Grafana Dashboards

```bash
# Access Grafana
kubectl port-forward svc/grafana 3000:3000 -n monitoring

# Default credentials
Username: admin
Password: admin123
```

Import dashboards:
- Node Exporter Full
- Kubernetes Cluster Monitoring
- DeepSeek RAG Custom Dashboard

### Alerting

Configure Alertmanager for notifications:

```yaml
# alertmanager.yml
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    send_resolved: true
```

## Security Configuration

### Network Policies

```bash
# Apply network policies
kubectl apply -f k8s/ingress.yaml  # Contains NetworkPolicy
```

### RBAC

```bash
# Create service account
kubectl create serviceaccount deepseek-rag -n deepseek-rag

# Create role and binding
kubectl apply -f - <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: deepseek-rag
  name: deepseek-rag-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: deepseek-rag-binding
  namespace: deepseek-rag
subjects:
- kind: ServiceAccount
  name: deepseek-rag
  namespace: deepseek-rag
roleRef:
  kind: Role
  name: deepseek-rag-role
  apiGroup: rbac.authorization.k8s.io
EOF
```

### Secrets Management

```bash
# Create secrets from files
kubectl create secret generic deepseek-rag-secrets \
  --from-literal=SESSION_SECRET=your-secret-key \
  --from-literal=API_KEY=your-api-key \
  -n deepseek-rag

# Or from external secret manager (AWS Secrets Manager, etc.)
kubectl apply -f - <<EOF
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
  namespace: deepseek-rag
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-west-2
EOF
```

## Backup and Recovery

### Database Backup (Redis)

```bash
# Redis backup
kubectl exec -it deployment/redis -n deepseek-rag -- redis-cli BGSAVE

# Copy backup
kubectl cp deepseek-rag/redis-pod:/data/dump.rdb ./backup/redis-backup-$(date +%Y%m%d).rdb
```

### Application Data Backup

```bash
# Backup persistent volumes
kubectl exec -it deployment/deepseek-rag-app -n deepseek-rag -- tar czf /tmp/app-backup.tar.gz /app/logs /app/cache /app/uploads

# Copy backup
kubectl cp deepseek-rag/app-pod:/tmp/app-backup.tar.gz ./backup/app-backup-$(date +%Y%m%d).tar.gz
```

### Disaster Recovery

```bash
# Full cluster backup using Velero
velero backup create deepseek-rag-backup --include-namespaces deepseek-rag

# Restore from backup
velero restore create --from-backup deepseek-rag-backup
```

## Performance Tuning

### Application Tuning

```bash
# Environment variables for performance
MEMORY_CACHE_SIZE=500
DISK_CACHE_SIZE_MB=2000
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
SIMILARITY_SEARCH_K=5
```

### Kubernetes Resource Optimization

```yaml
# High-performance configuration
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi

# JVM optimization for Ollama
env:
- name: JAVA_OPTS
  value: "-Xmx6g -Xms2g -XX:+UseG1GC"
```

### Storage Optimization

```bash
# Use SSD storage classes
storageClassName: fast-ssd

# Enable compression for Redis
redis-server --save 900 1 --save 300 10 --save 60 10000 --rdbcompression yes
```

## Troubleshooting

### Common Issues

#### 1. Ollama Model Download Failure

```bash
# Check Ollama logs
kubectl logs deployment/ollama -n deepseek-rag

# Manual model download
kubectl exec -it deployment/ollama -n deepseek-rag -- ollama pull deepseek-r1:1.5b
```

#### 2. Application Won't Start

```bash
# Check application logs
kubectl logs deployment/deepseek-rag-app -n deepseek-rag

# Check resource constraints
kubectl describe pod -l app=deepseek-rag -n deepseek-rag

# Check configuration
kubectl get configmap deepseek-rag-config -o yaml -n deepseek-rag
```

#### 3. Performance Issues

```bash
# Check resource usage
kubectl top pods -n deepseek-rag
kubectl top nodes

# Check cache hit ratios
curl http://localhost:8501/metrics | grep cache_hit_ratio

# Scale up if needed
kubectl scale deployment deepseek-rag-app --replicas=3 -n deepseek-rag
```

#### 4. Storage Issues

```bash
# Check PVC status
kubectl get pvc -n deepseek-rag

# Check disk usage
kubectl exec -it deployment/deepseek-rag-app -n deepseek-rag -- df -h

# Clean up old files
kubectl exec -it deployment/deepseek-rag-app -n deepseek-rag -- find /app/temp -type f -mtime +1 -delete
```

### Health Check Commands

```bash
# Application health
curl -f http://your-domain/health

# Ollama health
curl -f http://ollama-service:11434/api/tags

# Redis health
redis-cli ping

# Overall system status
kubectl get all -n deepseek-rag
```

### Log Analysis

```bash
# Application logs
kubectl logs -f deployment/deepseek-rag-app -n deepseek-rag

# Filter error logs
kubectl logs deployment/deepseek-rag-app -n deepseek-rag | grep ERROR

# Previous container logs
kubectl logs deployment/deepseek-rag-app -n deepseek-rag --previous
```

## Maintenance

### Regular Maintenance Tasks

1. **Daily**:
   - Check application health
   - Monitor resource usage
   - Review error logs

2. **Weekly**:
   - Update security patches
   - Clean up old cache files
   - Review performance metrics

3. **Monthly**:
   - Update base images
   - Review and update configurations
   - Performance optimization review

### Update Procedures

```bash
# Rolling update
kubectl set image deployment/deepseek-rag-app deepseek-rag=deepseek-rag:v1.1.0 -n deepseek-rag

# Check rollout status
kubectl rollout status deployment/deepseek-rag-app -n deepseek-rag

# Rollback if needed
kubectl rollout undo deployment/deepseek-rag-app -n deepseek-rag
```

## Support and Documentation

- **Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Operations**: See [OPERATIONS.md](OPERATIONS.md)
- **API Reference**: See [API.md](API.md)
- **Security**: See [SECURITY.md](SECURITY.md)

For additional support, check the GitHub issues or create a new issue with deployment details.