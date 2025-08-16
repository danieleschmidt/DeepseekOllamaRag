# Operations Runbook

## Overview

This runbook provides step-by-step procedures for operating the DeepSeek RAG application in production environments.

## System Overview

### Architecture Components

- **Application Layer**: Streamlit web application
- **LLM Layer**: Ollama + DeepSeek R1 model
- **Cache Layer**: Redis + Application caching
- **Storage Layer**: File system + Vector database
- **Monitoring Layer**: Prometheus + Grafana + Custom metrics

### Key Services

| Service | Purpose | Port | Health Check |
|---------|---------|------|--------------|
| deepseek-rag-app | Main application | 8501 | `/_stcore/health` |
| ollama | LLM inference | 11434 | `/api/tags` |
| redis | Caching | 6379 | `ping` |
| prometheus | Metrics | 9090 | `/-/healthy` |
| grafana | Dashboards | 3000 | `/api/health` |

## Daily Operations

### Morning Health Check

```bash
#!/bin/bash
# Morning health check script

echo "=== DeepSeek RAG Daily Health Check ==="
echo "Date: $(date)"
echo

# 1. Check application health
echo "1. Application Health:"
curl -s -f http://your-domain/_stcore/health && echo "✅ OK" || echo "❌ FAILED"

# 2. Check Ollama service
echo "2. Ollama Service:"
curl -s -f http://ollama-service:11434/api/tags > /dev/null && echo "✅ OK" || echo "❌ FAILED"

# 3. Check Redis
echo "3. Redis Cache:"
redis-cli ping > /dev/null && echo "✅ OK" || echo "❌ FAILED"

# 4. Check disk space
echo "4. Disk Space:"
df -h | grep -E "(logs|cache|uploads)" | while read line; do
    usage=$(echo $line | awk '{print $5}' | sed 's/%//')
    if [ $usage -gt 80 ]; then
        echo "⚠️  WARNING: $line"
    else
        echo "✅ OK: $line"
    fi
done

# 5. Check memory usage
echo "5. Memory Usage:"
free -h

# 6. Check running processes
echo "6. Service Status:"
if command -v kubectl >/dev/null 2>&1; then
    kubectl get pods -n deepseek-rag
else
    docker-compose ps
fi

echo
echo "=== Health Check Complete ==="
```

### Performance Monitoring

```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s http://your-domain/

# curl-format.txt content:
#     time_namelookup:  %{time_namelookup}\n
#        time_connect:  %{time_connect}\n
#     time_appconnect:  %{time_appconnect}\n
#    time_pretransfer:  %{time_pretransfer}\n
#       time_redirect:  %{time_redirect}\n
#  time_starttransfer:  %{time_starttransfer}\n
#                     ----------\n
#          time_total:  %{time_total}\n

# Check cache performance
curl -s http://your-domain/metrics | grep cache_hit_ratio

# Check system resources
if command -v kubectl >/dev/null 2>&1; then
    kubectl top pods -n deepseek-rag
    kubectl top nodes
else
    docker stats --no-stream
fi
```

## Incident Response

### Severity Levels

| Level | Description | Response Time | Escalation |
|-------|-------------|---------------|------------|
| P1 | Complete outage | 15 minutes | Immediate |
| P2 | Degraded performance | 30 minutes | 1 hour |
| P3 | Non-critical issues | 2 hours | 4 hours |
| P4 | Minor issues | 24 hours | 48 hours |

### Common Incident Types

#### 1. Application Down (P1)

**Symptoms:**
- Health check fails
- Users cannot access the application
- HTTP 503/502 errors

**Diagnosis:**
```bash
# Check application status
kubectl get pods -n deepseek-rag -l app=deepseek-rag
# or
docker-compose ps deepseek-rag

# Check logs
kubectl logs -f deployment/deepseek-rag-app -n deepseek-rag --tail=100
# or
docker-compose logs deepseek-rag

# Check resource usage
kubectl describe pod -l app=deepseek-rag -n deepseek-rag
# or
docker stats deepseek-rag
```

**Resolution:**
```bash
# Restart application
kubectl rollout restart deployment/deepseek-rag-app -n deepseek-rag
# or
docker-compose restart deepseek-rag

# If restart doesn't work, check for issues:
# 1. Check if Ollama is running
# 2. Check if Redis is accessible
# 3. Check if persistent volumes are mounted
# 4. Check resource limits
```

#### 2. Ollama Service Down (P1)

**Symptoms:**
- LLM requests fail
- "Cannot connect to Ollama" errors
- 500 errors on Q&A operations

**Diagnosis:**
```bash
# Check Ollama status
kubectl get pods -n deepseek-rag -l app=ollama
curl -f http://ollama-service:11434/api/tags

# Check Ollama logs
kubectl logs deployment/ollama -n deepseek-rag --tail=100
```

**Resolution:**
```bash
# Restart Ollama
kubectl rollout restart deployment/ollama -n deepseek-rag

# If model is missing, re-download
kubectl exec -it deployment/ollama -n deepseek-rag -- ollama pull deepseek-r1:1.5b

# Check available models
kubectl exec -it deployment/ollama -n deepseek-rag -- ollama list
```

#### 3. High Response Times (P2)

**Symptoms:**
- Slow page loading
- Timeouts on document processing
- User complaints about performance

**Diagnosis:**
```bash
# Check response times
curl -w "%{time_total}" -o /dev/null -s http://your-domain/

# Check resource usage
kubectl top pods -n deepseek-rag

# Check cache hit ratio
curl -s http://your-domain/metrics | grep cache_hit_ratio

# Check queue sizes
curl -s http://your-domain/metrics | grep queue_size
```

**Resolution:**
```bash
# Scale up application
kubectl scale deployment deepseek-rag-app --replicas=3 -n deepseek-rag

# Clear cache if hit ratio is low
redis-cli flushdb

# Restart services if memory usage is high
kubectl rollout restart deployment/deepseek-rag-app -n deepseek-rag
```

#### 4. Storage Issues (P2)

**Symptoms:**
- Disk space warnings
- Cannot upload files
- Cache write failures

**Diagnosis:**
```bash
# Check disk usage
kubectl exec -it deployment/deepseek-rag-app -n deepseek-rag -- df -h

# Check PVC status
kubectl get pvc -n deepseek-rag

# Check storage events
kubectl get events -n deepseek-rag --field-selector type=Warning
```

**Resolution:**
```bash
# Clean up old files
kubectl exec -it deployment/deepseek-rag-app -n deepseek-rag -- find /app/temp -type f -mtime +1 -delete
kubectl exec -it deployment/deepseek-rag-app -n deepseek-rag -- find /app/logs -name "*.log" -mtime +7 -delete

# Clean up cache if needed
redis-cli eval "for i=1,redis.call('dbsize') do redis.call('del', redis.call('randomkey')) end" 0

# Expand storage if necessary (cloud environments)
kubectl patch pvc deepseek-rag-uploads-pvc -n deepseek-rag -p '{"spec":{"resources":{"requests":{"storage":"50Gi"}}}}'
```

## Monitoring and Alerting

### Key Metrics to Monitor

#### Application Metrics
- **Response Time**: 95th percentile < 5 seconds
- **Error Rate**: < 1%
- **Uptime**: > 99.9%
- **Cache Hit Ratio**: > 80%

#### System Metrics
- **CPU Usage**: < 80%
- **Memory Usage**: < 85%
- **Disk Usage**: < 80%
- **Network Latency**: < 100ms

#### Business Metrics
- **Documents Processed**: Track daily volume
- **Questions Asked**: Track user engagement
- **Processing Time**: Average time per document

### Setting Up Alerts

```yaml
# Prometheus alert rules (monitoring/alert_rules.yml)
groups:
- name: critical-alerts
  rules:
  - alert: ApplicationDown
    expr: up{job="deepseek-rag"} == 0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "DeepSeek RAG application is down"
      
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
```

### Grafana Dashboards

Key dashboards to monitor:

1. **System Overview**
   - Service status
   - Resource usage
   - Request rates

2. **Application Performance**
   - Response times
   - Cache performance
   - Queue sizes

3. **Infrastructure Health**
   - CPU/Memory usage
   - Disk space
   - Network metrics

## Maintenance Procedures

### Weekly Maintenance

```bash
#!/bin/bash
# Weekly maintenance script

echo "=== Weekly Maintenance - $(date) ==="

# 1. Update container images (security patches)
kubectl set image deployment/deepseek-rag-app deepseek-rag=deepseek-rag:latest -n deepseek-rag

# 2. Clean up old logs
kubectl exec -it deployment/deepseek-rag-app -n deepseek-rag -- find /app/logs -name "*.log" -mtime +7 -delete

# 3. Optimize Redis
redis-cli bgrewriteaof

# 4. Check and clean up temp files
kubectl exec -it deployment/deepseek-rag-app -n deepseek-rag -- find /app/temp -type f -mtime +1 -delete

# 5. Backup configuration
kubectl get configmap deepseek-rag-config -o yaml -n deepseek-rag > backup/config-$(date +%Y%m%d).yaml

# 6. Check certificate expiry (if using TLS)
echo | openssl s_client -servername your-domain -connect your-domain:443 2>/dev/null | openssl x509 -noout -dates

echo "=== Weekly Maintenance Complete ==="
```

### Monthly Maintenance

```bash
#!/bin/bash
# Monthly maintenance script

echo "=== Monthly Maintenance - $(date) ==="

# 1. Full backup
velero backup create monthly-backup-$(date +%Y%m) --include-namespaces deepseek-rag

# 2. Security updates
# Update base images and rebuild
docker pull python:3.11-slim
docker pull ollama/ollama:latest
docker pull redis:7-alpine

# 3. Performance review
# Generate performance report
kubectl exec -it deployment/deepseek-rag-app -n deepseek-rag -- python -c "
from monitoring import performance_monitor
summary = performance_monitor.get_metrics_summary(hours=720)  # 30 days
print('Monthly Performance Summary:', summary)
"

# 4. Clean up old backups
find backup/ -name "*.yaml" -mtime +30 -delete

# 5. Update dependencies
# Check for security vulnerabilities
pip-audit --desc

echo "=== Monthly Maintenance Complete ==="
```

## Disaster Recovery

### Backup Procedures

#### Configuration Backup
```bash
# Backup all configurations
kubectl get all,configmap,secret,pvc -o yaml -n deepseek-rag > backup/full-config-$(date +%Y%m%d).yaml

# Backup specific components
kubectl get configmap deepseek-rag-config -o yaml -n deepseek-rag > backup/configmap-$(date +%Y%m%d).yaml
kubectl get secret deepseek-rag-secrets -o yaml -n deepseek-rag > backup/secrets-$(date +%Y%m%d).yaml
```

#### Data Backup
```bash
# Backup Redis data
kubectl exec deployment/redis -n deepseek-rag -- redis-cli bgsave
kubectl cp deepseek-rag/redis-pod:/data/dump.rdb backup/redis-$(date +%Y%m%d).rdb

# Backup application data
kubectl exec deployment/deepseek-rag-app -n deepseek-rag -- tar czf /tmp/app-data.tar.gz /app/uploads /app/cache
kubectl cp deepseek-rag/app-pod:/tmp/app-data.tar.gz backup/app-data-$(date +%Y%m%d).tar.gz
```

### Recovery Procedures

#### Complete Cluster Recovery
```bash
# 1. Restore namespace and RBAC
kubectl apply -f backup/full-config-$(date +%Y%m%d).yaml

# 2. Restore data
kubectl cp backup/redis-$(date +%Y%m%d).rdb deepseek-rag/redis-pod:/data/dump.rdb
kubectl cp backup/app-data-$(date +%Y%m%d).tar.gz deepseek-rag/app-pod:/tmp/app-data.tar.gz

# 3. Extract data
kubectl exec deployment/deepseek-rag-app -n deepseek-rag -- tar xzf /tmp/app-data.tar.gz -C /

# 4. Restart services
kubectl rollout restart deployment/redis -n deepseek-rag
kubectl rollout restart deployment/deepseek-rag-app -n deepseek-rag

# 5. Verify recovery
kubectl get pods -n deepseek-rag
curl -f http://your-domain/_stcore/health
```

#### Partial Recovery (Single Service)
```bash
# Restore single deployment
kubectl apply -f backup/deployment-deepseek-rag-app.yaml

# Restore configuration
kubectl apply -f backup/configmap-$(date +%Y%m%d).yaml

# Restart service
kubectl rollout restart deployment/deepseek-rag-app -n deepseek-rag
```

## Performance Tuning

### Application Tuning

```bash
# Optimize cache settings
export MEMORY_CACHE_SIZE=500
export DISK_CACHE_SIZE_MB=2000
export CACHE_TTL_SECONDS=7200

# Optimize document processing
export CHUNK_SIZE=1500
export CHUNK_OVERLAP=300
export SIMILARITY_SEARCH_K=5

# Optimize async processing
export ASYNC_WORKERS=4
export TASK_QUEUE_SIZE=200
```

### Database Tuning

```bash
# Redis optimization
redis-cli config set maxmemory-policy allkeys-lru
redis-cli config set maxmemory 1gb
redis-cli config set save "900 1 300 10 60 10000"

# Monitor Redis performance
redis-cli info stats
redis-cli info memory
```

### System Tuning

```bash
# Kubernetes resource optimization
kubectl patch deployment deepseek-rag-app -n deepseek-rag -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "deepseek-rag",
          "resources": {
            "requests": {"cpu": "500m", "memory": "1Gi"},
            "limits": {"cpu": "2000m", "memory": "4Gi"}
          }
        }]
      }
    }
  }
}'

# Enable horizontal pod autoscaling
kubectl autoscale deployment deepseek-rag-app --cpu-percent=70 --min=2 --max=10 -n deepseek-rag
```

## Security Operations

### Security Monitoring

```bash
# Check for security events
kubectl logs deployment/deepseek-rag-app -n deepseek-rag | grep -i security

# Monitor failed authentication attempts
grep "authentication.*failed" /app/logs/security.log

# Check rate limiting
redis-cli get rate_limit:violations

# Monitor file upload patterns
kubectl exec deployment/deepseek-rag-app -n deepseek-rag -- find /app/uploads -type f -mmin -60 | wc -l
```

### Security Updates

```bash
# Update base images
docker pull python:3.11-slim
docker build -t deepseek-rag:security-update .

# Scan for vulnerabilities
trivy image deepseek-rag:latest

# Update dependencies
pip-audit --fix
bandit -r . -f json -o security-report.json
```

## Contact Information

### Escalation Matrix

| Role | Contact | Responsibility |
|------|---------|----------------|
| On-Call Engineer | oncall@company.com | First response |
| Platform Team | platform@company.com | Infrastructure issues |
| Security Team | security@company.com | Security incidents |
| Product Team | product@company.com | Business impact |

### Emergency Procedures

1. **Critical Issues (P1)**:
   - Page on-call engineer immediately
   - Create incident in tracking system
   - Start incident bridge if needed

2. **Escalation Path**:
   - 15 min: Platform team
   - 30 min: Engineering manager
   - 45 min: Director of Engineering

3. **Communication**:
   - Status page updates every 15 minutes
   - Slack updates in #incidents channel
   - Customer communication if customer-facing

## Appendix

### Useful Commands

```bash
# Quick status check
kubectl get all -n deepseek-rag

# Resource usage
kubectl top pods -n deepseek-rag

# Recent events
kubectl get events -n deepseek-rag --sort-by='.lastTimestamp'

# Port forwarding for debugging
kubectl port-forward svc/deepseek-rag-service 8501:8501 -n deepseek-rag

# Execute commands in pods
kubectl exec -it deployment/deepseek-rag-app -n deepseek-rag -- bash

# View logs
kubectl logs -f deployment/deepseek-rag-app -n deepseek-rag
```

### Log Locations

| Component | Location | Purpose |
|-----------|----------|---------|
| Application | `/app/logs/app.log` | Main application logs |
| Security | `/app/logs/security.log` | Security events |
| Performance | `/app/logs/performance.log` | Performance metrics |
| Access | `/app/logs/access.log` | HTTP access logs |

### Configuration Files

| File | Purpose | Location |
|------|---------|----------|
| `config.py` | Application config | `/app/config.py` |
| `prometheus.yml` | Metrics config | `/etc/prometheus/prometheus.yml` |
| `redis.conf` | Cache config | `/etc/redis/redis.conf` |
| `nginx.conf` | Proxy config | `/etc/nginx/nginx.conf` |