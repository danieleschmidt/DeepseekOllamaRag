#!/bin/bash

# DeepSeek RAG Deployment Script
# This script handles deployment to various environments

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
APP_NAME="deepseek-rag"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
VERSION="${VERSION:-$(date +%Y%m%d-%H%M%S)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
    local           Deploy locally using Docker Compose
    k8s             Deploy to Kubernetes
    build           Build Docker images
    test            Run deployment tests
    cleanup         Clean up deployment resources
    status          Check deployment status
    logs            Show application logs

Options:
    -e, --env       Environment (dev, staging, prod) [default: dev]
    -v, --version   Version tag [default: current timestamp]
    -r, --registry  Docker registry [default: localhost:5000]
    -n, --namespace Kubernetes namespace [default: deepseek-rag]
    -h, --help      Show this help message

Examples:
    $0 build
    $0 local -e dev
    $0 k8s -e prod -v v1.2.3
    $0 status -e staging
EOF
}

# Parse command line arguments
parse_args() {
    ENVIRONMENT="dev"
    NAMESPACE="deepseek-rag"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -r|--registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            build|local|k8s|test|cleanup|status|logs)
                COMMAND="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    if [[ -z "${COMMAND:-}" ]]; then
        log_error "No command specified"
        usage
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    local missing_tools=()
    
    # Check required tools
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi
    
    if [[ "$COMMAND" == "k8s" ]] && ! command -v kubectl &> /dev/null; then
        missing_tools+=("kubectl")
    fi
    
    if [[ "$COMMAND" == "local" ]] && ! command -v docker-compose &> /dev/null; then
        missing_tools+=("docker-compose")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build main application image
    log_info "Building main application image..."
    docker build -t "${APP_NAME}:${VERSION}" -t "${APP_NAME}:latest" .
    
    # Tag for registry if specified
    if [[ "$DOCKER_REGISTRY" != "localhost:5000" ]]; then
        docker tag "${APP_NAME}:${VERSION}" "${DOCKER_REGISTRY}/${APP_NAME}:${VERSION}"
        docker tag "${APP_NAME}:latest" "${DOCKER_REGISTRY}/${APP_NAME}:latest"
    fi
    
    log_success "Docker images built successfully"
}

# Push images to registry
push_images() {
    if [[ "$DOCKER_REGISTRY" == "localhost:5000" ]]; then
        log_warning "Skipping push to localhost registry"
        return
    fi
    
    log_info "Pushing images to registry..."
    docker push "${DOCKER_REGISTRY}/${APP_NAME}:${VERSION}"
    docker push "${DOCKER_REGISTRY}/${APP_NAME}:latest"
    log_success "Images pushed successfully"
}

# Deploy locally using Docker Compose
deploy_local() {
    log_info "Deploying locally using Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Create environment file
    cat > .env << EOF
COMPOSE_PROJECT_NAME=${APP_NAME}-${ENVIRONMENT}
VERSION=${VERSION}
DOCKER_REGISTRY=${DOCKER_REGISTRY}
ENVIRONMENT=${ENVIRONMENT}
EOF
    
    # Create necessary directories
    mkdir -p logs cache temp uploads
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
        log_success "Application is running at http://localhost:8501"
    else
        log_error "Application health check failed"
        docker-compose logs deepseek-rag
        exit 1
    fi
}

# Deploy to Kubernetes
deploy_k8s() {
    log_info "Deploying to Kubernetes..."
    
    cd "$PROJECT_ROOT"
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Create namespace
    kubectl apply -f k8s/namespace.yaml
    
    # Apply ConfigMap and Secrets
    kubectl apply -f k8s/configmap.yaml
    
    # Apply PersistentVolumes
    kubectl apply -f k8s/persistent-volumes.yaml
    
    # Apply Deployments
    kubectl apply -f k8s/deployments.yaml
    
    # Apply Services
    kubectl apply -f k8s/services.yaml
    
    # Apply Ingress
    kubectl apply -f k8s/ingress.yaml
    
    # Wait for rollout
    log_info "Waiting for deployment rollout..."
    kubectl rollout status deployment/deepseek-rag-app -n "$NAMESPACE" --timeout=300s
    kubectl rollout status deployment/ollama -n "$NAMESPACE" --timeout=600s
    kubectl rollout status deployment/redis -n "$NAMESPACE" --timeout=180s
    
    # Check pod status
    kubectl get pods -n "$NAMESPACE"
    
    # Get service endpoints
    log_info "Getting service information..."
    kubectl get services -n "$NAMESPACE"
    
    log_success "Kubernetes deployment completed"
}

# Run deployment tests
run_tests() {
    log_info "Running deployment tests..."
    
    if [[ "$COMMAND" == "local" ]]; then
        # Test local deployment
        test_endpoint "http://localhost:8501"
    else
        # Test Kubernetes deployment
        local service_ip
        service_ip=$(kubectl get service deepseek-rag-loadbalancer -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
        
        if [[ -n "$service_ip" ]]; then
            test_endpoint "http://$service_ip"
        else
            log_warning "LoadBalancer IP not available, testing via port-forward"
            kubectl port-forward service/deepseek-rag-service 8501:8501 -n "$NAMESPACE" &
            local port_forward_pid=$!
            sleep 10
            test_endpoint "http://localhost:8501"
            kill $port_forward_pid
        fi
    fi
    
    log_success "Deployment tests completed"
}

# Test application endpoint
test_endpoint() {
    local endpoint="$1"
    
    log_info "Testing endpoint: $endpoint"
    
    # Health check
    if curl -f "$endpoint/_stcore/health" &> /dev/null; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        return 1
    fi
    
    # Basic functionality test
    local response_code
    response_code=$(curl -s -o /dev/null -w "%{http_code}" "$endpoint")
    
    if [[ "$response_code" == "200" ]]; then
        log_success "Application is responding correctly"
    else
        log_error "Application returned HTTP $response_code"
        return 1
    fi
}

# Show deployment status
show_status() {
    log_info "Checking deployment status..."
    
    case "$COMMAND" in
        local)
            log_info "Docker Compose status:"
            docker-compose ps
            ;;
        k8s)
            log_info "Kubernetes deployment status:"
            kubectl get all -n "$NAMESPACE"
            ;;
    esac
}

# Show application logs
show_logs() {
    log_info "Showing application logs..."
    
    case "$COMMAND" in
        local)
            docker-compose logs -f deepseek-rag
            ;;
        k8s)
            kubectl logs -f deployment/deepseek-rag-app -n "$NAMESPACE"
            ;;
    esac
}

# Cleanup deployment
cleanup() {
    log_info "Cleaning up deployment..."
    
    case "$COMMAND" in
        local)
            docker-compose down -v
            docker system prune -f
            ;;
        k8s)
            kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
            ;;
    esac
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    parse_args "$@"
    
    log_info "Starting deployment process..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    log_info "Command: $COMMAND"
    
    check_prerequisites
    
    case "$COMMAND" in
        build)
            build_images
            push_images
            ;;
        local)
            build_images
            deploy_local
            run_tests
            ;;
        k8s)
            build_images
            push_images
            deploy_k8s
            run_tests
            ;;
        test)
            run_tests
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        cleanup)
            cleanup
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            usage
            exit 1
            ;;
    esac
    
    log_success "Deployment process completed successfully!"
}

# Run main function with all arguments
main "$@"