#!/bin/bash
# =============================================================================
# SCHWABOT DEPLOYMENT SCRIPT
# =============================================================================
# Automated deployment script for Schwabot Trading System
#
# Usage:
#   ./scripts/deploy.sh [environment] [options]
#   ./scripts/deploy.sh production --backup
#   ./scripts/deploy.sh development --no-cache
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
ENV_FILE="$PROJECT_ROOT/.env"

# Default values
ENVIRONMENT="development"
BACKUP_BEFORE_DEPLOY=false
NO_CACHE=false
FORCE_DEPLOY=false
SKIP_TESTS=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [environment] [options]

Environments:
    development    Deploy development environment
    staging        Deploy staging environment
    production     Deploy production environment

Options:
    --backup       Create backup before deployment
    --no-cache     Build without Docker cache
    --force        Force deployment without confirmation
    --skip-tests   Skip running tests before deployment
    --help         Show this help message

Examples:
    $0 development
    $0 production --backup --no-cache
    $0 staging --force
EOF
}

# Function to parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            development|staging|production)
                ENVIRONMENT="$1"
                shift
                ;;
            --backup)
                BACKUP_BEFORE_DEPLOY=true
                shift
                ;;
            --no-cache)
                NO_CACHE=true
                shift
                ;;
            --force)
                FORCE_DEPLOY=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if .env file exists
    if [[ ! -f "$ENV_FILE" ]]; then
        print_error ".env file not found. Please run 'make setup' first."
        exit 1
    fi
    
    # Check if we're in the project root
    if [[ ! -f "$DOCKER_COMPOSE_FILE" ]]; then
        print_error "docker-compose.yml not found. Please run this script from the project root."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to create backup
create_backup() {
    if [[ "$BACKUP_BEFORE_DEPLOY" == true ]]; then
        print_status "Creating backup before deployment..."
        
        BACKUP_DIR="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$BACKUP_DIR"
        
        # Backup data directory
        if [[ -d "$PROJECT_ROOT/data" ]]; then
            cp -r "$PROJECT_ROOT/data" "$BACKUP_DIR/"
        fi
        
        # Backup config directory
        if [[ -d "$PROJECT_ROOT/config" ]]; then
            cp -r "$PROJECT_ROOT/config" "$BACKUP_DIR/"
        fi
        
        # Backup registry directory
        if [[ -d "$PROJECT_ROOT/registry" ]]; then
            cp -r "$PROJECT_ROOT/registry" "$BACKUP_DIR/"
        fi
        
        print_success "Backup created: $BACKUP_DIR"
    fi
}

# Function to run tests
run_tests() {
    if [[ "$SKIP_TESTS" == false ]]; then
        print_status "Running tests before deployment..."
        
        cd "$PROJECT_ROOT"
        
        # Run unit tests
        if ! python -m pytest test/ -v --tb=short; then
            print_error "Unit tests failed. Deployment aborted."
            exit 1
        fi
        
        # Run integration tests if they exist
        if [[ -d "$PROJECT_ROOT/test/integration" ]]; then
            if ! python -m pytest test/integration/ -v --tb=short; then
                print_error "Integration tests failed. Deployment aborted."
                exit 1
            fi
        fi
        
        print_success "All tests passed"
    else
        print_warning "Skipping tests as requested"
    fi
}

# Function to build Docker images
build_images() {
    print_status "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build arguments
    BUILD_ARGS=""
    if [[ "$NO_CACHE" == true ]]; then
        BUILD_ARGS="--no-cache"
    fi
    
    # Build the main application image
    if ! docker build $BUILD_ARGS -t schwabot:latest .; then
        print_error "Failed to build Docker image"
        exit 1
    fi
    
    print_success "Docker images built successfully"
}

# Function to deploy services
deploy_services() {
    print_status "Deploying services to $ENVIRONMENT environment..."
    
    cd "$PROJECT_ROOT"
    
    # Set environment-specific variables
    export SCHWABOT_ENVIRONMENT="$ENVIRONMENT"
    
    # Deploy using Docker Compose
    if ! docker-compose -f "$DOCKER_COMPOSE_FILE" up -d; then
        print_error "Failed to deploy services"
        exit 1
    fi
    
    print_success "Services deployed successfully"
}

# Function to wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for the main application
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s http://localhost:5000/health > /dev/null 2>&1; then
            print_success "Application is ready"
            break
        fi
        
        print_status "Waiting for application... (attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    if [[ $attempt -gt $max_attempts ]]; then
        print_error "Application failed to start within expected time"
        docker-compose logs schwabot
        exit 1
    fi
}

# Function to verify deployment
verify_deployment() {
    print_status "Verifying deployment..."
    
    # Check if all containers are running
    if ! docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "Up"; then
        print_error "Some services are not running"
        docker-compose -f "$DOCKER_COMPOSE_FILE" ps
        exit 1
    fi
    
    # Check application health
    if ! curl -f -s http://localhost:5000/health > /dev/null 2>&1; then
        print_error "Application health check failed"
        exit 1
    fi
    
    print_success "Deployment verification completed"
}

# Function to show deployment status
show_status() {
    print_status "Deployment Status:"
    echo ""
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
    echo ""
    print_status "Service URLs:"
    echo "  Application: http://localhost:5000"
    echo "  Web Interface: http://localhost:8080"
    echo "  Grafana: http://localhost:3000"
    echo "  Prometheus: http://localhost:9090"
    echo ""
    print_status "Logs:"
    echo "  View logs: docker-compose logs -f"
    echo "  Stop services: docker-compose down"
}

# Function to confirm deployment
confirm_deployment() {
    if [[ "$FORCE_DEPLOY" == false ]]; then
        echo ""
        print_warning "About to deploy to $ENVIRONMENT environment"
        echo "Environment: $ENVIRONMENT"
        echo "Backup: $BACKUP_BEFORE_DEPLOY"
        echo "No cache: $NO_CACHE"
        echo "Skip tests: $SKIP_TESTS"
        echo ""
        read -p "Do you want to continue? (y/N): " -n 1 -r
        echo ""
        
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Deployment cancelled"
            exit 0
        fi
    fi
}

# Main deployment function
main() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  SCHWABOT DEPLOYMENT SCRIPT${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Check prerequisites
    check_prerequisites
    
    # Confirm deployment
    confirm_deployment
    
    # Create backup if requested
    create_backup
    
    # Run tests
    run_tests
    
    # Build images
    build_images
    
    # Deploy services
    deploy_services
    
    # Wait for services
    wait_for_services
    
    # Verify deployment
    verify_deployment
    
    # Show status
    show_status
    
    echo ""
    print_success "Deployment to $ENVIRONMENT completed successfully!"
}

# Run main function with all arguments
main "$@" 