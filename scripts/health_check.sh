#!/bin/bash
# =============================================================================
# SCHWABOT HEALTH CHECK SCRIPT
# =============================================================================
# System health monitoring script for Schwabot Trading System
#
# Usage:
#   ./scripts/health_check.sh [options]
#   ./scripts/health_check.sh --full
#   ./scripts/health_check.sh --monitor --interval 30
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
LOG_FILE="$PROJECT_ROOT/logs/health_check.log"
ALERT_FILE="$PROJECT_ROOT/logs/health_alerts.log"

# Default values
CHECK_TYPE="basic"
MONITOR_MODE=false
INTERVAL=60
VERBOSE=false
SEND_ALERTS=false

# Health check results
HEALTH_STATUS="healthy"
ISSUES_FOUND=0
CRITICAL_ISSUES=0

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

# Function to log messages
log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    if [[ "$level" == "ERROR" ]] || [[ "$level" == "CRITICAL" ]]; then
        echo "[$timestamp] [$level] $message" >> "$ALERT_FILE"
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [options]

Options:
    --basic              Basic health check (default)
    --full               Full comprehensive health check
    --monitor            Continuous monitoring mode
    --interval SECONDS   Monitoring interval in seconds (default: 60)
    --verbose            Enable verbose output
    --alerts             Send alerts for issues
    --help               Show this help message

Examples:
    $0 --basic
    $0 --full --verbose
    $0 --monitor --interval 30 --alerts
EOF
}

# Function to parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --basic)
                CHECK_TYPE="basic"
                shift
                ;;
            --full)
                CHECK_TYPE="full"
                shift
                ;;
            --monitor)
                MONITOR_MODE=true
                shift
                ;;
            --interval)
                INTERVAL="$2"
                shift 2
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --alerts)
                SEND_ALERTS=true
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

# Function to check system resources
check_system_resources() {
    print_status "Checking system resources..."
    
    # Check CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    if (( $(echo "$cpu_usage > 80" | bc -l) )); then
        print_warning "High CPU usage: ${cpu_usage}%"
        log_message "WARNING" "High CPU usage: ${cpu_usage}%"
        ((ISSUES_FOUND++))
    else
        print_success "CPU usage: ${cpu_usage}%"
    fi
    
    # Check memory usage
    local mem_info=$(free -m | awk 'NR==2{printf "%.2f", $3*100/$2}')
    if (( $(echo "$mem_info > 85" | bc -l) )); then
        print_warning "High memory usage: ${mem_info}%"
        log_message "WARNING" "High memory usage: ${mem_info}%"
        ((ISSUES_FOUND++))
    else
        print_success "Memory usage: ${mem_info}%"
    fi
    
    # Check disk usage
    local disk_usage=$(df / | awk 'NR==2{print $5}' | sed 's/%//')
    if [[ $disk_usage -gt 90 ]]; then
        print_error "Critical disk usage: ${disk_usage}%"
        log_message "CRITICAL" "Critical disk usage: ${disk_usage}%"
        ((CRITICAL_ISSUES++))
    elif [[ $disk_usage -gt 80 ]]; then
        print_warning "High disk usage: ${disk_usage}%"
        log_message "WARNING" "High disk usage: ${disk_usage}%"
        ((ISSUES_FOUND++))
    else
        print_success "Disk usage: ${disk_usage}%"
    fi
}

# Function to check Docker services
check_docker_services() {
    print_status "Checking Docker services..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker not installed"
        log_message "ERROR" "Docker not installed"
        ((CRITICAL_ISSUES++))
        return
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose not installed"
        log_message "ERROR" "Docker Compose not installed"
        ((CRITICAL_ISSUES++))
        return
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon not running"
        log_message "CRITICAL" "Docker daemon not running"
        ((CRITICAL_ISSUES++))
        return
    fi
    
    # Check Schwabot containers
    local containers_running=0
    local expected_containers=0
    
    if [[ -f "$PROJECT_ROOT/docker-compose.yml" ]]; then
        cd "$PROJECT_ROOT"
        
        # Count expected containers
        expected_containers=$(docker-compose ps -q | wc -l)
        
        # Count running containers
        containers_running=$(docker-compose ps -q --filter "status=running" | wc -l)
        
        if [[ $containers_running -lt $expected_containers ]]; then
            print_error "Not all containers are running ($containers_running/$expected_containers)"
            log_message "CRITICAL" "Not all containers are running ($containers_running/$expected_containers)"
            ((CRITICAL_ISSUES++))
            
            # Show container status
            docker-compose ps
        else
            print_success "All containers running ($containers_running/$expected_containers)"
        fi
    else
        print_warning "docker-compose.yml not found"
        log_message "WARNING" "docker-compose.yml not found"
        ((ISSUES_FOUND++))
    fi
}

# Function to check application health
check_application_health() {
    print_status "Checking application health..."
    
    # Check if application is responding
    if curl -f -s http://localhost:5000/health > /dev/null 2>&1; then
        print_success "Application health endpoint responding"
    else
        print_error "Application health endpoint not responding"
        log_message "CRITICAL" "Application health endpoint not responding"
        ((CRITICAL_ISSUES++))
    fi
    
    # Check API endpoints
    local endpoints=("http://localhost:5000/api/status" "http://localhost:5000/api/version")
    for endpoint in "${endpoints[@]}"; do
        if curl -f -s "$endpoint" > /dev/null 2>&1; then
            print_success "API endpoint responding: $endpoint"
        else
            print_warning "API endpoint not responding: $endpoint"
            log_message "WARNING" "API endpoint not responding: $endpoint"
            ((ISSUES_FOUND++))
        fi
    done
}

# Function to check database connectivity
check_database_connectivity() {
    print_status "Checking database connectivity..."
    
    # Check PostgreSQL
    if command -v psql &> /dev/null; then
        if psql -h localhost -p 5432 -U schwabot_user -d schwabot_production -c "SELECT 1;" &> /dev/null; then
            print_success "PostgreSQL connection OK"
        else
            print_warning "PostgreSQL connection failed"
            log_message "WARNING" "PostgreSQL connection failed"
            ((ISSUES_FOUND++))
        fi
    fi
    
    # Check Redis
    if command -v redis-cli &> /dev/null; then
        if redis-cli -h localhost -p 6379 ping &> /dev/null; then
            print_success "Redis connection OK"
        else
            print_warning "Redis connection failed"
            log_message "WARNING" "Redis connection failed"
            ((ISSUES_FOUND++))
        fi
    fi
}

# Function to check log files
check_log_files() {
    print_status "Checking log files..."
    
    local log_dir="$PROJECT_ROOT/logs"
    if [[ ! -d "$log_dir" ]]; then
        print_warning "Log directory not found: $log_dir"
        log_message "WARNING" "Log directory not found: $log_dir"
        ((ISSUES_FOUND++))
        return
    fi
    
    # Check for recent errors in log files
    local error_count=0
    for log_file in "$log_dir"/*.log; do
        if [[ -f "$log_file" ]]; then
            local recent_errors=$(grep -c "ERROR\|CRITICAL\|FATAL" "$log_file" 2>/dev/null || echo "0")
            if [[ $recent_errors -gt 0 ]]; then
                print_warning "Found $recent_errors errors in $(basename "$log_file")"
                log_message "WARNING" "Found $recent_errors errors in $(basename "$log_file")"
                ((ISSUES_FOUND++))
                ((error_count++))
            fi
        fi
    done
    
    if [[ $error_count -eq 0 ]]; then
        print_success "No recent errors found in log files"
    fi
}

# Function to check configuration files
check_configuration() {
    print_status "Checking configuration files..."
    
    # Check if .env file exists
    if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
        print_error ".env file not found"
        log_message "CRITICAL" ".env file not found"
        ((CRITICAL_ISSUES++))
    else
        print_success ".env file exists"
    fi
    
    # Check if config directory exists
    if [[ ! -d "$PROJECT_ROOT/config" ]]; then
        print_error "Config directory not found"
        log_message "CRITICAL" "Config directory not found"
        ((CRITICAL_ISSUES++))
    else
        print_success "Config directory exists"
    fi
    
    # Check for required config files
    local required_configs=("schwabot_config.yaml" "pipeline.yaml")
    for config in "${required_configs[@]}"; do
        if [[ -f "$PROJECT_ROOT/config/$config" ]]; then
            print_success "Config file exists: $config"
        else
            print_warning "Config file missing: $config"
            log_message "WARNING" "Config file missing: $config"
            ((ISSUES_FOUND++))
        fi
    done
}

# Function to check network connectivity
check_network_connectivity() {
    print_status "Checking network connectivity..."
    
    # Check internet connectivity
    if ping -c 1 8.8.8.8 &> /dev/null; then
        print_success "Internet connectivity OK"
    else
        print_error "No internet connectivity"
        log_message "CRITICAL" "No internet connectivity"
        ((CRITICAL_ISSUES++))
    fi
    
    # Check DNS resolution
    if nslookup google.com &> /dev/null; then
        print_success "DNS resolution OK"
    else
        print_error "DNS resolution failed"
        log_message "CRITICAL" "DNS resolution failed"
        ((CRITICAL_ISSUES++))
    fi
}

# Function to check trading system specific items
check_trading_system() {
    print_status "Checking trading system components..."
    
    # Check if trading is enabled
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        if grep -q "SCHWABOT_LIVE_TRADING_ENABLED=true" "$PROJECT_ROOT/.env"; then
            print_warning "Live trading is enabled - extra caution required"
            log_message "WARNING" "Live trading is enabled"
        else
            print_success "Live trading is disabled (safe mode)"
        fi
    fi
    
    # Check API keys (without exposing them)
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        local api_keys_configured=0
        if grep -q "BINANCE_API_KEY=" "$PROJECT_ROOT/.env"; then
            ((api_keys_configured++))
        fi
        if grep -q "COINBASE_API_KEY=" "$PROJECT_ROOT/.env"; then
            ((api_keys_configured++))
        fi
        
        if [[ $api_keys_configured -gt 0 ]]; then
            print_success "API keys configured: $api_keys_configured"
        else
            print_warning "No API keys configured"
            log_message "WARNING" "No API keys configured"
            ((ISSUES_FOUND++))
        fi
    fi
}

# Function to send alerts
send_alerts() {
    if [[ "$SEND_ALERTS" == true ]] && [[ $CRITICAL_ISSUES -gt 0 ]]; then
        print_status "Sending alerts..."
        
        # Email alert (if configured)
        if command -v mail &> /dev/null; then
            local alert_subject="Schwabot Health Check Alert"
            local alert_body="Critical issues found: $CRITICAL_ISSUES"
            # mail -s "$alert_subject" admin@example.com <<< "$alert_body"
        fi
        
        # Slack alert (if configured)
        if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
            local slack_message="{\"text\":\"🚨 Schwabot Health Check Alert: $CRITICAL_ISSUES critical issues found\"}"
            curl -X POST -H 'Content-type: application/json' --data "$slack_message" "$SLACK_WEBHOOK_URL" &> /dev/null || true
        fi
        
        print_success "Alerts sent"
    fi
}

# Function to generate health report
generate_health_report() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Determine overall health status
    if [[ $CRITICAL_ISSUES -gt 0 ]]; then
        HEALTH_STATUS="critical"
    elif [[ $ISSUES_FOUND -gt 0 ]]; then
        HEALTH_STATUS="warning"
    else
        HEALTH_STATUS="healthy"
    fi
    
    # Generate report
    cat << EOF
==========================================
SCHWABOT HEALTH CHECK REPORT
==========================================
Timestamp: $timestamp
Overall Status: $HEALTH_STATUS
Issues Found: $ISSUES_FOUND
Critical Issues: $CRITICAL_ISSUES

System Resources:
- CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%
- Memory Usage: $(free -m | awk 'NR==2{printf "%.2f", $3*100/$2}')%
- Disk Usage: $(df / | awk 'NR==2{print $5}')

Services:
- Docker: $(docker info &> /dev/null && echo "Running" || echo "Not Running")
- Application: $(curl -f -s http://localhost:5000/health > /dev/null 2>&1 && echo "Healthy" || echo "Unhealthy")

EOF
    
    # Log the report
    log_message "INFO" "Health check completed - Status: $HEALTH_STATUS, Issues: $ISSUES_FOUND, Critical: $CRITICAL_ISSUES"
}

# Function to run basic health check
run_basic_check() {
    print_status "Running basic health check..."
    
    check_system_resources
    check_docker_services
    check_application_health
    check_configuration
}

# Function to run full health check
run_full_check() {
    print_status "Running full health check..."
    
    run_basic_check
    check_database_connectivity
    check_log_files
    check_network_connectivity
    check_trading_system
}

# Function to run monitoring mode
run_monitoring_mode() {
    print_status "Starting monitoring mode (interval: ${INTERVAL}s)..."
    
    while true; do
        clear
        echo "=========================================="
        echo "SCHWABOT HEALTH MONITORING"
        echo "=========================================="
        echo "Press Ctrl+C to stop monitoring"
        echo ""
        
        # Reset counters
        ISSUES_FOUND=0
        CRITICAL_ISSUES=0
        
        # Run health check
        if [[ "$CHECK_TYPE" == "full" ]]; then
            run_full_check
        else
            run_basic_check
        fi
        
        # Generate report
        generate_health_report
        
        # Send alerts if needed
        send_alerts
        
        # Wait for next check
        sleep "$INTERVAL"
    done
}

# Main health check function
main() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  SCHWABOT HEALTH CHECK${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Create log directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Run health check based on type
    if [[ "$MONITOR_MODE" == true ]]; then
        run_monitoring_mode
    else
        if [[ "$CHECK_TYPE" == "full" ]]; then
            run_full_check
        else
            run_basic_check
        fi
        
        # Generate report
        generate_health_report
        
        # Send alerts if needed
        send_alerts
        
        echo ""
        if [[ "$HEALTH_STATUS" == "healthy" ]]; then
            print_success "Health check completed - System is healthy!"
        elif [[ "$HEALTH_STATUS" == "warning" ]]; then
            print_warning "Health check completed - Issues found: $ISSUES_FOUND"
        else
            print_error "Health check completed - Critical issues found: $CRITICAL_ISSUES"
        fi
    fi
}

# Run main function with all arguments
main "$@" 