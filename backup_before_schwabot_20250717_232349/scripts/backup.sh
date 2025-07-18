#!/bin/bash
# =============================================================================
# SCHWABOT BACKUP SCRIPT
# =============================================================================
# Automated backup script for Schwabot Trading System
#
# Usage:
#   ./scripts/backup.sh [options]
#   ./scripts/backup.sh --full
#   ./scripts/backup.sh --incremental --compress
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
BACKUP_DIR="$PROJECT_ROOT/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="schwabot_backup_$TIMESTAMP"

# Default values
BACKUP_TYPE="full"
COMPRESS_BACKUP=false
UPLOAD_TO_CLOUD=false
RETENTION_DAYS=30
VERBOSE=false

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
Usage: $0 [options]

Options:
    --full              Create full backup (default)
    --incremental       Create incremental backup
    --compress          Compress backup files
    --upload            Upload backup to cloud storage
    --retention DAYS    Set retention period in days (default: 30)
    --verbose           Enable verbose output
    --help              Show this help message

Examples:
    $0 --full --compress
    $0 --incremental --retention 7
    $0 --full --upload --compress
EOF
}

# Function to parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --full)
                BACKUP_TYPE="full"
                shift
                ;;
            --incremental)
                BACKUP_TYPE="incremental"
                shift
                ;;
            --compress)
                COMPRESS_BACKUP=true
                shift
                ;;
            --upload)
                UPLOAD_TO_CLOUD=true
                shift
                ;;
            --retention)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            --verbose)
                VERBOSE=true
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
    
    # Check if we're in the project root
    if [[ ! -d "$PROJECT_ROOT" ]]; then
        print_error "Project root not found"
        exit 1
    fi
    
    # Create backup directory if it doesn't exist
    mkdir -p "$BACKUP_DIR"
    
    # Check available disk space
    local available_space=$(df "$BACKUP_DIR" | awk 'NR==2 {print $4}')
    local required_space=1048576  # 1GB in KB
    
    if [[ $available_space -lt $required_space ]]; then
        print_warning "Low disk space available: ${available_space}KB"
        print_warning "Recommended: at least 1GB free space"
    fi
    
    print_success "Prerequisites check passed"
}

# Function to create full backup
create_full_backup() {
    print_status "Creating full backup..."
    
    local backup_path="$BACKUP_DIR/$BACKUP_NAME"
    mkdir -p "$backup_path"
    
    # Backup data directory
    if [[ -d "$PROJECT_ROOT/data" ]]; then
        print_status "Backing up data directory..."
        cp -r "$PROJECT_ROOT/data" "$backup_path/"
    fi
    
    # Backup config directory
    if [[ -d "$PROJECT_ROOT/config" ]]; then
        print_status "Backing up config directory..."
        cp -r "$PROJECT_ROOT/config" "$backup_path/"
    fi
    
    # Backup registry directory
    if [[ -d "$PROJECT_ROOT/registry" ]]; then
        print_status "Backing up registry directory..."
        cp -r "$PROJECT_ROOT/registry" "$backup_path/"
    fi
    
    # Backup logs directory
    if [[ -d "$PROJECT_ROOT/logs" ]]; then
        print_status "Backing up logs directory..."
        cp -r "$PROJECT_ROOT/logs" "$backup_path/"
    fi
    
    # Backup environment file
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        print_status "Backing up environment file..."
        cp "$PROJECT_ROOT/.env" "$backup_path/"
    fi
    
    # Create backup manifest
    create_backup_manifest "$backup_path"
    
    print_success "Full backup created: $backup_path"
}

# Function to create incremental backup
create_incremental_backup() {
    print_status "Creating incremental backup..."
    
    # Find the most recent backup
    local latest_backup=$(find "$BACKUP_DIR" -maxdepth 1 -type d -name "schwabot_backup_*" | sort | tail -n 1)
    
    if [[ -z "$latest_backup" ]]; then
        print_warning "No previous backup found, creating full backup instead"
        create_full_backup
        return
    fi
    
    local backup_path="$BACKUP_DIR/$BACKUP_NAME"
    mkdir -p "$backup_path"
    
    # Create incremental backup using rsync
    print_status "Creating incremental backup from: $latest_backup"
    
    # Backup data directory incrementally
    if [[ -d "$PROJECT_ROOT/data" ]]; then
        rsync -av --link-dest="$latest_backup/data" "$PROJECT_ROOT/data/" "$backup_path/data/"
    fi
    
    # Backup config directory incrementally
    if [[ -d "$PROJECT_ROOT/config" ]]; then
        rsync -av --link-dest="$latest_backup/config" "$PROJECT_ROOT/config/" "$backup_path/config/"
    fi
    
    # Backup registry directory incrementally
    if [[ -d "$PROJECT_ROOT/registry" ]]; then
        rsync -av --link-dest="$latest_backup/registry" "$PROJECT_ROOT/registry/" "$backup_path/registry/"
    fi
    
    # Create backup manifest
    create_backup_manifest "$backup_path"
    
    print_success "Incremental backup created: $backup_path"
}

# Function to create backup manifest
create_backup_manifest() {
    local backup_path="$1"
    local manifest_file="$backup_path/backup_manifest.txt"
    
    cat > "$manifest_file" << EOF
Schwabot Backup Manifest
========================
Backup Type: $BACKUP_TYPE
Created: $(date)
Timestamp: $TIMESTAMP
Backup Name: $BACKUP_NAME

Directories Backed Up:
$(find "$backup_path" -type d | sort)

Files Backed Up:
$(find "$backup_path" -type f | sort)

Backup Size: $(du -sh "$backup_path" | cut -f1)
EOF
    
    print_status "Backup manifest created: $manifest_file"
}

# Function to compress backup
compress_backup() {
    if [[ "$COMPRESS_BACKUP" == true ]]; then
        print_status "Compressing backup..."
        
        local backup_path="$BACKUP_DIR/$BACKUP_NAME"
        local compressed_file="$BACKUP_DIR/${BACKUP_NAME}.tar.gz"
        
        if tar -czf "$compressed_file" -C "$BACKUP_DIR" "$BACKUP_NAME"; then
            # Remove uncompressed backup
            rm -rf "$backup_path"
            print_success "Backup compressed: $compressed_file"
            BACKUP_NAME="${BACKUP_NAME}.tar.gz"
        else
            print_error "Failed to compress backup"
            exit 1
        fi
    fi
}

# Function to upload to cloud storage
upload_to_cloud() {
    if [[ "$UPLOAD_TO_CLOUD" == true ]]; then
        print_status "Uploading backup to cloud storage..."
        
        # Check if AWS CLI is available
        if command -v aws &> /dev/null; then
            local backup_file="$BACKUP_DIR/$BACKUP_NAME"
            local s3_bucket="${SCHWABOT_S3_BUCKET:-schwabot-backups}"
            
            if aws s3 cp "$backup_file" "s3://$s3_bucket/"; then
                print_success "Backup uploaded to S3: s3://$s3_bucket/$BACKUP_NAME"
            else
                print_error "Failed to upload backup to S3"
            fi
        else
            print_warning "AWS CLI not found, skipping cloud upload"
        fi
    fi
}

# Function to clean old backups
clean_old_backups() {
    print_status "Cleaning backups older than $RETENTION_DAYS days..."
    
    local deleted_count=0
    
    # Find and delete old backups
    while IFS= read -r -d '' backup; do
        local backup_age=$(( ( $(date +%s) - $(stat -c %Y "$backup") ) / 86400 ))
        
        if [[ $backup_age -gt $RETENTION_DAYS ]]; then
            if [[ "$VERBOSE" == true ]]; then
                print_status "Deleting old backup: $backup (age: ${backup_age} days)"
            fi
            
            rm -rf "$backup"
            ((deleted_count++))
        fi
    done < <(find "$BACKUP_DIR" -maxdepth 1 -type d -name "schwabot_backup_*" -print0)
    
    # Clean compressed backups
    while IFS= read -r -d '' backup; do
        local backup_age=$(( ( $(date +%s) - $(stat -c %Y "$backup") ) / 86400 ))
        
        if [[ $backup_age -gt $RETENTION_DAYS ]]; then
            if [[ "$VERBOSE" == true ]]; then
                print_status "Deleting old compressed backup: $backup (age: ${backup_age} days)"
            fi
            
            rm -f "$backup"
            ((deleted_count++))
        fi
    done < <(find "$BACKUP_DIR" -maxdepth 1 -type f -name "schwabot_backup_*.tar.gz" -print0)
    
    print_success "Cleaned $deleted_count old backups"
}

# Function to verify backup
verify_backup() {
    print_status "Verifying backup integrity..."
    
    local backup_path="$BACKUP_DIR/$BACKUP_NAME"
    
    if [[ "$COMPRESS_BACKUP" == true ]]; then
        backup_path="$BACKUP_DIR/${BACKUP_NAME}.tar.gz"
        
        # Test compressed backup
        if ! tar -tzf "$backup_path" > /dev/null 2>&1; then
            print_error "Compressed backup verification failed"
            exit 1
        fi
    else
        # Check if backup directory exists and has content
        if [[ ! -d "$backup_path" ]] || [[ -z "$(ls -A "$backup_path")" ]]; then
            print_error "Backup verification failed"
            exit 1
        fi
    fi
    
    print_success "Backup verification passed"
}

# Function to show backup status
show_backup_status() {
    print_status "Backup Status:"
    echo ""
    echo "Backup Type: $BACKUP_TYPE"
    echo "Backup Name: $BACKUP_NAME"
    echo "Backup Location: $BACKUP_DIR"
    echo "Compressed: $COMPRESS_BACKUP"
    echo "Uploaded to Cloud: $UPLOAD_TO_CLOUD"
    echo ""
    
    # Show backup size
    if [[ "$COMPRESS_BACKUP" == true ]]; then
        local backup_file="$BACKUP_DIR/${BACKUP_NAME}.tar.gz"
        if [[ -f "$backup_file" ]]; then
            echo "Backup Size: $(du -sh "$backup_file" | cut -f1)"
        fi
    else
        local backup_path="$BACKUP_DIR/$BACKUP_NAME"
        if [[ -d "$backup_path" ]]; then
            echo "Backup Size: $(du -sh "$backup_path" | cut -f1)"
        fi
    fi
    
    echo ""
    print_status "Available Backups:"
    ls -la "$BACKUP_DIR" | grep "schwabot_backup"
}

# Main backup function
main() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  SCHWABOT BACKUP SCRIPT${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Check prerequisites
    check_prerequisites
    
    # Create backup based on type
    if [[ "$BACKUP_TYPE" == "full" ]]; then
        create_full_backup
    else
        create_incremental_backup
    fi
    
    # Compress backup if requested
    compress_backup
    
    # Upload to cloud if requested
    upload_to_cloud
    
    # Verify backup
    verify_backup
    
    # Clean old backups
    clean_old_backups
    
    # Show backup status
    show_backup_status
    
    echo ""
    print_success "Backup completed successfully!"
}

# Run main function with all arguments
main "$@" 