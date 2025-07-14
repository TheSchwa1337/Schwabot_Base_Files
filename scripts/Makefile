# =============================================================================
# SCHWABOT MAKEFILE
# =============================================================================
# Build automation and development tasks for Schwabot Trading System
#
# Usage:
#   make install          # Install dependencies
#   make test             # Run all tests
#   make lint             # Run linting
#   make build            # Build Docker image
#   make deploy           # Deploy with Docker Compose
#   make clean            # Clean build artifacts
# =============================================================================

# Variables
PYTHON := python3
PIP := pip3
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PYTEST := pytest
FLAKE8 := flake8
BLACK := black
MYPY := mypy

# Project info
PROJECT_NAME := schwabot
VERSION := $(shell grep -oP 'version="\K[^"]+' setup.py 2>/dev/null || echo "2.1.0")
DOCKER_IMAGE := $(PROJECT_NAME):$(VERSION)

# Directories
SRC_DIR := core
TEST_DIR := test
DOCS_DIR := docs
SCRIPTS_DIR := scripts
CONFIG_DIR := config

# Files
REQUIREMENTS := requirements.txt
PYPROJECT := pyproject.toml
DOCKERFILE := Dockerfile
DOCKER_COMPOSE_FILE := docker-compose.yml

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

.PHONY: help install test lint build deploy clean setup dev prod backup restore

# Default target
help:
	@echo "$(BLUE)Schwabot Trading System - Available Commands:$(NC)"
	@echo ""
	@echo "$(GREEN)Setup & Installation:$(NC)"
	@echo "  setup          - Initial project setup"
	@echo "  install        - Install Python dependencies"
	@echo "  install-dev    - Install development dependencies"
	@echo ""
	@echo "$(GREEN)Testing & Quality:$(NC)"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  lint           - Run linting (flake8, black, mypy)"
	@echo "  format         - Format code with black"
	@echo ""
	@echo "$(GREEN)Build & Deploy:$(NC)"
	@echo "  build          - Build Docker image"
	@echo "  deploy         - Deploy with Docker Compose"
	@echo "  deploy-dev     - Deploy development environment"
	@echo "  deploy-prod    - Deploy production environment"
	@echo ""
	@echo "$(GREEN)Maintenance:$(NC)"
	@echo "  clean          - Clean build artifacts"
	@echo "  backup         - Create system backup"
	@echo "  restore        - Restore from backup"
	@echo "  logs           - View system logs"
	@echo "  status         - Check system status"

# =============================================================================
# SETUP & INSTALLATION
# =============================================================================

setup: ## Initial project setup
	@echo "$(BLUE)Setting up Schwabot project...$(NC)"
	@mkdir -p data logs backups config/templates
	@cp env.example .env
	@echo "$(GREEN)✓ Project setup complete$(NC)"
	@echo "$(YELLOW)Please edit .env with your configuration$(NC)"

install: ## Install Python dependencies
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	$(PIP) install -r $(REQUIREMENTS)
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov flake8 black mypy tox
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

# =============================================================================
# TESTING & QUALITY
# =============================================================================

test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/ -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ All tests completed$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/test_*.py -v -m "not integration"
	@echo "$(GREEN)✓ Unit tests completed$(NC)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/test_*.py -v -m "integration"
	@echo "$(GREEN)✓ Integration tests completed$(NC)"

lint: ## Run linting
	@echo "$(BLUE)Running linting...$(NC)"
	$(FLAKE8) $(SRC_DIR)/ $(TEST_DIR)/ --max-line-length=100 --ignore=E203,W503
	$(BLACK) --check $(SRC_DIR)/ $(TEST_DIR)/
	$(MYPY) $(SRC_DIR)/
	@echo "$(GREEN)✓ Linting completed$(NC)"

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	$(BLACK) $(SRC_DIR)/ $(TEST_DIR)/
	@echo "$(GREEN)✓ Code formatting completed$(NC)"

# =============================================================================
# BUILD & DEPLOY
# =============================================================================

build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	$(DOCKER) build -t $(DOCKER_IMAGE) -f $(DOCKERFILE) .
	@echo "$(GREEN)✓ Docker image built: $(DOCKER_IMAGE)$(NC)"

deploy: ## Deploy with Docker Compose
	@echo "$(BLUE)Deploying with Docker Compose...$(NC)"
	$(DOCKER_COMPOSE) -f $(DOCKER_COMPOSE_FILE) up -d
	@echo "$(GREEN)✓ Deployment completed$(NC)"

deploy-dev: ## Deploy development environment
	@echo "$(BLUE)Deploying development environment...$(NC)"
	$(DOCKER_COMPOSE) -f $(DOCKER_COMPOSE_FILE) -f docker-compose.dev.yml up -d
	@echo "$(GREEN)✓ Development deployment completed$(NC)"

deploy-prod: ## Deploy production environment
	@echo "$(BLUE)Deploying production environment...$(NC)"
	$(DOCKER_COMPOSE) -f $(DOCKER_COMPOSE_FILE) -f docker-compose.prod.yml up -d
	@echo "$(GREEN)✓ Production deployment completed$(NC)"

# =============================================================================
# MAINTENANCE
# =============================================================================

clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	$(DOCKER) system prune -f
	@echo "$(GREEN)✓ Cleanup completed$(NC)"

backup: ## Create system backup
	@echo "$(BLUE)Creating system backup...$(NC)"
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	@cp -r data/ backups/$(shell date +%Y%m%d_%H%M%S)/data/
	@cp -r config/ backups/$(shell date +%Y%m%d_%H%M%S)/config/
	@cp -r registry/ backups/$(shell date +%Y%m%d_%H%M%S)/registry/
	@echo "$(GREEN)✓ Backup created$(NC)"

restore: ## Restore from backup (usage: make restore BACKUP=20231201_120000)
	@echo "$(BLUE)Restoring from backup...$(NC)"
	@if [ -z "$(BACKUP)" ]; then \
		echo "$(RED)Error: Please specify backup directory with BACKUP=YYYYMMDD_HHMMSS$(NC)"; \
		exit 1; \
	fi
	@if [ ! -d "backups/$(BACKUP)" ]; then \
		echo "$(RED)Error: Backup directory backups/$(BACKUP) not found$(NC)"; \
		exit 1; \
	fi
	@cp -r backups/$(BACKUP)/data/ data/
	@cp -r backups/$(BACKUP)/config/ config/
	@cp -r backups/$(BACKUP)/registry/ registry/
	@echo "$(GREEN)✓ Restore completed$(NC)"

logs: ## View system logs
	@echo "$(BLUE)Viewing system logs...$(NC)"
	$(DOCKER_COMPOSE) -f $(DOCKER_COMPOSE_FILE) logs -f

status: ## Check system status
	@echo "$(BLUE)Checking system status...$(NC)"
	$(DOCKER_COMPOSE) -f $(DOCKER_COMPOSE_FILE) ps
	@echo "$(GREEN)✓ Status check completed$(NC)"

# =============================================================================
# UTILITY TARGETS
# =============================================================================

check-env: ## Check environment configuration
	@echo "$(BLUE)Checking environment configuration...$(NC)"
	@if [ ! -f ".env" ]; then \
		echo "$(RED)Error: .env file not found. Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Environment configuration OK$(NC)"

security-check: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	$(PIP) install safety
	safety check
	@echo "$(GREEN)✓ Security checks completed$(NC)"

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	$(PIP) install sphinx sphinx-rtd-theme
	sphinx-build -b html docs/ docs/_build/html
	@echo "$(GREEN)✓ Documentation generated$(NC)"

# =============================================================================
# DEVELOPMENT TARGETS
# =============================================================================

dev: ## Start development environment
	@echo "$(BLUE)Starting development environment...$(NC)"
	$(DOCKER_COMPOSE) -f $(DOCKER_COMPOSE_FILE) -f docker-compose.dev.yml up

prod: ## Start production environment
	@echo "$(BLUE)Starting production environment...$(NC)"
	$(DOCKER_COMPOSE) -f $(DOCKER_COMPOSE_FILE) -f docker-compose.prod.yml up -d

stop: ## Stop all services
	@echo "$(BLUE)Stopping all services...$(NC)"
	$(DOCKER_COMPOSE) -f $(DOCKER_COMPOSE_FILE) down
	@echo "$(GREEN)✓ All services stopped$(NC)"

restart: ## Restart all services
	@echo "$(BLUE)Restarting all services...$(NC)"
	$(DOCKER_COMPOSE) -f $(DOCKER_COMPOSE_FILE) restart
	@echo "$(GREEN)✓ All services restarted$(NC)" 