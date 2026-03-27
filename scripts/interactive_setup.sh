#!/bin/bash

# --- Styling ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}====================================================${NC}"
echo -e "${CYAN}      PII Remover Proxy - Interactive Setup        ${NC}"
echo -e "${BLUE}====================================================${NC}"

# Ensure we are in the project root
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: Run this script from the project root.${NC}"
    exit 1
fi

# --- Dependency Check ---
echo -e "\n${BLUE}--- Dependency Check ---${NC}"

# Python 3 Check
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version)
    echo -e "${GREEN}[OK] Python 3 found: ${PY_VERSION}${NC}"
else
    echo -e "${RED}[ERROR] Python 3 not found. Please install Python 3.12+.${NC}"
    exit 1
fi

# uv Check (Optional)
if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version)
    echo -e "${GREEN}[OK] uv found: ${UV_VERSION}${NC}"
else
    echo -e "${YELLOW}[INFO] uv not found. Will use standard pip/venv for local setup.${NC}"
fi

# Docker Check (Optional)
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    echo -e "${GREEN}[OK] Docker found: ${DOCKER_VERSION}${NC}"
else
    echo -e "${YELLOW}[INFO] Docker not found. Containerization features will be disabled.${NC}"
fi

# Load existing .env if it exists
if [ -f ".env" ]; then
    echo -e "${YELLOW}Existing .env found. Using it for defaults.${NC}"
    source .env
else
    echo -e "${YELLOW}No .env found. Creating one from .env.example.${NC}"
    cp .env.example .env
    source .env
fi

# Function to prompt for a value with a default
prompt_val() {
    local var_name=$1
    local prompt_text=$2
    local default_val=$3
    
    read -p "$(echo -e "${CYAN}${prompt_text} [${default_val}]: ${NC}")" input_val
    if [ -z "$input_val" ]; then
        printf -v "$var_name" '%s' "$default_val"
    else
        printf -v "$var_name" '%s' "$input_val"
    fi
}

# --- Hardware Detection & GPU Selection ---
echo -e "\n${BLUE}--- Hardware Configuration ---${NC}"
# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}Detection: NVIDIA GPU found.${NC}"
    DET_USE_GPU="true"
else
    echo -e "${YELLOW}Detection: No NVIDIA GPU found.${NC}"
    DET_USE_GPU="false"
fi

# Prompt for GPU preference
prompt_val "USE_GPU" "Enable GPU acceleration?" "${DET_USE_GPU}"

# Verification of GPU if selected
if [ "$USE_GPU" == "true" ]; then
    if [ "$DET_USE_GPU" == "false" ]; then
        echo -e "${RED}Warning: You enabled GPU but no NVIDIA hardware was detected.${NC}"
        echo -e "${YELLOW}Falling back to CPU is recommended.${NC}"
        read -p "Are you sure you want to keep GPU enabled? (y/N): " keep_gpu
        if [[ ! $keep_gpu =~ ^[Yy]$ ]]; then
            USE_GPU="false"
            echo -e "${GREEN}Switched to CPU mode.${NC}"
        fi
    fi
fi

# --- Service Configuration ---
echo -e "\n${BLUE}--- Service Configuration ---${NC}"
prompt_val "HOST" "Binding Host" "${HOST:-0.0.0.0}"
prompt_val "PORT" "Binding Port" "${PORT:-8000}"
prompt_val "LOG_LEVEL" "Log Level (DEBUG|INFO|WARNING|ERROR)" "${LOG_LEVEL:-INFO}"
prompt_val "DEBUG" "Debug Mode (true|false)" "${DEBUG:-false}"
prompt_val "PROXY_AUTH_TOKEN" "Secure Proxy Token" "${PROXY_AUTH_TOKEN:-$(openssl rand -hex 16 2>/dev/null || echo "default_secure_token")}"

# --- LLM Provider Keys ---
echo -e "\n${BLUE}--- LLM Provider Keys (Optional) ---${NC}"
prompt_val "OPENAI_API_KEY" "OpenAI API Key" "${OPENAI_API_KEY}"
prompt_val "ANTHROPIC_API_KEY" "Anthropic API Key" "${ANTHROPIC_API_KEY}"
prompt_val "GEMINI_API_KEY" "Google Gemini API Key" "${GEMINI_API_KEY}"

# Determine DOCKER_BASE_IMAGE for compose
if [ "$USE_GPU" == "true" ]; then
    DOCKER_BASE_IMAGE="nvidia/cuda:13.0.1-runtime-ubuntu24.04"
else
    DOCKER_BASE_IMAGE="python:3.13-slim"
fi

# --- Save to .env ---
cat <<EOF > .env
# LLM Provider Keys
OPENAI_API_KEY=${OPENAI_API_KEY}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
GEMINI_API_KEY=${GEMINI_API_KEY}

# Proxy Authentication
PROXY_AUTH_TOKEN=${PROXY_AUTH_TOKEN}

# App Configuration
DEBUG=${DEBUG}
LOG_LEVEL=${LOG_LEVEL}
USE_GPU=${USE_GPU}
DOCKER_BASE_IMAGE=${DOCKER_BASE_IMAGE}
HOST=${HOST}
PORT=${PORT}
EOF

echo -e "\n${GREEN}Success: .env file updated.${NC}"

# --- Python Setup ---
echo -e "\n${BLUE}--- Python Environment Setup ---${NC}"
read -p "Do you want to run the automated Python setup (scripts/setup.py)? (y/n): " run_setup
if [[ $run_setup =~ ^[Yy]$ ]]; then
    if [ ! -d ".venv" ]; then
        echo -e "${YELLOW}No .venv found. Creating one...${NC}"
        if command -v uv &> /dev/null; then
            uv venv .venv
        else
            python3 -m venv .venv
        fi
    fi
    
    echo -e "${CYAN}Running setup.py...${NC}"
    .venv/bin/python3 scripts/setup.py --silent
fi

echo -e "\n${GREEN}====================================================${NC}"
echo -e "${GREEN}      Setup Complete! You are ready to go.          ${NC}"
echo -e "${GREEN}====================================================${NC}"
echo -e "To start the proxy locally: ${YELLOW}source .venv/bin/activate && python3 -m src.main${NC}"
echo -e "To use Docker: ${YELLOW}docker compose up -d --build${NC}"
