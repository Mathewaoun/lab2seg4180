#!/bin/bash
# Quick Start & Testing Guide

# ============================================
# SETUP (run once)
# ============================================
cd /Users/matto/Downloads/model-service
source .venv/bin/activate
pip install -r requirements.txt

# ============================================
# TEST THE PROJECT
# ============================================

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_api.py -v

# Run with coverage
python -m pytest tests/ --cov=model_service

# ============================================
# RUN THE API SERVER
# ============================================

# Start the server (runs on http://localhost:5001)
python app.py

# In another terminal, test the server:

# Health check
curl http://localhost:5001/health | python -m json.tool

# Make prediction (requires valid image)
curl -X POST http://localhost:5001/predict \
  -F "file=@test-image.png" \
  -H "X-API-Key: lab2-secret"

# ============================================
# DOCKER
# ============================================

# Build Docker image
docker build -t house-segmentation-service:latest .

# Run Docker container
docker run -p 5001:5000 \
  -e REQUIRE_API_KEY=true \
  -e MODEL_SERVICE_API_KEY=lab2-secret \
  house-segmentation-service:latest

# ============================================
# GIT REPOSITORY COMMANDS
# ============================================

# View commit history
git log --oneline

# View changes
git status
git diff

# Push to GitHub (configure remote first)
git remote add origin https://github.com/USERNAME/REPO.git
git branch -M main
git push -u origin main

# ============================================
# DEACTIVATE VENV (when done)
# ============================================
deactivate
