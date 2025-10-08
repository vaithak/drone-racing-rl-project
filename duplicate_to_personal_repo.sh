#!/bin/bash

# Script to duplicate isaac_quad_sim2real to a personal GitHub repository
# Usage: ./duplicate_to_personal_repo.sh YOUR_GITHUB_USERNAME

if [ $# -eq 0 ]; then
    echo "Usage: $0 YOUR_GITHUB_USERNAME"
    echo "Example: $0 johnsmith"
    exit 1
fi

GITHUB_USERNAME=$1
ORIGINAL_DIR="/home/vineet/isaac_quad_sim2real"
NEW_DIR="/home/vineet/ese651_project"
REPO_NAME="ese651_project"

echo "========================================="
echo "Duplicating project to personal GitHub"
echo "GitHub Username: $GITHUB_USERNAME"
echo "New Directory: $NEW_DIR"
echo "Repository Name: $REPO_NAME"
echo "========================================="

# Step 1: Create a copy of the project
echo "Step 1: Creating copy of project..."
if [ -d "$NEW_DIR" ]; then
    echo "Directory $NEW_DIR already exists!"
    read -p "Do you want to remove it and continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$NEW_DIR"
    else
        echo "Exiting..."
        exit 1
    fi
fi

cp -r "$ORIGINAL_DIR" "$NEW_DIR"
cd "$NEW_DIR"

# Step 2: Remove old git history
echo "Step 2: Removing old git history..."
rm -rf .git

# Step 3: Clean up files you might not want
echo "Step 3: Cleaning up unnecessary files..."
# Remove large output directories (they're in .gitignore anyway)
rm -rf outputs/* logs/* runs/* wandb/*
# Remove plot files if you don't need them
read -p "Remove plot files and images? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f *.png *.pdf plot_*.py
fi

# Step 4: Initialize new repository
echo "Step 4: Initializing new repository..."
git init

# Step 5: Add all files
echo "Step 5: Adding files..."
git add .

# Step 6: Create initial commit
echo "Step 6: Creating initial commit..."
git commit -m "Initial commit - Quadcopter simulation project for ESE 651"

# Step 7: Create GitHub repository and push
echo "Step 7: Setting up GitHub remote..."
echo ""
echo "Now you need to:"
echo "1. Go to https://github.com/new"
echo "2. Create a new repository called '$REPO_NAME'"
echo "3. Make it private or public as you prefer"
echo "4. DON'T initialize with README, .gitignore, or license"
echo ""
read -p "Press Enter when you've created the repository on GitHub..."

# Add remote and push
git remote add origin "git@github.com:$GITHUB_USERNAME/$REPO_NAME.git"
git branch -M main
git push -u origin main

echo ""
echo "========================================="
echo "✅ Successfully duplicated to your personal GitHub!"
echo "Repository: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo "Local directory: $NEW_DIR"
echo "========================================="
echo ""
echo "You can now:"
echo "  cd $NEW_DIR"
echo "  git status"