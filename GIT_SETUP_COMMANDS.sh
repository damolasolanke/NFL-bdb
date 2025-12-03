#!/bin/bash
# Git Setup Commands for NFL Big Data Bowl 2026 Project
# Run these commands to push the refactored project to GitHub

echo "=========================================="
echo "NFL Big Data Bowl 2026 - Git Setup"
echo "=========================================="
echo ""

# Step 1: Initialize Git repository
echo "Step 1: Initializing Git repository..."
git init
echo "✓ Git repository initialized"
echo ""

# Step 2: Add all files
echo "Step 2: Adding files to Git..."
git add .
echo "✓ Files added"
echo ""

# Step 3: Create initial commit
echo "Step 3: Creating initial commit..."
git commit -m "Initial commit: Refactored NFL BDB project

- Refactored code into src/nfl_bdb/ package structure
- Added CLI entrypoints (train, infer, eval, submit)
- Created comprehensive documentation
- Added requirements.txt and pyproject.toml
- Created tests structure
- Archived old/experimental files
- Updated all imports to new structure"
echo "✓ Initial commit created"
echo ""

# Step 4: Instructions for GitHub
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Create a new repository on GitHub"
echo "2. Note the repository URL (e.g., https://github.com/username/nfl-bdb.git)"
echo "3. Run the following commands:"
echo ""
echo "   git remote add origin <repository-url>"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "=========================================="
echo "Current Status:"
echo "=========================================="
git status
echo ""
echo "To see what will be committed:"
echo "  git status"
echo ""
echo "To see the commit:"
echo "  git log --oneline"
echo ""
