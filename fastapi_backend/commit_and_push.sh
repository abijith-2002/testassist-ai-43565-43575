#!/bin/bash
set -e

cd "$(dirname "$0")"

# Initialize git if not already
if [ ! -d .git ]; then
    git init
fi

# Set up remote if not already set
REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")
if [ -z "$REMOTE_URL" ]; then
    # Placeholder: user must set actual remote repo here.
    # Example: git remote add origin https://github.com/your-org/fastapi_backend.git
    echo "No remote 'origin' set. Please add your git remote with:"
    echo "    git remote add origin <YOUR_REMOTE_REPO_URL>"
    exit 1
fi

# Stage all changes
git add .

# Commit with clear message
git commit -m "Push all local changes for FastAPI backend: Includes API logic, Gemini integration, REST endpoints, and Q&A answer file."

# Push to the remote
git push origin main || git push origin master

echo "All local changes have been pushed to remote repository."
