#!/bin/bash

# Function to calculate next Saturday's date
get_next_saturday() {
    # Get today's day of week (0-6, Sunday = 0)
    current_day=$(date +%w)
    
    # Calculate days until next Saturday (6 = Saturday)
    days_until_saturday=$(( (6 - current_day + 7) % 7 ))
    
    # If today is Saturday, add 7 to get next Saturday
    if [ $days_until_saturday -eq 0 ]; then
        days_until_saturday=7
    fi
    
    # Get next Saturday's date in ISO 8601 format
    next_saturday=$(date -v+"$days_until_saturday"d +%Y-%m-%dT23:59:59Z 2>/dev/null)
    
    # If above fails (Linux), try alternative date command
    if [ $? -ne 0 ]; then
        next_saturday=$(date -d "+$days_until_saturday days" +%Y-%m-%dT23:59:59Z)
    fi
    
    echo "$next_saturday"
}

# Function to check if gh CLI is installed
check_gh_cli() {
    if ! command -v gh &> /dev/null; then
        echo "Error: GitHub CLI (gh) is not installed"
        echo "Please install it from: https://cli.github.com/"
        exit 1
    fi
}

# Function to check if user is authenticated
check_auth() {
    if ! gh auth status &> /dev/null; then
        echo "Error: Not authenticated with GitHub"
        echo "Please run: gh auth login"
        exit 1
    fi
}

# Function to get repository information
get_repo_info() {
    repo_info=$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null)
    if [ $? -ne 0 ]; then
        echo "Error: Not in a GitHub repository or repository not found"
        exit 1
    fi
    echo "$repo_info"
}

# Main script
main() {
    # Check prerequisites
    check_gh_cli
    check_auth
    
    # Get repository information
    repo_info=$(get_repo_info)
    owner=$(echo "$repo_info" | cut -d'/' -f1)
    repo=$(echo "$repo_info" | cut -d'/' -f2)
    
    # Get next Saturday's date
    due_date=$(get_next_saturday)
    
    # Format date for display (YYYY-MM-DD)
    display_date=$(echo "$due_date" | cut -d'T' -f1)
    
    # Create milestone
    echo "Creating milestone for next Saturday ($display_date)..."
    
    response=$(gh api "repos/$owner/$repo/milestones" \
        --method POST \
        -f title="$display_date" \
        -f state="open" \
        -f description="$display_date" \
        -f due_on="$due_date" 2>&1)
    
    if [ $? -eq 0 ]; then
        echo "Successfully created milestone!"
        echo "Title: $display_date"
        echo "Due date: $display_date"
    else
        echo "Error creating milestone: $response"
        exit 1
    fi
}

# Run the script
main
