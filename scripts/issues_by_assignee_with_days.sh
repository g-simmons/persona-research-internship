#!/bin/bash

# Check for required commands
for cmd in jq gh; do
  if ! command -v "$cmd" &> /dev/null; then
    echo "Error: $cmd is not installed. Please install it first." >&2
    exit 1
  fi
done

# Function to calculate days between dates for BSD date
calculate_days() {
    assigned_date="$1"
    # Convert GitHub ISO 8601 date to seconds since epoch
    assigned_seconds=$(date -j -f "%Y-%m-%dT%H:%M:%SZ" "${assigned_date}" "+%s")
    current_seconds=$(date "+%s")
    echo $(( (current_seconds - assigned_seconds) / 86400 ))
}

# First, get all issues with assignees
gh issue list --limit 1000 --json number,assignees \
  --jq 'map(select(.assignees | length > 0) | .number)[]' | \
while read -r issue_number; do
  # For each issue, get the timeline events to find assignment date
  gh api "/repos/:owner/:repo/issues/$issue_number/timeline" --jq '
    # Find all assigned events
    map(select(.event == "assigned")) |
    # Group by assignee
    group_by(.assignee.login) |
    # For each assignee, get their earliest assignment
    map({
      assignee: .[0].assignee.login,
      assigned_at: min_by(.created_at).created_at
    }) | .[]
  ' | while read -r event; do
    # Parse the JSON event
    assignee=$(echo "$event" | jq -r '.assignee')
    assigned_at=$(echo "$event" | jq -r '.assigned_at')
    
    # Get the issue details
    gh issue view "$issue_number" --json title \
      --jq '[.title] | @tsv' | \
    while IFS=$'\t' read -r title; do
      # Calculate days elapsed since assignment
      days=$(calculate_days "$assigned_at")
      
      # Clean the title
      title="${title//$'\n'/ }"
      title="${title//$'\t'/ }"
      
      # Output in format for sorting
      echo "$assignee|$issue_number|$days|$title"
    done
  done
done | sort | \
awk -F'|' '
  BEGIN { prev="" }
  {
    if ($1 != prev) {
      if (prev != "") print ""
      print $1 ":"
      prev=$1
    }
    printf "  #%-4s (%3d days) %s\n", $2, $3, $4
  }'
