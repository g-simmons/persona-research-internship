#!/usr/bin/env python3
import json
import subprocess
from datetime import datetime
from collections import defaultdict
import sys

def fetch_closed_issues(repo):
    """Fetch closed issues from GitHub using gh CLI"""
    try:
        # Using gh cli to fetch closed issues in JSON format
        cmd = ['gh', 'issue', 'list', '--repo', repo, '--state', 'closed', '--json', 'title,closedAt,number']
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to fetch issues. Make sure 'gh' is installed and you're authenticated.")
        print(f"Error details: {e.stderr}")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Failed to parse GitHub CLI output")
        sys.exit(1)

def get_week_number(date_str):
    """Convert ISO date string to week number"""
    date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    return date.strftime('%Y-W%W')

def organize_by_week(issues):
    """Organize issues by week"""
    weekly_issues = defaultdict(list)
    
    for issue in issues:
        week = get_week_number(issue['closedAt'])
        weekly_issues[week].append({
            'number': issue['number'],
            'title': issue['title']
        })
    
    return dict(sorted(weekly_issues.items()))

def print_report(weekly_issues):
    """Print formatted report to terminal"""
    print("\n📊 Weekly Closed Issues Report")
    print("=" * 50)
    
    for week, issues in weekly_issues.items():
        print(f"\n📅 Week {week}")
        print(f"Total issues closed: {len(issues)}" + "*" * len(issues))
        print("-" * 40)
        
        for issue in issues:
            print(f"#{issue['number']}: {issue['title']}")
        
        print()

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py owner/repository")
        print("Example: python script.py facebook/react")
        sys.exit(1)
    
    repo = sys.argv[1]
    print(f"Fetching closed issues for {repo}...")
    
    issues = fetch_closed_issues(repo)
    if not issues:
        print("No closed issues found.")
        return
    
    weekly_issues = organize_by_week(issues)
    print_report(weekly_issues)

if __name__ == "__main__":
    main()