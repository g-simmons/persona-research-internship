#!/bin/bash

gh issue list --limit 1000 --json assignees,title,number \
  --jq '
    .[] | 
    # Only output if there are assignees
    select((.assignees | length) > 0) |
    # Store the parent object
    . as $issue |
    # Handle each assignee separately
    .assignees[] | {
      assignee: .login,
      number: $issue.number,
      title: $issue.title
    } | [.assignee, .number, .title] | @tsv
  ' \
  | sort \
  | awk -F'\t' '
    BEGIN {prev=""}
    {
      # Clean the title by removing newlines and tabs
      gsub(/[\n\t]/, " ", $3)
      
      if ($1 != prev) {
        if (prev != "") print ""
        print $1 ":"
        prev=$1
      }
      printf "  #%-4s %s\n", $2, $3
    }'