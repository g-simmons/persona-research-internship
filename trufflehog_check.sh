#!/bin/sh

# Clear the output file
> trufflehog_output.txt

trufflehog filesystem ./test_secrets.txt > trufflehog_output.txt  # No --only-modified

if grep -q "Found verified result" trufflehog_output.txt; then
  cat trufflehog_output.txt
  exit 1
fi

if grep -q "Found unverified result" trufflehog_output.txt; then
  cat trufflehog_output.txt
  exit 1
fi

cat trufflehog_output.txt