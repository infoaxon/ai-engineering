#!/bin/bash

# Source directory (current directory if not specified)
SOURCE_DIR="${1:-$(pwd)}"

# Target directory (must be specified)
TARGET_DIR="$2"

if [ -z "$TARGET_DIR" ]; then
  echo "Usage: $0 [source_dir] target_dir"
  exit 1
fi

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Find and move all PDFs
find "$SOURCE_DIR" -type f -iname '*.pdf' -exec mv -i {} "$TARGET_DIR" \;

echo "All PDF files have been moved to $TARGET_DIR."
