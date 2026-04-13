#!/bin/bash
# Deletes generated chart HTML files from the charts/ folder

FORCE=${1:-0}
CHARTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/charts"

files=$(find "$CHARTS_DIR" -maxdepth 1 -name "*.html" 2>/dev/null)

if [ -z "$files" ]; then
    echo "No chart HTML files found in $CHARTS_DIR."
    exit 0
fi

count=$(echo "$files" | wc -l)
echo "Found $count file(s) in $CHARTS_DIR:"
echo "$files"
echo ""

if [[ "$FORCE" == "1" ]]; then
    echo "$files" | xargs rm -f
    echo "Deleted $count file(s)."
else
    read -p "Delete all? [y/N] " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        echo "$files" | xargs rm -f
        echo "Deleted $count file(s)."
    else
        echo "Cancelled."
    fi
fi
