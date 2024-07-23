#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <parent_directory>"
    exit 1
fi

PARENT_DIR=$1
OUTPUT_DIR=$2

# Check if the parent directory exists
if [ ! -d "$PARENT_DIR" ]; then
    echo "Parent directory does not exist: $PARENT_DIR"
    exit 1
fi

# Create the output directory if it does not exist
mkdir -p "$OUTPUT_DIR"

# Find and iterate over each .wav file in the parent directory and its subdirectories
find "$PARENT_DIR" -type f -name "*.wav" | while read -r input_file; do
    # Get the base name of the input file (e.g., audio_01.wav)
    base_name=$(basename "$input_file")

    # Run deepFilter command
    deepFilter "$input_file" --output-dir "$OUTPUT_DIR"
    echo "processing $base_name"
done

echo "Processing complete. Filtered files are saved in $OUTPUT_DIR"