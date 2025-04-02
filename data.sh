#!/bin/bash
set -e

mkdir -p datasets

# --- Download and unzip the first kddcup.data.gz file ---
echo "Downloading kddcup.data.gz..."
wget -O datasets/kddcup.data.gz http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz

echo "Unzipping kddcup.data.gz to kddcup.data.txt..."
gunzip -c datasets/kddcup.data.gz > datasets/kddcup.data.txt

# --- Download and unzip the corrected.gz file ---
echo "Downloading corrected.gz..."
wget -O datasets/corrected.gz http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz

echo "Unzipping corrected.gz to corrected.txt..."
gunzip -c datasets/corrected.gz > datasets/corrected.txt

echo "Cleaning up: Deleting original .gz files..."
rm -f datasets/kddcup.data.gz datasets/corrected.gz

echo "All files have been downloaded, unzipped, and cleaned up in the 'datasets' directory."
