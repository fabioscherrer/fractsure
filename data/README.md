# Data Folder

This directory holds local datasets and derived artifacts.

## Recommended layout

- data/raw: immutable source datasets after download
- data/processed: transformed datasets generated from pipelines

## HBFMID target path

Copy the YOLO dataset so this file exists:

- data/raw/hbfmid/data.yaml

with sibling train/valid/test image and label folders.

## Versioning rule

Raw data files are ignored by Git.
Track only DVC metadata files (.dvc and generated .gitignore files) in Git.
