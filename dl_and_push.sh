#!/bin/bash

# Hard-coded Feishu document URL
FEISHU_URL="https://ndro4zkb6p.feishu.cn/docx/FaBXdERDvoleXDxSiMXcMBegnvd"

# Extract the document ID from the URL
DOC_ID=$(basename "$FEISHU_URL")

# Download the document using feishu2md
echo "Downloading document..."
feishu2md dl "$FEISHU_URL"

# Check if the download was successful
if [ $? -ne 0 ]; then
    echo "Failed to download the document."
    exit 1
fi

# Rename the downloaded file to README.md
echo "Renaming the downloaded file..."
mv "${DOC_ID}.md" "README.md"

# Check if the rename was successful
if [ $? -ne 0 ]; then
    echo "Failed to rename the file."
    exit 1
fi

# Add the file to the Git staging area
echo "Adding file to Git..."
git add .

# Check if the add was successful
if [ $? -ne 0 ]; then
    echo "Failed to add the file to Git."
    exit 1
fi

# Commit the changes with a message
echo "Committing changes..."
git commit -m 'add paper'

# Check if the commit was successful
if [ $? -ne 0 ]; then
    echo "Failed to commit the changes."
    exit 1
fi

# Push the changes to the remote repository
echo "Pushing changes to remote repository..."
git push

# Check if the push was successful
if [ $? -ne 0 ]; then
    echo "Failed to push the changes."
    exit 1
fi

echo "Process completed successfully."