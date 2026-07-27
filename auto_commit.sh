#!/bin/bash
cd /home/polo/Escritorio/PHD/code/photextra_pipeline || exit 1
if [ -n "$(git status --porcelain)" ]; then
  git add -A
  git commit -m "auto-backup: $(date '+%Y-%m-%d %H:%M:%S')" >> /home/polo/Escritorio/PHD/code/photextra_pipeline/auto_commit.log 2>&1
fi
