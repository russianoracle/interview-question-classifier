#!/bin/bash
# COMMANDS.sh — Все команды для подготовки датасета, обучения и скачивания

# ════════════════════════════════════════════════════════════════════════════════
# 📍 Current Directory
# ════════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT="/Users/artemgusarov/Downloads/PROJECTS/AIssistant"
DATASET_DIR="$PROJECT_ROOT/training/question-classifier"

# ════════════════════════════════════════════════════════════════════════════════
# 1️⃣  DOWNLOAD ADDITIONAL SOURCES (Parallel)
# ════════════════════════════════════════════════════════════════════════════════

# Downloads all found sources simultaneously:
# - YouTube channels (IT Отец, System Design, Mock interviews)
# - GitHub repositories (10 largest interview Q&A repos)
# - Habr articles (4 top interview collections)
# - Podcast transcripts (Frontend Weekend, MoscowPython, Hexlet)
#
# Output: raw_sources/ directory with all files organized by type

cd "$DATASET_DIR/dataset"
bash download_all_sources.sh

# What you get:
# ✅ raw_sources/youtube/it-otec/*.vtt (40+ videos)
# ✅ raw_sources/youtube/system-design/*.vtt (38 videos)
# ✅ raw_sources/youtube/mock-interviews/*.vtt
# ✅ raw_sources/github/ (10 repos cloned)
# ✅ raw_sources/habr/*.html (4 articles)
# ✅ raw_sources/podcasts/*.vtt (3 channels)
#
# Runs in parallel — completes faster than sequential


# ════════════════════════════════════════════════════════════════════════════════
# 2️⃣  PROCESS NEW SOURCES INTO DATASET (Optional expansion)
# ════════════════════════════════════════════════════════════════════════════════

# If you ran download_all_sources.sh, process new sources:

cd "$DATASET_DIR/dataset"

# A. Add new YouTube subtitle files
# (if you want to expand beyond current 162 sources)

# B. Add new questions from custom sources
# Edit raw_new_questions.txt with questions found from raw_sources/
# Then:
python3 add_questions.py

# C. Re-normalize the expanded dataset
python3 normalize_dataset.py

# Result: new dataset_normalized.jsonl with more samples
# (original 27.4K + new samples from sources)


# ════════════════════════════════════════════════════════════════════════════════
# 3️⃣  PREPARE FOR TRAINING
# ════════════════════════════════════════════════════════════════════════════════

# Copy dataset to accessible location
cp "$DATASET_DIR/dataset/processed/dataset_normalized.jsonl" \
   "$DATASET_DIR/dataset_normalized.jsonl"

# Copy notebook to main directory
cp "$DATASET_DIR/train_colab.ipynb" \
   "$DATASET_DIR/train_colab_ready.ipynb"

# Verify files exist:
ls -lh "$DATASET_DIR/dataset_normalized.jsonl"      # Should be ~5.8 MB
ls -lh "$DATASET_DIR/train_colab.ipynb"              # Should be ~100+ KB


# ════════════════════════════════════════════════════════════════════════════════
# 4️⃣  LOCAL TRAINING (if you have PyTorch + GPU)
# ════════════════════════════════════════════════════════════════════════════════

# Install dependencies
pip install torch transformers scikit-learn pandas numpy coremltools tqdm

# Run Jupyter notebook
cd "$DATASET_DIR"
jupyter notebook train_colab.ipynb

# Then:
# 1. Update DATASET_PATH to point to dataset_normalized.jsonl
# 2. Run all cells sequentially
# 3. Wait for training to complete (~1-2 hours on GPU)
# 4. Download outputs (best_model.pth, .mlpackage, etc.)


# ════════════════════════════════════════════════════════════════════════════════
# 5️⃣  GOOGLE COLAB TRAINING (Recommended)
# ════════════════════════════════════════════════════════════════════════════════

# 1. Open Google Colab: colab.research.google.com
# 2. Click "File" → "Open notebook" → "Upload"
# 3. Select: train_colab.ipynb
# 4. Upload dataset_normalized.jsonl to Colab:
#    - Click folder icon (left sidebar)
#    - Upload files
#    - Upload dataset_normalized.jsonl
# 5. In notebook cell with DATASET_PATH, set:
#    DATASET_PATH = "dataset_normalized.jsonl"
# 6. Click "Runtime" → "Run all" (or run cells one by one)
# 7. After training, download outputs:
#    - best_model.pth
#    - QuestionClassifier.mlpackage (for iOS)
#    - question_classifier_traced.pt (for server)


# ════════════════════════════════════════════════════════════════════════════════
# 6️⃣  DOWNLOAD YOUTUBE VIDEOS ONLY (Alternative)
# ════════════════════════════════════════════════════════════════════════════════

# If you only want YouTube subtitles (not full sources), use:

cd "$DATASET_DIR/dataset"

# Download by channel (gets all videos from IT Отец channel)
bash download_subs.sh "channel_id_1" "channel_id_2" ...

# Or download specific YouTube video IDs
bash download_subs.sh \
    "dQw4w9WgXcQ" \
    "9bZkp7q19f0" \
    "jNQXAC9IVRw"
# Creates: subtitles/{video_id}/{title}.vtt files


# ════════════════════════════════════════════════════════════════════════════════
# 📊 CHECK DATASET STATUS
# ════════════════════════════════════════════════════════════════════════════════

echo "=== Dataset Status ==="
echo "Dataset file size:"
ls -lh "$DATASET_DIR/dataset/processed/dataset_normalized.jsonl"

echo "\nDataset line count:"
wc -l "$DATASET_DIR/dataset/processed/dataset_normalized.jsonl"

echo "\nFirst 3 samples:"
head -3 "$DATASET_DIR/dataset/processed/dataset_normalized.jsonl"

echo "\nClass distribution:"
grep -o '"head2_label":"[^"]*"' "$DATASET_DIR/dataset/processed/dataset_normalized.jsonl" | sort | uniq -c

echo "\nDataset statistics:"
cat "$DATASET_DIR/dataset/processed/stats.json" 2>/dev/null || echo "(No stats.json)"


# ════════════════════════════════════════════════════════════════════════════════
# 📁 DIRECTORY STRUCTURE
# ════════════════════════════════════════════════════════════════════════════════

echo "=== Directory Structure ==="
tree -L 3 "$DATASET_DIR" -I '__pycache__|*.pyc|subtitles|raw_sources' 2>/dev/null || \
find "$DATASET_DIR" -maxdepth 3 -type f \( -name "*.py" -o -name "*.ipynb" -o -name "*.md" -o -name "*.sh" -o -name "*.json" \) | sort


# ════════════════════════════════════════════════════════════════════════════════
# 🎯 QUICK START (Summary)
# ════════════════════════════════════════════════════════════════════════════════

echo "
╔════════════════════════════════════════════════════════════════════════════╗
║                    INTERVIEW QUESTION CLASSIFIER — QUICK START              ║
╚════════════════════════════════════════════════════════════════════════════╝

✅ DATASET READY: $(wc -l < "$DATASET_DIR/dataset/processed/dataset_normalized.jsonl") samples
✅ NOTEBOOK READY: train_colab.ipynb
✅ DOWNLOAD SCRIPT READY: download_all_sources.sh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION A: Train Immediately (Recommended for now)
  1. Go to Google Colab: colab.research.google.com
  2. Upload train_colab.ipynb
  3. Upload dataset_normalized.jsonl
  4. Run notebook (GPU, ~1-2 hours)
  5. Download trained model

OPTION B: Expand Dataset + Train
  1. Run: bash download_all_sources.sh
     (Downloads YouTube, GitHub, Habr, Podcasts in parallel)
  2. Process new sources through build_dataset.py
  3. Re-normalize with normalize_dataset.py
  4. Upload expanded dataset_normalized.jsonl to Colab
  5. Train on larger dataset

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 Key Files:
  • train_colab.ipynb ..................... Full training pipeline
  • dataset/processed/dataset_normalized.jsonl ... Ready-to-train data (27.4K)
  • dataset/download_all_sources.sh ....... Download all sources in parallel
  • README_TRAINING.md .................... Full documentation
  • QUICKSTART.md ......................... Setup guide

📊 Dataset: 27,400 samples | 3:1 actionable ratio | 5.8 MB
🎯 Model: DistilBERT + 3 heads (Punctuation, Actionable, Query Span)
⏱️  Training: ~1-2 hours on GPU (Google Colab T4)

╚════════════════════════════════════════════════════════════════════════════╝
"
