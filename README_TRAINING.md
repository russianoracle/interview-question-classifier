# Interview Question Classifier — Training Guide

Подготовка датасета и обучение 3-head DistilBERT модели на интервью-вопросах из YouTube и GitHub.

## 📁 Files Overview

### Dataset
- **`dataset/processed/dataset_normalized.jsonl`** — готовый датасет (27,400 samples)
  - 20,550 actionable (вопросы)
  - 6,850 non_actionable (утверждения/ответы)
  - Ratio: 3:1 (сбалансировано)
  - 94.6% с пунктуацией
  - Источники: 162 YouTube VTT файла + 128 curated questions

### Training Notebook
- **`train_colab.ipynb`** — полный Jupyter notebook для Google Colab
  - Data loading and preprocessing
  - Model architecture (3-head DistilBERT)
  - Training loop with early stopping
  - Evaluation on test set
  - CoreML export for iOS

### Data Collection
- **`dataset/download_subs.sh`** — скачивает YouTube субтитры по видео ID
- **`dataset/download_all_sources.sh`** — параллельно скачивает все найденные источники

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
1. Download `train_colab.ipynb`
2. Upload to Google Colab
3. Upload `dataset_normalized.jsonl` to Colab environment
4. Update `DATASET_PATH` in the notebook to point to the file
5. Run cells sequentially

### Option 2: Local Training
```bash
# Install dependencies
pip install torch transformers datasets scikit-learn coremltools tqdm

# Run notebook locally (requires Jupyter)
jupyter notebook train_colab.ipynb
```

## 📊 Dataset Statistics

```
Total samples: 27,400
├── Actionable: 20,550 (75.0%)
│   ├── Curated: 128 (0.5%)
│   └── VTT-based: 20,422 (74.5%)
├── Non-actionable: 6,850 (25.0%)
└── Ratio: 3.0:1

Has punctuation: 26,357 (96.2%)
With query_start > 0: 1,754 (6.4%)

Split:
├── Train: 21,920 (80%)
├── Validation: 2,740 (10%)
└── Test: 2,740 (10%)
```

## 🔧 Model Architecture

### Backbone
- **DistilBERT** (base, multilingual, cased)
- Parameters: 66M (shared)
- Max sequence length: 128 tokens

### Three Classification Heads

#### Head 1: Punctuation/Capitalization (12 classes)
- Pre-trained, frozen during training
- Used for preprocessing (restore_stream)
- Not actively trained in this script

#### Head 2: Actionable Classification (Binary)
- **Input:** [CLS] token representation
- **Output:** 2 classes (non_actionable, actionable)
- **Loss:** CrossEntropyLoss with class weights
- **Main training objective** (70% weight)

#### Head 3: Query Span Detection (Token-level, Binary)
- **Input:** All token representations
- **Output:** 2 classes per token (0: not query, 1: query start)
- **Loss:** CrossEntropyLoss
- **Auxiliary task** (30% weight)
- Helps the model learn where questions begin

## 📈 Training Configuration

```python
Epochs: 3
Batch size: 32
Learning rate: 2e-5
Optimizer: AdamW
Warmup steps: 500
Max grad norm: 1.0
Early stopping patience: 2 epochs
Loss weights: Head 2 (70%) + Head 3 (30%)
```

## 📤 Model Outputs

After training, you get:
- `best_model.pth` — best checkpoint
- `question_classifier_weights.pth` — final weights
- `model_config.json` — configuration
- `question_classifier_traced.pt` — TorchScript version
- `QuestionClassifier.mlpackage` — CoreML for iOS

## 🔍 Inference Example

```python
result = predict(
    "Какие вопросы обычно задают на собеседовании?",
    model,
    tokenizer,
    device
)
# Output:
# {
#     'text': '...',
#     'head2_class': 'actionable',
#     'head2_confidence': 0.95,
#     'head2_scores': {
#         'non_actionable': 0.05,
#         'actionable': 0.95,
#     }
# }
```

## 📥 Downloading Additional Sources

Run the parallel download script to get all found sources:

```bash
cd dataset
bash download_all_sources.sh
```

This downloads in parallel:
- **YouTube videos** from IT Отец, System Design, Mock Interviews channels
- **GitHub repositories** with interview questions
- **Habr articles** with aggregated data
- **Podcast transcripts** (Frontend Weekend, MoscowPython, Hexlet)

Output: `raw_sources/` directory with subdirectories for each source type

## 🔄 Data Pipeline

```
Raw sources (YouTube, GitHub, Habr, Podcasts)
    ↓
VTT parsing + delta-extraction (rolling-caption dedup)
    ↓
Punctuation restoration (Apple NL + PunctuationModel)
    ↓
Question filtering (regex patterns + curated)
    ↓
Normalization (downsample to 3:1 ratio, preserve curated)
    ↓
dataset_normalized.jsonl (27,400 samples, ready to train)
```

## 📝 JSON Format

Each line in `dataset_normalized.jsonl`:

```json
{
  "text": "Какие основные паттерны проектирования ты знаешь?",
  "source": "curated",
  "head2_label": "actionable",
  "head3_query_start": 2,
  "has_punct": true
}
```

Fields:
- **text**: Full sentence/utterance
- **source**: YouTube ID or "curated"
- **head2_label**: "actionable" or "non_actionable"
- **head3_query_start**: Word offset where question starts (0+ if query present, null otherwise)
- **has_punct**: Whether text has punctuation/capitalization

## 🎯 Expected Results

Based on the dataset composition:
- **Validation F1:** Expected ~0.85-0.92 (3:1 imbalance still challenging)
- **Test Accuracy:** Expected ~88-92%
- **Precision (actionable):** ~0.88
- **Recall (actionable):** ~0.92

## 🐛 Troubleshooting

### CUDA out of memory
- Reduce batch size: `BATCH_SIZE = 16` or `8`
- Reduce max length: `MAX_LEN = 64`

### Low accuracy
- Check dataset stratification in splits
- Verify class weights are applied correctly
- Increase num_epochs
- Lower learning_rate to 1e-5

### CoreML export fails
- Ensure TorchScript export succeeded first
- Check coremltools version compatibility
- May need to simplify model (remove Head 3)

## 🔗 Integration

### iOS/Swift
1. Import `QuestionClassifier.mlpackage` into Xcode
2. Use MLModel to load and run predictions
3. See AIssistant/Core/Speech/SentenceEndLSTMPredictor.swift for integration pattern

### Server/Backend
1. Load weights: `torch.load('question_classifier_weights.pth')`
2. Initialize model: `ThreeHeadQuestionClassifier(MODEL_NAME)`
3. Use TorchScript version for production if available

## 📚 References

- Original dataset: `dataset/processed/dataset.jsonl` (74,688 samples before normalization)
- Stats: `dataset/processed/stats.json`
- Build script: `dataset/build_dataset.py`
- Normalization: `dataset/normalize_dataset.py`

---

**Дата подготовки:** 2026-04-07
**Версия датасета:** 1.0 (normalized)
**Модель:** DistilBERT + 3 heads (Punctuation, Actionable, Query Span)
