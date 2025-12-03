
# Create additional utilities: requirements file and usage guide

requirements = '''# Requirements for Hate Speech Classification with BERT

# Core Deep Learning
torch>=2.0.0
transformers>=4.30.0

# Data Processing
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Progress Bars
tqdm>=4.65.0

# Optional but recommended
datasets>=2.12.0  # For loading pre-built hate speech datasets
matplotlib>=3.7.0  # For visualization
seaborn>=0.12.0   # For confusion matrix plots

# Installation command:
# pip install torch transformers pandas numpy scikit-learn tqdm
'''

with open('requirements.txt', 'w') as f:
    f.write(requirements)

print("? Requirements file created: requirements.txt")

# Create usage guide
usage_guide = '''# Hate Speech Classification - Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install torch transformers pandas numpy scikit-learn tqdm
```

### 2. Prepare Your Dataset
Your dataset should be a CSV file with two columns:
- `text`: The text content
- `label`: 0 for not hate speech, 1 for hate speech

Example format:
```csv
text,label
"I love this community",0
"You fucking idiot go die",1
"Great work everyone",0
```

### 3. Modify the Code
In `hate_speech_classifier.py`, replace the dummy data section with your actual dataset:

```python
# Replace this:
df = pd.DataFrame({...})  # Dummy data

# With this:
df = pd.read_csv('your_dataset.csv')
```

### 4. Run the Training
```bash
python hate_speech_classifier.py
```

## Configuration Options

Edit the CONFIG dictionary in the main() function:

```python
CONFIG = {
    'base_model': 'bert-base-uncased',  # or 'bert-large-uncased'
    'max_length': 128,                   # Max sequence length
    'batch_size': 16,                    # Adjust based on GPU memory
    'learning_rate': 2e-5,               # Learning rate
    'epochs': 3,                         # Number of training epochs
    'sample_ratio': 0.3,                 # Use 30% of data (adjust as needed)
    'train_split': 0.8,                  # 80% train, 20% validation
    'random_seed': 42
}
```

## Key Features

### 1. Extended Vocabulary
Automatically adds 200+ hate speech terms and variants including:
- Profanity: fuck, f*ck, f**k, etc.
- Slurs: faggot, f@ggot, etc.
- Racial slurs and their variants
- Contextual hate patterns

### 2. TF-IDF Dataset Sampling
- Reduces training data while maintaining accuracy
- Selects most informative samples
- Maintains class balance
- Adjust `sample_ratio` (0.1 = 10%, 0.5 = 50%, etc.)

### 3. Training Pipeline
- Automatic train/validation split
- Progress bars for monitoring
- Saves best model automatically
- Comprehensive evaluation metrics

## Using the Trained Model

After training, use the model for inference:

```python
from hate_speech_classifier import predict_text
import torch
from transformers import BertTokenizer

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('best_hate_speech_model.pt'))
model = model.to(device)

# Predict
text = "Your text here"
prediction, confidence = predict_text(text, model, tokenizer, device)
print(f"Prediction: {prediction}, Confidence: {confidence}")
```

## Popular Datasets

You can use these public hate speech datasets:

1. **Davidson et al. Hate Speech Dataset**
   - 25k tweets labeled as hate speech, offensive, or neither
   - Available on GitHub and Kaggle

2. **HateXplain**
   - 20k posts with rationales
   - Multiple annotators per post

3. **OLID (Offensive Language Identification Dataset)**
   - 14k tweets in English
   - Hierarchical annotation

4. **Jigsaw Toxic Comment Dataset**
   - Available on Kaggle
   - Multiple toxicity types

## Performance Tips

1. **For limited GPU memory:**
   - Reduce `batch_size` to 8 or 4
   - Use `bert-base-uncased` instead of `bert-large-uncased`

2. **For better accuracy:**
   - Increase `epochs` to 4-5
   - Increase `sample_ratio` to use more data
   - Use `bert-large-uncased`

3. **For faster training:**
   - Decrease `sample_ratio` to 0.2 or 0.1
   - Reduce `max_length` to 64
   - Use fewer epochs

## Output Files

After training, you'll have:
- `best_hate_speech_model.pt` - Trained model weights
- Training logs in console
- Classification report and confusion matrix

## Troubleshooting

**CUDA out of memory:**
- Reduce batch_size
- Reduce max_length
- Use CPU (slower but works)

**Low accuracy:**
- Increase sample_ratio
- Train for more epochs
- Check dataset quality and balance

**Slow training:**
- Enable GPU if available
- Reduce sample_ratio
- Use smaller BERT model
'''

with open('USAGE_GUIDE.md', 'w') as f:
    f.write(usage_guide)

print("? Usage guide created: USAGE_GUIDE.md")

print("\n" + "="*60)
print("ALL FILES CREATED SUCCESSFULLY!")
print("="*60)
print("\nGenerated files:")
print("1. hate_speech_classifier.py - Main implementation")
print("2. requirements.txt - Dependencies")
print("3. USAGE_GUIDE.md - Complete usage instructions")
print("\nNext steps:")
print("1. Install requirements: pip install -r requirements.txt")
print("2. Prepare your dataset (CSV with 'text' and 'label' columns)")
print("3. Update the data loading section in the code")
print("4. Run: python hate_speech_classifier.py")