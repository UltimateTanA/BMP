"""
Hate Speech Classification using BERT with Extended Vocabulary and TF-IDF Dataset Selection
=========================================================================================

This implementation includes:
1. Extended vocabulary with hate speech terms and variants
2. TF-IDF-based intelligent dataset sampling
3. BERT fine-tuning for hate speech classification
4. Comprehensive training and evaluation pipeline
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 1. EXTENDED VOCABULARY WITH HATE SPEECH TERMS
# =====================================================================

class ExtendedVocabulary:
    """Extended vocabulary including hate speech terms and their variants"""
    
    def __init__(self):
        # Base hate speech terms with variants
        self.hate_terms = {
            # Profanity
            'fuck': ['fuck', 'fck', 'f*ck', 'f**k', 'f***', 'fuk', 'fuck', 'phuck', 
                     'fucking', 'fucked', 'fucker', 'fuckers', 'fking', 'f@ck'],
            
            # Homophobic slurs
            'faggot': ['faggot', 'fag', 'f@ggot', 'f*ggot', 'fagg', 'fagot', 
                       'faggots', 'fags', 'fag0t', 'phag', 'phagget'],
            
            # Racial slurs
            'nigger': ['nigger', 'nigga', 'n*gger', 'n***a', 'nig', 'nigg', 
                       'niggas', 'n1gger', 'n1gga', 'niqqa', 'niqqer'],
            
            # Other offensive terms
            'bitch': ['bitch', 'b*tch', 'b1tch', 'biatch', 'biotch', 'bitches',
                      'bytch', 'b!tch', 'beotch'],
            
            'cunt': ['cunt', 'c*nt', 'kunt', 'cunts', 'c@nt', 'cvnt'],
            
            'whore': ['whore', 'wh*re', 'hore', 'whores', 'wh0re', 'hoar'],
            
            'retard': ['retard', 'retarded', 'r3tard', 'ret@rd', 'retards',
                       'retart', 'ritard'],
            
            'dick': ['dick', 'd*ck', 'dik', 'dicks', 'd1ck', 'dikk', 'dck'],
            
            'ass': ['ass', 'a$$', '@ss', 'arse', 'asshole', 'a**', 'azzhole',
                    'assh0le', 'arsehole'],
            
            'shit': ['shit', 'sh*t', 'sht', 'shyt', 'sh1t', 'shiit', 'shite',
                     'shitt', 'sh!t'],
            
            'pussy': ['pussy', 'pus*y', 'puss', 'pussies', 'pusy', 'pu$$y'],
            
            'slut': ['slut', 'sl*t', 'sluts', 'slt', 'slutt', 'sl@t'],
            
            'bastard': ['bastard', 'bast@rd', 'bastards', 'basterd', 'b@stard'],
            
            'damn': ['damn', 'd*mn', 'dam', 'damm', 'dammit', 'd@mn'],
            
            'hell': ['hell', 'h*ll', 'hel', 'helll', 'h3ll'],
        }
        
        # Contextual hate patterns
        self.hate_patterns = [
            'kill yourself', 'kys', 'die', 'hang yourself', 'shoot yourself',
            'white power', 'white supremacy', 'master race', 'inferior race',
            'go back to', 'terrorist', 'subhuman', 'vermin', 'scum',
            'trash', 'garbage', 'worthless', 'disease', 'cancer'
        ]
        
        self.all_terms = self._flatten_terms()
    
    def _flatten_terms(self):
        """Flatten all hate terms into a single list"""
        terms = []
        for base, variants in self.hate_terms.items():
            terms.extend(variants)
        terms.extend(self.hate_patterns)
        return list(set(terms))
    
    def get_vocabulary_list(self):
        """Return list of all hate speech terms"""
        return self.all_terms


# =====================================================================
# 2. TF-IDF BASED DATASET SAMPLING
# =====================================================================

class TFIDFDatasetSampler:
    """Sample most informative datapoints using TF-IDF"""
    
    def __init__(self, sample_ratio=0.3, max_features=5000):
        """
        Args:
            sample_ratio: Fraction of dataset to keep (0.1 = 10%)
            max_features: Maximum number of TF-IDF features
        """
        self.sample_ratio = sample_ratio
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    
    def calculate_informativeness(self, texts):
        """Calculate informativeness score for each text"""
        # Fit TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Calculate informativeness: sum of TF-IDF values per document
        informativeness_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        return informativeness_scores
    
    def sample_dataset(self, df, text_column='text', label_column='label'):
        """
        Sample most informative datapoints from dataset
        
        Args:
            df: DataFrame with text and labels
            text_column: Name of text column
            label_column: Name of label column
        
        Returns:
            Sampled DataFrame
        """
        print(f"Original dataset size: {len(df)}")
        
        # Calculate informativeness scores
        scores = self.calculate_informativeness(df[text_column].values)
        df['informativeness_score'] = scores
        
        # Sample separately from each class to maintain balance
        sampled_dfs = []
        for label in df[label_column].unique():
            label_df = df[df[label_column] == label].copy()
            n_samples = max(1, int(len(label_df) * self.sample_ratio))
            
            # Sort by informativeness and take top samples
            sampled_label_df = label_df.nlargest(n_samples, 'informativeness_score')
            sampled_dfs.append(sampled_label_df)
        
        sampled_df = pd.concat(sampled_dfs, ignore_index=True)
        sampled_df = sampled_df.drop('informativeness_score', axis=1)
        
        print(f"Sampled dataset size: {len(sampled_df)} ({len(sampled_df)/len(df)*100:.1f}%)")
        print(f"Label distribution:")
        print(sampled_df[label_column].value_counts())
        
        return sampled_df


# =====================================================================
# 3. BERT TOKENIZER WITH EXTENDED VOCABULARY
# =====================================================================

class ExtendedBERTTokenizer:
    """BERT tokenizer with extended hate speech vocabulary"""
    
    def __init__(self, base_model='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(base_model)
        self.vocab_extension = ExtendedVocabulary()
        
        # Add new tokens to tokenizer
        new_tokens = self.vocab_extension.get_vocabulary_list()
        num_added = self.tokenizer.add_tokens(new_tokens)
        print(f"Added {num_added} new tokens to vocabulary")
        
        self.original_vocab_size = len(self.tokenizer) - num_added
        self.extended_vocab_size = len(self.tokenizer)
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_vocab_size(self):
        return self.extended_vocab_size


# =====================================================================
# 4. DATASET CLASS
# =====================================================================

class HateSpeechDataset(Dataset):
    """PyTorch Dataset for hate speech classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# =====================================================================
# 5. BERT MODEL WITH EXTENDED VOCABULARY
# =====================================================================

class ExtendedBERTClassifier(nn.Module):
    """BERT classifier with extended vocabulary"""
    
    def __init__(self, base_model='bert-base-uncased', num_labels=2, 
                 extended_vocab_size=None, dropout=0.3):
        super(ExtendedBERTClassifier, self).__init__()
        
        # Load base BERT model
        self.bert = BertForSequenceClassification.from_pretrained(
            base_model,
            num_labels=num_labels
        )
        
        # Resize token embeddings for extended vocabulary
        if extended_vocab_size:
            self.bert.resize_token_embeddings(extended_vocab_size)
            print(f"Resized embeddings to {extended_vocab_size}")
        
        # Add dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels  # pass labels to get loss
        )
        return outputs



# =====================================================================
# 6. TRAINING PIPELINE
# =====================================================================

class HateSpeechTrainer:
    """Training pipeline for hate speech classification"""
    
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=2e-5, epochs=3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        
        # Optimizer and scheduler
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs.loss
                total_loss += loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        return avg_loss, accuracy, f1, predictions, true_labels
    
    def train(self):
        """Full training loop"""
        print("\nStarting training...")
        best_val_acc = 0
        
        for epoch in range(self.epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc, val_f1, predictions, true_labels = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_acc:.4f}")
            print(f"Val F1 Score: {val_f1:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_hate_speech_model.pt')
                print(f"\n? Saved best model with accuracy: {val_acc:.4f}")
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
        print(f"{'='*60}")
        
        return predictions, true_labels


# =====================================================================
# 7. MAIN EXECUTION PIPELINE
# =====================================================================

def main():
    """Main execution function"""
    
    # Configuration
    CONFIG = {
        'base_model': 'bert-base-uncased',
        'max_length': 128,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 3,
        'sample_ratio': 0.3,  # Use 30% of data
        'train_split': 0.8,
        'random_seed': 42
    }
    
    # Set random seeds
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*60)
    print("HATE SPEECH CLASSIFICATION WITH BERT")
    print("="*60)
    
    # ===== STEP 1: LOAD DATA =====
    print("\n[STEP 1] Loading dataset...")
    # Replace this with your actual dataset loading
    # Expected format: DataFrame with 'text' and 'label' columns
    # label: 0 = not hate speech, 1 = hate speech
    
    # Example: Creating dummy data (replace with actual data loading)
    
    df = pd.read_csv('hateSpeechDataset.csv')
    # Ensure columns: 'text', 'label'
    
    
    # Dummy data for demonstration
    print("NOTE: Using dummy data. Replace with actual dataset!")
    df = pd.DataFrame({
        'text': [
            'I love everyone',
            'You are a fucking idiot',
            'Great day today',
            'Kill yourself f*ggot',
            'Nice work on the project',
            'You stupid n***er',
        ] * 100,  # Repeat for demo
        'label': [0, 1, 0, 1, 0, 1] * 100
    })
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # ===== STEP 2: TF-IDF SAMPLING =====
    print(f"\n[STEP 2] Sampling dataset using TF-IDF (ratio={CONFIG['sample_ratio']})...")
    sampler = TFIDFDatasetSampler(sample_ratio=CONFIG['sample_ratio'])
    df_sampled = sampler.sample_dataset(df, text_column='text', label_column='label')
    
    # ===== STEP 3: EXTENDED TOKENIZER =====
    print(f"\n[STEP 3] Initializing BERT tokenizer with extended vocabulary...")
    extended_tokenizer = ExtendedBERTTokenizer(base_model=CONFIG['base_model'])
    tokenizer = extended_tokenizer.get_tokenizer()
    vocab_size = extended_tokenizer.get_vocab_size()
    
    # ===== STEP 4: TRAIN/VAL SPLIT =====
    print(f"\n[STEP 4] Splitting into train/validation sets...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df_sampled['text'].values,
        df_sampled['label'].values,
        test_size=1-CONFIG['train_split'],
        random_state=CONFIG['random_seed'],
        stratify=df_sampled['label'].values
    )
    
    print(f"Train set: {len(train_texts)} samples")
    print(f"Val set: {len(val_texts)} samples")
    
    # ===== STEP 5: CREATE DATASETS =====
    print(f"\n[STEP 5] Creating PyTorch datasets...")
    train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer, CONFIG['max_length'])
    val_dataset = HateSpeechDataset(val_texts, val_labels, tokenizer, CONFIG['max_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    # ===== STEP 6: INITIALIZE MODEL =====
    print(f"\n[STEP 6] Initializing BERT model with extended vocabulary...")
    model = ExtendedBERTClassifier(
        base_model=CONFIG['base_model'],
        num_labels=2,
        extended_vocab_size=vocab_size
    )
    
    # ===== STEP 7: TRAIN MODEL =====
    print(f"\n[STEP 7] Training model...")
    trainer = HateSpeechTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=CONFIG['learning_rate'],
        epochs=CONFIG['epochs']
    )
    
    predictions, true_labels = trainer.train()
    
    # ===== STEP 8: EVALUATION =====
    print(f"\n[STEP 8] Final evaluation...")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, 
                                target_names=['Not Hate Speech', 'Hate Speech']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predictions))
    
    # ===== STEP 9: INFERENCE EXAMPLE =====
    print(f"\n[STEP 9] Testing inference on sample texts...")
    
    def predict_text(text, model, tokenizer, device, max_length=128):
        """Predict hate speech for a single text"""
        model.eval()
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
        
        return prediction, confidence
    
    test_texts = [
        "I hope you have a great day!",
        "You fucking piece of shit",
        "Let's work together on this",
        "Go kill yourself f@ggot"
    ]
    
    print("\nSample predictions:")
    for text in test_texts:
        pred, conf = predict_text(text, model, tokenizer, device)
        label = "HATE SPEECH" if pred == 1 else "NOT HATE SPEECH"
        print(f"\nText: {text}")
        print(f"Prediction: {label} (confidence: {conf:.4f})")
    
    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)


if __name__ == "__main__":
    main()
