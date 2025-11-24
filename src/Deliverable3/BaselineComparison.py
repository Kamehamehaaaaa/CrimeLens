"""
Baseline Models for CrimeLens Comparison
Implements: Cosine Similarity, Node2Vec + Logistic Regression, MLP (no graph)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# BASELINE 1: Feature-based MLP (No Graph Structure)
# ============================================================

class FeatureExtractor:
    """Extract tabular features from crime scenes (ignores graph structure)."""
    
    def __init__(self):
        self.encoders = {}
        self.fitted = False
        
    def fit(self, df: pd.DataFrame):
        """Fit encoders on the dataset."""
        categorical_cols = ['suspect', 'victim', 'object', 'location', 'action']
        
        for col in categorical_cols:
            self.encoders[col] = LabelEncoder()
            self.encoders[col].fit(list(df[col].unique()) + ['<UNK>'])
        
        self.encoders['crime_type'] = LabelEncoder()
        self.encoders['crime_type'].fit(df['crime_type'].unique())
        self.fitted = True
        
    def transform_scene(self, scene_df: pd.DataFrame) -> np.ndarray:
        """
        Transform a scene into a feature vector.
        Aggregates multiple events into single vector.
        """
        features = []
        
        # Encode first event's entities (simplified)
        first_event = scene_df.iloc[0]
        
        for col in ['suspect', 'victim', 'object', 'location', 'action']:
            try:
                enc_val = self.encoders[col].transform([first_event[col]])[0]
            except:
                enc_val = self.encoders[col].transform(['<UNK>'])[0]
            features.append(enc_val)
        
        # Add aggregated features
        features.append(len(scene_df))  # Number of events
        features.append(scene_df['object'].nunique())  # Unique objects
        features.append(scene_df['action'].nunique())  # Unique actions
        features.append(scene_df['location'].nunique())  # Unique locations
        
        return np.array(features, dtype=np.float32)
    
    def get_label(self, scene_df: pd.DataFrame) -> int:
        """Get crime type label for a scene."""
        crime_type = scene_df['crime_type'].iloc[0]
        return self.encoders['crime_type'].transform([crime_type])[0]
    
    def process_dataset(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Process entire dataset into feature matrix and labels."""
        if not self.fitted:
            self.fit(df)
        
        X, y = [], []
        for scene_id in df['scene_id'].unique():
            scene_df = df[df['scene_id'] == scene_id]
            X.append(self.transform_scene(scene_df))
            y.append(self.get_label(scene_df))
        
        return np.array(X), np.array(y)


class MLPBaseline(nn.Module):
    """Simple MLP that ignores graph structure."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


class MLPBaselineTrainer:
    """Train and evaluate MLP baseline."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, lr: float = 0.01):
        self.model = MLPBaseline(input_dim, hidden_dim, num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 100) -> Dict:
        
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        
        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            logits = self.model(X_train_t)
            loss = self.criterion(logits, y_train_t)
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            preds = logits.argmax(dim=1).numpy()
            train_acc = accuracy_score(y_train, preds)
            history['train_loss'].append(loss.item())
            history['train_acc'].append(train_acc)
            
            if X_val is not None:
                val_acc = self.evaluate(X_val, y_val)['accuracy']
                history['val_acc'].append(val_acc)
        
        return history
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(X))
            preds = logits.argmax(dim=1).numpy()
            probs = F.softmax(logits, dim=1).numpy()
        
        return {
            'accuracy': accuracy_score(y, preds),
            'predictions': preds,
            'probabilities': probs
        }


# ============================================================
# BASELINE 2: Logistic Regression on Features
# ============================================================

class LogisticRegressionBaseline:
    """Simple logistic regression on tabular features."""
    
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, multi_class='multinomial')
        self.scaler = StandardScaler()
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)
        
        return {
            'accuracy': accuracy_score(y, preds),
            'predictions': preds,
            'probabilities': probs
        }


# ============================================================
# BASELINE 3: Random Forest on Features
# ============================================================

class RandomForestBaseline:
    """Random Forest on tabular features."""
    
    def __init__(self, n_estimators: int = 100):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        
        return {
            'accuracy': accuracy_score(y, preds),
            'predictions': preds,
            'probabilities': probs,
            'feature_importance': self.model.feature_importances_
        }


# ============================================================
# BASELINE 4: Cosine Similarity Based
# ============================================================

class CosineSimilarityBaseline:
    """
    Uses cosine similarity between scene embeddings.
    Embeddings are mean of one-hot entity encodings.
    """
    
    def __init__(self):
        self.encoders = {}
        self.train_embeddings = None
        self.train_labels = None
        
    def fit(self, df: pd.DataFrame):
        """Create one-hot encodings for all entities."""
        for col in ['suspect', 'victim', 'object', 'location', 'action']:
            self.encoders[col] = LabelEncoder()
            self.encoders[col].fit(df[col].unique())
        
        self.encoders['crime_type'] = LabelEncoder()
        self.encoders['crime_type'].fit(df['crime_type'].unique())
        
        # Compute embedding dimension
        self.embed_dim = sum(len(enc.classes_) for enc in self.encoders.values() 
                           if enc != self.encoders['crime_type'])
    
    def _scene_to_embedding(self, scene_df: pd.DataFrame) -> np.ndarray:
        """Convert scene to one-hot embedding."""
        embedding = np.zeros(self.embed_dim)
        offset = 0
        
        for col in ['suspect', 'victim', 'object', 'location', 'action']:
            enc = self.encoders[col]
            n_classes = len(enc.classes_)
            
            for val in scene_df[col].unique():
                try:
                    idx = enc.transform([val])[0]
                    embedding[offset + idx] = 1
                except:
                    pass
            offset += n_classes
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and create embeddings for training data."""
        self.fit(df)
        
        embeddings = []
        labels = []
        
        for scene_id in df['scene_id'].unique():
            scene_df = df[df['scene_id'] == scene_id]
            embeddings.append(self._scene_to_embedding(scene_df))
            crime_type = scene_df['crime_type'].iloc[0]
            labels.append(self.encoders['crime_type'].transform([crime_type])[0])
        
        self.train_embeddings = np.array(embeddings)
        self.train_labels = np.array(labels)
        
        return self.train_embeddings, self.train_labels
    
    def predict(self, scene_df: pd.DataFrame, k: int = 3) -> Dict:
        """Predict crime type using k-NN with cosine similarity."""
        query_embed = self._scene_to_embedding(scene_df)
        
        # Compute similarities
        similarities = np.dot(self.train_embeddings, query_embed)
        
        # Get top-k neighbors
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        top_k_labels = self.train_labels[top_k_idx]
        top_k_sims = similarities[top_k_idx]
        
        # Majority vote
        from collections import Counter
        vote = Counter(top_k_labels).most_common(1)[0][0]
        
        return {
            'prediction': vote,
            'top_k_similarities': top_k_sims,
            'top_k_labels': top_k_labels
        }
    
    def evaluate(self, df: pd.DataFrame) -> Dict:
        """Evaluate on a dataset."""
        correct = 0
        total = 0
        
        for scene_id in df['scene_id'].unique():
            scene_df = df[df['scene_id'] == scene_id]
            pred = self.predict(scene_df)['prediction']
            
            true_label = self.encoders['crime_type'].transform(
                [scene_df['crime_type'].iloc[0]]
            )[0]
            
            if pred == true_label:
                correct += 1
            total += 1
        
        return {'accuracy': correct / total}


# ============================================================
# BASELINE COMPARISON RUNNER
# ============================================================

class BaselineComparison:
    """Run all baselines and compare results."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
        
    def run_all(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                verbose: bool = True) -> pd.DataFrame:
        """Run all baselines and return comparison table."""
        
        # Prepare feature-based data
        feature_extractor = FeatureExtractor()
        feature_extractor.fit(train_df)
        
        X_train, y_train = feature_extractor.process_dataset(train_df)
        X_test, y_test = feature_extractor.process_dataset(test_df)
        
        num_classes = len(feature_extractor.encoders['crime_type'].classes_)
        
        if verbose:
            print("="*60)
            print("BASELINE COMPARISON")
            print("="*60)
            print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
            print(f"Features: {X_train.shape[1]}, Classes: {num_classes}")
            print("="*60)
        
        # 1. Logistic Regression
        if verbose:
            print("\n[1] Logistic Regression...")
        lr_model = LogisticRegressionBaseline()
        lr_model.fit(X_train, y_train)
        lr_results = lr_model.evaluate(X_test, y_test)
        self.results['Logistic Regression'] = lr_results
        if verbose:
            print(f"    Accuracy: {lr_results['accuracy']:.3f}")
        
        # 2. Random Forest
        if verbose:
            print("\n[2] Random Forest...")
        rf_model = RandomForestBaseline()
        rf_model.fit(X_train, y_train)
        rf_results = rf_model.evaluate(X_test, y_test)
        self.results['Random Forest'] = rf_results
        if verbose:
            print(f"    Accuracy: {rf_results['accuracy']:.3f}")
        
        # 3. MLP (No Graph)
        if verbose:
            print("\n[3] MLP (No Graph Structure)...")
        mlp_trainer = MLPBaselineTrainer(
            input_dim=X_train.shape[1],
            hidden_dim=64,
            num_classes=num_classes
        )
        mlp_trainer.fit(X_train, y_train, X_test, y_test, epochs=100)
        mlp_results = mlp_trainer.evaluate(X_test, y_test)
        self.results['MLP (No Graph)'] = mlp_results
        if verbose:
            print(f"    Accuracy: {mlp_results['accuracy']:.3f}")
        
        # 4. Cosine Similarity k-NN
        if verbose:
            print("\n[4] Cosine Similarity k-NN...")
        cosine_model = CosineSimilarityBaseline()
        cosine_model.fit_transform(train_df)
        cosine_results = cosine_model.evaluate(test_df)
        self.results['Cosine Similarity'] = cosine_results
        if verbose:
            print(f"    Accuracy: {cosine_results['accuracy']:.3f}")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Test Accuracy': [r['accuracy'] for r in self.results.values()]
        })
        
        if verbose:
            print("\n" + "="*60)
            print("COMPARISON SUMMARY")
            print("="*60)
            print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def add_gnn_result(self, name: str, accuracy: float, 
                       suspect_auc: float = None, suspect_ap: float = None):
        """Add GNN results for comparison."""
        self.results[name] = {
            'accuracy': accuracy,
            'suspect_auc': suspect_auc,
            'suspect_ap': suspect_ap
        }
    
    def get_comparison_table(self) -> pd.DataFrame:
        """Get formatted comparison table for report."""
        rows = []
        for model_name, metrics in self.results.items():
            row = {
                'Model': model_name,
                'Crime Type Accuracy': f"{metrics['accuracy']:.3f}"
            }
            if 'suspect_auc' in metrics and metrics['suspect_auc']:
                row['Suspect AUC'] = f"{metrics['suspect_auc']:.3f}"
            if 'suspect_ap' in metrics and metrics['suspect_ap']:
                row['Suspect AP'] = f"{metrics['suspect_ap']:.3f}"
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_comparison(self, save_path: str = None):
        """Plot bar chart comparison of models."""
        import matplotlib.pyplot as plt
        
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#3498DB'] * len(models)
        # Highlight GNN models
        for i, m in enumerate(models):
            if 'GNN' in m or 'Graph' in m:
                colors[i] = '#E74C3C'
        
        bars = ax.bar(models, accuracies, color=colors, alpha=0.8)
        
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_title('Model Comparison: Crime Type Classification', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{acc:.3f}', ha='center', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
        return fig