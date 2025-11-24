"""
CrimeLens Complete Pipeline - Updated with Working GNN
Integrates all components for Deliverable 3

Run this script to:
1. Train GNN with metrics logging
2. Run baseline comparisons  
3. Generate explanations
4. Save all artifacts for report
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# GRAPH BUILDER
# ============================================================

class CrimeGraphBuilder:
    """Builds graphs where events are nodes with action/object/location features."""
    
    def __init__(self):
        self.action_encoder = LabelEncoder()
        self.crime_encoder = LabelEncoder()
        self.object_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.fitted = False
        
    def fit(self, df: pd.DataFrame):
        self.action_encoder.fit(df['action'].unique().tolist() + ['<PAD>'])
        self.crime_encoder.fit(df['crime_type'].unique())
        self.object_encoder.fit(df['object'].unique().tolist() + ['<PAD>'])
        self.location_encoder.fit(df['location'].unique().tolist() + ['<PAD>'])
        
        self.n_actions = len(self.action_encoder.classes_)
        self.n_crimes = len(self.crime_encoder.classes_)
        self.n_objects = len(self.object_encoder.classes_)
        self.n_locations = len(self.location_encoder.classes_)
        self.fitted = True
        
    def build_graph(self, scene_df: pd.DataFrame) -> Data:
        scene_df = scene_df.sort_values('event_id').reset_index(drop=True)
        n_events = len(scene_df)
        
        actions, objects, locations = [], [], []
        for _, event in scene_df.iterrows():
            actions.append(self.action_encoder.transform([event['action']])[0])
            objects.append(self.object_encoder.transform([event['object']])[0])
            locations.append(self.location_encoder.transform([event['location']])[0])
        
        # Fully connected graph
        edge_src, edge_dst = [], []
        for i in range(n_events):
            for j in range(n_events):
                if i != j:
                    edge_src.append(i)
                    edge_dst.append(j)
        
        if len(edge_src) == 0:
            edge_src, edge_dst = [0], [0]
        
        data = Data()
        data.action = torch.tensor(actions, dtype=torch.long)
        data.object = torch.tensor(objects, dtype=torch.long)
        data.location = torch.tensor(locations, dtype=torch.long)
        data.edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        data.y = torch.tensor([self.crime_encoder.transform([scene_df['crime_type'].iloc[0]])[0]], dtype=torch.long)
        data.batch = torch.zeros(n_events, dtype=torch.long)
        data.num_nodes = n_events
        data.metadata = {
            'suspect': scene_df['suspect'].iloc[0],
            'victim': scene_df['victim'].iloc[0],
            'persons': list(pd.concat([scene_df['suspect'], scene_df['victim']]).unique()),
            'objects': list(scene_df['object'].unique()),
            'locations': list(scene_df['location'].unique()),
            'n_events': n_events
        }
        return data
    
    def process_dataset(self, df: pd.DataFrame) -> List[Data]:
        if not self.fitted:
            self.fit(df)
        return [self.build_graph(df[df['scene_id'] == sid]) for sid in df['scene_id'].unique()]


# ============================================================
# GNN MODEL
# ============================================================

class CrimeGNN(nn.Module):
    """GNN with embeddings + one-hot actions for crime classification."""
    
    def __init__(self, n_actions: int, n_objects: int, n_locations: int, 
                 n_crimes: int, embed_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.n_crimes = n_crimes
        self.n_actions = n_actions
        
        self.action_embed = nn.Embedding(n_actions, embed_dim)
        self.object_embed = nn.Embedding(n_objects, embed_dim)
        self.location_embed = nn.Embedding(n_locations, embed_dim)
        
        input_dim = embed_dim * 3 + n_actions
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_crimes)
        )
        
        self.suspect_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        action_emb = self.action_embed(data.action)
        object_emb = self.object_embed(data.object)
        location_emb = self.location_embed(data.location)
        action_onehot = F.one_hot(data.action, self.n_actions).float()
        
        x = torch.cat([action_emb, object_emb, location_emb, action_onehot], dim=-1)
        x = F.relu(self.input_proj(x))
        
        x1 = F.dropout(F.relu(self.norm1(self.conv1(x, data.edge_index))), p=0.3, training=self.training)
        x = x + x1
        x2 = F.dropout(F.relu(self.norm2(self.conv2(x, data.edge_index))), p=0.3, training=self.training)
        x = x + x2
        x3 = F.relu(self.norm3(self.conv3(x, data.edge_index)))
        x = x + x3
        
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        action_global = global_max_pool(action_onehot, batch)
        x_graph = torch.cat([x_mean, x_max, action_global], dim=-1)
        
        return {
            'crime_logits': self.classifier(x_graph),
            'suspect_scores': self.suspect_predictor(x).squeeze(-1),
            'node_embeddings': x,
            'graph_embedding': x_graph
        }


# ============================================================
# TRAINER WITH METRICS LOGGING
# ============================================================

class CrimeGNNTrainer:
    """Trainer with comprehensive metrics logging."""
    
    def __init__(self, model, lr=0.005, weight_decay=1e-4, save_dir="results"):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.criterion = nn.CrossEntropyLoss()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.best_acc = 0
        self.best_state = None
        
    def train_epoch(self, graphs: List[Data]) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct = 0, 0
        
        for idx in np.random.permutation(len(graphs)):
            data = graphs[idx]
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = self.criterion(out['crime_logits'], data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            correct += (out['crime_logits'].argmax(dim=1) == data.y).sum().item()
        
        self.scheduler.step()
        return total_loss / len(graphs), correct / len(graphs)
    
    @torch.no_grad()
    def evaluate(self, graphs: List[Data]) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct = 0, 0
        for data in graphs:
            out = self.model(data)
            total_loss += self.criterion(out['crime_logits'], data.y).item()
            correct += (out['crime_logits'].argmax(dim=1) == data.y).sum().item()
        return total_loss / len(graphs), correct / len(graphs)
    
    def fit(self, train_graphs, val_graphs, epochs=200, patience=40, verbose=True):
        no_improve = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_graphs)
            val_loss, val_acc = self.evaluate(val_graphs)
            
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
            
            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        if self.best_state:
            self.model.load_state_dict(self.best_state)
        
        self._save_artifacts()
        return self.history
    
    def _save_artifacts(self):
        # Save history
        pd.DataFrame(self.history).to_csv(os.path.join(self.save_dir, "training_history.csv"), index=False)
        
        # Save model
        torch.save(self.best_state, os.path.join(self.save_dir, "best_model.pt"))
        
        # Plot training curves
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.history['epoch'], self.history['train_loss'], 'b-', label='Train')
        axes[0].plot(self.history['epoch'], self.history['val_loss'], 'r-', label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history['epoch'], self.history['train_acc'], 'b-', label='Train')
        axes[1].plot(self.history['epoch'], self.history['val_acc'], 'r-', label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training & Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "training_curves.png"), dpi=150)
        plt.close()
        
        print(f"Artifacts saved to {self.save_dir}/")


# ============================================================
# BASELINE COMPARISON
# ============================================================

class BaselineComparison:
    """Run and compare all baseline models."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
        
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create tabular features for baselines."""
        scene_features, scene_labels = [], []
        
        for scene_id in df['scene_id'].unique():
            scene = df[df['scene_id'] == scene_id]
            features = {
                'n_events': len(scene),
                'n_unique_actions': scene['action'].nunique(),
                'n_unique_objects': scene['object'].nunique(),
            }
            for action in self.df['action'].unique():
                features[f'action_{action}'] = int(action in scene['action'].values)
            for obj in self.df['object'].unique():
                features[f'object_{obj}'] = int(obj in scene['object'].values)
            
            scene_features.append(features)
            scene_labels.append(scene['crime_type'].iloc[0])
        
        X = pd.DataFrame(scene_features).values
        y = LabelEncoder().fit_transform(scene_labels)
        return X, y
    
    def run_all(self, train_df: pd.DataFrame, test_df: pd.DataFrame, verbose=True) -> pd.DataFrame:
        X_train, y_train = self._prepare_features(train_df)
        X_test, y_test = self._prepare_features(test_df)
        
        if verbose:
            print("\n" + "="*50)
            print("BASELINE COMPARISON")
            print("="*50)
        
        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        self.results['Logistic Regression'] = {'accuracy': lr.score(X_test, y_test)}
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        self.results['Random Forest'] = {'accuracy': rf.score(X_test, y_test)}
        
        if verbose:
            for name, res in self.results.items():
                print(f"{name}: {res['accuracy']:.3f}")
        
        return pd.DataFrame([{'Model': k, 'Accuracy': v['accuracy']} for k, v in self.results.items()])
    
    def add_gnn_result(self, accuracy: float):
        self.results['GNN (Ours)'] = {'accuracy': accuracy}
    
    def plot_comparison(self, save_path: str = None):
        models = list(self.results.keys())
        accs = [self.results[m]['accuracy'] for m in models]
        colors = ['#E74C3C' if 'GNN' in m else '#3498DB' for m in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(models, accs, color=colors, alpha=0.8)
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Comparison: Crime Type Classification')
        ax.set_ylim(0, 1.1)
        
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{acc:.3f}', ha='center')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()
        return fig


# ============================================================
# HYPOTHESIS GENERATOR & VISUALIZER
# ============================================================

class HypothesisGenerator:
    """Generate and visualize crime hypotheses."""
    
    def __init__(self, model, graph_builder):
        self.model = model
        self.graph_builder = graph_builder
        
    @torch.no_grad()
    def generate(self, scene_df: pd.DataFrame) -> Dict:
        self.model.eval()
        graph = self.graph_builder.build_graph(scene_df)
        output = self.model(graph)
        
        crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().numpy()
        crime_types = self.graph_builder.crime_encoder.classes_
        
        return {
            'crime_predictions': {ct: float(crime_probs[i]) for i, ct in enumerate(crime_types)},
            'top_prediction': crime_types[np.argmax(crime_probs)],
            'confidence': float(crime_probs.max()),
            'actual_crime': scene_df['crime_type'].iloc[0]
        }
    
    def visualize(self, scene_df: pd.DataFrame, save_path: str = None):
        hypothesis = self.generate(scene_df)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        crimes = list(hypothesis['crime_predictions'].keys())
        probs = list(hypothesis['crime_predictions'].values())
        colors = ['#E74C3C' if c == hypothesis['actual_crime'] else '#3498DB' for c in crimes]
        
        bars = ax.barh(crimes, probs, color=colors, alpha=0.8)
        ax.set_xlabel('Probability')
        ax.set_title(f"Crime Type Prediction\nActual: {hypothesis['actual_crime']} | Predicted: {hypothesis['top_prediction']}")
        ax.set_xlim(0, 1)
        
        for bar, prob in zip(bars, probs):
            ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2, f'{prob*100:.1f}%', va='center')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
        
        return hypothesis


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_complete_pipeline(df: pd.DataFrame, save_dir: str = "results", epochs: int = 200):
    """Run the complete CrimeLens training and evaluation pipeline."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*70)
    print("CRIMELENS COMPLETE PIPELINE")
    print("="*70)
    
    # Step 1: Data Preparation
    print("\n[STEP 1] Data Preparation")
    print("-"*50)
    
    scene_ids = df['scene_id'].unique()
    train_val_ids, test_ids = train_test_split(scene_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=0.15, random_state=42)
    
    train_df = df[df['scene_id'].isin(train_ids)]
    val_df = df[df['scene_id'].isin(val_ids)]
    test_df = df[df['scene_id'].isin(test_ids)]
    
    print(f"Total scenes: {len(scene_ids)}")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # Step 2: Build Graphs
    print("\n[STEP 2] Building Graphs")
    print("-"*50)
    
    graph_builder = CrimeGraphBuilder()
    graph_builder.fit(df)
    
    train_graphs = graph_builder.process_dataset(train_df)
    val_graphs = graph_builder.process_dataset(val_df)
    test_graphs = graph_builder.process_dataset(test_df)
    
    print(f"Graphs built - Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    print(f"Actions: {graph_builder.n_actions}, Objects: {graph_builder.n_objects}, Crime types: {graph_builder.n_crimes}")
    
    # Step 3: Train GNN
    print("\n[STEP 3] Training GNN")
    print("-"*50)
    
    model = CrimeGNN(
        n_actions=graph_builder.n_actions,
        n_objects=graph_builder.n_objects,
        n_locations=graph_builder.n_locations,
        n_crimes=graph_builder.n_crimes,
        embed_dim=32,
        hidden_dim=128
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = CrimeGNNTrainer(model, lr=0.005, save_dir=save_dir)
    trainer.fit(train_graphs, val_graphs, epochs=epochs, patience=40)
    
    # Test evaluation
    test_loss, test_acc = trainer.evaluate(test_graphs)
    print(f"\nTest Accuracy: {test_acc:.3f}")
    
    # Step 4: Run Baselines
    print("\n[STEP 4] Running Baselines")
    print("-"*50)
    
    baseline = BaselineComparison(df)
    baseline.run_all(train_df, test_df)
    baseline.add_gnn_result(test_acc)
    baseline.plot_comparison(save_path=os.path.join(save_dir, "model_comparison.png"))
    
    # Step 5: Generate Sample Hypothesis
    print("\n[STEP 5] Sample Hypothesis")
    print("-"*50)
    
    hypothesis_gen = HypothesisGenerator(model, graph_builder)
    test_scene = df[df['scene_id'] == test_ids[0]]
    hypothesis = hypothesis_gen.visualize(test_scene, save_path=os.path.join(save_dir, "hypothesis_example.png"))
    
    print(f"Predicted: {hypothesis['top_prediction']} ({hypothesis['confidence']*100:.1f}%)")
    print(f"Actual: {hypothesis['actual_crime']}")
    
    # Step 6: Save Summary
    print("\n[STEP 6] Saving Summary")
    print("-"*50)
    
    summary = {
        'dataset': {'total_scenes': len(scene_ids), 'train': len(train_ids), 'val': len(val_ids), 'test': len(test_ids)},
        'model': {'parameters': sum(p.numel() for p in model.parameters()), 'hidden_dim': 128},
        'results': {'gnn_accuracy': test_acc, **{k: v['accuracy'] for k, v in baseline.results.items()}}
    }
    
    with open(os.path.join(save_dir, "experiment_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Comparison table
    comparison_df = pd.DataFrame([
        {'Model': k, 'Accuracy': f"{v['accuracy']:.3f}"} 
        for k, v in baseline.results.items()
    ])
    comparison_df.to_csv(os.path.join(save_dir, "model_comparison.csv"), index=False)
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    for model_name, res in sorted(baseline.results.items(), key=lambda x: -x[1]['accuracy']):
        bar = "█" * int(res['accuracy'] * 40)
        print(f"{model_name:25s} | {res['accuracy']:.3f} | {bar}")
    
    print(f"\nAll artifacts saved to: {save_dir}/")
    
    return {
        'model': model,
        'graph_builder': graph_builder,
        'trainer': trainer,
        'test_acc': test_acc,
        'baseline_results': baseline.results,
        'summary': summary
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Option 1: Use your own data
    from Architecture_2_generator import DataManager
    dm = DataManager("/Users/rohitbogulla/Desktop/Sem 3/Applied ML 2/CrimeLens/data/realistic_crime_data.csv")
    df = dm.getData()
    
    # Run pipeline
    results = run_complete_pipeline(df, save_dir="results", epochs=200)