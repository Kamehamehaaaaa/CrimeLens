"""
Working Crime GNN - Simplified architecture that actually learns
Focus: Use action sequences as the primary signal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class SimpleGraphBuilder:
    """
    Simplified graph builder that captures action patterns.
    Key insight: Crime type is determined by the SEQUENCE of actions.
    """
    
    def __init__(self):
        self.action_encoder = LabelEncoder()
        self.crime_encoder = LabelEncoder()
        self.object_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.fitted = False
        
    def fit(self, df: pd.DataFrame):
        """Fit encoders on the dataset."""
        self.action_encoder.fit(df['action'].unique().tolist() + ['<PAD>'])
        self.crime_encoder.fit(df['crime_type'].unique())
        self.object_encoder.fit(df['object'].unique().tolist() + ['<PAD>'])
        self.location_encoder.fit(df['location'].unique().tolist() + ['<PAD>'])
        
        self.n_actions = len(self.action_encoder.classes_)
        self.n_crimes = len(self.crime_encoder.classes_)
        self.n_objects = len(self.object_encoder.classes_)
        self.n_locations = len(self.location_encoder.classes_)
        
        self.fitted = True
        print(f"Fitted: {self.n_actions} actions, {self.n_crimes} crime types, "
              f"{self.n_objects} objects, {self.n_locations} locations")
    
    def build_graph(self, scene_df: pd.DataFrame) -> Data:
        """
        Build a simple graph where:
        - Each EVENT is a node (not each entity)
        - Nodes are connected sequentially (event chain)
        - Node features = [action, object, location] embeddings
        """
        scene_df = scene_df.sort_values('event_id').reset_index(drop=True)
        n_events = len(scene_df)
        
        # Node features: action_id, object_id, location_id for each event
        actions = []
        objects = []
        locations = []
        
        for _, event in scene_df.iterrows():
            actions.append(self.action_encoder.transform([event['action']])[0])
            objects.append(self.object_encoder.transform([event['object']])[0])
            locations.append(self.location_encoder.transform([event['location']])[0])
        
        # Create sequential edges (event_i -> event_i+1)
        # Plus fully connected for small graphs
        edge_src = []
        edge_dst = []
        
        for i in range(n_events):
            for j in range(n_events):
                if i != j:
                    edge_src.append(i)
                    edge_dst.append(j)
        
        # If no edges (single event), add self-loop
        if len(edge_src) == 0:
            edge_src = [0]
            edge_dst = [0]
        
        data = Data()
        data.action = torch.tensor(actions, dtype=torch.long)
        data.object = torch.tensor(objects, dtype=torch.long)
        data.location = torch.tensor(locations, dtype=torch.long)
        data.edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        data.y = torch.tensor([self.crime_encoder.transform([scene_df['crime_type'].iloc[0]])[0]], 
                             dtype=torch.long)
        data.batch = torch.zeros(n_events, dtype=torch.long)
        data.num_nodes = n_events
        
        # Metadata
        data.metadata = {
            'suspect': scene_df['suspect'].iloc[0],
            'victim': scene_df['victim'].iloc[0],
            'n_events': n_events
        }
        
        return data
    
    def process_dataset(self, df: pd.DataFrame) -> List[Data]:
        if not self.fitted:
            self.fit(df)
        
        graphs = []
        for scene_id in df['scene_id'].unique():
            scene_df = df[df['scene_id'] == scene_id]
            graphs.append(self.build_graph(scene_df))
        return graphs


class WorkingCrimeGNN(nn.Module):
    """
    Improved GNN that combines:
    - Learned embeddings for generalization
    - One-hot action features for direct signal
    - Deeper network with residual connections
    """
    
    def __init__(self, n_actions: int, n_objects: int, n_locations: int, 
                 n_crimes: int, embed_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        
        self.n_crimes = n_crimes
        self.n_actions = n_actions
        
        # Embeddings for each feature type
        self.action_embed = nn.Embedding(n_actions, embed_dim)
        self.object_embed = nn.Embedding(n_objects, embed_dim)
        self.location_embed = nn.Embedding(n_locations, embed_dim)
        
        # Input: embeddings + one-hot action (direct signal)
        input_dim = embed_dim * 3 + n_actions
        
        # Project to hidden dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers with residual connections
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Classifier with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + n_actions, hidden_dim),  # +n_actions for global action features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_crimes)
        )
        
    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        # Get embeddings
        action_emb = self.action_embed(data.action)
        object_emb = self.object_embed(data.object)
        location_emb = self.location_embed(data.location)
        
        # One-hot encode actions for direct signal
        action_onehot = F.one_hot(data.action, self.n_actions).float()
        
        # Concatenate all features
        x = torch.cat([action_emb, object_emb, location_emb, action_onehot], dim=-1)
        x = self.input_proj(x)
        x = F.relu(x)
        
        # GNN layers with residual connections
        x1 = self.conv1(x, data.edge_index)
        x1 = self.norm1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.3, training=self.training)
        x = x + x1  # Residual
        
        x2 = self.conv2(x, data.edge_index)
        x2 = self.norm2(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.3, training=self.training)
        x = x + x2  # Residual
        
        x3 = self.conv3(x, data.edge_index)
        x3 = self.norm3(x3)
        x3 = F.relu(x3)
        x = x + x3  # Residual
        
        # Global pooling
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        # Global action feature: which actions are present in this scene?
        action_global = global_max_pool(action_onehot, batch)
        
        # Combine all for classification
        x_graph = torch.cat([x_mean, x_max, action_global], dim=-1)
        
        # Classification
        logits = self.classifier(x_graph)
        
        return {'crime_logits': logits, 'graph_embedding': x_graph}


class SimpleTrainer:
    """Improved trainer with learning rate scheduling."""
    
    def __init__(self, model, lr=0.005, weight_decay=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        self.best_acc = 0
        self.best_state = None
        
    def train_epoch(self, graphs: List[Data]) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        
        indices = np.random.permutation(len(graphs))
        
        for idx in indices:
            data = graphs[idx]
            self.optimizer.zero_grad()
            
            out = self.model(data)
            loss = self.criterion(out['crime_logits'], data.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = out['crime_logits'].argmax(dim=1)
            correct += (pred == data.y).sum().item()
        
        self.scheduler.step()
        return total_loss / len(graphs), correct / len(graphs)
    
    @torch.no_grad()
    def evaluate(self, graphs: List[Data]) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        
        for data in graphs:
            out = self.model(data)
            loss = self.criterion(out['crime_logits'], data.y)
            total_loss += loss.item()
            pred = out['crime_logits'].argmax(dim=1)
            correct += (pred == data.y).sum().item()
        
        return total_loss / len(graphs), correct / len(graphs)
    
    def fit(self, train_graphs, val_graphs, epochs=200, patience=40, verbose=True):
        no_improve = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_graphs)
            val_loss, val_acc = self.evaluate(val_graphs)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if verbose and epoch % 20 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:3d} | LR: {lr:.5f} | Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
            
            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        if self.best_state:
            self.model.load_state_dict(self.best_state)
        
        return self.history


def run_working_gnn(df: pd.DataFrame, verbose: bool = True):
    """
    Run the working GNN pipeline.
    Returns model, builder, and test accuracy.
    """
    if verbose:
        print("="*60)
        print("WORKING CRIME GNN")
        print("="*60)
    
    # Build graphs
    builder = SimpleGraphBuilder()
    builder.fit(df)
    graphs = builder.process_dataset(df)
    
    if verbose:
        print(f"Built {len(graphs)} graphs")
    
    # Split data
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    train_graphs, val_graphs = train_test_split(train_graphs, test_size=0.15, random_state=42)
    
    if verbose:
        print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    # Create model
    model = WorkingCrimeGNN(
        n_actions=builder.n_actions,
        n_objects=builder.n_objects,
        n_locations=builder.n_locations,
        n_crimes=builder.n_crimes,
        embed_dim=32,
        hidden_dim=64
    )
    
    if verbose:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = SimpleTrainer(model, lr=0.01, weight_decay=1e-4)
    history = trainer.fit(train_graphs, val_graphs, epochs=300, patience=50, verbose=verbose)
    
    # Test
    test_loss, test_acc = trainer.evaluate(test_graphs)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Best Val Accuracy: {trainer.best_acc:.3f}")
        print(f"Test Accuracy: {test_acc:.3f}")
    
    return {
        'model': model,
        'builder': builder,
        'trainer': trainer,
        'test_acc': test_acc,
        'history': history
    }


# ============================================================
# COMPARISON WITH BASELINES
# ============================================================

def compare_all_models(df: pd.DataFrame):
    """Run GNN and all baselines, return comparison."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    print("="*70)
    print("FULL MODEL COMPARISON")
    print("="*70)
    
    # 1. Run GNN
    print("\n[1/4] Training GNN...")
    gnn_results = run_working_gnn(df, verbose=False)
    print(f"      GNN Test Accuracy: {gnn_results['test_acc']:.3f}")
    
    # 2. Prepare tabular data for baselines
    print("\n[2/4] Preparing baseline features...")
    
    # Aggregate scene features
    scene_features = []
    scene_labels = []
    
    for scene_id in df['scene_id'].unique():
        scene = df[df['scene_id'] == scene_id]
        
        # Feature: counts of each action type
        action_counts = scene['action'].value_counts()
        object_counts = scene['object'].value_counts()
        
        # Create feature vector
        features = {
            'n_events': len(scene),
            'n_unique_actions': scene['action'].nunique(),
            'n_unique_objects': scene['object'].nunique(),
            'n_unique_locations': scene['location'].nunique(),
        }
        
        # Add action presence features
        for action in df['action'].unique():
            features[f'action_{action}'] = int(action in scene['action'].values)
        
        # Add object presence features  
        for obj in df['object'].unique():
            features[f'object_{obj}'] = int(obj in scene['object'].values)
        
        scene_features.append(features)
        scene_labels.append(scene['crime_type'].iloc[0])
    
    feature_df = pd.DataFrame(scene_features)
    X = feature_df.values
    y = LabelEncoder().fit_transform(scene_labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Random Forest
    print("\n[3/4] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"      Random Forest Accuracy: {rf_acc:.3f}")
    
    # 4. Logistic Regression
    print("\n[4/4] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_acc = lr.score(X_test, y_test)
    print(f"      Logistic Regression Accuracy: {lr_acc:.3f}")
    
    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    results = {
        'GNN': gnn_results['test_acc'],
        'Random Forest': rf_acc,
        'Logistic Regression': lr_acc,
        'Random Baseline': 1.0 / len(df['crime_type'].unique())
    }
    
    for model, acc in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(acc * 40)
        print(f"{model:25s} | {acc:.3f} | {bar}")
    
    return results


if __name__ == "__main__":    
    # Generate data
    print("Generating realistic crime data...")
    from Architecture_2_generator import DataManager
    
    # Generate data
    print("Generating realistic crime data...")
    dm = DataManager("/Users/rohitbogulla/Desktop/Sem 3/Applied ML 2/CrimeLens/data/realistic_crime_data.csv")
    df = dm.getData()     
    # Compare all models
    results = compare_all_models(df)


# """
# Working Crime GNN - Simplified architecture that actually learns
# Focus: Use action sequences as the primary signal
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from typing import List, Dict, Tuple
# import warnings
# warnings.filterwarnings('ignore')


# class SimpleGraphBuilder:
#     """
#     Simplified graph builder that captures action patterns.
#     Key insight: Crime type is determined by the SEQUENCE of actions.
#     """
    
#     def __init__(self):
#         self.action_encoder = LabelEncoder()
#         self.crime_encoder = LabelEncoder()
#         self.object_encoder = LabelEncoder()
#         self.location_encoder = LabelEncoder()
#         self.fitted = False
        
#     def fit(self, df: pd.DataFrame):
#         """Fit encoders on the dataset."""
#         self.action_encoder.fit(df['action'].unique().tolist() + ['<PAD>'])
#         self.crime_encoder.fit(df['crime_type'].unique())
#         self.object_encoder.fit(df['object'].unique().tolist() + ['<PAD>'])
#         self.location_encoder.fit(df['location'].unique().tolist() + ['<PAD>'])
        
#         self.n_actions = len(self.action_encoder.classes_)
#         self.n_crimes = len(self.crime_encoder.classes_)
#         self.n_objects = len(self.object_encoder.classes_)
#         self.n_locations = len(self.location_encoder.classes_)
        
#         self.fitted = True
#         print(f"Fitted: {self.n_actions} actions, {self.n_crimes} crime types, "
#               f"{self.n_objects} objects, {self.n_locations} locations")
    
#     def build_graph(self, scene_df: pd.DataFrame) -> Data:
#         """
#         Build a simple graph where:
#         - Each EVENT is a node (not each entity)
#         - Nodes are connected sequentially (event chain)
#         - Node features = [action, object, location] embeddings
#         """
#         scene_df = scene_df.sort_values('event_id').reset_index(drop=True)
#         n_events = len(scene_df)
        
#         # Node features: action_id, object_id, location_id for each event
#         actions = []
#         objects = []
#         locations = []
        
#         for _, event in scene_df.iterrows():
#             actions.append(self.action_encoder.transform([event['action']])[0])
#             objects.append(self.object_encoder.transform([event['object']])[0])
#             locations.append(self.location_encoder.transform([event['location']])[0])
        
#         # Create sequential edges (event_i -> event_i+1)
#         # Plus fully connected for small graphs
#         edge_src = []
#         edge_dst = []
        
#         for i in range(n_events):
#             for j in range(n_events):
#                 if i != j:
#                     edge_src.append(i)
#                     edge_dst.append(j)
        
#         # If no edges (single event), add self-loop
#         if len(edge_src) == 0:
#             edge_src = [0]
#             edge_dst = [0]
        
#         data = Data()
#         data.action = torch.tensor(actions, dtype=torch.long)
#         data.object = torch.tensor(objects, dtype=torch.long)
#         data.location = torch.tensor(locations, dtype=torch.long)
#         data.edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
#         data.y = torch.tensor([self.crime_encoder.transform([scene_df['crime_type'].iloc[0]])[0]], 
#                              dtype=torch.long)
#         data.batch = torch.zeros(n_events, dtype=torch.long)
#         data.num_nodes = n_events
        
#         # Metadata
#         data.metadata = {
#             'suspect': scene_df['suspect'].iloc[0],
#             'victim': scene_df['victim'].iloc[0],
#             'n_events': n_events
#         }
        
#         return data
    
#     def process_dataset(self, df: pd.DataFrame) -> List[Data]:
#         if not self.fitted:
#             self.fit(df)
        
#         graphs = []
#         for scene_id in df['scene_id'].unique():
#             scene_df = df[df['scene_id'] == scene_id]
#             graphs.append(self.build_graph(scene_df))
#         return graphs


# class WorkingCrimeGNN(nn.Module):
#     """
#     Simplified GNN that focuses on learning action patterns.
#     """
    
#     def __init__(self, n_actions: int, n_objects: int, n_locations: int, 
#                  n_crimes: int, embed_dim: int = 32, hidden_dim: int = 64):
#         super().__init__()
        
#         self.n_crimes = n_crimes
        
#         # Embeddings for each feature type
#         self.action_embed = nn.Embedding(n_actions, embed_dim)
#         self.object_embed = nn.Embedding(n_objects, embed_dim)
#         self.location_embed = nn.Embedding(n_locations, embed_dim)
        
#         # Combine embeddings
#         input_dim = embed_dim * 3
        
#         # Simple GNN layers
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
#         # Classifier
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(hidden_dim, n_crimes)
#         )
        
#     def forward(self, data: Data) -> Dict[str, torch.Tensor]:
#         # Get embeddings
#         action_emb = self.action_embed(data.action)
#         object_emb = self.object_embed(data.object)
#         location_emb = self.location_embed(data.location)
        
#         # Concatenate to form node features
#         x = torch.cat([action_emb, object_emb, location_emb], dim=-1)
        
#         # GNN layers
#         x = F.relu(self.conv1(x, data.edge_index))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = F.relu(self.conv2(x, data.edge_index))
        
#         # Global pooling
#         batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)
#         x_mean = global_mean_pool(x, batch)
#         x_max = global_max_pool(x, batch)
#         x_graph = torch.cat([x_mean, x_max], dim=-1)
        
#         # Classification
#         logits = self.classifier(x_graph)
        
#         return {'crime_logits': logits, 'graph_embedding': x_graph}


# class SimpleTrainer:
#     """Simple trainer with proper batching."""
    
#     def __init__(self, model, lr=0.01, weight_decay=1e-4):
#         self.model = model
#         self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#         self.criterion = nn.CrossEntropyLoss()
#         self.history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
#         self.best_acc = 0
#         self.best_state = None
        
#     def train_epoch(self, graphs: List[Data]) -> Tuple[float, float]:
#         self.model.train()
#         total_loss = 0
#         correct = 0
        
#         indices = np.random.permutation(len(graphs))
        
#         for idx in indices:
#             data = graphs[idx]
#             self.optimizer.zero_grad()
            
#             out = self.model(data)
#             loss = self.criterion(out['crime_logits'], data.y)
            
#             loss.backward()
#             self.optimizer.step()
            
#             total_loss += loss.item()
#             pred = out['crime_logits'].argmax(dim=1)
#             correct += (pred == data.y).sum().item()
        
#         return total_loss / len(graphs), correct / len(graphs)
    
#     @torch.no_grad()
#     def evaluate(self, graphs: List[Data]) -> Tuple[float, float]:
#         self.model.eval()
#         total_loss = 0
#         correct = 0
        
#         for data in graphs:
#             out = self.model(data)
#             loss = self.criterion(out['crime_logits'], data.y)
#             total_loss += loss.item()
#             pred = out['crime_logits'].argmax(dim=1)
#             correct += (pred == data.y).sum().item()
        
#         return total_loss / len(graphs), correct / len(graphs)
    
#     def fit(self, train_graphs, val_graphs, epochs=300, patience=50, verbose=True):
#         no_improve = 0
        
#         for epoch in range(epochs):
#             train_loss, train_acc = self.train_epoch(train_graphs)
#             val_loss, val_acc = self.evaluate(val_graphs)
            
#             self.history['train_loss'].append(train_loss)
#             self.history['train_acc'].append(train_acc)
#             self.history['val_acc'].append(val_acc)
            
#             if val_acc > self.best_acc:
#                 self.best_acc = val_acc
#                 self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
#                 no_improve = 0
#             else:
#                 no_improve += 1
            
#             if verbose and epoch % 20 == 0:
#                 print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
#                       f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
            
#             if no_improve >= patience:
#                 if verbose:
#                     print(f"Early stopping at epoch {epoch}")
#                 break
        
#         if self.best_state:
#             self.model.load_state_dict(self.best_state)
        
#         return self.history


# def run_working_gnn(df: pd.DataFrame, verbose: bool = True):
#     """
#     Run the working GNN pipeline.
#     Returns model, builder, and test accuracy.
#     """
#     if verbose:
#         print("="*60)
#         print("WORKING CRIME GNN")
#         print("="*60)
    
#     # Build graphs
#     builder = SimpleGraphBuilder()
#     builder.fit(df)
#     graphs = builder.process_dataset(df)
    
#     if verbose:
#         print(f"Built {len(graphs)} graphs")
    
#     # Split data
#     train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
#     train_graphs, val_graphs = train_test_split(train_graphs, test_size=0.15, random_state=42)
    
#     if verbose:
#         print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
#     # Create model
#     model = WorkingCrimeGNN(
#         n_actions=builder.n_actions,
#         n_objects=builder.n_objects,
#         n_locations=builder.n_locations,
#         n_crimes=builder.n_crimes,
#         embed_dim=32,
#         hidden_dim=64
#     )
    
#     if verbose:
#         print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
#     # Train
#     trainer = SimpleTrainer(model, lr=0.01, weight_decay=1e-4)
#     history = trainer.fit(train_graphs, val_graphs, epochs=300, patience=50, verbose=verbose)
    
#     # Test
#     test_loss, test_acc = trainer.evaluate(test_graphs)
    
#     if verbose:
#         print(f"\n{'='*60}")
#         print(f"FINAL RESULTS")
#         print(f"{'='*60}")
#         print(f"Best Val Accuracy: {trainer.best_acc:.3f}")
#         print(f"Test Accuracy: {test_acc:.3f}")
    
#     return {
#         'model': model,
#         'builder': builder,
#         'trainer': trainer,
#         'test_acc': test_acc,
#         'history': history
#     }


# # ============================================================
# # COMPARISON WITH BASELINES
# # ============================================================

# def compare_all_models(df: pd.DataFrame):
#     """Run GNN and all baselines, return comparison."""
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.linear_model import LogisticRegression
    
#     print("="*70)
#     print("FULL MODEL COMPARISON")
#     print("="*70)
    
#     # 1. Run GNN
#     print("\n[1/4] Training GNN...")
#     gnn_results = run_working_gnn(df, verbose=False)
#     print(f"      GNN Test Accuracy: {gnn_results['test_acc']:.3f}")
    
#     # 2. Prepare tabular data for baselines
#     print("\n[2/4] Preparing baseline features...")
    
#     # Aggregate scene features
#     scene_features = []
#     scene_labels = []
    
#     for scene_id in df['scene_id'].unique():
#         scene = df[df['scene_id'] == scene_id]
        
#         # Feature: counts of each action type
#         action_counts = scene['action'].value_counts()
#         object_counts = scene['object'].value_counts()
        
#         # Create feature vector
#         features = {
#             'n_events': len(scene),
#             'n_unique_actions': scene['action'].nunique(),
#             'n_unique_objects': scene['object'].nunique(),
#             'n_unique_locations': scene['location'].nunique(),
#         }
        
#         # Add action presence features
#         for action in df['action'].unique():
#             features[f'action_{action}'] = int(action in scene['action'].values)
        
#         # Add object presence features  
#         for obj in df['object'].unique():
#             features[f'object_{obj}'] = int(obj in scene['object'].values)
        
#         scene_features.append(features)
#         scene_labels.append(scene['crime_type'].iloc[0])
    
#     feature_df = pd.DataFrame(scene_features)
#     X = feature_df.values
#     y = LabelEncoder().fit_transform(scene_labels)
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # 3. Random Forest
#     print("\n[3/4] Training Random Forest...")
#     rf = RandomForestClassifier(n_estimators=100, random_state=42)
#     rf.fit(X_train, y_train)
#     rf_acc = rf.score(X_test, y_test)
#     print(f"      Random Forest Accuracy: {rf_acc:.3f}")
    
#     # 4. Logistic Regression
#     print("\n[4/4] Training Logistic Regression...")
#     lr = LogisticRegression(max_iter=1000, random_state=42)
#     lr.fit(X_train, y_train)
#     lr_acc = lr.score(X_test, y_test)
#     print(f"      Logistic Regression Accuracy: {lr_acc:.3f}")
    
#     # Summary
#     print("\n" + "="*70)
#     print("COMPARISON SUMMARY")
#     print("="*70)
    
#     results = {
#         'GNN': gnn_results['test_acc'],
#         'Random Forest': rf_acc,
#         'Logistic Regression': lr_acc,
#         'Random Baseline': 1.0 / len(df['crime_type'].unique())
#     }
    
#     for model, acc in sorted(results.items(), key=lambda x: -x[1]):
#         bar = "█" * int(acc * 40)
#         print(f"{model:25s} | {acc:.3f} | {bar}")
    
#     return results


# if __name__ == "__main__":
#     from Architecture_2_generator import DataManager
    
#     # Generate data
#     print("Generating realistic crime data...")
#     dm = DataManager("/Users/rohitbogulla/Desktop/Sem 3/Applied ML 2/CrimeLens/data/realistic_crime_data.csv")
#     df = dm.getData()    
#     # Compare all models