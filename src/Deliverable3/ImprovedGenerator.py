"""
Improved Crime GNN with better architecture and training
Fixes issues with the original HeteroGNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, global_mean_pool, global_add_pool
from torch_geometric.utils import to_undirected
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional


# ============================================================
# IMPROVED GRAPH BUILDER - Homogeneous Graph (More Stable)
# ============================================================

class ImprovedGraphBuilder:
    """
    Builds a homogeneous graph instead of heterogeneous.
    More stable for training and often performs better.
    
    All entities become nodes with type embeddings.
    All relationships become edges with type embeddings.
    """
    
    def __init__(self):
        self.node_encoder = LabelEncoder()
        self.node_type_encoder = LabelEncoder()
        self.edge_type_encoder = LabelEncoder()
        self.crime_type_encoder = LabelEncoder()
        self.action_encoder = LabelEncoder()
        self.fitted = False
        
    def fit(self, df: pd.DataFrame):
        """Learn all unique entities from the dataset."""
        # Collect all nodes (persons, objects, locations)
        all_persons = pd.concat([df['suspect'], df['victim']]).unique().tolist()
        all_objects = df['object'].unique().tolist()
        all_locations = df['location'].unique().tolist()
        
        all_nodes = all_persons + all_objects + all_locations + ['<UNK>']
        self.node_encoder.fit(all_nodes)
        
        # Node types
        self.node_type_encoder.fit(['person', 'object', 'location'])
        
        # Edge types
        all_actions = df['action'].unique().tolist()
        edge_types = all_actions + ['uses', 'at', 'involves', '<UNK>']
        self.edge_type_encoder.fit(edge_types)
        
        # Crime types
        self.crime_type_encoder.fit(df['crime_type'].unique())
        
        # Actions for edge features
        self.action_encoder.fit(all_actions + ['<UNK>'])
        
        self.n_nodes = len(self.node_encoder.classes_)
        self.n_node_types = len(self.node_type_encoder.classes_)
        self.n_edge_types = len(self.edge_type_encoder.classes_)
        self.n_crime_types = len(self.crime_type_encoder.classes_)
        self.n_actions = len(self.action_encoder.classes_)
        
        # Store node type mapping
        self.node_to_type = {}
        for p in all_persons:
            self.node_to_type[p] = 'person'
        for o in all_objects:
            self.node_to_type[o] = 'object'
        for l in all_locations:
            self.node_to_type[l] = 'location'
        
        self.fitted = True
        
        print(f"Graph Builder fitted:")
        print(f"  Nodes: {self.n_nodes}, Node types: {self.n_node_types}")
        print(f"  Edge types: {self.n_edge_types}, Crime types: {self.n_crime_types}")
        
    def _safe_encode(self, encoder: LabelEncoder, value: str) -> int:
        """Encode value, using <UNK> for unseen values."""
        try:
            return encoder.transform([value])[0]
        except ValueError:
            if '<UNK>' in encoder.classes_:
                return encoder.transform(['<UNK>'])[0]
            return 0
    
    def build_graph(self, scene_df: pd.DataFrame) -> Data:
        """
        Build a homogeneous graph from a crime scene.
        
        Nodes: All unique persons, objects, locations in the scene
        Edges: Actions between persons, uses (person->object), at (person->location)
        """
        # Collect unique entities in this scene
        persons = list(pd.concat([scene_df['suspect'], scene_df['victim']]).unique())
        objects = list(scene_df['object'].unique())
        locations = list(scene_df['location'].unique())
        
        all_entities = persons + objects + locations
        entity_to_local_idx = {e: i for i, e in enumerate(all_entities)}
        
        # Node features: [node_id, node_type]
        node_ids = []
        node_types = []
        
        for entity in all_entities:
            node_ids.append(self._safe_encode(self.node_encoder, entity))
            node_type = self.node_to_type.get(entity, 'object')
            node_types.append(self._safe_encode(self.node_type_encoder, node_type))
        
        # Build edges
        edge_src = []
        edge_dst = []
        edge_types = []
        edge_actions = []
        
        for _, event in scene_df.iterrows():
            suspect = event['suspect']
            victim = event['victim']
            obj = event['object']
            loc = event['location']
            action = event['action']
            
            sus_idx = entity_to_local_idx[suspect]
            vic_idx = entity_to_local_idx[victim]
            obj_idx = entity_to_local_idx[obj]
            loc_idx = entity_to_local_idx[loc]
            
            action_enc = self._safe_encode(self.action_encoder, action)
            
            # Edge 1: Suspect -> Victim (action)
            edge_src.append(sus_idx)
            edge_dst.append(vic_idx)
            edge_types.append(self._safe_encode(self.edge_type_encoder, action))
            edge_actions.append(action_enc)
            
            # Edge 2: Suspect -> Object (uses)
            edge_src.append(sus_idx)
            edge_dst.append(obj_idx)
            edge_types.append(self._safe_encode(self.edge_type_encoder, 'uses'))
            edge_actions.append(action_enc)
            
            # Edge 3: Suspect -> Location (at)
            edge_src.append(sus_idx)
            edge_dst.append(loc_idx)
            edge_types.append(self._safe_encode(self.edge_type_encoder, 'at'))
            edge_actions.append(action_enc)
            
            # Edge 4: Victim -> Location (at)
            edge_src.append(vic_idx)
            edge_dst.append(loc_idx)
            edge_types.append(self._safe_encode(self.edge_type_encoder, 'at'))
            edge_actions.append(action_enc)
            
            # Edge 5: Object -> Location (involves)
            edge_src.append(obj_idx)
            edge_dst.append(loc_idx)
            edge_types.append(self._safe_encode(self.edge_type_encoder, 'involves'))
            edge_actions.append(action_enc)
        
        # Create PyG Data object
        data = Data()
        
        # Node features
        data.x = torch.tensor(node_ids, dtype=torch.long)
        data.node_type = torch.tensor(node_types, dtype=torch.long)
        
        # Edge index (make bidirectional for better message passing)
        edge_index = torch.tensor([edge_src + edge_dst, edge_dst + edge_src], dtype=torch.long)
        data.edge_index = edge_index
        
        # Edge features (duplicate for bidirectional)
        data.edge_type = torch.tensor(edge_types + edge_types, dtype=torch.long)
        data.edge_action = torch.tensor(edge_actions + edge_actions, dtype=torch.long)
        
        # Graph label
        crime_type = scene_df['crime_type'].iloc[0]
        data.y = torch.tensor([self._safe_encode(self.crime_type_encoder, crime_type)], dtype=torch.long)
        
        # Batch index for pooling (single graph)
        data.batch = torch.zeros(len(all_entities), dtype=torch.long)
        
        # Metadata
        data.num_nodes = len(all_entities)
        data.metadata = {
            'entities': all_entities,
            'persons': persons,
            'objects': objects,
            'locations': locations,
            'suspect': scene_df['suspect'].iloc[0],
            'victim': scene_df['victim'].iloc[0]
        }
        
        return data
    
    def process_dataset(self, df: pd.DataFrame) -> List[Data]:
        """Convert entire dataset to list of graphs."""
        if not self.fitted:
            self.fit(df)
        
        graphs = []
        for scene_id in df['scene_id'].unique():
            scene_df = df[df['scene_id'] == scene_id]
            graph = self.build_graph(scene_df)
            graphs.append(graph)
        
        return graphs


# ============================================================
# IMPROVED GNN MODEL
# ============================================================

class ImprovedCrimeGNN(nn.Module):
    """
    Improved GNN with:
    - Node and edge type embeddings
    - Multiple GNN layer options (SAGE, GAT, GCN)
    - Better aggregation and readout
    - Residual connections
    """
    
    def __init__(self, 
                 n_nodes: int,
                 n_node_types: int,
                 n_edge_types: int,
                 n_actions: int,
                 n_crime_types: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 gnn_type: str = 'sage'):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_crime_types = n_crime_types
        
        # Embeddings
        self.node_embed = nn.Embedding(n_nodes, hidden_dim)
        self.node_type_embed = nn.Embedding(n_node_types, hidden_dim // 4)
        self.edge_type_embed = nn.Embedding(n_edge_types, hidden_dim // 4)
        self.action_embed = nn.Embedding(n_actions, hidden_dim // 4)
        
        # Initial projection
        self.input_proj = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if gnn_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Graph-level classifier
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_crime_types)
        )
        
        # Node-level suspect predictor
        self.suspect_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
    
    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        # Get embeddings
        x = self.node_embed(data.x)
        node_type_emb = self.node_type_embed(data.node_type)
        
        # Concatenate node features
        x = torch.cat([x, node_type_emb], dim=-1)
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Message passing with residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_new = conv(x, data.edge_index)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new  # Residual connection
        
        # Graph-level readout (mean + max pooling)
        graph_mean = global_mean_pool(x, data.batch)
        graph_max = global_mean_pool(x, data.batch)  # Can also use global_max_pool
        graph_embed = torch.cat([graph_mean, graph_max], dim=-1)
        
        # Predictions
        crime_logits = self.graph_classifier(graph_embed)
        
        # Suspect scores (only for person nodes)
        suspect_scores = self.suspect_predictor(x).squeeze(-1)
        
        return {
            'crime_logits': crime_logits,
            'suspect_scores': suspect_scores,
            'node_embeddings': x,
            'graph_embedding': graph_embed
        }


# ============================================================
# IMPROVED TRAINER
# ============================================================

class ImprovedTrainer:
    """Trainer with better optimization and monitoring."""
    
    def __init__(self, model: ImprovedCrimeGNN, lr: float = 0.001, weight_decay: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10
        )
        self.criterion = nn.CrossEntropyLoss()
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.best_val_acc = 0
        self.best_state = None
    
    def train_epoch(self, graphs: List[Data]) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        
        # Shuffle graphs
        indices = np.random.permutation(len(graphs))
        
        for idx in indices:
            graph = graphs[idx]
            self.optimizer.zero_grad()
            
            output = self.model(graph)
            loss = self.criterion(output['crime_logits'], graph.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output['crime_logits'].argmax(dim=1)
            correct += (pred == graph.y).sum().item()
        
        return total_loss / len(graphs), correct / len(graphs)
    
    @torch.no_grad()
    def evaluate(self, graphs: List[Data]) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        
        for graph in graphs:
            output = self.model(graph)
            loss = self.criterion(output['crime_logits'], graph.y)
            total_loss += loss.item()
            
            pred = output['crime_logits'].argmax(dim=1)
            correct += (pred == graph.y).sum().item()
        
        return total_loss / len(graphs), correct / len(graphs)
    
    def fit(self, train_graphs: List[Data], val_graphs: List[Data],
            epochs: int = 200, patience: int = 30, verbose: bool = True):
        
        no_improve = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_graphs)
            val_loss, val_acc = self.evaluate(val_graphs)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_acc)
            
            # Best model tracking
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")
            
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if self.best_state:
            self.model.load_state_dict(self.best_state)
        
        print(f"\nBest Val Accuracy: {self.best_val_acc:.3f}")
        return self.history


# ============================================================
# QUICK TEST FUNCTION
# ============================================================

def test_improved_gnn(df: pd.DataFrame):
    """Quick test of the improved GNN."""
    from sklearn.model_selection import train_test_split
    
    print("="*60)
    print("TESTING IMPROVED GNN")
    print("="*60)
    
    # Build graphs
    builder = ImprovedGraphBuilder()
    builder.fit(df)
    graphs = builder.process_dataset(df)
    
    print(f"\nBuilt {len(graphs)} graphs")
    
    # Split
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    train_graphs, val_graphs = train_test_split(train_graphs, test_size=0.15, random_state=42)
    
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    # Create model
    model = ImprovedCrimeGNN(
        n_nodes=builder.n_nodes,
        n_node_types=builder.n_node_types,
        n_edge_types=builder.n_edge_types,
        n_actions=builder.n_actions,
        n_crime_types=builder.n_crime_types,
        hidden_dim=128,
        num_layers=3,
        dropout=0.3,
        gnn_type='sage'
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = ImprovedTrainer(model, lr=0.001)
    trainer.fit(train_graphs, val_graphs, epochs=200, patience=30)
    
    # Test
    test_loss, test_acc = trainer.evaluate(test_graphs)
    print(f"\nTest Accuracy: {test_acc:.3f}")
    
    return model, builder, trainer, test_acc


# if __name__ == "__main__":
#     # Test with sample data
#     from realistic_data_generator import generate_realistic_data
#     df = generate_realistic_data(n_scenes=200)
#     test_improved_gnn(df)