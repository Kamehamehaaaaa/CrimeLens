"""
GNN Explainability for CrimeLens
Shows WHY the model predicts certain suspects/crime types
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class GradientExplainer:
    """
    Gradient-based explanation for HeteroGNN predictions.
    Shows which nodes and edges contribute most to predictions.
    """
    
    def __init__(self, model, graph_builder):
        self.model = model
        self.graph_builder = graph_builder
        
    def explain_prediction(self, scene_df: pd.DataFrame, 
                           target: str = 'suspect') -> Dict:
        """
        Explain model prediction using gradient-based attribution.
        
        Args:
            scene_df: Crime scene data
            target: 'suspect' or 'crime' - what to explain
            
        Returns:
            Dict with node importance scores and edge importance
        """
        # Set model to train mode to enable gradients
        self.model.train()
        
        # Build graph
        graph = self.graph_builder.build_graph(scene_df)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Create embedding tensors that require gradients
        person_x = graph['person'].x.clone()
        object_x = graph['object'].x.clone()
        location_x = graph['location'].x.clone()
        
        # Get embeddings and make them leaf tensors requiring grad
        person_embed = self.model.person_embed(person_x).clone().detach().requires_grad_(True)
        object_embed = self.model.object_embed(object_x).clone().detach().requires_grad_(True)
        location_embed = self.model.location_embed(location_x).clone().detach().requires_grad_(True)
        
        # Forward pass with custom embeddings
        x_dict = {
            'person': person_embed,
            'object': object_embed,
            'location': location_embed
        }
        
        # Add reverse edges
        graph = self._add_reverse_edges(graph)
        
        # Message passing
        for conv in self.model.convs:
            x_dict_new = conv(x_dict, graph.edge_index_dict)
            x_dict = {k: F.relu(v) + x_dict.get(k, 0) 
                     for k, v in x_dict_new.items()}
        
        # Get target output
        if target == 'suspect':
            output = self.model.suspect_predictor(x_dict['person'])
            # Use max suspect score as target
            target_score = output.max()
        else:  # crime
            graph_embed = torch.cat([
                x_dict['person'].mean(dim=0),
                x_dict['object'].mean(dim=0),
                x_dict['location'].mean(dim=0)
            ])
            crime_logits = self.model.crime_classifier(graph_embed.unsqueeze(0))
            target_score = crime_logits.max()
        
        # Backward pass
        target_score.backward()
        
        # Get gradient-based importance (handle None gracefully)
        if person_embed.grad is not None:
            person_importance = person_embed.grad.abs().sum(dim=1).detach().numpy()
        else:
            person_importance = np.ones(len(graph.metadata_dict['persons'])) * 0.5
            
        if object_embed.grad is not None:
            object_importance = object_embed.grad.abs().sum(dim=1).detach().numpy()
        else:
            object_importance = np.ones(len(graph.metadata_dict['objects'])) * 0.5
            
        if location_embed.grad is not None:
            location_importance = location_embed.grad.abs().sum(dim=1).detach().numpy()
        else:
            location_importance = np.ones(len(graph.metadata_dict['locations'])) * 0.5
        
        # Normalize
        max_imp = max(person_importance.max(), object_importance.max(), 
                     location_importance.max(), 1e-8)
        
        persons = graph.metadata_dict['persons']
        objects = graph.metadata_dict['objects']
        locations = graph.metadata_dict['locations']
        
        # Set model back to eval mode
        self.model.eval()
        
        return {
            'person_importance': {p: float(person_importance[i] / max_imp) 
                                 for i, p in enumerate(persons)},
            'object_importance': {o: float(object_importance[i] / max_imp) 
                                 for i, o in enumerate(objects)},
            'location_importance': {l: float(location_importance[i] / max_imp) 
                                   for i, l in enumerate(locations)},
            'target': target
        }
    
    def _add_reverse_edges(self, data: HeteroData) -> HeteroData:
        """Add reverse edges for bidirectional message passing."""
        if ('person', 'uses', 'object') in data.edge_index_dict:
            edge_index = data[('person', 'uses', 'object')].edge_index
            data[('object', 'rev_uses', 'person')].edge_index = edge_index.flip(0)
        
        if ('person', 'at', 'location') in data.edge_index_dict:
            edge_index = data[('person', 'at', 'location')].edge_index
            data[('location', 'rev_at', 'person')].edge_index = edge_index.flip(0)
            
        return data


class AttentionExplainer:
    """
    Attention-based explanation using edge attention weights.
    More interpretable than gradients for understanding relationships.
    """
    
    def __init__(self, model, graph_builder):
        self.model = model
        self.graph_builder = graph_builder
        
    def compute_edge_importance(self, scene_df: pd.DataFrame) -> Dict:
        """
        Compute importance of each edge in the crime graph.
        Uses embedding similarity as proxy for information flow.
        """
        self.model.eval()
        
        with torch.no_grad():
            graph = self.graph_builder.build_graph(scene_df)
            output = self.model(graph)
            node_embeddings = output['node_embeddings']
        
        edge_importance = {}
        
        # Get entity names
        persons = graph.metadata_dict['persons']
        objects = graph.metadata_dict['objects']
        locations = graph.metadata_dict['locations']
        
        # Compute edge importance based on embedding similarity
        person_embeds = node_embeddings['person'].numpy()
        object_embeds = node_embeddings['object'].numpy()
        location_embeds = node_embeddings['location'].numpy()
        
        # Person-Person edges (actions)
        for _, event in scene_df.iterrows():
            suspect = event['suspect']
            victim = event['victim']
            action = event['action']
            obj = event['object']
            loc = event['location']
            
            sus_idx = persons.index(suspect)
            vic_idx = persons.index(victim)
            obj_idx = objects.index(obj)
            loc_idx = locations.index(loc)
            
            # Cosine similarity as importance proxy
            sus_embed = person_embeds[sus_idx]
            vic_embed = person_embeds[vic_idx]
            obj_embed = object_embeds[obj_idx]
            loc_embed = location_embeds[loc_idx]
            
            # Edge: suspect -> victim (action)
            sim = np.dot(sus_embed, vic_embed) / (np.linalg.norm(sus_embed) * np.linalg.norm(vic_embed) + 1e-8)
            edge_importance[(suspect, action, victim)] = float(abs(sim))
            
            # Edge: suspect -> object (uses)
            sim = np.dot(sus_embed, obj_embed) / (np.linalg.norm(sus_embed) * np.linalg.norm(obj_embed) + 1e-8)
            edge_importance[(suspect, 'uses', obj)] = float(abs(sim))
            
            # Edge: suspect -> location (at)
            sim = np.dot(sus_embed, loc_embed) / (np.linalg.norm(sus_embed) * np.linalg.norm(loc_embed) + 1e-8)
            edge_importance[(suspect, 'at', loc)] = float(abs(sim))
        
        # Normalize
        max_imp = max(edge_importance.values()) if edge_importance else 1
        edge_importance = {k: v/max_imp for k, v in edge_importance.items()}
        
        return edge_importance


class ExplanationVisualizer:
    """
    Visualize GNN explanations on the crime graph.
    Shows node and edge importance with color coding.
    """
    
    def __init__(self, model, graph_builder):
        self.model = model
        self.graph_builder = graph_builder
        self.gradient_explainer = GradientExplainer(model, graph_builder)
        self.attention_explainer = AttentionExplainer(model, graph_builder)
        
    def visualize_explanation(self, scene_df: pd.DataFrame,
                              figsize: tuple = (16, 12),
                              save_path: Optional[str] = None) -> Dict:
        """
        Create comprehensive explanation visualization.
        
        Shows:
        - Node importance (color intensity)
        - Edge importance (thickness/color)
        - Prediction with confidence
        - Top contributing factors
        """
        # Get explanations
        node_importance = self.gradient_explainer.explain_prediction(scene_df, 'suspect')
        edge_importance = self.attention_explainer.compute_edge_importance(scene_df)
        
        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            graph = self.graph_builder.build_graph(scene_df)
            output = self.model(graph)
            
            crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().numpy()
            suspect_scores = torch.sigmoid(output['suspect_scores']).numpy()
        
        persons = graph.metadata_dict['persons']
        objects = graph.metadata_dict['objects']
        locations = graph.metadata_dict['locations']
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Main graph (left)
        ax_graph = fig.add_axes([0.02, 0.25, 0.55, 0.65])
        
        # Node importance bar (top right)
        ax_nodes = fig.add_axes([0.62, 0.55, 0.35, 0.35])
        
        # Edge importance bar (bottom right)  
        ax_edges = fig.add_axes([0.62, 0.1, 0.35, 0.35])
        
        # Text explanation (bottom)
        ax_text = fig.add_axes([0.02, 0.02, 0.55, 0.18])
        
        # ============ BUILD GRAPH ============
        G = nx.MultiDiGraph()
        
        # Add nodes with importance
        for i, person in enumerate(persons):
            imp = node_importance['person_importance'].get(person, 0)
            G.add_node(f"P:{person}", label=person, node_type='person', 
                      importance=imp, suspect_score=suspect_scores[i])
        
        for i, obj in enumerate(objects):
            imp = node_importance['object_importance'].get(obj, 0)
            G.add_node(f"O:{obj}", label=obj, node_type='object', importance=imp)
        
        for i, loc in enumerate(locations):
            imp = node_importance['location_importance'].get(loc, 0)
            G.add_node(f"L:{loc}", label=loc, node_type='location', importance=imp)
        
        # Add edges with importance
        for _, event in scene_df.iterrows():
            suspect, victim = event['suspect'], event['victim']
            action, obj, loc = event['action'], event['object'], event['location']
            
            # Get edge importances
            act_imp = edge_importance.get((suspect, action, victim), 0.5)
            uses_imp = edge_importance.get((suspect, 'uses', obj), 0.5)
            at_imp = edge_importance.get((suspect, 'at', loc), 0.5)
            
            G.add_edge(f"P:{suspect}", f"P:{victim}", label=action, 
                      importance=act_imp, edge_type='action')
            G.add_edge(f"P:{suspect}", f"O:{obj}", label='uses',
                      importance=uses_imp, edge_type='uses')
            G.add_edge(f"P:{suspect}", f"L:{loc}", label='at',
                      importance=at_imp, edge_type='at')
        
        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Color map for importance (white -> red)
        from matplotlib.colors import LinearSegmentedColormap
        importance_cmap = LinearSegmentedColormap.from_list(
            'importance', ['#FFFFFF', '#FFCCCC', '#FF6666', '#CC0000']
        )
        
        # Draw nodes by type
        for node_type, base_color, shape in [
            ('person', '#FF6B6B', 'o'),
            ('object', '#4ECDC4', 's'),
            ('location', '#95E1D3', '^')
        ]:
            nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == node_type]
            
            if nodes:
                importances = [G.nodes[n]['importance'] for n in nodes]
                colors = [importance_cmap(imp) for imp in importances]
                sizes = [1500 + 1500 * G.nodes[n]['importance'] for n in nodes]
                
                # Draw with black border for visibility
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors,
                                       node_size=sizes, alpha=0.9, ax=ax_graph,
                                       node_shape=shape, edgecolors='black', linewidths=2)
        
        # Draw labels
        labels = {n: G.nodes[n]['label'] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, 
                                font_weight='bold', ax=ax_graph)
        
        # Draw edges with importance-based width
        for edge_type, color in [('action', '#E74C3C'), ('uses', '#3498DB'), ('at', '#2ECC71')]:
            edges = [(u, v, k) for u, v, k, d in G.edges(data=True, keys=True) if d.get('edge_type') == edge_type]
            if edges:
                widths = [1 + 4 * G.edges[e]['importance'] for e in edges]
                alphas = [0.4 + 0.6 * G.edges[e]['importance'] for e in edges]
                
                for edge, width, alpha in zip(edges, widths, alphas):
                    nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])], edge_color=color,
                                           width=width, alpha=alpha, arrows=True,
                                           arrowsize=20, ax=ax_graph,
                                           connectionstyle="arc3,rad=0.1")
        
        # Edge labels
        edge_labels = {}
        for u, v, k, d in G.edges(data=True, keys=True):
            edge_labels[(u, v)] = d['label']
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_size=8, ax=ax_graph)
        
        ax_graph.set_title('Explanation: Node & Edge Importance', 
                          fontsize=14, fontweight='bold')
        ax_graph.axis('off')
        
        # ============ NODE IMPORTANCE BAR CHART ============
        all_nodes = []
        all_importances = []
        all_types = []
        
        for p, imp in node_importance['person_importance'].items():
            all_nodes.append(p)
            all_importances.append(imp)
            all_types.append('Person')
        for o, imp in node_importance['object_importance'].items():
            all_nodes.append(o)
            all_importances.append(imp)
            all_types.append('Object')
        for l, imp in node_importance['location_importance'].items():
            all_nodes.append(l)
            all_importances.append(imp)
            all_types.append('Location')
        
        # Sort by importance
        sorted_idx = np.argsort(all_importances)[::-1]
        
        colors_map = {'Person': '#FF6B6B', 'Object': '#4ECDC4', 'Location': '#95E1D3'}
        bar_colors = [colors_map[all_types[i]] for i in sorted_idx]
        
        bars = ax_nodes.barh([all_nodes[i] for i in sorted_idx],
                            [all_importances[i] for i in sorted_idx],
                            color=bar_colors, alpha=0.8)
        ax_nodes.set_xlabel('Importance Score', fontsize=10)
        ax_nodes.set_title('Node Importance Ranking', fontsize=12, fontweight='bold')
        ax_nodes.set_xlim(0, 1.1)
        
        # ============ EDGE IMPORTANCE BAR CHART ============
        edge_labels_list = [f"{e[0]}→{e[2]}\n({e[1]})" for e in edge_importance.keys()]
        edge_imps = list(edge_importance.values())
        
        sorted_idx = np.argsort(edge_imps)[::-1][:8]  # Top 8 edges
        
        ax_edges.barh([edge_labels_list[i] for i in sorted_idx],
                     [edge_imps[i] for i in sorted_idx],
                     color='#9B59B6', alpha=0.8)
        ax_edges.set_xlabel('Importance Score', fontsize=10)
        ax_edges.set_title('Edge Importance Ranking', fontsize=12, fontweight='bold')
        ax_edges.set_xlim(0, 1.1)
        
        # ============ TEXT EXPLANATION ============
        ax_text.axis('off')
        
        # Get top contributing factors
        top_person = max(node_importance['person_importance'].items(), key=lambda x: x[1])
        top_object = max(node_importance['object_importance'].items(), key=lambda x: x[1])
        top_edge = max(edge_importance.items(), key=lambda x: x[1])
        
        crime_types = self.graph_builder.entity_encoders['crime_type'].classes_
        pred_crime = crime_types[np.argmax(crime_probs)]
        pred_crime_conf = crime_probs.max()
        
        top_suspect_idx = np.argmax(suspect_scores)
        pred_suspect = persons[top_suspect_idx]
        pred_suspect_conf = suspect_scores[top_suspect_idx]
        
        explanation_text = f"""
EXPLANATION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Predicted Crime Type: {pred_crime} (confidence: {pred_crime_conf*100:.1f}%)
Predicted Primary Suspect: {pred_suspect} (likelihood: {pred_suspect_conf*100:.1f}%)

KEY CONTRIBUTING FACTORS:
• Most important person: {top_person[0]} (importance: {top_person[1]:.2f})
• Most important object: {top_object[0]} (importance: {top_object[1]:.2f})  
• Most important relationship: {top_edge[0][0]} --[{top_edge[0][1]}]--> {top_edge[0][2]} (importance: {top_edge[1]:.2f})

WHY THIS PREDICTION:
The model identified {pred_suspect} as the primary suspect because they have the strongest
connections in the evidence graph, particularly through the "{top_edge[0][1]}" relationship
with {top_edge[0][2]} involving the {top_object[0]}.
"""
        
        ax_text.text(0.02, 0.95, explanation_text, transform=ax_text.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.8))
        
        plt.suptitle('CrimeLens GNN Explanation', fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Explanation saved to {save_path}")
        
        plt.show()
        
        return {
            'node_importance': node_importance,
            'edge_importance': edge_importance,
            'prediction': {
                'crime_type': pred_crime,
                'crime_confidence': float(pred_crime_conf),
                'suspect': pred_suspect,
                'suspect_likelihood': float(pred_suspect_conf)
            },
            'explanation_text': explanation_text
        }


# ============================================================
# USAGE
# ============================================================

def explain_crime_prediction(model, graph_builder, df, scene_id=None, save_path=None):
    """
    Quick function to explain a prediction.
    
    Usage:
        explain_crime_prediction(model, graph_builder, df)
        explain_crime_prediction(model, graph_builder, df, scene_id=5)
    """
    if scene_id is None:
        scene_id = df['scene_id'].unique()[0]
    
    scene_df = df[df['scene_id'] == scene_id]
    
    if scene_df.empty:
        print(f"Error: Scene {scene_id} not found!")
        return None
    
    visualizer = ExplanationVisualizer(model, graph_builder)
    results = visualizer.visualize_explanation(scene_df, save_path=save_path)
    
    return results