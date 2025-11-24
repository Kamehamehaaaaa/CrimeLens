"""
Enhanced CrimeGNN Trainer with Comprehensive Metrics Logging
Generates training curves and saves all metrics for reporting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional


class MetricsLogger:
    """Logs and visualizes training metrics."""
    
    def __init__(self, save_dir: str = "results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'val_ap': [],
            'suspect_auc': [],
            'suspect_ap': []
        }
        
    def log(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for an epoch."""
        self.history['epoch'].append(epoch)
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def save_history(self, filename: str = "training_history.csv"):
        """Save metrics history to CSV."""
        df = pd.DataFrame(self.history)
        path = os.path.join(self.save_dir, filename)
        df.to_csv(path, index=False)
        print(f"Training history saved to {path}")
        return df
    
    def save_json(self, filename: str = "training_history.json"):
        """Save metrics history to JSON."""
        path = os.path.join(self.save_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {path}")
    
    def plot_training_curves(self, save: bool = True, show: bool = True):
        """Generate publication-ready training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = self.history['epoch']
        
        # Plot 1: Loss curves
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if self.history['val_loss'] and self.history['val_loss'][0] is not None:
            ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        ax2 = axes[0, 1]
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        if self.history['val_acc'] and self.history['val_acc'][0] is not None:
            ax2.plot(epochs, self.history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Crime Classification Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Plot 3: AUC curves
        ax3 = axes[1, 0]
        if self.history['val_auc'] and self.history['val_auc'][0] is not None:
            ax3.plot(epochs, self.history['val_auc'], 'g-', label='Crime Type AUC', linewidth=2)
        if self.history['suspect_auc'] and self.history['suspect_auc'][0] is not None:
            ax3.plot(epochs, self.history['suspect_auc'], 'm-', label='Suspect Ranking AUC', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('AUC-ROC', fontsize=12)
        ax3.set_title('Validation AUC Scores', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Plot 4: Average Precision curves
        ax4 = axes[1, 1]
        if self.history['val_ap'] and self.history['val_ap'][0] is not None:
            ax4.plot(epochs, self.history['val_ap'], 'g--', label='Crime Type AP', linewidth=2)
        if self.history['suspect_ap'] and self.history['suspect_ap'][0] is not None:
            ax4.plot(epochs, self.history['suspect_ap'], 'm--', label='Suspect Ranking AP', linewidth=2)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Average Precision', fontsize=12)
        ax4.set_title('Validation Average Precision', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.suptitle('CrimeLens GNN Training Progress', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            path = os.path.join(self.save_dir, "training_curves.png")
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to {path}")
        
        if show:
            plt.show()
        
        return fig


class EnhancedCrimeGNNTrainer:
    """Enhanced trainer with comprehensive metrics logging."""
    
    def __init__(self, model, graph_builder, lr: float = 0.001, 
                 weight_decay: float = 1e-4, save_dir: str = "results"):
        self.model = model
        self.graph_builder = graph_builder
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.logger = MetricsLogger(save_dir)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def _compute_metrics(self, graphs: List[HeteroData], 
                         return_predictions: bool = False) -> Dict[str, float]:
        """Compute comprehensive metrics on a set of graphs."""
        self.model.eval()
        
        all_crime_preds = []
        all_crime_labels = []
        all_suspect_scores = []
        all_suspect_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for graph in graphs:
                output = self.model(graph)
                
                # Crime classification
                crime_logits = output['crime_logits']
                crime_label = graph.y
                loss = self.criterion(crime_logits, crime_label)
                total_loss += loss.item()
                
                crime_pred = crime_logits.argmax(dim=1).item()
                crime_probs = F.softmax(crime_logits, dim=1).squeeze().numpy()
                
                all_crime_preds.append(crime_pred)
                all_crime_labels.append(crime_label.item())
                
                # Suspect scoring (binary: is this person the actual suspect?)
                suspect_scores = torch.sigmoid(output['suspect_scores']).numpy()
                persons = graph.metadata_dict['persons']
                actual_suspect = graph.metadata_dict['suspect']
                
                for i, person in enumerate(persons):
                    all_suspect_scores.append(suspect_scores[i])
                    all_suspect_labels.append(1 if person == actual_suspect else 0)
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / len(graphs),
            'acc': accuracy_score(all_crime_labels, all_crime_preds)
        }
        
        # AUC and AP for suspect ranking (if we have both classes)
        if len(set(all_suspect_labels)) > 1:
            metrics['suspect_auc'] = roc_auc_score(all_suspect_labels, all_suspect_scores)
            metrics['suspect_ap'] = average_precision_score(all_suspect_labels, all_suspect_scores)
        else:
            metrics['suspect_auc'] = None
            metrics['suspect_ap'] = None
        
        # Multi-class AUC for crime type (if possible)
        if len(set(all_crime_labels)) > 1:
            try:
                # One-vs-rest AUC
                from sklearn.preprocessing import label_binarize
                n_classes = len(self.graph_builder.entity_encoders['crime_type'].classes_)
                crime_labels_bin = label_binarize(all_crime_labels, classes=range(n_classes))
                if crime_labels_bin.shape[1] > 1:
                    metrics['crime_auc'] = roc_auc_score(
                        crime_labels_bin, 
                        np.eye(n_classes)[all_crime_preds],
                        average='weighted',
                        multi_class='ovr'
                    )
                else:
                    metrics['crime_auc'] = None
            except:
                metrics['crime_auc'] = None
        else:
            metrics['crime_auc'] = None
            
        if return_predictions:
            return metrics, {
                'crime_preds': all_crime_preds,
                'crime_labels': all_crime_labels,
                'suspect_scores': all_suspect_scores,
                'suspect_labels': all_suspect_labels
            }
        
        return metrics
    
    def train_epoch(self, graphs: List[HeteroData]) -> Tuple[float, float]:
        """Train for one epoch, return loss and accuracy."""
        self.model.train()
        total_loss = 0
        correct = 0
        
        for graph in graphs:
            self.optimizer.zero_grad()
            output = self.model(graph)
            
            loss = self.criterion(output['crime_logits'], graph.y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output['crime_logits'].argmax(dim=1)
            correct += (pred == graph.y).sum().item()
        
        return total_loss / len(graphs), correct / len(graphs)
    
    def fit(self, train_graphs: List[HeteroData], 
            val_graphs: List[HeteroData] = None,
            epochs: int = 100, 
            log_interval: int = 5,
            early_stopping_patience: int = 20,
            verbose: bool = True) -> Dict:
        """
        Train the model with comprehensive logging.
        
        Returns dict with best metrics and model state.
        """
        best_val_acc = 0
        best_epoch = 0
        patience_counter = 0
        
        print("="*60)
        print("TRAINING STARTED")
        print("="*60)
        print(f"Train graphs: {len(train_graphs)}, Val graphs: {len(val_graphs) if val_graphs else 0}")
        print(f"Epochs: {epochs}, Log interval: {log_interval}")
        print("="*60)
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_graphs)
            
            # Validation
            if val_graphs:
                val_metrics = self._compute_metrics(val_graphs)
                val_loss = val_metrics['loss']
                val_acc = val_metrics['acc']
                val_auc = val_metrics.get('crime_auc')
                val_ap = None  # Compute if needed
                suspect_auc = val_metrics.get('suspect_auc')
                suspect_ap = val_metrics.get('suspect_ap')
            else:
                val_loss = val_acc = val_auc = val_ap = suspect_auc = suspect_ap = None
            
            # Log metrics
            self.logger.log(epoch, {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'val_ap': val_ap,
                'suspect_auc': suspect_auc,
                'suspect_ap': suspect_ap
            })
            
            # Print progress
            if verbose and epoch % log_interval == 0:
                msg = f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f}"
                if val_graphs:
                    msg += f" | Val Acc: {val_acc:.3f}"
                    if suspect_auc:
                        msg += f" | Suspect AUC: {suspect_auc:.3f}"
                print(msg)
            
            # Early stopping & best model tracking
            if val_graphs and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch} (best epoch: {best_epoch})")
                break
        
        print("="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        # Final evaluation
        final_train = self._compute_metrics(train_graphs)
        print(f"\nFinal Train Metrics:")
        print(f"  Loss: {final_train['loss']:.4f}, Acc: {final_train['acc']:.3f}")
        
        if val_graphs:
            final_val = self._compute_metrics(val_graphs)
            print(f"\nFinal Validation Metrics:")
            print(f"  Loss: {final_val['loss']:.4f}, Acc: {final_val['acc']:.3f}")
            if final_val.get('suspect_auc'):
                print(f"  Suspect AUC: {final_val['suspect_auc']:.3f}, AP: {final_val['suspect_ap']:.3f}")
        
        # Save artifacts
        self.logger.save_history()
        self.logger.save_json()
        self.logger.plot_training_curves()
        
        # Save best model
        model_path = os.path.join(self.save_dir, "best_model.pt")
        torch.save(self.best_state, model_path)
        print(f"\nBest model saved to {model_path}")
        
        return {
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
            'final_train_metrics': final_train,
            'final_val_metrics': final_val if val_graphs else None,
            'history': self.logger.history
        }
    
    def evaluate_test(self, test_graphs: List[HeteroData]) -> Dict:
        """Evaluate on test set and return comprehensive metrics."""
        # Load best model
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)
        
        metrics, predictions = self._compute_metrics(test_graphs, return_predictions=True)
        
        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60)
        print(f"Crime Classification Accuracy: {metrics['acc']:.3f}")
        if metrics.get('suspect_auc'):
            print(f"Suspect Ranking AUC: {metrics['suspect_auc']:.3f}")
            print(f"Suspect Ranking AP: {metrics['suspect_ap']:.3f}")
        
        return {'metrics': metrics, 'predictions': predictions}