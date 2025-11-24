"""
CrimeLens Complete Pipeline
Run this script to:
1. Train HeteroGNN with metrics logging
2. Run baseline comparisons
3. Generate explanations
4. Save all artifacts for report
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from Architecture_2_generator import CrimeGraphBuilder, CrimeHeteroGNN  # Your existing code
from EnhancedTrainer import EnhancedCrimeGNNTrainer
from BaselineComparison import BaselineComparison
from Explainer import ExplanationVisualizer

from SimpleGraphGenerator import SimpleGraphBuilder, WorkingCrimeGNN

import warnings
warnings.filterwarnings('ignore')

# Import our modules (assuming they're in same directory or installed)
# from crime_gnn import CrimeGraphBuilder, CrimeHeteroGNN
# from enhanced_trainer import EnhancedCrimeGNNTrainer, MetricsLogger
# from baselines import BaselineComparison, FeatureExtractor
# from gnn_explainer import explain_crime_prediction, ExplanationVisualizer


def run_complete_pipeline(df: pd.DataFrame, 
                          save_dir: str = "results",
                          epochs: int = 150,
                          hidden_dim: int = 64,
                          test_size: float = 0.2,
                          val_size: float = 0.1):
    """
    Run the complete CrimeLens training and evaluation pipeline.
    
    Args:
        df: Crime scene DataFrame with columns:
            scene_id, event_id, crime_type, suspect, victim, object, location, action
        save_dir: Directory to save all artifacts
        epochs: Training epochs
        hidden_dim: GNN hidden dimension
        test_size: Fraction for test set
        val_size: Fraction for validation set
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*70)
    print("CRIMELENS COMPLETE PIPELINE")
    print("="*70)
    
    # ============================================================
    # STEP 1: Data Preparation
    # ============================================================
    print("\n[STEP 1] Data Preparation")
    print("-"*50)
    
    scene_ids = df['scene_id'].unique()
    print(f"Total scenes: {len(scene_ids)}")
    print(f"Total events: {len(df)}")
    print(f"Crime types: {df['crime_type'].unique().tolist()}")
    
    # Split by scene_id
    train_val_ids, test_ids = train_test_split(
        scene_ids, test_size=test_size, random_state=42
    )
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size/(1-test_size), random_state=42
    )
    
    train_df = df[df['scene_id'].isin(train_ids)]
    val_df = df[df['scene_id'].isin(val_ids)]
    test_df = df[df['scene_id'].isin(test_ids)]
    
    print(f"Train scenes: {len(train_ids)}, Val scenes: {len(val_ids)}, Test scenes: {len(test_ids)}")
    
    # ============================================================
    # STEP 2: Build Graphs
    # ============================================================
    print("\n[STEP 2] Building Graph Representations")
    print("-"*50)


    builder = SimpleGraphBuilder()
    builder.fit(df)
    graphs = builder.process_dataset(df)
    
    print(f"Built {len(graphs)} graphs")
    
    # Split data
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    train_graphs, val_graphs = train_test_split(train_graphs, test_size=0.15, random_state=42)
    
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

    graph_builder = builder
    
    # graph_builder = CrimeGraphBuilder()
    # graph_builder.fit(df)  # Fit on all data to get complete vocabulary
    
    # train_graphs = graph_builder.process_dataset(train_df)
    # val_graphs = graph_builder.process_dataset(val_df)
    # test_graphs = graph_builder.process_dataset(test_df)
    
    # print(f"Train graphs: {len(train_graphs)}")
    # print(f"Val graphs: {len(val_graphs)}")
    # print(f"Test graphs: {len(test_graphs)}")
    # print(f"Vocabulary sizes: {graph_builder.vocab_sizes}")
    
    # # ============================================================
    # # STEP 3: Train HeteroGNN
    # # ============================================================
    # print("\n[STEP 3] Training Heterogeneous GNN")
    # print("-"*50)
    
    # # model = CrimeHeteroGNN(
    # #     vocab_sizes=graph_builder.vocab_sizes,
    # #     hidden_dim=hidden_dim,
    # #     num_layers=3,
    # #     dropout=0.2
    # # )

    # model = WorkingCrimeGNN(
        
    # )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = EnhancedCrimeGNNTrainer(
        model=model,
        graph_builder=graph_builder,
        lr=0.01,
        weight_decay=1e-4,
        save_dir=save_dir
    )
    
    training_results = trainer.fit(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        epochs=epochs,
        log_interval=10,
        early_stopping_patience=30,
        verbose=True
    )
    
    # Test evaluation
    test_results = trainer.evaluate_test(test_graphs)
    
    # ============================================================
    # STEP 4: Run Baseline Comparisons
    # ============================================================
    print("\n[STEP 4] Running Baseline Comparisons")
    print("-"*50)
    
    baseline_comparison = BaselineComparison(df)
    baseline_df = baseline_comparison.run_all(train_df, test_df, verbose=True)
    
    # Add GNN results
    baseline_comparison.add_gnn_result(
        name="HeteroGNN (Ours)",
        accuracy=test_results['metrics']['acc'],
        suspect_auc=test_results['metrics'].get('suspect_auc'),
        suspect_ap=test_results['metrics'].get('suspect_ap')
    )
    
    # Save comparison table
    comparison_table = baseline_comparison.get_comparison_table()
    comparison_table.to_csv(os.path.join(save_dir, "model_comparison.csv"), index=False)
    print(f"\nComparison table saved to {os.path.join(save_dir, 'model_comparison.csv')}")
    
    # Plot comparison
    baseline_comparison.plot_comparison(save_path=os.path.join(save_dir, "model_comparison.png"))
    
    # ============================================================
    # STEP 5: Generate Explanations
    # ============================================================
    print("\n[STEP 5] Generating Model Explanations")
    print("-"*50)
    
    # Load best model
    model.load_state_dict(trainer.best_state)
    
    # Explain a test scene
    test_scene_id = test_ids[0]
    test_scene_df = df[df['scene_id'] == test_scene_id]
    
    print(f"Explaining prediction for Scene {test_scene_id}:")
    print(test_scene_df[['suspect', 'victim', 'object', 'action', 'location']].to_string())
    
    explainer = ExplanationVisualizer(model, graph_builder)
    explanation = explainer.visualize_explanation(
        test_scene_df,
        save_path=os.path.join(save_dir, "explanation_example.png")
    )
    
    # ============================================================
    # STEP 6: Generate Summary Report
    # ============================================================
    print("\n[STEP 6] Generating Summary Report")
    print("-"*50)
    
    summary = {
        'Dataset': {
            'Total Scenes': len(scene_ids),
            'Total Events': len(df),
            'Train/Val/Test Split': f"{len(train_ids)}/{len(val_ids)}/{len(test_ids)}",
            'Crime Types': len(df['crime_type'].unique()),
            'Unique Persons': len(pd.concat([df['suspect'], df['victim']]).unique()),
            'Unique Objects': df['object'].nunique(),
            'Unique Locations': df['location'].nunique()
        },
        'Model': {
            'Architecture': 'Heterogeneous GraphSAGE',
            'Hidden Dimension': hidden_dim,
            'Layers': 3,
            'Parameters': sum(p.numel() for p in model.parameters())
        },
        'Training': {
            'Epochs (actual)': training_results['best_epoch'],
            'Best Val Accuracy': f"{training_results['best_val_acc']:.3f}"
        },
        'Test Results': {
            'Crime Classification Accuracy': f"{test_results['metrics']['acc']:.3f}",
            'Suspect Ranking AUC': f"{test_results['metrics'].get('suspect_auc', 'N/A')}",
            'Suspect Ranking AP': f"{test_results['metrics'].get('suspect_ap', 'N/A')}"
        }
    }
    
    # Save summary
    summary_path = os.path.join(save_dir, "experiment_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CRIMELENS EXPERIMENT SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        for section, items in summary.items():
            f.write(f"\n{section}\n")
            f.write("-"*40 + "\n")
            for key, value in items.items():
                f.write(f"  {key}: {value}\n")
    
    print(f"Summary saved to {summary_path}")
    
    # ============================================================
    # FINAL OUTPUT
    # ============================================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nArtifacts saved to: {save_dir}/")
    print("  - training_history.csv")
    print("  - training_history.json")
    print("  - training_curves.png")
    print("  - best_model.pt")
    print("  - model_comparison.csv")
    print("  - model_comparison.png")
    print("  - explanation_example.png")
    print("  - experiment_summary.txt")
    
    print("\n" + "="*70)
    print("FINAL COMPARISON TABLE")
    print("="*70)
    print(comparison_table.to_string(index=False))
    
    return {
        'model': model,
        'graph_builder': graph_builder,
        'trainer': trainer,
        'training_results': training_results,
        'test_results': test_results,
        'baseline_comparison': comparison_table,
        'explanation': explanation,
        'summary': summary
    }


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Load your data
    # Option 1: From your DataManager
    # dm = DataManager("/path/to/data")
    # df = dm.getData()
    
    # Option 2: Sample data for testing
    data = {
        'scene_id': [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10],
        'event_id': [1,2]*10,
        'crime_type': ['homicide','homicide','robbery','robbery','assault','assault',
                      'robbery','robbery','homicide','homicide','assault','assault',
                      'robbery','robbery','homicide','homicide','assault','assault',
                      'robbery','robbery'],
        'suspect': ['John','John','Evan','Evan','Marcus','Marcus','Alice','Alice',
                   'Bob','Bob','Carol','Carol','Dan','Dan','Eve','Eve','Frank','Frank',
                   'Grace','Grace'],
        'victim': ['Maria','Maria','Nina','Nina','Tom','Tom','Sam','Sam',
                  'Lisa','Lisa','Mike','Mike','Nancy','Nancy','Oscar','Oscar',
                  'Paul','Paul','Quinn','Quinn'],
        'object': ['knife','knife','gun','wallet','bat','bat','knife','jewelry',
                  'poison','poison','fist','fist','crowbar','wallet','gun','gun',
                  'bat','bat','knife','knife'],
        'location': ['home','home','street','parking','bar','alley','home','store',
                    'office','office','club','club','warehouse','warehouse','park','park',
                    'gym','gym','apartment','apartment'],
        'action': ['stabbed','fled','threatened','robbed','attacked','fled',
                  'stabbed','stole','poisoned','fled','punched','fled',
                  'attacked','robbed','shot','fled','beat','fled','stabbed','fled'],
    }
    df = pd.DataFrame(data)
    
    # Run pipeline
    results = run_complete_pipeline(
        df=df,
        save_dir="results",
        epochs=150,
        hidden_dim=64
    )