from MainRunner_2 import run_complete_pipeline
from Architecture_2_generator import DataManager

# Load your data
dm = DataManager("/Users/rohitbogulla/Desktop/Sem 3/Applied ML 2/CrimeLens/data/realistic_crime_data.csv")
df = dm.getData()

# Run everything
results = run_complete_pipeline(df, save_dir="results", epochs=200)


# from ImprovedGenerator import ImprovedGraphBuilder, ImprovedCrimeGNN, ImprovedTrainer, test_improved_gnn
# from SimpleGraphGenerator import SimpleGraphBuilder, SimpleTrainer, WorkingCrimeGNN, run_working_gnn
# # Generate data
# dm = DataManager("/Users/rohitbogulla/Desktop/Sem 3/Applied ML 2/CrimeLens/data/realistic_crime_data.csv")
# df = dm.getData()

# # Quick test
# model, builder, trainer, test_acc = run_working_gnn(df)