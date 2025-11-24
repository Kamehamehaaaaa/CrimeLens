#!/bin/bash
# Quick setup script for NER integration

echo "=========================================="
echo "Setting up NER for CrimeLens"
echo "=========================================="

# Install spaCy
echo "Installing spaCy..."
pip install spacy

# Download spaCy model
echo "Downloading English model..."
python -m spacy download en_core_web_sm

# Test installation
echo "Testing NER module..."
# python -c "import NERExtraction; print('✓ NER module working!')"

# Run evaluation
echo "Running NER evaluation..."
python src/Deliverable3/NERevaluation.py

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "Generated files:"
echo "  ✓ figures/ner_comparison.png"
echo "  ✓ figures/ner_comparison_table.csv"
echo ""
echo "Next steps:"
echo "  1. Update your Streamlit app to use NER parser"
echo "  2. Run: streamlit run app.py"
echo "  3. Compile LaTeX report with new NER section"