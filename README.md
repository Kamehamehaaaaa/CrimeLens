# 🕵️‍♀️ CrimeLens — Intelligent Crime Scene Reconstruction

**Author:** Rohit Bogulla
**Course:** Applied Machine Learning II (EEE 6778)  
**Deliverable 1:** From Pitch to Prototype – Foundation and Design Blueprint  
**Date:** October 19 2025  

---

## 📖 Overview
CrimeLens transforms narrative and structured evidence into an interactive reasoning system.  
By combining **Whisper-based transcription**, **graph neural networks**, and **transformer reasoning**, the project reconstructs plausible crime-scene narratives and visual layouts from multimodal inputs such as podcasts, witness text, and open police records.

---

## 🧩 Repository Structure
```
CrimeLens/
├── data/                # raw or sample data
│   ├── sample_transcripts.csv
│   ├── chicago_incidents_sample.csv
│   └── synthetic_crime_scenes.csv
├── notebooks/           # Jupyter notebooks
│   ├── setup.ipynb              # verifies env, loads data, plots
│   ├── graph_quicklook.ipynb    # inspects JSON evidence graph
│   └── whisper_transcription.ipynb      # Whisper transcription
├── src/                 # helper scripts
│   ├── data_utils.py
│   └── graph_builder.py
│   └── syntheticData.py
├── ui/                  # Streamlit placeholder
│   └── app.py
├── results/             # exploratory outputs
├── docs/                # diagrams or visuals (project_idea, progress1)
├── requirements.txt     # minimal pip dependencies
└── README.md            # project overview
```
---

## ⚙️ Setup and Verification

### 1️⃣ Create Environment
```bash
pip install -r requirements.txt
```
### 2️⃣ Run Setup Notebook
```
jupyter notebook notebooks/setup.ipynb
```

✅ Expected output:
	•	Loads sample_transcripts.csv
	•	Prints summary stats (rows, episodes, duration)
	•	Displays a matplotlib histogram of segment durations
	
### 3️⃣ Preview UI (optional)
```
streamlit run ui/app.py
```
## 🧠 Data Sources

| **Type** | **Source** | **Usage** |
|-----------|-------------|-----------|
| Audio → Text | Public true-crime podcasts (e.g., *Serial* S1) → OpenAI Whisper transcripts | Training and testing narrative reasoning |
| Tabular | Open Chicago Police Incidents (sample CSV) | Structured evidence integration |
| Synthethic | Manually constructed crime scenes | Synthethic crime scene data |

## 🧠 Core Pipeline
Audio/Text Data
      ↓
Whisper Transcription + Parsing
      ↓
Graph Builder (JSON Relations)
      ↓
Graph Encoder (GNN / R-GCN)
      ↓
Temporal Transformer Reasoning
      ↓
Narrative & Scene Generator (LLM + Diffusion)
      ↓
Streamlit Interface for Visualization

🖥️ User Interface (Prototype)

The Streamlit dashboard will support:
	•	Uploading CSV/JSON evidence
	•	Generating and ranking hypotheses with confidence scores
	•	Visualizing 2D scene layouts (heatmaps)
	•	Reading explainable narratives and provenance chains
	
## 📆 Planned Timeline

| **Week** | **Focus** | **Expected Outcome** |
|-----------|------------|----------------------|
| Oct 20 – 26 | Data collection + Whisper pipeline | Working transcript and data loader |
| Oct 27 – Nov 2 | Baseline GNN + UI prototype | End-to-end demo running |
| Nov 3 – 16 | Model tuning + interpretability | Improved accuracy and visuals |
| Nov 17 – 30 | UI integration + testing | Stable interactive dashboard |
| Dec 1 – 11 | Final report and presentation | Complete CrimeLens demo |

---

## 🤖 Responsible AI Reflection
CrimeLens processes open, publicly available sources (e.g., podcast audio and public crime datasets) strictly for **educational and research purposes**.  
All transcripts are anonymized to protect privacy and avoid misrepresentation.  
The system does **not** make real-world investigative claims or decisions — its outputs are meant to demonstrate AI reasoning and visualization methods only.  
Provenance tracking, transparency, and interpretability are prioritized to ensure responsible development and reproducibility.

---

## 📚 References
1. OpenAI Whisper: *Robust Speech Recognition via Large-Scale Weak Supervision* (2022)  
2. Chicago Data Portal – *Crimes - 2001 to Present*  
3. Hamilton, W. et al. (2017). *Inductive Representation Learning on Large Graphs (GraphSAGE)*.  
4. Vaswani, A. et al. (2017). *Attention Is All You Need*.  
5. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*.

---

## ⚖️ License
This project is released under the **MIT License**.  
You are free to use, modify, and distribute this work for educational or research purposes, provided appropriate credit is given to the author.  
See the [LICENSE](LICENSE) file for full terms.
	
📫 Contact
Rohit Bogulla
📧 <rbogulla@ufl.edu>