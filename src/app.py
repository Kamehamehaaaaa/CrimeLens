# ui/app.py — CrimeLens Streamlit App (heuristics + optional GNN)
# ---------------------------------------------------------------
# Features:
# 1) Text → Triples (regex heuristics)
# 2) Optional CSV edges (source,relation,target)
# 3) Build/visualize knowledge graph (NetworkX)
# 4) Missing-link suggestions (Jaccard + Resource Allocation Index)
# 5) Toggle to use trained GraphSAGE if models/gnn_model.pt exists
#
# Notes:
# - GNN mode requires torch + torch_geometric installed AND the model checkpoint you saved during training.
# - If anything is missing/mismatched, the app falls back to heuristic suggestions with a clear message.

import re
import io
import json
import itertools as it
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

# ---- Optional GNN imports (guarded) ----
TORCH_OK, PYG_OK = True, True
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    TORCH_OK = False

try:
    # Only loaded if torch import worked
    if TORCH_OK:
        from torch_geometric.nn import SAGEConv
        from torch_geometric.utils import from_networkx
    else:
        PYG_OK = False
except Exception:
    PYG_OK = False


# =========================
# UI CONFIG
# =========================
st.set_page_config(page_title="CrimeLens", page_icon="🕵️‍♀️", layout="wide")
st.title("🕵️‍♀️ CrimeLens — Interactive Knowledge Graph + Link Prediction")

st.markdown(
    """
This prototype converts free-form crime text or tabular edges into a **knowledge graph**,
then suggests plausible **missing links**. You can toggle a **trained GNN** (GraphSAGE) if available.
"""
)

# =========================
# Helpers: Text → Triples
# =========================
ENTITY_TYPES = ["person", "object", "location", "org", "time", "unknown"]

def guess_entity_type(token: str) -> str:
    t = token.strip()
    if not t:
        return "unknown"
    # naive cues
    if re.search(r"\b(street|avenue|road|room|park|alley|cafe|station|lot|apartment)\b", t, re.I):
        return "location"
    if re.search(r"\b(inc|corp|company|police|dept|department|bank|store|agency)\b", t, re.I):
        return "org"
    if re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|am|pm|\d{1,2}:\d{2})\b", t, re.I):
        return "time"
    if re.search(r"\b(he|she|suspect|witness|investigator|officer|mr\.|ms\.|mrs\.)\b", t, re.I):
        return "person"
    if re.search(r"\b(gun|bag|object|knife|phone|car|vehicle|money|wallet|evidence|fingerprints)\b", t, re.I):
        return "object"
    # default
    return "unknown"

def clean_token(t: str) -> str:
    t = t.strip()
    t = re.sub(r'[^\w\s\-/:#]', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t[:64]

PATTERNS = [
    # subject saw object
    (re.compile(r"\b([A-Z][\w\- ]+)\s+saw\s+(?:a|the)?\s*([A-Z]?\w[\w\- ]+)", re.I), "saw"),
    # dropped
    (re.compile(r"\b([A-Z][\w\- ]+)\s+dropped\s+(?:a|the)?\s*([A-Z]?\w[\w\- ]+)", re.I), "dropped"),
    # found in
    (re.compile(r"\b([A-Z][\w\- ]+)\s+(?:was\s+)?found\s+in\s+([A-Z]?\w[\w\- ]+)", re.I), "found_in"),
    # near
    (re.compile(r"\b([A-Z][\w\- ]+)\s+(?:was\s+)?near\s+([A-Z]?\w[\w\- ]+)", re.I), "near"),
    # went to / at
    (re.compile(r"\b([A-Z][\w\- ]+)\s+went\s+to\s+([A-Z]?\w[\w\- ]+)", re.I), "at"),
    (re.compile(r"\b([A-Z][\w\- ]+)\s+at\s+([A-Z]?\w[\w\- ]+)", re.I), "at"),
]

def text_to_triples(text: str):
    triples = []
    for pat, rel in PATTERNS:
        for m in pat.finditer(text):
            a = clean_token(m.group(1))
            b = clean_token(m.group(2))
            if a and b and a.lower() != b.lower():
                triples.append((a, rel.lower(), b))
    # lightweight dedup
    triples = list(dict.fromkeys(triples))
    return triples

# =========================
# Build Graph + Features
# =========================
def triples_to_graph(triples, csv_df=None):
    G = nx.DiGraph()
    # text triples
    for s, r, o in triples:
        if s not in G:
            G.add_node(s, label=s, type=guess_entity_type(s))
        if o not in G:
            G.add_node(o, label=o, type=guess_entity_type(o))
        G.add_edge(s, o, relation=r)
    # csv edges
    if csv_df is not None and len(csv_df):
        cols = [c.lower().strip() for c in csv_df.columns]
        colmap = {c: i for i, c in enumerate(cols)}
        def getcol(name):
            # accept common variants
            for k in ["source", "src", "from", "u", "node_u"]:
                if k in colmap and name == "source":
                    return csv_df.iloc[:, colmap[k]]
            for k in ["target", "dst", "to", "v", "node_v"]:
                if k in colmap and name == "target":
                    return csv_df.iloc[:, colmap[k]]
            if "relation" in colmap and name == "relation":
                return csv_df.iloc[:, colmap["relation"]]
            return None

        s_col = getcol("source")
        t_col = getcol("target")
        r_col = getcol("relation")
        if s_col is not None and t_col is not None:
            for i in range(len(csv_df)):
                s = clean_token(str(s_col.iloc[i]))
                o = clean_token(str(t_col.iloc[i]))
                rel = str(r_col.iloc[i]).strip().lower() if r_col is not None else "related_to"
                if s and o and s.lower() != o.lower():
                    if s not in G:
                        G.add_node(s, label=s, type=guess_entity_type(s))
                    if o not in G:
                        G.add_node(o, label=o, type=guess_entity_type(o))
                    G.add_edge(s, o, relation=rel)
    return G

def nx_draw(G: nx.DiGraph):
    if G.number_of_nodes() == 0:
        st.info("Graph is empty.")
        return
    plt.figure(figsize=(7, 5))
    pos = nx.spring_layout(G, seed=42)
    # color by type
    colors = []
    for n, d in G.nodes(data=True):
        t = d.get("type", "unknown")
        palette = {
            "person": "#1f77b4",
            "object": "#ff7f0e",
            "location": "#2ca02c",
            "org": "#d62728",
            "time": "#9467bd",
            "unknown": "#7f7f7f",
        }
        colors.append(palette.get(t, "#7f7f7f"))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=600, alpha=0.9)
    nx.draw_networkx_labels(G, pos, {n: n for n in G.nodes()}, font_size=9)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", width=1.5, alpha=0.8)
    # edge labels (relation)
    edge_labels = {(u, v): d.get("relation", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.axis("off")
    st.pyplot(plt.gcf())
    plt.close()

def build_features(G: nx.DiGraph):
    """Return (nodes, nid_map, X_features) with simple structural + type features."""
    nodes = list(G.nodes())
    nid = {n: i for i, n in enumerate(nodes)}
    deg = np.array([G.degree(n) for n in nodes], dtype=np.float32).reshape(-1, 1)

    types = [G.nodes[n].get("type", "unknown") for n in nodes]
    type_to_idx = {t: i for i, t in enumerate(ENTITY_TYPES)}
    T = np.zeros((len(nodes), len(ENTITY_TYPES)), dtype=np.float32)
    for i, t in enumerate(types):
        T[i, type_to_idx.get(t, type_to_idx["unknown"])] = 1.0

    X = np.hstack([deg, T])  # [deg + one-hot types] = 1 + len(ENTITY_TYPES)
    return nodes, nid, X

def graph_to_edge_index(G: nx.DiGraph, nid: dict):
    # Undirected for link prediction embeddings
    und = G.to_undirected()
    edges = np.array(list(und.edges()), dtype=object)
    if edges.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    u = np.array([nid[a] for a, b in edges], dtype=np.int64)
    v = np.array([nid[b] for a, b in edges], dtype=np.int64)
    return np.vstack([u, v])

# =========================
# Heuristic Link Suggestion
# =========================
def heuristic_missing_links(G: nx.Graph, topk=10):
    und = G.to_undirected()
    preds = []
    # Jaccard
    try:
        for u, v, score in nx.jaccard_coefficient(und):
            preds.append((u, v, float(score), "jaccard"))
    except Exception:
        pass
    # Resource Allocation
    try:
        for u, v, score in nx.resource_allocation_index(und):
            preds.append((u, v, float(score), "resource_allocation"))
    except Exception:
        pass

    # Combine by max score per pair
    key = lambda p: tuple(sorted((p[0], p[1])))
    best = {}
    for u, v, s, m in preds:
        if (u, v) in und.edges or u == v:
            continue
        k = key((u, v))
        if k not in best or s > best[k][2]:
            best[k] = (u, v, s, m)
    ranked = sorted(best.values(), key=lambda x: x[2], reverse=True)[:topk]
    return ranked

# =========================
# GNN: GraphSAGE (optional)
# =========================
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.scorer = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        z = self.conv2(h, edge_index)
        return z

    def score_pairs(self, z, pairs_idx: torch.Tensor):
        a = z[pairs_idx[:, 0]]
        b = z[pairs_idx[:, 1]]
        return self.scorer(torch.cat([a, b], dim=-1)).squeeze(-1)

def gnn_missing_links(G: nx.DiGraph, nodes, nid, X, topk=10, model_path="models/gnn_model.pt"):
    """Returns (ranked_pairs, message). Falls back with message if something mismatches."""
    if not TORCH_OK or not PYG_OK:
        return None, "PyTorch / PyG not available; showing heuristic suggestions instead."
    if not Path(model_path).exists():
        return None, f"Model checkpoint not found at {model_path}; showing heuristic suggestions instead."

    in_dim = X.shape[1]
    try:
        model = GraphSAGE(in_dim=in_dim)
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
    except Exception as e:
        return None, f"Could not load model ({e}); showing heuristic suggestions instead."

    # Build edge_index and candidate non-edges
    edge_index = graph_to_edge_index(G, nid)
    x = torch.tensor(X, dtype=torch.float32)
    if edge_index.size == 0:
        return None, "Graph has no edges; cannot run GNN. Showing heuristic suggestions instead."

    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Embeddings
    with torch.no_grad():
        z = model(x, edge_index)

    und = G.to_undirected()
    all_pairs = []
    N = len(nodes)
    present = set(tuple(sorted(e)) for e in und.edges)
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) not in present:
                all_pairs.append((i, j))
    if not all_pairs:
        return [], "No candidate missing links."

    pairs_idx = torch.tensor(all_pairs, dtype=torch.long)
    with torch.no_grad():
        scores = model.score_pairs(z, pairs_idx).cpu().numpy()

    ranked_idx = np.argsort(-scores)[:topk]
    ranked = []
    for k in ranked_idx:
        i, j = all_pairs[k]
        ranked.append((nodes[i], nodes[j], float(scores[k]), "GraphSAGE"))
    return ranked, None

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Inputs")
text_input = st.sidebar.text_area(
    "Paste crime narrative text",
    height=180,
    placeholder="e.g., On 5th Avenue the witness saw a man drop a bag near the cafe. The object was found in Room 2.",
)

txt_file = st.sidebar.file_uploader("Or upload a .txt file", type=["txt"])
csv_file = st.sidebar.file_uploader("Optional edges CSV (source,relation,target)", type=["csv"])

col1, col2 = st.sidebar.columns(2)
use_gnn = col1.checkbox("Use trained GraphSAGE (if available)", value=False)
topk = col2.number_input("Top-K suggestions", min_value=3, max_value=50, value=10, step=1)

run = st.sidebar.button("Build Graph & Predict")

# =========================
# MAIN EXECUTION
# =========================
triples = []
csv_df = None

if run:
    # Load text
    text = text_input or ""
    if txt_file is not None:
        try:
            text = txt_file.read().decode("utf-8", errors="ignore")
        except Exception:
            text = ""

    # Parse to triples
    if text.strip():
        triples = text_to_triples(text)
        st.success(f"Extracted {len(triples)} triples from text.")
        if triples:
            st.dataframe(pd.DataFrame(triples, columns=["subject", "relation", "object"]))
    else:
        st.info("No text provided. You can still upload a CSV of edges below.")

    # CSV edges
    if csv_file is not None:
        try:
            csv_df = pd.read_csv(csv_file)
            st.success(f"Loaded CSV edges: {len(csv_df)} rows")
            st.dataframe(csv_df.head())
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            csv_df = None

    # Build graph
    G = triples_to_graph(triples, csv_df)
    st.subheader("📈 Knowledge Graph")
    st.caption(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    nx_draw(G)

    # Save graph JSON (for export)
    graph_json = {
        "nodes": [{"id": n, "type": G.nodes[n].get("type", "unknown")} for n in G.nodes()],
        "edges": [{"source": u, "target": v, "relation": d.get("relation", "")} for u, v, d in G.edges(data=True)],
    }
    st.download_button(
        "Download Graph JSON",
        data=json.dumps(graph_json, indent=2),
        file_name="crimelens_graph.json",
        mime="application/json",
        use_container_width=True
    )

    # Build features
    nodes, nid, X = build_features(G)

    # Predictions
    st.subheader("🔮 Missing Link Suggestions")
    predictions = None
    note = None

    if use_gnn:
        preds, note = gnn_missing_links(G, nodes, nid, X, topk=topk)
        if note:
            st.warning(note)
        if preds is not None:
            predictions = preds

    if predictions is None:
        # Heuristic fallback
        predictions = heuristic_missing_links(G, topk=topk)

    if predictions:
        df_pred = pd.DataFrame(predictions, columns=["node_u", "node_v", "score", "method"])
        st.dataframe(df_pred)
        # simple textual summary
        st.write("**Top suggestions:**")
        for u, v, s, m in predictions[: min(5, len(predictions))]:
            st.markdown(f"- Connect **{u} ↔ {v}** (score: `{s:.3f}`, method: {m})")
    else:
        st.info("No candidate missing links found (graph may already be dense).")

else:
    st.info("👈 Provide text and/or CSV, choose options, then click **Build Graph & Predict**.")


# =========================
# FOOTER
# =========================
st.markdown(
    """
---
**Tips**
- CSV format: columns like `source,relation,target` (case-insensitive).
- GNN toggle loads `models/gnn_model.pt`. If missing or incompatible, heuristics are used.
- Node features: degree + one-hot entity type (person/object/location/org/time/unknown).
"""
)
