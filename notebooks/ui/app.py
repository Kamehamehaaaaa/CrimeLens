import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import re
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# ---------------------------
# Robust repo path detection
# ---------------------------
THIS = Path(__file__).resolve()
# If running from repo_root/ui/app.py -> root is parents[1]
# If running from repo_root/notebooks/notebooks/ui/app.py -> root is parents[3]
CANDIDATES = [
    THIS.parents[1],           # repo_root if ui/ is at root
    THIS.parents[2],           # repo_root if extra level
    THIS.parents[3],           # repo_root if under notebooks/notebooks/ui
]
ROOT = None
for cand in CANDIDATES:
    if (cand / "models").exists() and (cand / "notebooks").exists():
        ROOT = cand
        break
if ROOT is None:
    # last resort: pick two levels up
    ROOT = THIS.parents[2]

MODELS = ROOT / "models"
RESULTS = ROOT / "results"
DATA_DIRS = [
    ROOT / "notebooks" / "data",
    ROOT / "data",
    ROOT / "notebooks" / "notebooks" / "data",
]

MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Model class (must match training)
# ---------------------------
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim=6, h=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, h)
        self.conv2 = SAGEConv(h, h)
        self.scorer = torch.nn.Sequential(
            torch.nn.Linear(2*h, h),
            torch.nn.ReLU(),
            torch.nn.Linear(h, 1)
        )
    def forward(self, x, edge_index):
        z = F.relu(self.conv1(x, edge_index))
        z = self.conv2(z, edge_index)
        return z
    def score(self, z, pairs):
        a = z[pairs[:,0]]
        b = z[pairs[:,1]]
        return self.scorer(torch.cat([a,b], dim=-1)).squeeze(-1)

# ---------------------------
# Tokenizer & feature builder (same as notebook 2)
# ---------------------------
STOP = set("""
THE A AN AND OF IN ON TO BY FOR WITH FROM THIS THAT THOSE THESE IS ARE WAS WERE IT ITS AS AT OR
""".split())

KEY_TERMS = {
    "suspect","witness","police","bank","check","forgery","detective","evidence","room","object",
    "parker","mabel","peabody","jury","arrest","signature","certificate","money","crime","trial","court"
}

def extract_tokens(text: str):
    toks = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", str(text))
    keep=[]
    for t in toks:
        if t.upper() == t and len(t)>3:
            continue
        if t[0].isupper() and t.upper() not in STOP:
            keep.append(t.lower())
        elif t.lower() in KEY_TERMS:
            keep.append(t.lower())
    return keep

def node_type(tok):
    if tok in {"mabel","parker","peabody","clark","lang","sweeney","patrick","mcmahon"}:
        return "person"
    if tok in {"suspect","witness","detective","jury","prisoner"}:
        return "role"
    if tok in {"bank","room","object","check","evidence","signature","court","money"}:
        return "artifact"
    return "other"

def find_transcript():
    for d in DATA_DIRS:
        f = d / "true_stories_transcript.csv"
        if f.exists():
            return f
    # fallback to sample if exists
    for d in DATA_DIRS:
        f = d / "sample_transcripts.csv"
        if f.exists():
            return f
    return None

def build_graph_and_features():
    csv = find_transcript()
    if csv is None:
        raise FileNotFoundError("Could not find transcript CSV in expected data folders.")
    df = pd.read_csv(csv)
    df["tokens"] = df["text"].map(extract_tokens)

    # co-occurrence
    window = 5
    co_counts = {}
    for i in range(len(df)):
        bag = set(itertools.chain.from_iterable(df.loc[i:min(i+window, len(df)-1), "tokens"].values))
        bag = sorted(bag)
        for a_idx in range(len(bag)):
            for b_idx in range(a_idx+1, len(bag)):
                k = (bag[a_idx], bag[b_idx])
                co_counts[k] = co_counts.get(k, 0) + 1

    G = nx.Graph()
    for (a,b),w in co_counts.items():
        if w >= 2:
            G.add_edge(a,b,weight=w)

    if G.number_of_nodes() > 0:
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(comps[0]).copy()

    nodes = sorted(G.nodes())
    nid = {n:i for i,n in enumerate(nodes)}

    # features
    types = [node_type(n) for n in nodes]
    type_vocab = sorted(set(types))
    t2i = {t:i for i,t in enumerate(type_vocab)}

    deg = np.array([G.degree(n) for n in nodes], dtype=float)
    deg_norm = (deg - deg.mean()) / (deg.std() + 1e-9)
    freq = {}
    for toks in df["tokens"]:
        for t in toks:
            freq[t] = freq.get(t,0)+1
    tf = np.array([freq.get(n,0) for n in nodes], dtype=float)
    tf = tf / (tf.max() + 1e-9)

    X = np.zeros((len(nodes), len(type_vocab)+2), dtype=np.float32)
    for i,t in enumerate(types):
        X[i, t2i[t]] = 1.0
    X[:, len(type_vocab)]   = deg_norm
    X[:, len(type_vocab)+1] = tf

    # edge_index
    ei = np.array([[nid[u], nid[v]] for u,v in G.edges()], dtype=np.int64).T
    if ei.size == 0:
        ei = np.empty((2,0), dtype=np.int64)

    return G, nodes, X, ei

# ---------------------------
# Load or rebuild artifacts
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_or_make_artifacts():
    nodes_path = MODELS / "nodes.npy"
    emb_path   = MODELS / "embeddings.npy"
    meta_path  = MODELS / "meta.json"
    weights = MODELS / "gnn_model.pt"

    if nodes_path.exists() and emb_path.exists():
        nodes = np.load(nodes_path, allow_pickle=True).tolist()
        emb = np.load(emb_path)
        # build a light graph for ego viz (optional)
        G = nx.Graph()
        G.add_nodes_from(nodes)
        return nodes, emb, G

    # rebuild from transcript + model
    G, nodes, X, edge_index = build_graph_and_features()
    in_dim = X.shape[1]

    if not weights.exists():
        # no weights; just use normalized X as "embeddings"
        emb = (X - X.mean(0)) / (X.std(0) + 1e-9)
    else:
        # load model and compute embeddings
        # try meta.json for hidden size; else default 64
        h = 64
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                h = int(meta.get("hidden_dim", 64))
            except Exception:
                pass
        model = GraphSAGE(in_dim=in_dim, h=h)
        state = torch.load(weights, map_location="cpu")
        # allow relaxed loading if shapes differ slightly (older checkpoints)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            pass  # fine — we only need conv layers for embeddings
        x_t = torch.from_numpy(X)
        if edge_index.size == 0:
            emb = model.conv1(x_t, torch.empty((2,0), dtype=torch.long))
            emb = model.conv2(F.relu(emb), torch.empty((2,0), dtype=torch.long)).detach().numpy()
        else:
            ei_t = torch.from_numpy(edge_index)
            with torch.no_grad():
                emb = model(x_t, ei_t).detach().numpy()

    np.save(nodes_path, np.array(nodes, dtype=object))
    np.save(emb_path, emb)
    # write a minimal meta
    try:
        json.dump({"in_dim": int(in_dim), "hidden_dim": int(emb.shape[1])}, open(meta_path, "w"))
    except Exception:
        pass

    # lightweight graph for viz
    H = nx.Graph()
    H.add_nodes_from(nodes)
    return nodes, emb, H

nodes, emb, G = load_or_make_artifacts()
nid = {n:i for i,n in enumerate(nodes)}

# ---------------------------
# UI
# ---------------------------
st.title("🕵️‍♀️ CrimeLens — Interactive Explorer")
st.caption(f"Artifacts folder: `{MODELS}`")

q = st.text_input("Entity (exact token, e.g., mabel, crime, parker)", value="mabel").strip().lower()
k = st.slider("Top-K neighbors", 3, 25, 12)

if q:
    if q not in nid:
        st.warning(f"‘{q}’ not in vocabulary. Try: mabel, parker, crime, court, evidence, bank, jury…")
    else:
        qi = nid[q]
        v = emb[qi]
        sims = (emb @ v) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(v) + 1e-9)
        order = np.argsort(sims)[::-1]
        neigh_idx = [i for i in order if i != qi][:k]
        table = [(nodes[i], float(sims[i])) for i in neigh_idx]
        st.subheader(f"Nearest to “{q}”")
        st.dataframe({"entity":[t for t,_ in table], "cosine":[s for _,s in table]})

        # ego graph (connect neighbors above median sim)
        H = nx.Graph()
        H.add_node(q)
        for i, s in table:
            H.add_node(i)
            H.add_edge(q, i, w=s)
        thr = max(0.0, float(np.median([s for _,s in table]) - 0.05))
        idxs = [nid[t] for t,_ in table]
        for a in range(len(idxs)):
            for b in range(a+1, len(idxs)):
                s = float((emb[idxs[a]] @ emb[idxs[b]]) /
                          (np.linalg.norm(emb[idxs[a]])*np.linalg.norm(emb[idxs[b]]) + 1e-9))
                if s >= thr:
                    H.add_edge(nodes[idxs[a]], nodes[idxs[b]], w=s)

        fig = plt.figure(figsize=(8,6))
        pos = nx.spring_layout(H, seed=42)
        nx.draw_networkx_nodes(H, pos, node_color="#89CFF0",
                               node_size=[300 + 120*(H.degree(n)) for n in H.nodes()])
        nx.draw_networkx_edges(H, pos, alpha=0.4)
        nx.draw_networkx_labels(H, pos, font_size=9)
        plt.axis("off")
        st.pyplot(fig)

st.markdown("---")
st.caption("If `nodes.npy` / `embeddings.npy` are missing, the app rebuilds them from the transcript and model weights.")