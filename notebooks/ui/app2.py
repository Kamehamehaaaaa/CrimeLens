# ui/app.py — CrimeLens Prototype Workbench (updated)

import os, io, json, re, itertools, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn.functional as F

# ────────────────────────────────────────────────────────────────────────────────
# Optional deps
# ────────────────────────────────────────────────────────────────────────────────
try:
    import whisper  # pip install openai-whisper
    WHISPER_OK = True
except Exception:
    WHISPER_OK = False

try:
    from torch_geometric.nn import SAGEConv
    PYG_OK = True
except Exception:
    PYG_OK = False

st.set_page_config(page_title="CrimeLens", layout="wide")

# ────────────────────────────────────────────────────────────────────────────────
# Robust repo root detection
# ────────────────────────────────────────────────────────────────────────────────
THIS = Path(__file__).resolve()
CANDIDATES = [THIS.parents[i] for i in range(1, 6)]
ROOT = None
for cand in CANDIDATES:
    if (cand / "models").exists():
        ROOT = cand
        break
if ROOT is None:
    # fallback: …/notebooks
    ROOT = THIS.parents[2]

MODELS  = ROOT / "models"
RESULTS = ROOT / "results"
DATA_DIRS = [
    ROOT / "notebooks" / "data",
    ROOT / "data",
    ROOT / "notebooks" / "notebooks" / "data",
]

MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────
# Tokenizer & feature helpers
# ────────────────────────────────────────────────────────────────────────────────
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
        if t.upper() == t and len(t)>3:     # drop acronyms
            continue
        if t[0].isupper() and t.upper() not in STOP:
            keep.append(t.lower())
        elif t.lower() in KEY_TERMS:
            keep.append(t.lower())
    return keep

def node_type(tok: str):
    if tok in {"mabel","parker","peabody","clark","lang","sweeney","patrick","mcmahon"}:
        return "person"
    if tok in {"suspect","witness","detective","jury","prisoner"}:
        return "role"
    if tok in {"bank","room","object","check","evidence","signature","court","money"}:
        return "artifact"
    return "other"

# ────────────────────────────────────────────────────────────────────────────────
# GraphSAGE model (same layout as Notebook 2)
# ────────────────────────────────────────────────────────────────────────────────
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim=6, h=64):
        super().__init__()
        if not PYG_OK:
            raise RuntimeError("torch_geometric not available.")
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
    def score_pairs(self, z, pairs):
        a = z[pairs[:,0]]
        b = z[pairs[:,1]]
        return self.scorer(torch.cat([a,b], dim=-1)).squeeze(-1)  # logits

# ────────────────────────────────────────────────────────────────────────────────
# Transcript discovery & loading
# ────────────────────────────────────────────────────────────────────────────────
def find_transcript():
    for d in DATA_DIRS:
        f = d / "true_stories_transcript.csv"
        if f.exists():
            return f
    for d in DATA_DIRS:
        f = d / "sample_transcripts.csv"
        if f.exists():
            return f
    return None

@st.cache_data(show_spinner=False)
def load_base_dataframe():
    csv = find_transcript()
    if csv is None:
        return pd.DataFrame(columns=["text"])
    df = pd.read_csv(csv)
    if "text" not in df.columns:
        # minimal fallback
        return pd.DataFrame({"text": df.iloc[:,0].astype(str)})
    return df[["text"]]

# ────────────────────────────────────────────────────────────────────────────────
# Build graph + features from a dataframe (needs "text")
# ────────────────────────────────────────────────────────────────────────────────
def build_graph_and_features(df: pd.DataFrame):
    if "text" not in df.columns:
        raise ValueError("Input dataframe must have a 'text' column.")
    df = df.copy()
    df["tokens"] = df["text"].astype(str).map(extract_tokens)

    # Co-occurrence counts in sliding window
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

    # Features: one-hot type + degree z-score + term frequency
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

    edge_index = np.array([[nid[u], nid[v]] for u,v in G.edges()], dtype=np.int64).T
    if edge_index.size == 0:
        edge_index = np.empty((2,0), dtype=np.int64)

    meta = {
        "node_count": int(len(nodes)),
        "in_dim": int(X.shape[1]),
        "type_vocab": type_vocab,
    }
    return G, nodes, X, edge_index, meta

# ────────────────────────────────────────────────────────────────────────────────
# Save & load artifacts (NO nx.write_gpickle; use pickle instead)
# ────────────────────────────────────────────────────────────────────────────────
def save_artifacts(G, nodes, emb, meta):
    # 1) Edges CSV (for debugging/repro)
    edges_df = pd.DataFrame([(u, v, d.get("weight", 1.0)) for u, v, d in G.edges(data=True)],
                            columns=["u","v","weight"])
    edges_df.to_csv(RESULTS / "graph_edges.csv", index=False)

    # 2) Graph pickle
    with open(MODELS / "graph.pkl", "wb") as f:
        pickle.dump(G, f)

    # 3) Nodes & embeddings
    np.save(MODELS / "nodes.npy", np.array(nodes, dtype=object))
    np.save(MODELS / "embeddings.npy", emb)

    # 4) Meta JSON
    with open(MODELS / "meta.json", "w") as f:
        json.dump(meta, f)

def load_saved_graph():
    pkl = MODELS / "graph.pkl"
    nodes_npy = MODELS / "nodes.npy"
    emb_npy   = MODELS / "embeddings.npy"
    meta_json = MODELS / "meta.json"

    if pkl.exists() and nodes_npy.exists() and emb_npy.exists():
        with open(pkl, "rb") as f:
            G = pickle.load(f)
        nodes = np.load(nodes_npy, allow_pickle=True).tolist()
        emb = np.load(emb_npy)
        meta = {}
        if meta_json.exists():
            try:
                meta = json.load(open(meta_json))
            except Exception:
                meta = {}
        return True, (G, nodes, emb, meta)
    return False, None

def compute_embeddings_from_features(X, edge_index):
    """If model weights exist, use GraphSAGE; else z-scored X."""
    weights = MODELS / "gnn_model.pt"
    in_dim = X.shape[1]
    if PYG_OK and weights.exists():
        model = GraphSAGE(in_dim=in_dim, h=64)
        state = torch.load(weights, map_location="cpu")
        model.load_state_dict(state, strict=False)
        x_t = torch.from_numpy(X)
        if edge_index.size == 0:
            # empty graph → just pass through convs with empty edges
            ei = torch.empty((2,0), dtype=torch.long)
            with torch.no_grad():
                z = model(x_t, ei).detach().numpy()
        else:
            ei = torch.from_numpy(edge_index)
            with torch.no_grad():
                z = model(x_t, ei).detach().numpy()
        return z
    # fallback: normalized features
    return (X - X.mean(0)) / (X.std(0) + 1e-9)

# ────────────────────────────────────────────────────────────────────────────────
# Heatmap helper
# ────────────────────────────────────────────────────────────────────────────────
def plot_heatmap(G, topN=25):
    deg_sorted = sorted(G.degree, key=lambda x: -x[1])[:topN]
    labels = [n for n,_ in deg_sorted]
    idx = {n:i for i,n in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=float)
    for u,v,d in G.edges(data=True):
        if u in idx and v in idx:
            w = float(d.get("weight",1.0))
            M[idx[u], idx[v]] = w
            M[idx[v], idx[u]] = w
    fig = plt.figure(figsize=(7,6))
    plt.imshow(M, interpolation="nearest")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.title("Co-occurrence Heatmap (Top nodes)")
    plt.tight_layout()
    return fig

# ────────────────────────────────────────────────────────────────────────────────
# Whisper transcription (optional)
# ────────────────────────────────────────────────────────────────────────────────
def transcribe_audio(file_bytes: bytes, filename: str) -> str:
    if not WHISPER_OK:
        return ""
    audio_path = RESULTS / filename
    with open(audio_path, "wb") as f:
        f.write(file_bytes)
    model = whisper.load_model("base")
    out = model.transcribe(str(audio_path))
    return out.get("text","")

# ────────────────────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────────────────────
st.title("🕵️‍♀️ CrimeLens — Prototype Workbench")
tabs = st.tabs([
    "1) Input",
    "2) Hypotheses",
    "3) Graph View",
    "4) Heatmap",
    "5) Narrative & Provenance",
])

# Tab 1 — INPUT
with tabs[0]:
    st.subheader("Add Evidence")

    base_df = load_base_dataframe()
    st.caption(f"Loaded baseline transcript rows: {len(base_df)}")

    user_text = st.text_area(
        "Witness / officer notes (append to evidence)",
        height=120, placeholder="Type a paragraph…"
    )

    up_csv  = st.file_uploader("Upload CSV with a 'text' column",  type=["csv"])
    up_json = st.file_uploader("Upload JSON (list of objects with a 'text' field)", type=["json"])
    up_audio= st.file_uploader("Upload audio (optional) for transcription", type=["wav","mp3","m4a","flac","ogg"])

    ev_texts = []

    if up_csv is not None:
        try:
            dfu = pd.read_csv(up_csv)
            if "text" in dfu.columns:
                ev_texts.extend(dfu["text"].astype(str).tolist())
                st.success(f"Loaded {len(dfu)} rows from CSV.")
            else:
                st.warning("CSV must contain a 'text' column.")
        except Exception as e:
            st.error(f"CSV read error: {e}")

    if up_json is not None:
        try:
            js = json.load(io.TextIOWrapper(up_json, encoding="utf-8"))
            if isinstance(js, list):
                for obj in js:
                    if isinstance(obj, dict) and "text" in obj:
                        ev_texts.append(str(obj["text"]))
            elif isinstance(js, dict) and "text" in js:
                ev_texts.append(str(js["text"]))
            st.success(f"Parsed {len(ev_texts)} text entries from JSON.")
        except Exception as e:
            st.error(f"JSON read error: {e}")

    if up_audio is not None:
        if WHISPER_OK:
            with st.spinner("Transcribing audio…"):
                tx = transcribe_audio(up_audio.read(), up_audio.name)
            if tx.strip():
                st.info("Audio transcript preview:")
                st.write(tx[:500] + ("…" if len(tx)>500 else ""))
                ev_texts.append(tx)
        else:
            st.warning("Whisper not installed; skipping transcription.")

    if user_text.strip():
        ev_texts.append(user_text.strip())

    # combine with baseline
    if len(ev_texts) > 0:
        add_df = pd.DataFrame({"text": ev_texts})
        working_df = pd.concat([base_df[["text"]], add_df], ignore_index=True)
    else:
        working_df = base_df[["text"]].copy()

    st.session_state["working_df"] = working_df
    st.write("**Evidence rows in memory:**", len(working_df))

    # Build/Refresh
    if st.button("Build/Refresh Graph & Embeddings", type="primary"):
        G, nodes, X, edge_index, meta = build_graph_and_features(working_df)
        emb = compute_embeddings_from_features(X, edge_index)
        save_artifacts(G, nodes, emb, meta)
        st.session_state["G"] = G
        st.session_state["nodes"] = nodes
        st.session_state["emb"] = emb
        st.session_state["meta"] = meta
        st.success(f"Graph ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

# Ensure artifacts exist for other tabs
if "G" not in st.session_state:
    ok, pack = load_saved_graph()
    if ok:
        G, nodes, emb, meta = pack
    else:
        base_df = load_base_dataframe()
        G, nodes, X, edge_index, meta = build_graph_and_features(base_df)
        emb = compute_embeddings_from_features(X, edge_index)
        save_artifacts(G, nodes, emb, meta)
    st.session_state["G"] = G
    st.session_state["nodes"] = nodes
    st.session_state["emb"] = emb
    st.session_state["meta"] = meta

G    = st.session_state["G"]
nodes= st.session_state["nodes"]
emb  = st.session_state["emb"]
nid  = {n:i for i,n in enumerate(nodes)}

# Tab 2 — HYPOTHESES
with tabs[1]:
    st.subheader("Generate & Rank Hypotheses")
    topk = st.slider("How many candidate links?", 5, 50, 15)
    if len(nodes) < 2:
        st.warning("Graph is too small.")
    else:
        A = nx.to_numpy_array(G, nodelist=nodes)
        cand = np.transpose(np.where(np.triu(1 - A - np.eye(len(nodes)), k=1)))
        if cand.size == 0:
            st.info("No candidate non-edges to score.")
        else:
            vnorm = np.linalg.norm(emb, axis=1) + 1e-9
            sims = np.array([(emb[a] @ emb[b]) / (vnorm[a]*vnorm[b]) for a,b in cand], dtype=float)
            order = np.argsort(sims)[::-1][:topk]
            rows = [{
                "node_a": nodes[int(cand[i][0])],
                "node_b": nodes[int(cand[i][1])],
                "confidence": float(1/(1+np.exp(-5*sims[i]))),  # squashed (0,1)
            } for i in order]
            st.dataframe(pd.DataFrame(rows))
            if st.button("Save hypotheses CSV"):
                out_path = RESULTS / "hypotheses.csv"
                pd.DataFrame(rows).to_csv(out_path, index=False)
                st.success(f"Saved {out_path}")

# Tab 3 — GRAPH VIEW
with tabs[2]:
    st.subheader("Neighborhood Explorer")
    token = st.selectbox("Choose an entity", options=nodes, index=0 if nodes else 0)
    k = st.slider("Neighborhood size", 5, 40, 20)
    if token:
        qi = nid[token]
        v  = emb[qi]
        sims = (emb @ v) / (np.linalg.norm(emb, axis=1)*np.linalg.norm(v) + 1e-9)
        idx = np.argsort(sims)[::-1]
        neigh = [nodes[i] for i in idx if i != qi][:k]
        H = nx.Graph()
        H.add_node(token)
        for nb in neigh:
            H.add_edge(token, nb, w=1.0)
        fig = plt.figure(figsize=(8,6))
        pos = nx.spring_layout(H, seed=42)
        nx.draw_networkx_nodes(H, pos, node_color="#89CFF0",
                               node_size=[300+120*H.degree(n) for n in H.nodes()])
        nx.draw_networkx_edges(H, pos, alpha=0.5)
        nx.draw_networkx_labels(H, pos, font_size=9)
        plt.title(f"Neighborhood graph around '{token}'")
        plt.axis("off")
        st.pyplot(fig)

# Tab 4 — HEATMAP
with tabs[3]:
    st.subheader("2D Scene Layout — Co-occurrence Heatmap")
    topN = st.slider("Top-N nodes by degree", 10, 60, 25)
    if G.number_of_nodes() >= 2:
        fig = plot_heatmap(G, topN=topN)
        st.pyplot(fig)
        if st.button("Save heatmap PNG"):
            path = RESULTS / "cooccurrence_heatmap.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            st.success(f"Saved {path}")

# Tab 5 — NARRATIVE & PROVENANCE
with tabs[4]:
    st.subheader("Explainable Narrative & Provenance")
    c1, c2 = st.columns(2)
    with c1:
        src = st.selectbox("Source entity", options=nodes, index=0)
    with c2:
        dst = st.selectbox("Target entity", options=nodes, index=min(1, len(nodes)-1))
    if src and dst:
        try:
            path = nx.shortest_path(G, source=src, target=dst)
            st.markdown("**Provenance chain (shortest path):** " + " → ".join(path))
        except Exception:
            st.warning("No path found between selected nodes.")
            path = []
        if path:
            narrative = (
                f"In the current evidence graph, **{src}** is connected to **{dst}** "
                f"via {len(path)-1} step(s): " + " → ".join(path) + ". "
                "These links reflect co-occurrence in the transcript and any added notes, "
                "and may indicate a contextual relationship worth review."
            )
            st.write(narrative)
    st.caption(
        "Confidence comes from embedding similarity; when running with GraphSAGE weights, "
        "replace cosine with model logits → sigmoid for calibrated probabilities."
    )