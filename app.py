"""
app.py — Streamlit frontend for Rose's Memories.

Run:
    .venv/bin/streamlit run app.py
Requires:
    .venv/bin/uvicorn serve:app --reload
"""

from datetime import datetime

import requests
import streamlit as st
import streamlit.components.v1 as components

from config import API_URL

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ROSE // QUERY INTERFACE",
    page_icon="■",
    layout="wide",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
html, body, [class*="css"], .stApp {
    font-family: 'Courier New', Courier, monospace !important;
    background-color: #fff !important;
    color: #000 !important;
}
.block-container {
    padding: 2rem 2rem 1rem 2rem !important;
    max-width: 100% !important;
}
.stTextInput > div > div > input {
    background-color: #fff !important;
    color: #000 !important;
    border: 1px solid #000 !important;
    border-radius: 0 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 13px !important;
}
.stTextInput > div > div > input::placeholder { color: #bbb !important; }
.stTextInput > div > div > input:focus {
    box-shadow: none !important;
    border-color: #000 !important;
}
.stButton > button {
    background-color: #000 !important;
    color: #fff !important;
    border: 1px solid #000 !important;
    border-radius: 0 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 12px !important;
    font-weight: bold !important;
    letter-spacing: 2px !important;
    height: 42px !important;
}
.stButton > button:hover { background-color: #333 !important; }
.stSpinner > div { border-top-color: #000 !important; }
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def fetch_kg_graph() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/kg", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def esc(s: str) -> str:
    """Minimal HTML escape for user-visible strings."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── Session state ─────────────────────────────────────────────────────────────

if "kg_graph"   not in st.session_state:
    st.session_state.kg_graph   = fetch_kg_graph()
if "history"    not in st.session_state:
    st.session_state.history    = []
if "input_key"  not in st.session_state:
    st.session_state.input_key  = 0


# ── KG SVG builder ────────────────────────────────────────────────────────────

def build_kg_html(kg_graph: dict | None, highlighted_triples: list[dict]) -> str:
    if not kg_graph:
        return """<html><body style="margin:0;background:#fff;">
        <div style="color:#ccc;font-family:monospace;font-size:11px;padding:20px;
                    border:1px solid #eee;">KG NOT LOADED</div>
        </body></html>"""

    nodes = kg_graph["nodes"]
    edges = kg_graph["edges"]

    h_nodes: set[str] = set()
    h_edges: set[tuple] = set()
    for t in highlighted_triples:
        h_nodes.add(t["canonical_subject"])
        h_nodes.add(t["canonical_object"])
        h_edges.add((t["canonical_subject"], t["canonical_object"]))

    node_pos  = {n["id"]: (n["x"], n["y"]) for n in nodes}
    node_size = {n["id"]: n["size"]         for n in nodes}

    W, H, PAD = 900, 520, 28

    def sx(x): return PAD + x * (W - 2 * PAD)
    def sy(y): return PAD + y * (H - 2 * PAD)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 {W} {H}"'
        f' style="background:#fff;display:block;">'
    ]

    for lit in (False, True):
        for e in edges:
            s, t = e["source"], e["target"]
            if s not in node_pos or t not in node_pos:
                continue
            is_lit = (s, t) in h_edges
            if is_lit != lit:
                continue
            x1, y1 = sx(node_pos[s][0]), sy(node_pos[s][1])
            x2, y2 = sx(node_pos[t][0]), sy(node_pos[t][1])
            color  = "#000" if is_lit else "#e8e8e8"
            weight = "1.2"  if is_lit else "0.5"
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
                f' stroke="{color}" stroke-width="{weight}"/>'
            )
            if is_lit:
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                label  = esc(e.get("label", "")[:22])
                parts.append(
                    f'<text x="{mx:.1f}" y="{my - 3:.1f}" fill="#888" font-size="7"'
                    f' font-family="monospace" text-anchor="middle">{label}</text>'
                )

    for lit in (False, True):
        for n in nodes:
            nid = n["id"]
            if nid not in node_pos:
                continue
            is_lit = nid in h_nodes
            if is_lit != lit:
                continue
            x   = sx(n["x"])
            y   = sy(n["y"])
            sz  = node_size[nid]
            color = "#000" if is_lit else "#ddd"
            half  = sz / 2
            parts.append(
                f'<rect x="{x - half:.1f}" y="{y - half:.1f}"'
                f' width="{sz:.1f}" height="{sz:.1f}" fill="{color}"/>'
            )
            if is_lit:
                label = esc(nid[:20])
                parts.append(
                    f'<text x="{x:.1f}" y="{y - half - 3:.1f}" fill="#000"'
                    f' font-size="8" font-family="monospace" text-anchor="middle">{label}</text>'
                )

    parts.append('</svg>')
    return (
        '<html><body style="margin:0;padding:0;background:#fff;overflow:hidden;">'
        + "".join(parts)
        + "</body></html>"
    )


# ── History HTML builder ──────────────────────────────────────────────────────

def build_history_html(history: list[dict]) -> str:
    if not history:
        return """
        <html><body style="margin:0;padding:16px;background:#fff;
            font-family:'Courier New',monospace;color:#ccc;font-size:12px;">
        AWAITING QUERY...
        </body></html>"""

    entries_html = []
    for i, entry in enumerate(history):
        is_latest   = (i == 0)
        border_top  = "2px solid #000" if is_latest else "1px solid #e8e8e8"
        q_size      = "13px" if is_latest else "12px"
        ans_size    = "13px" if is_latest else "12px"
        log_bg      = "#f5f5f5" if is_latest else "#fafafa"
        log_border  = "#ccc"   if is_latest else "#e8e8e8"

        # Header
        html  = f'<div style="border-top:{border_top};padding:14px 0 0 0;margin-bottom:6px;">'
        html += (f'<span style="color:#aaa;font-size:10px;">[{esc(entry["ts"])}]</span> '
                 f'<span style="font-size:{q_size};font-weight:bold;">&gt; {esc(entry["question"])}</span>')
        html += '</div>'

        if "error" in entry:
            html += f'<div style="color:#c00;font-size:11px;margin-bottom:12px;">ERR: {esc(entry["error"])}</div>'
            entries_html.append(html)
            continue

        # Execution log
        log_lines = entry.get("log", [])
        if log_lines:
            log_inner = "".join(
                f'<div style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                f'<span style="color:#bbb;">[{j+1:02d}]</span>  {esc(line)}</div>'
                for j, line in enumerate(log_lines)
            )
            html += (
                f'<div style="background:{log_bg};border-left:2px solid {log_border};'
                f'padding:5px 10px;margin-bottom:8px;font-size:10px;color:#888;">'
                f'{log_inner}</div>'
            )

        # Answer
        html += (
            f'<div style="font-size:{ans_size};line-height:1.75;margin-bottom:8px;">'
            f'{esc(entry["answer"])}</div>'
        )

        # Retrieval map — all candidates colored by score, reranked survivors outlined
        candidates = entry.get("candidates", [])
        if candidates and is_latest:
            blocks = ""
            for c in candidates:
                score = c["score"]
                bg    = f"rgba(0,0,0,{score:.2f})"
                tc    = "#fff" if score > 0.5 else "#555"
                bdr   = "2px solid #000" if c.get("reranked") else "1px solid #e8e8e8"
                blocks += (
                    f'<div style="flex:0 0 calc(20% - 4px);min-width:0;height:70px;overflow:hidden;'
                    f'background:{bg};border:{bdr};padding:4px;box-sizing:border-box;">'
                    f'<div style="color:{tc};font-size:8px;opacity:0.8;white-space:nowrap;'
                    f'overflow:hidden;text-overflow:ellipsis;">{esc(c["source"])}</div>'
                    f'<div style="color:{tc};font-size:9px;font-weight:bold;margin:1px 0;">[{score:.2f}]</div>'
                    f'<div style="color:{tc};font-size:8px;opacity:0.7;overflow:hidden;'
                    f'display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;">'
                    f'{esc(c["text"][:55])}</div>'
                    f'</div>'
                )
            n_reranked = sum(1 for c in candidates if c.get("reranked"))
            html += (
                f'<div style="margin-bottom:8px;">'
                f'<div style="font-size:9px;color:#aaa;letter-spacing:2px;margin-bottom:4px;">'
                f'RETRIEVAL MAP &nbsp;<span style="font-weight:normal;letter-spacing:0;color:#bbb;">'
                f'{len(candidates)} RETRIEVED → {n_reranked} RERANKED &nbsp;'
                f'<span style="border:2px solid #000;padding:0 2px;color:#000;">■</span>'
                f' = RERANKED</span></div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:4px;">{blocks}</div>'
                f'</div>'
            )

        # KG triples used
        kg_triples = entry.get("kg_triples", [])
        if kg_triples and is_latest:
            triple_rows = "".join(
                f'<div style="border-top:1px solid #eee;padding:3px 0;font-size:10px;color:#555;">'
                f'{esc(t["canonical_subject"])} &nbsp;—&nbsp; '
                f'<span style="color:#aaa;">{esc(t["predicate"])}</span>'
                f' &nbsp;—&nbsp; {esc(t["canonical_object"])}'
                f'</div>'
                for t in kg_triples[:8]
            )
            more = len(kg_triples) - 8
            html += (
                f'<div style="margin-bottom:8px;">'
                f'<div style="font-size:9px;color:#aaa;letter-spacing:2px;margin-bottom:4px;">'
                f'KG TRIPLES USED</div>'
                f'{triple_rows}'
                + (f'<div style="font-size:9px;color:#ccc;padding-top:3px;">+{more} MORE</div>' if more > 0 else "")
                + '</div>'
            )

        # Meta footer
        meta = []
        if entry.get("sources"):
            meta.append("SRC: " + " | ".join(sorted(entry["sources"])))
        n_kg = len(kg_triples)
        if n_kg:
            meta.append(f"KG: {n_kg} TRIPLES")
        if meta:
            html += (
                f'<div style="font-size:10px;color:#bbb;margin-bottom:6px;letter-spacing:1px;">'
                + " &nbsp;/&nbsp; ".join(meta)
                + '</div>'
            )

        entries_html.append(html)

    body = "\n".join(entries_html)
    return f"""
    <html>
    <body style="margin:0;padding:0 16px 16px 0;background:#fff;
                 font-family:'Courier New',Courier,monospace;color:#000;
                 overflow-y:auto;box-sizing:border-box;">
    {body}
    </body>
    </html>"""


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="border-bottom:2px solid #000;padding-bottom:10px;margin-bottom:14px;">
  <span style="font-size:10px;color:#aaa;letter-spacing:3px;">PROJECT-ROSE //</span>
  <span style="font-size:18px;font-weight:bold;letter-spacing:4px;"> MEMORY QUERY INTERFACE</span>
</div>
""", unsafe_allow_html=True)

health = fetch_health()
if health:
    kg_label = "ENABLED" if health.get("kg") else "DISABLED"
    rl_str = ""
    if st.session_state.history and st.session_state.history[0].get("rate_limit"):
        rl = st.session_state.history[0]["rate_limit"]
        rl_color = "#000" if rl["pct"] > 20 else "#c00"
        rl_str = (
            f' &nbsp;/&nbsp; TPM: <span style="color:{rl_color}">{rl["pct"]}% REMAINING</span>'
            f' <span style="color:#ccc">({rl["remaining"]:,} / {rl["limit"]:,})</span>'
        )
    st.markdown(
        f'<div style="font-size:10px;color:#aaa;letter-spacing:1px;margin-bottom:14px;">'
        f'STATUS: <span style="color:#000">READY</span> &nbsp;/&nbsp; '
        f'INDEX: <span style="color:#000">{health["chunks"]} CHUNKS</span> &nbsp;/&nbsp; '
        f'KG: <span style="color:#000">{kg_label}</span>{rl_str}'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div style="font-size:10px;color:#c00;letter-spacing:1px;margin-bottom:14px;">'
        'STATUS: API OFFLINE — run: .venv/bin/uvicorn serve:app'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Query row ─────────────────────────────────────────────────────────────────

c1, c2 = st.columns([6, 1])
with c1:
    question = st.text_input(
        "",
        placeholder="ENTER QUERY...",
        label_visibility="collapsed",
        key=f"q_{st.session_state.input_key}",
    )
with c2:
    ask = st.button("EXECUTE", use_container_width=True)

st.markdown('<div style="border-top:1px solid #e8e8e8;margin:10px 0 14px 0;"></div>',
            unsafe_allow_html=True)

# ── Execute ───────────────────────────────────────────────────────────────────

if ask and question.strip():
    with st.spinner(""):
        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={"question": question},
                timeout=30,
            )
            resp.raise_for_status()
            entry = resp.json()
        except requests.exceptions.ConnectionError:
            entry = {"error": "API OFFLINE"}
        except Exception as e:
            entry = {"error": str(e)}

        entry["question"] = question
        entry["ts"]       = datetime.now().strftime("%H:%M:%S")
        st.session_state.history.insert(0, entry)

    st.session_state.input_key += 1   # clears the text input on rerun
    st.rerun()

# ── Two-column layout ─────────────────────────────────────────────────────────

col_left, col_right = st.columns([5, 7])

with col_left:
    st.markdown(
        '<div style="font-size:9px;color:#aaa;letter-spacing:3px;margin-bottom:6px;">QUERY LOG</div>',
        unsafe_allow_html=True,
    )
    components.html(
        build_history_html(st.session_state.history),
        height=620,
        scrolling=True,
    )

with col_right:
    st.markdown(
        '<div style="font-size:9px;color:#aaa;letter-spacing:3px;margin-bottom:6px;">KNOWLEDGE GRAPH</div>',
        unsafe_allow_html=True,
    )
    highlighted = []
    if st.session_state.history and "kg_triples" in st.session_state.history[0]:
        highlighted = st.session_state.history[0]["kg_triples"]

    components.html(
        build_kg_html(st.session_state.kg_graph, highlighted),
        height=548,
        scrolling=False,
    )
