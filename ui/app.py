import html
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Fact Layer", page_icon=None, layout="wide")
API_URL = "http://localhost:8000"

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: #f7f8fa; }
    [data-testid="stAppViewContainer"] .stMarkdown, [data-testid="stAppViewContainer"] .stCaption,
    [data-testid="stAppViewContainer"] label, [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] [data-baseweb="tab-list"] button,
    [data-testid="stAppViewContainer"] [data-baseweb="select"] * { color: #1f2937 !important; }
    [data-testid="stSidebar"] { background: #101827; }
    [data-testid="stSidebar"] * { color: #e7edf5; }
    [data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] > div { color: #111827 !important; background: #ffffff !important; }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: #1f2937 !important; }
    .hero { padding: 1.5rem 0 1rem; }
    .eyebrow { color: #2563eb; font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .hero h1 { color: #111827; font-size: 2.2rem; margin: .2rem 0; }
    .hero p { color: #667085; font-size: 1.05rem; }
    .case-card { background: white; border: 1px solid #e4e7ec; border-radius: 12px; padding: 1rem; min-height: 130px; }
    .case-card h3 { margin: 0; color: #111827; font-size: 1rem; }
    .case-card p { color: #667085; font-size: .9rem; }
    .evidence { border-left: 3px solid #2563eb; background: #f8fafc; padding: .7rem .9rem; margin: .5rem 0; color: #344054; }
    .label-text, .relationship-text, .empty-state, .provider-status { color: var(--text-color, #1f2937); }
    .evidence strong, .evidence em { color: #1f2937; }
    @media (prefers-color-scheme: dark) {
      .case-card, .evidence { background: #182230; border-color: #344054; }
      .case-card h3, .case-card p, .evidence, .evidence strong, .evidence em { color: #e7edf5; }
      .hero h1, .hero p, .label-text, .relationship-text, .empty-state, .provider-status { color: #e7edf5; }
    }
    [data-theme="dark"] .case-card, [data-theme="dark"] .evidence { background: #182230; border-color: #344054; }
    [data-theme="dark"] .case-card h3, [data-theme="dark"] .case-card p,
    [data-theme="dark"] .evidence, [data-theme="dark"] .evidence strong,
    [data-theme="dark"] .evidence em, [data-theme="dark"] .hero h1,
    [data-theme="dark"] .hero p, [data-theme="dark"] .label-text,
    [data-theme="dark"] .relationship-text, [data-theme="dark"] .empty-state,
    [data-theme="dark"] .provider-status { color: #e7edf5; }
    svg text { fill: #1f2937 !important; }
    .stAlert p, .stAlert [data-testid="stMarkdownContainer"] { color: inherit !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
def api_get(path, timeout=30):
    response = requests.get(f"{API_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


try:
    provider_status = api_get("/health", timeout=5)
    provider_label = provider_status.get("provider", "unknown")
    provider_state = "ready" if provider_status.get("configured") else "not configured"
except requests.RequestException:
    provider_label, provider_state = "unavailable", "API offline"


@st.fragment(run_every="2s")
def render_job(job_id):
    try:
        job = api_get(f"/upload/{job_id}", timeout=10)
    except requests.RequestException as exc:
        st.error(f"Could not read job status: {exc}")
        return
    status = job["status"]
    st.progress(job.get("progress", 0) / 100, text=f"{status.title()} - {job.get('progress', 0)}%")
    st.caption(
        f"{job.get('chunks_completed', 0)}/{job.get('chunks_total') or '?'} chunks completed "
        f"  |  job {job_id}"
    )
    if status == "success":
        st.success(f"Finished {job['filename']}.")
    elif status == "partial":
        quota = (job.get("result") or {}).get("quota")
        if quota:
            st.warning(quota.get("message", "Gemini quota limited part of this upload."))
            st.caption("Successful evidence remains available. Upload again later to retry failed chunks.")
        else:
            st.warning("Finished with chunk-level failures. The successful evidence remains available.")
    elif status == "failed":
        errors = (job.get("result") or {}).get("errors") or []
        error = job.get("error") or (errors[0] if errors else None)
        if isinstance(error, dict) and error.get("code") == "provider_quota":
            st.error(error.get("message", "Gemini quota is temporarily exhausted."))
            st.caption("No automatic retry was started. Use offline demo or your last successful results.")
        elif isinstance(error, dict):
            st.error(error.get("message", "Processing failed."))
        else:
            st.error(error or "Processing failed.")


def evidence_block(fact):
    evidence = fact.get("evidence", [])
    for item in evidence:
        excerpt = html.escape(item.get("excerpt") or "No excerpt returned.")
        source = html.escape(item.get("document_name") or "Unknown document")
        page = item.get("page") or "?"
        st.markdown(
            f'<div class="evidence"><strong>{source}</strong> - page {page}<br>'
            f'<em>"{excerpt}"</em></div>',
            unsafe_allow_html=True,
        )
        if provider_status.get("error"):
            st.caption(provider_status["error"])
        document_name = item.get("document_name")
        if document_name:
            page_number = item.get("page") or 1
            try:
                preview = api_get(
                    f"/source-preview?document_name={quote(document_name)}&page={page_number}",
                    timeout=10,
                )
            except requests.RequestException:
                preview = None
            if preview:
                if preview["kind"] == "pdf":
                    st.caption(
                        f"PDF source: page {preview['page']} of {preview['page_count']}. "
                        "Use the page controls in the viewer; browser PDF deep links vary by browser."
                    )
                    st.markdown(
                        f'<iframe src="{API_URL}/source-file/{quote(document_name)}#page={page_number}" '
                        'width="100%" height="420" style="border:1px solid #d0d5dd;border-radius:8px"></iframe>',
                        unsafe_allow_html=True,
                    )
                else:
                    with st.expander("Preview text source"):
                        st.code(preview["text"][:12000])


def claim_block(claim, label):
    if isinstance(claim, dict):
        text = html.escape(str(claim.get("text") or claim))
        st.markdown(f'<div class="label-text"><strong>{html.escape(label)}:</strong> {text}</div>', unsafe_allow_html=True)
        if claim.get("value"):
            st.caption(f"Value: {claim['value']}")
        evidence_block(claim)
    else:
        st.markdown(
            f'<div class="label-text"><strong>{html.escape(label)}:</strong> '
            f'{html.escape(str(claim))}</div>',
            unsafe_allow_html=True,
        )


def relation_card(relation, label):
    st.markdown(f'<div class="label-text"><strong>{html.escape(label)}</strong></div>', unsafe_allow_html=True)
    explanation = html.escape(relation.get("explanation") or "No explanation returned.")
    st.markdown(f'<div class="relationship-text">{explanation}</div>', unsafe_allow_html=True)
    with st.expander("Inspect compared claims"):
        claim_block(relation.get("fact_1"), "Claim A")
        claim_block(relation.get("fact_2"), "Claim B")


def render_graph(graph):
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    positions = {
        node["id"]: (80 + (index % 3) * 245, 70 + (index // 3) * 105)
        for index, node in enumerate(nodes)
    }
    lines = []
    for edge in edges:
        if edge["source"] not in positions or edge["target"] not in positions:
            continue
        x1, y1 = positions[edge["source"]]
        x2, y2 = positions[edge["target"]]
        lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{edge["color"]}" stroke-width="3" />'
        )
    circles = []
    for node in nodes:
        x, y = positions[node["id"]]
        color = "#6b7280" if node["kind"] == "failure" else "#2563eb"
        label = html.escape((node.get("label") or "")[:42])
        value = html.escape(str(node.get("value") or ""))
        circles.append(
            f'<g><circle cx="{x}" cy="{y}" r="27" fill="{color}" opacity=".92"/>'
            f'<text x="{x}" y="{y + 48}" text-anchor="middle" font-size="12" fill="#344054">{label}</text>'
            f'<text x="{x}" y="{y + 64}" text-anchor="middle" font-size="11" fill="#667085">{value}</text></g>'
        )
    legend = " ".join(
        f'<span style="color:{item["color"]};font-weight:700">● {html.escape(item["label"])}</span>'
        for item in graph.get("legend", {}).values()
    )
    svg = (
        '<div style="overflow:auto;background:#fff;border:1px solid #e4e7ec;border-radius:12px;padding:8px">'
        f'<div style="display:flex;gap:18px;flex-wrap:wrap;font-size:12px">{legend}</div>'
        f'<svg viewBox="0 0 780 {max(170, ((len(nodes) + 2) // 3) * 105 + 80)}" '
        'width="100%" role="img" aria-label="Interactive relationship graph">'
        + "".join(lines)
        + "".join(circles)
        + "</svg></div>"
    )
    components.html(svg, height=320, scrolling=True)


st.markdown(
    '<div class="hero"><div class="eyebrow">Evidence-first document intelligence</div>'
    '<h1>Fact Knowledge Layer</h1>'
    '<p>Extract grounded claims, compare them across documents, and review uncertainty without losing the source.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Process documents")
    st.markdown(
        f'<div class="provider-status"><strong>LLM provider:</strong> {html.escape(provider_label)} '
        f'({html.escape(provider_state)})</div>',
        unsafe_allow_html=True,
    )
    if provider_status.get("error"):
        st.caption(provider_status["error"])
    if provider_status.get("model"):
        st.caption(f"Model: {provider_status['model']}")
    uploaded_files = st.file_uploader(
        "Choose one or more source documents", type=["pdf", "txt"], accept_multiple_files=True
    )
    if uploaded_files and st.button("Start extraction", type="primary", use_container_width=True):
        try:
            response = requests.post(
                f"{API_URL}/uploads",
                files=[
                    ("files", (uploaded_file.name, uploaded_file.getvalue()))
                    for uploaded_file in uploaded_files
                ],
                timeout=30,
            )
            response.raise_for_status()
            st.session_state["job_ids"] = [job["job_id"] for job in response.json()["jobs"]]
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")
    if st.button("Load offline demo", use_container_width=True):
        try:
            response = requests.post(f"{API_URL}/demo", timeout=10)
            response.raise_for_status()
            st.session_state["reasoning"] = response.json()["result"]
            st.session_state["job_ids"] = []
            st.session_state["demo_mode"] = True
            st.session_state["last_successful_reasoning"] = response.json()["result"]
            st.success("Loaded synthetic demo data; previous upload errors were cleared.")
        except requests.RequestException as exc:
            st.error(f"Demo load failed: {exc}")
    if st.session_state.get("last_successful_reasoning") and st.button(
        "Use last successful results", use_container_width=True
    ):
        st.session_state["reasoning"] = st.session_state["last_successful_reasoning"]
        st.session_state["job_ids"] = []
        st.info("Showing the last successful comparison; no Gemini call was made.")
    for job_id in st.session_state.get("job_ids", []):
        render_job(job_id)
    st.markdown("---")
    st.caption("PDF calls are bounded for free-tier safety. Quota failures are concise and never retried automatically.")

facts = []
try:
    facts = api_get("/facts", timeout=20).get("facts", [])
except requests.RequestException as exc:
    st.warning(f"API unavailable: {exc}")

left, right = st.columns([1.4, 1])
with left:
    st.subheader("Grounded fact register")
    st.caption(f"{len(facts)} extracted claims currently in memory")
    if facts:
        query = st.text_input("Filter claims", placeholder="Search claim text, value, or source")
        if query:
            needle = query.lower()
            facts = [
                fact
                for fact in facts
                if needle in fact.get("text", "").lower()
                or needle in (fact.get("value") or "").lower()
                or any(
                    needle in (item.get("document_name") or "").lower()
                    for item in fact.get("evidence", [])
                )
            ]
            st.caption(f"{len(facts)} claims match the filter")
        if not facts:
            st.info("No claims match that filter.")
        else:
            rows = [
                {
                    "Claim": fact.get("text", ""),
                    "Value": fact.get("value") or "",
                    "Source": (
                        f"{fact.get('evidence', [{}])[0].get('document_name', 'Unknown')} "
                        f"p.{fact.get('evidence', [{}])[0].get('page', '?')}"
                    ),
                }
                for fact in facts
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            selected = st.selectbox(
                "Inspect source evidence",
                range(len(facts)),
                format_func=lambda index: facts[index].get("text", "")[:100],
            )
            evidence_block(facts[selected])
    else:
        st.info("Upload one or more documents to populate the fact register.")

with right:
    st.subheader("Review coverage")
    st.markdown(
        '<div class="case-card"><h3>1. Corroboration</h3><p>Independent documents support the same claim.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="case-card"><h3>2. Genuine contradiction</h3><p>Claims cannot both be true in the same scope.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="case-card"><h3>3. Explained by context</h3><p>Period, unit, currency, or scope reconciles the difference.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="case-card"><h3>4. Extraction or reasoning failure</h3><p>Uncertainty is surfaced instead of hidden.</p></div>',
        unsafe_allow_html=True,
    )

st.divider()
st.subheader("Cross-document review")
if st.button("Run comparison", type="primary"):
    with st.spinner("Comparing retrieved candidate pairs..."):
        try:
            st.session_state["reasoning"] = api_get("/corroborations", timeout=180)
            st.session_state["last_successful_reasoning"] = st.session_state["reasoning"]
            st.session_state["demo_mode"] = False
        except requests.RequestException as exc:
            st.error(f"Comparison failed: {exc}")

data = st.session_state.get("reasoning")
if data:
    if data.get("demo"):
        st.info("Offline demo mode: these facts are synthetic and clearly labeled.")
    st.subheader("Interactive relationship graph")
    try:
        graph = api_get("/graph", timeout=20)
        render_graph(graph)
        graph_facts = [node for node in graph.get("nodes", []) if node.get("kind") == "fact"]
        if graph_facts:
            selected_node = st.selectbox(
                "Select a graph fact to inspect its evidence",
                graph_facts,
                format_func=lambda node: node.get("label", "")[:100],
            )
            evidence_block(selected_node)
        graph_edges = [edge for edge in graph.get("edges", []) if edge["source"] != edge["target"]]
        if graph_edges:
            selected_edge = st.selectbox(
                "Select a relationship edge",
                graph_edges,
                format_func=lambda edge: edge.get("label", ""),
            )
            st.caption(selected_edge.get("explanation") or "No relationship explanation returned.")
    except requests.RequestException as exc:
        st.warning(f"Graph unavailable: {exc}")
    corroborations = data.get("corroborations", [])
    contradictions = data.get("contradictions", [])
    failures = data.get("failures", [])
    tabs = st.tabs(["Corroboration", "Genuine contradiction", "Explained by context", "Failures"])
    with tabs[0]:
        if corroborations:
            for item in corroborations:
                relation_card(item, "Independent support")
        else:
            st.info("No corroboration was returned for the retrieved candidates.")
    with tabs[1]:
        genuine = [item for item in contradictions if item.get("type") == "genuine_contradiction"]
        if genuine:
            for item in genuine:
                relation_card(item, "Potential conflict")
        else:
            st.info("No genuine contradiction was returned.")
    with tabs[2]:
        contextual = [item for item in contradictions if item.get("type") == "explained_by_context"]
        if contextual:
            for item in contextual:
                relation_card(item, "Context reconciles the claims")
        else:
            st.info("No context-explained contradiction was returned.")
    with tabs[3]:
        if failures:
            for failure in failures:
                st.error(f"{failure.get('type', 'failure')}: {failure.get('description', '')}")
        else:
            st.info("No extraction or reasoning failures were reported.")
