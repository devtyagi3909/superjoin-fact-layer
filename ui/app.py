import html

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Fact Layer", page_icon=None, layout="wide")
API_URL = "http://localhost:8000"

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: #f7f8fa; }
    [data-testid="stSidebar"] { background: #101827; }
    [data-testid="stSidebar"] * { color: #e7edf5; }
    .hero { padding: 1.5rem 0 1rem; }
    .eyebrow { color: #2563eb; font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .hero h1 { color: #111827; font-size: 2.2rem; margin: .2rem 0; }
    .hero p { color: #667085; font-size: 1.05rem; }
    .case-card { background: white; border: 1px solid #e4e7ec; border-radius: 12px; padding: 1rem; min-height: 130px; }
    .case-card h3 { margin: 0; color: #111827; font-size: 1rem; }
    .case-card p { color: #667085; font-size: .9rem; }
    .evidence { border-left: 3px solid #2563eb; background: #f8fafc; padding: .7rem .9rem; margin: .5rem 0; color: #344054; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path, timeout=30):
    response = requests.get(f"{API_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


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
        st.warning("Finished with chunk-level failures. The successful evidence remains available.")
    elif status == "failed":
        st.error(job.get("error") or "Processing failed.")


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


def claim_block(claim, label):
    if isinstance(claim, dict):
        st.markdown(f"**{label}:** {claim.get('text') or claim}")
        if claim.get("value"):
            st.caption(f"Value: {claim['value']}")
        evidence_block(claim)
    else:
        st.write(f"**{label}:** {claim}")


def relation_card(relation, label):
    st.markdown(f"**{label}**")
    st.write(relation.get("explanation") or "No explanation returned.")
    with st.expander("Inspect compared claims"):
        claim_block(relation.get("fact_1"), "Claim A")
        claim_block(relation.get("fact_2"), "Claim B")


st.markdown(
    '<div class="hero"><div class="eyebrow">Evidence-first document intelligence</div>'
    '<h1>Fact Knowledge Layer</h1>'
    '<p>Extract grounded claims, compare them across documents, and review uncertainty without losing the source.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Process a PDF")
    uploaded_file = st.file_uploader("Choose a source document", type=["pdf"])
    if uploaded_file and st.button("Start extraction", type="primary", use_container_width=True):
        try:
            response = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                timeout=30,
            )
            response.raise_for_status()
            st.session_state["job_id"] = response.json()["job_id"]
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")
    if st.session_state.get("job_id"):
        render_job(st.session_state["job_id"])
    st.markdown("---")
    st.caption("The API uses bounded PDF chunks and parallel extraction workers. No credentials are stored in the UI.")

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
        st.info("Upload a PDF to populate the fact register.")

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
        except requests.RequestException as exc:
            st.error(f"Comparison failed: {exc}")

data = st.session_state.get("reasoning")
if data:
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
