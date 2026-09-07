import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Superjoin Fact Layer", layout="wide")

st.title("📄 Superjoin Fact Knowledge Layer")
st.markdown("AI agents for finance - Extract and compare facts from financial documents.")

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload PDF or Spreadsheet", type=["pdf", "xlsx", "csv"])
    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing document using LLM..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post("http://localhost:8000/upload", files=files)
                if response.status_code == 200:
                    st.success(f"Document '{uploaded_file.name}' processed successfully!")
                else:
                    st.error("Error processing document")

st.header("🔍 Extracted Facts")
if st.button("Refresh Facts"):
    with st.spinner("Fetching facts..."):
        response = requests.get("http://localhost:8000/facts")
        if response.status_code == 200:
            facts = response.json().get("facts", [])
            if facts:
                df_facts = []
                for f in facts:
                    evidence_doc = f["evidence"][0]["document_name"] if f.get("evidence") else "Unknown"
                    page_num = f["evidence"][0].get("page", "") if f.get("evidence") else ""
                    excerpt = f["evidence"][0]["excerpt"] if f.get("evidence") else ""
                    
                    df_facts.append({
                        "Fact": f["text"],
                        "Value": f["value"],
                        "Source": f"{evidence_doc} (Page {page_num})",
                        "Excerpt": excerpt
                    })
                st.dataframe(pd.DataFrame(df_facts), use_container_width=True)
            else:
                st.info("No facts extracted yet. Please upload and process a document.")
        else:
            st.error("Failed to fetch facts.")

st.header("🧠 Reasoning Engine (Corroborations & Contradictions)")
if st.button("Run Reasoning Engine"):
    with st.spinner("Analyzing relationships between facts..."):
        response = requests.get("http://localhost:8000/corroborations")
        if response.status_code == 200:
            data = response.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("✅ Corroborations")
                corroborations = data.get("corroborations", [])
                if corroborations:
                    for c in corroborations:
                        with st.expander(f"Match: {c.get('fact_1', '')[:30]}..."):
                            st.write(f"**Fact 1:** {c.get('fact_1')}")
                            st.write(f"**Fact 2:** {c.get('fact_2')}")
                            st.success(f"**Explanation:** {c.get('explanation')}")
                else:
                    st.info("No corroborations found.")
            
            with col2:
                st.subheader("⚠️ Contradictions")
                contradictions = data.get("contradictions", [])
                if contradictions:
                    for c in contradictions:
                        with st.expander(f"Conflict: {c.get('fact_1', '')[:30]}..."):
                            st.write(f"**Fact 1:** {c.get('fact_1')}")
                            st.write(f"**Fact 2:** {c.get('fact_2')}")
                            if c.get("type") == "explained_by_context":
                                st.warning(f"**Contextual Explanation:** {c.get('explanation')}")
                            else:
                                st.error(f"**Genuine Contradiction:** {c.get('explanation')}")
                else:
                    st.info("No contradictions found.")
                    
            if data.get("failures"):
                st.subheader("❌ Extraction Failures")
                for f in data.get("failures", []):
                    st.error(f.get("description", "Unknown error"))
