import streamlit as st
from pathlib import Path
from rag_engine import ResumeRAG
from config import UPLOAD_DIR

st.set_page_config(page_title="AI Resume Assistant", page_icon="📄", layout="wide")

st.title("📄 AI Resume Assistant")
st.caption("RAG-powered resume screening assistant for job-relevant candidate analysis")

# Session state
if "rag" not in st.session_state:
    st.session_state.rag = ResumeRAG()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []

with st.sidebar:
    st.header("Job Requirement")
    job_description = st.text_area(
        "Paste the Job Description",
        height=280,
        placeholder="Example: Python Developer with 2+ years experience, FastAPI/Flask, SQL, REST APIs, Git..."
    )

    st.divider()
    st.header("Resume Upload")
    uploaded_files = st.file_uploader(
        "Upload multiple PDF resumes",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        current_names = []

        for uploaded_file in uploaded_files:
            file_path = UPLOAD_DIR / uploaded_file.name
            file_path.write_bytes(uploaded_file.getbuffer())
            current_names.append(uploaded_file.name)

        st.session_state.uploaded_names = current_names
        st.success(f"{len(uploaded_files)} resume(s) uploaded.")

    create_kb = st.button(
        "🧠 Create Knowledge Base",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files
    )

    if create_kb:
        with st.spinner("Extracting resumes, creating chunks and embeddings..."):
            try:
                result = st.session_state.rag.build_knowledge_base(uploaded_files)
                st.session_state.kb_ready = True
                st.success(
                    f"Knowledge base created: {result['documents']} resumes, "
                    f"{result['chunks']} chunks."
                )
            except Exception as e:
                st.error(f"Could not create knowledge base: {e}")

    if st.session_state.kb_ready:
        st.success("Knowledge Base: Ready")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🔄 Reset Knowledge Base", use_container_width=True):
        st.session_state.rag.reset()
        st.session_state.kb_ready = False
        st.session_state.chat_history = []
        st.rerun()

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Candidate Screening")
    if job_description:
        st.info("Job description loaded.")
    else:
        st.warning("Add a job description from the sidebar.")

    if st.session_state.kb_ready and job_description:
        if st.button("🏆 Rank Candidates", type="primary"):
            with st.spinner("Comparing resumes against the job description..."):
                try:
                    ranking = st.session_state.rag.rank_candidates(
                        job_description
                    )
                    st.session_state.ranking = ranking
                except Exception as e:
                    st.error(f"Ranking failed: {e}")

        if "ranking" in st.session_state:
            st.markdown("### Ranking")
            for idx, candidate in enumerate(st.session_state.ranking, 1):
                with st.container(border=True):
                    st.markdown(
                        f"**#{idx} — {candidate['candidate_name']}**"
                    )
                    st.write(f"**Match score:** {candidate['score']}/100")
                    st.write(f"**Reason:** {candidate['reason']}")
                    if candidate.get("strengths"):
                        st.write(
                            "**Relevant strengths:** "
                            + ", ".join(candidate["strengths"])
                        )
                    if candidate.get("gaps"):
                        st.write(
                            "**Job-relevant gaps:** "
                            + ", ".join(candidate["gaps"])
                        )

with col2:
    st.subheader("💬 Ask the Resume Assistant")

    if not st.session_state.kb_ready:
        st.info("Upload resumes and click 'Create Knowledge Base' first.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.write(
                            f"**{source['candidate']}** — "
                            f"{source['file']} (page {source['page']})"
                        )

    prompt = st.chat_input(
        "Ask about candidates, skills, experience, gaps, etc.",
        disabled=not st.session_state.kb_ready
    )

    if prompt:
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching resumes and generating answer..."):
                try:
                    answer, sources = st.session_state.rag.answer(
                        prompt,
                        job_description=job_description,
                        chat_history=st.session_state.chat_history[:-1]
                    )
                    st.markdown(answer)

                    if sources:
                        with st.expander("Sources"):
                            for source in sources:
                                st.write(
                                    f"**{source['candidate']}** — "
                                    f"{source['file']} (page {source['page']})"
                                )

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Answer generation failed: {e}")
