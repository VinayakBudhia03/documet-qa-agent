import streamlit as st
import tempfile, os
from src.ingest import load_and_split
from src.rag_chain import build_vectorstore, build_conversational_chain

st.set_page_config(page_title="Document Q&A Agent", page_icon="📄")
st.title("📄 Intelligent Document Q&A Agent")

with st.sidebar:
    persona = st.selectbox("Response style", ["default", "beginner_tutor", "technical_expert"])
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    process_btn = st.button("Process document")

if "chain" not in st.session_state:
    st.session_state.chain = None
    st.session_state.messages = []

if process_btn and uploaded_file:
    with st.spinner("Reading and indexing document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        chunks = load_and_split(tmp_path)
        vectorstore = build_vectorstore(chunks)
        chain, memory = build_conversational_chain(vectorstore, persona=persona)

        st.session_state.chain = chain
        st.session_state.messages = []
        os.unlink(tmp_path)
    st.success(f"Indexed {len(chunks)} chunks. Ask away!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.chain:
    if question := st.chat_input("Ask a question about the document"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain.invoke({"question": question})
                answer = result["answer"]
                st.write(answer)
                with st.expander("Sources used"):
                    for doc in result.get("source_documents", []):
                        st.caption(f"Page {doc.metadata.get('page', '?')}: {doc.page_content[:200]}...")
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a PDF and click 'Process document' to start.")