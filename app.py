import streamlit as st
import ollama

from rag import (
    extract_text_from_pdf,
    create_chunks,
    create_embeddings,
    create_vector_database,
    search_chunks
)


st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄"
)


st.title("AI PDF Chatbot")

st.write(
    "Upload a PDF and ask questions about it."
)


# Session state
if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None


# PDF upload
pdf_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if pdf_file:

    if st.button("Process PDF"):

        with st.spinner("Processing PDF..."):

            # Extract text
            text = extract_text_from_pdf(
                pdf_file
            )

            # Create chunks
            chunks = create_chunks(text)

            # Create embeddings
            embeddings = create_embeddings(
                chunks
            )

            # Create FAISS database
            index = create_vector_database(
                embeddings
            )

            # Save in session
            st.session_state.chunks = chunks
            st.session_state.index = index

        st.success(
            f"PDF processed successfully! "
            f"{len(chunks)} chunks created."
        )


# Chat section

if st.session_state.chunks:

    st.subheader("Ask Questions")

    question = st.text_input(
        "Ask something about your PDF"
    )

    if st.button("Ask"):

        if question:

            with st.spinner("Searching PDF..."):

                relevant_chunks = search_chunks(
                    question,
                    st.session_state.chunks,
                    st.session_state.index
                )

            context = "\n\n".join(
                relevant_chunks
            )

            prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY
the information provided in the context.

If the answer is not present in the context,
say:

"I could not find this information in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""

            with st.spinner("Generating answer..."):

                response = ollama.chat(
                    model="llama3.2:1b",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

            answer = response["message"]["content"]

            st.subheader("Answer")

            st.write(answer)