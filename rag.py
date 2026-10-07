import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_file):

    text = ""

    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")

    for page in pdf:
        text += page.get_text()

    pdf.close()

    return text


def create_chunks(text, chunk_size=500):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        chunks.append(chunk)

    return chunks


def create_embeddings(chunks):

    embeddings = model.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    return embeddings


def create_vector_database(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def search_chunks(question, chunks, index, top_k=3):

    question_embedding = model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for i in indices[0]:

        if i < len(chunks):
            results.append(chunks[i])

    return results