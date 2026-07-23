from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Load a PDF and split it into overlapping chunks."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks

if __name__ == "__main__":
    chunks = load_and_split("data/sample.pdf")
    print(f"Loaded {len(chunks)} chunks")
    print(chunks[0].page_content[:300])