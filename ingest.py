from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "knowledgehub_documents"


def create_embeddings():
    """Initialize the local embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def create_vector_store_connection():
    """Connect to the persistent ChromaDB collection."""

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    embeddings = create_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    return vector_store


def load_documents(filenames=None):
    """Load all PDFs or only the specified PDF files."""

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    documents = []

    if filenames is None:
        pdf_paths = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    else:
        pdf_paths = []

        for filename in filenames:
            pdf_path = DOCUMENTS_DIR / filename

            if not pdf_path.is_file():
                raise FileNotFoundError(f"PDF not found: {filename}")

            pdf_paths.append(pdf_path)

    for pdf_path in pdf_paths:
        loader = PyMuPDFLoader(str(pdf_path))
        pdf_documents = loader.load()

        for document in pdf_documents:
            document.metadata["source"] = pdf_path.name

        documents.extend(pdf_documents)

    return documents


def split_documents(documents):
    """Split document text into chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    return splitter.split_documents(documents)


def create_vector_store(chunks):
    """
    Index document chunks.

    Existing chunks belonging to the same source PDFs are replaced.
    Chunks belonging to other PDFs are preserved.
    """

    vector_store = create_vector_store_connection()

    if not chunks:
        return vector_store

    source_files = {
        chunk.metadata.get("source")
        for chunk in chunks
        if chunk.metadata.get("source")
    }

    for source_file in source_files:

        existing_records = vector_store.get(
            where={
                "source": source_file
            }
        )

        existing_ids = existing_records.get(
            "ids",
            []
        )

        if existing_ids:
            vector_store.delete(
                ids=existing_ids
            )

    vector_store.add_documents(
        chunks
    )

    return vector_store


def list_documents():
    """Return all PDF filenames saved in the documents directory."""

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    return sorted(
        pdf.name
        for pdf in DOCUMENTS_DIR.glob("*.pdf")
        if pdf.is_file()
    )


def list_indexed_documents():
    """
    Return PDF filenames that have indexed chunks in ChromaDB.

    A PDF is considered indexed when its source filename exists
    in the ChromaDB collection metadata.
    """

    vector_store = create_vector_store_connection()

    records = vector_store.get(
        include=["metadatas"]
    )

    indexed_files = set()

    for metadata in records.get("metadatas", []):

        if not metadata:
            continue

        source = metadata.get("source")

        if source and source.lower().endswith(".pdf"):
            indexed_files.add(source)

    return sorted(indexed_files)


def delete_document(filename):
    """
    Delete a PDF and its indexed chunks.

    Returns the number of indexed chunks removed.
    """

    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or not filename.lower().endswith(".pdf")
    ):
        raise ValueError(
            "Invalid PDF filename."
        )

    pdf_path = DOCUMENTS_DIR / filename

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {filename}"
        )

    vector_store = create_vector_store_connection()

    records = vector_store.get(
        where={
            "source": filename
        },
        include=["metadatas"]
    )

    chunk_ids = records.get(
        "ids",
        []
    )

    if chunk_ids:
        vector_store.delete(
            ids=chunk_ids
        )

    pdf_path.unlink()

    return len(chunk_ids)


def remove_stale_documents(vector_store):
    """
    Remove indexed chunks whose source PDFs no longer exist.
    """

    records = vector_store.get(
        include=["metadatas"]
    )

    metadata_list = records.get(
        "metadatas",
        []
    )

    ids = records.get(
        "ids",
        []
    )

    stale_ids = []

    for record_id, metadata in zip(
        ids,
        metadata_list
    ):

        if not metadata:
            continue

        source = metadata.get("source")

        if not source:
            continue

        pdf_path = DOCUMENTS_DIR / source

        if not pdf_path.is_file():
            stale_ids.append(record_id)

    if stale_ids:

        vector_store.delete(
            ids=stale_ids
        )

        print(
            f"Removed {len(stale_ids)} stale chunks."
        )

    else:
        print("No stale chunks found.")


def inspect_collection():
    """Print ChromaDB collection statistics."""

    vector_store = create_vector_store_connection()

    print("\n--- COLLECTION DEBUG ---")

    print(
        "Collection count:",
        vector_store._collection.count()
    )

    print(
        "Collection metadata:",
        vector_store._collection.metadata
    )

    records = vector_store.get(
        include=["metadatas"]
    )

    sources = {}

    for metadata in records.get("metadatas", []):

        source = (
            metadata.get("source", "Unknown")
            if metadata
            else "Unknown"
        )

        sources[source] = (
            sources.get(source, 0) + 1
        )

    print("Chunks by source:")

    for source, count in sorted(sources.items()):
        print(f"{source}: {count}")

    print("--- END DEBUG ---\n")


def main():
    """Index all PDFs in the documents directory."""

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    documents = load_documents()

    if not documents:
        print("No PDF documents found.")
        print(
            f"Add PDF files to: {DOCUMENTS_DIR}"
        )
        return

    chunks = split_documents(
        documents
    )

    print(
        f"Document pages loaded: {len(documents)}"
    )

    print(
        f"Text chunks created: {len(chunks)}"
    )

    create_vector_store(
        chunks
    )

    print(
        "Documents indexed successfully!"
    )


if __name__ == "__main__":
    main()