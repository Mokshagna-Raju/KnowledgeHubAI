from pathlib import Path
import time

import streamlit as st

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import re
import os
from langchain_groq import ChatGroq


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "knowledgehub_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:3b"


# --------------------------------------------------
# VECTOR STORE
# --------------------------------------------------

@st.cache_resource
def create_vector_store():
    """Initialize the embedding model and connect to ChromaDB."""

    start = time.perf_counter()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    embedding_time = time.perf_counter()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    end = time.perf_counter()

    print(
        f"Embedding initialization: "
        f"{embedding_time - start:.3f} seconds"
    )

    print(
        f"ChromaDB initialization: "
        f"{end - embedding_time:.3f} seconds"
    )

    return vector_store


# --------------------------------------------------
# LANGUAGE MODEL
# --------------------------------------------------

@st.cache_resource
def create_llm(model_name, num_predict=128):
    """Initialize the configured local or hosted language model."""

    provider = os.getenv("LLM_PROVIDER")

if not provider:
    try:
        provider = st.secrets.get("LLM_PROVIDER", "ollama")
    except Exception:
        provider = "ollama"

provider = provider.lower()

    print(f"Initializing LLM provider: {provider}")

    if provider == "groq":
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except (KeyError, FileNotFoundError):
            api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Configure it in Streamlit secrets."
            )

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            api_key=api_key,
            temperature=0,
            max_tokens=num_predict,
        )

    if provider == "ollama":
        return ChatOllama(
            model=model_name,
            temperature=0,
            num_predict=num_predict,
            keep_alive="10m",
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider}"
    )

# --------------------------------------------------
# CROSS-ENCODER RERANKER
# --------------------------------------------------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Cross-encoder scores are relevance logits, not probabilities. A score below
# this configurable floor is treated as insufficient evidence for answering.
# Start at 0.0 and tune using questions against your own PDFs if needed.
RERANKER_MIN_SCORE = float(os.getenv("RERANKER_MIN_SCORE", "0.0"))


@st.cache_resource
def create_reranker():
    """Load the reranking model once and reuse it."""

    print(f"Loading reranker: {RERANKER_MODEL}")

    start = time.perf_counter()

    reranker = CrossEncoder(RERANKER_MODEL)

    print(
        f"Reranker initialization: "
        f"{time.perf_counter() - start:.3f} seconds"
    )

    return reranker


# --------------------------------------------------
# HYBRID SEARCH + RERANKING
# --------------------------------------------------

def retrieve_documents(
    question,
    vector_store,
    k=3,
    selected_documents=None
):
    """
    Retrieve document chunks using:
    1. ChromaDB semantic search
    2. BM25 keyword search
    3. Reciprocal Rank Fusion
    4. Cross-encoder reranking
    """

    if selected_documents is not None and not selected_documents:
        return []

    print("\n--- HYBRID RETRIEVAL DEBUG ---")
    print("Question:", question)
    print("Selected documents:", selected_documents)

    # --------------------------------------------------
    # 1. BUILD THE DOCUMENT FILTER
    # --------------------------------------------------

    source_filter = None

    if selected_documents is not None:

        if len(selected_documents) == 1:

            source_filter = {
                "source": selected_documents[0]
            }

        else:

            source_filter = {
                "$or": [
                    {"source": filename}
                    for filename in selected_documents
                ]
            }

    # --------------------------------------------------
    # 2. SEMANTIC SEARCH
    # --------------------------------------------------

    semantic_kwargs = {
        "k": 12
    }

    if source_filter is not None:
        semantic_kwargs["filter"] = source_filter

    semantic_results = (
        vector_store.similarity_search_with_score(
            question,
            **semantic_kwargs
        )
    )

    semantic_documents = [
        document
        for document, distance in semantic_results
    ]

    # --------------------------------------------------
    # 3. LOAD DOCUMENTS FOR KEYWORD SEARCH
    # --------------------------------------------------

    get_kwargs = {
        "include": [
            "documents",
            "metadatas"
        ]
    }

    if source_filter is not None:
        get_kwargs["where"] = source_filter

    records = vector_store.get(
        **get_kwargs
    )

    corpus_documents = []

    texts = records.get("documents") or []
    metadatas = records.get("metadatas") or []

    for text, metadata in zip(texts, metadatas):

        if not text:
            continue

        corpus_documents.append(
            Document(
                page_content=text,
                metadata=metadata or {}
            )
        )

    # --------------------------------------------------
    # 4. BM25 KEYWORD SEARCH
    # --------------------------------------------------

    def tokenize(text):
        return re.findall(
            r"\b[\w.-]+\b",
            text.lower()
        )

    keyword_documents = []

    tokenized_documents = [
        tokenize(document.page_content)
        for document in corpus_documents
    ]

    valid_indices = [
        index
        for index, tokens in enumerate(tokenized_documents)
        if tokens
    ]

    if valid_indices:

        valid_documents = [
            corpus_documents[index]
            for index in valid_indices
        ]

        valid_tokens = [
            tokenized_documents[index]
            for index in valid_indices
        ]

        bm25 = BM25Okapi(
            valid_tokens
        )

        query_tokens = tokenize(
            question
        )

        if query_tokens:

            scores = bm25.get_scores(
                query_tokens
            )

            ranked_indices = sorted(
                range(len(scores)),
                key=lambda index: scores[index],
                reverse=True
            )

            keyword_documents = [
                valid_documents[index]
                for index in ranked_indices[:12]
                if scores[index] > 0
            ]

    # --------------------------------------------------
    # 5. COMBINE RESULTS USING RRF
    # --------------------------------------------------

    # RRF combines rankings instead of directly comparing
    # BM25 scores with ChromaDB distance values.

    rrf_scores = {}
    candidate_documents = {}

    def add_ranked_results(documents, rank_constant=60):

        for rank, document in enumerate(
            documents,
            start=1
        ):

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get(
                "page"
            )

            normalized_content = " ".join(
                document.page_content.lower().split()
            )

            # Preserve source and page attribution.
            key = (
                source,
                page,
                normalized_content
            )

            candidate_documents[key] = document

            rrf_scores[key] = (
                rrf_scores.get(key, 0.0)
                + 1.0 / (rank_constant + rank)
            )

    add_ranked_results(
        semantic_documents
    )

    add_ranked_results(
        keyword_documents
    )

    if not candidate_documents:

        print("No retrieval candidates found.")
        print("--- END HYBRID RETRIEVAL ---\n")

        return []

    fused_results = sorted(
        candidate_documents.items(),
        key=lambda item: item[1] is not None
        and rrf_scores[item[0]],
        reverse=True
    )

    candidates = [
        document
        for key, document in fused_results[:12]
    ]

    print(
        f"Semantic candidates: {len(semantic_documents)}"
    )

    print(
        f"BM25 candidates: {len(keyword_documents)}"
    )

    print(
        f"Combined candidates: {len(candidates)}"
    )

    # --------------------------------------------------
    # 6. RERANK THE CANDIDATES
    # --------------------------------------------------

    reranker = create_reranker()

    pairs = [
        (
            question,
            document.page_content
        )
        for document in candidates
    ]

    rerank_scores = reranker.predict(
        pairs
    )

    reranked_results = sorted(
        zip(candidates, rerank_scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    # Reject the entire retrieval result when even its best passage is below
    # the relevance floor. This prevents unrelated PDFs from being shown as
    # sources for questions such as asking about a person not in the documents.
    if not reranked_results or float(reranked_results[0][1]) < RERANKER_MIN_SCORE:
        best_score = (
            float(reranked_results[0][1]) if reranked_results else None
        )
        print(
            f"No sufficiently relevant passages. Best reranker score: "
            f"{best_score}; required: {RERANKER_MIN_SCORE}"
        )
        print("--- END HYBRID RETRIEVAL ---\n")
        return []

    # --------------------------------------------------
    # 7. SELECT FINAL CHUNKS
    # --------------------------------------------------

    final_documents = []
    seen_content = set()

    for document, score in reranked_results:

        content = " ".join(
            document.page_content.lower().split()
        )

        # Prevent identical text from occupying multiple
        # positions in the final context.
        if content in seen_content:
            continue

        seen_content.add(
            content
        )

        final_documents.append(
            document
        )

        print(
            f"Rerank score: {float(score):.4f} | "
            f"Source: {document.metadata.get('source', 'Unknown')} | "
            f"Page: {document.metadata.get('page', 'Unknown')}"
        )

        if len(final_documents) >= k:
            break

    print(
        "Final chunks retained:",
        len(final_documents)
    )

    print("--- END HYBRID RETRIEVAL ---\n")

    return final_documents


# --------------------------------------------------
# BUILD CONTEXT
# --------------------------------------------------

def build_context(documents):
    """Combine retrieved chunks and attach source information."""

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        source = document.metadata.get(
            "source",
            "Unknown document"
        )

        page = document.metadata.get("page")

        if page is not None:
            page_number = int(page) + 1
            location = f"{source}, page {page_number}"
        else:
            location = source

        context_parts.append(
            f"[Source {index}: {location}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(context_parts)


def contextualize_question(question, chat_history, llm):
    """
    Rewrite a follow-up question as a standalone question
    using the previous conversation.

    This function does not answer the question. It only
    clarifies the user's intent for document retrieval.
    """

    if not chat_history:
        return question

    history_lines = []

    # Use the most recent conversation messages.
    for message in chat_history[-10:]:

        role = message.get("role")
        content = message.get("content", "")

        if role == "user":
            history_lines.append(
                f"User: {content}"
            )

        elif role == "assistant":
            history_lines.append(
                f"Assistant: {content}"
            )

    if not history_lines:
        return question

    history_text = "\n".join(history_lines)

    prompt = ChatPromptTemplate.from_template(
        """
You rewrite follow-up questions into standalone questions
for a document-search system.

Use the conversation history to resolve references such as
"it", "they", "that", "its advantages", and similar expressions.

Rules:
1. Preserve the user's original intent.
2. If the question is already standalone, keep its meaning.
3. Do not answer the question.
4. Do not introduce facts that are not in the conversation.
5. Return only the rewritten question.
6. If clarification is unnecessary, return the original question.

Conversation history:
{history}

Latest user question:
{question}

Standalone question:
"""
    )

    chain = prompt | llm

    response = chain.invoke({
        "history": history_text,
        "question": question
    })

    rewritten_question = response.content.strip()

    # Avoid using an empty response as the retrieval query.
    if not rewritten_question:
        return question

    return rewritten_question


# --------------------------------------------------
# RAG CHAIN
# --------------------------------------------------

@st.cache_resource
def create_rag_chain(model_name):
    """Create the language model and RAG prompt."""

    vector_store = create_vector_store()

    llm = create_llm(model_name)

    prompt = ChatPromptTemplate.from_template(
        """
You are a question-answering assistant for uploaded documents.

Answer the user's question using only the information
provided in the retrieved context.

Rules:
1. Answer only what the user asks.
2. Use information from the retrieved context.
3. Do not invent facts or make unsupported assumptions.
4. If the context does not contain enough information,
   say: "I couldn't find enough information in the uploaded documents."
5. Keep the answer clear and concise.
6. When multiple documents are provided, combine their information
   only when the retrieved context supports the comparison.
7. Distinguish facts from different source documents when necessary.

Retrieved context:
{context}

User question:
{question}

Answer:
"""
    )

    chain = prompt | llm

    return vector_store, chain


# --------------------------------------------------
# FORMAT SOURCE INFORMATION
# --------------------------------------------------

def collect_sources(documents):
    """Extract source filenames, page numbers, and content."""

    sources = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown document"
        )

        page = document.metadata.get("page")

        if page is not None:
            page = int(page) + 1

        sources.append({
            "document": source,
            "page": page,
            "content": document.page_content
        })

    return sources


# --------------------------------------------------
# ASK A QUESTION
# --------------------------------------------------

def ask_question(
    question,
    selected_documents=None,
    chat_history=None
):
    """
    Answer questions using selected documents and
    optional conversation history.
    """

    total_start = time.perf_counter()

    chain_start = time.perf_counter()

    vector_store, chain = create_rag_chain(
        LLM_MODEL
    )

    llm = create_llm(
        LLM_MODEL
    )

    chain_end = time.perf_counter()

    timings = {
        "chain_access_seconds": round(
            chain_end - chain_start,
            3
        )
    }

    # --------------------------------------------------
    # UNDERSTAND FOLLOW-UP QUESTIONS
    # --------------------------------------------------

    contextualization_start = time.perf_counter()

    standalone_question = contextualize_question(
        question=question,
        chat_history=chat_history or [],
        llm=llm
    )

    contextualization_end = time.perf_counter()

    timings["contextualization_seconds"] = round(
        contextualization_end - contextualization_start,
        3
    )

    # --------------------------------------------------
    # RETRIEVE DOCUMENTS
    # --------------------------------------------------

    retrieval_start = time.perf_counter()

    documents = retrieve_documents(
        question=standalone_question,
        vector_store=vector_store,
        k=3,
        selected_documents=selected_documents
    )

    retrieval_end = time.perf_counter()

    timings["retrieval_seconds"] = round(
        retrieval_end - retrieval_start,
        3
    )

    # --------------------------------------------------
    # HANDLE MISSING INFORMATION
    # --------------------------------------------------

    if not documents:

        timings["total_seconds"] = round(
            time.perf_counter() - total_start,
            3
        )

        if selected_documents is not None and not selected_documents:

            answer = (
                "No documents are selected. "
                "Please select at least one indexed PDF."
            )

        else:

            answer = (
                "I couldn't find enough information "
                "in the uploaded documents."
            )

        return {
            "answer": answer,
            "sources": [],
            "timings": timings
        }

    # --------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------

    context = build_context(
        documents
    )

    # --------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------

    generation_start = time.perf_counter()

    response = chain.invoke({
        "context": context,
        "question": standalone_question
    })

    generation_end = time.perf_counter()

    timings["generation_seconds"] = round(
        generation_end - generation_start,
        3
    )

    # --------------------------------------------------
    # COLLECT SOURCES
    # --------------------------------------------------

    sources = collect_sources(
        documents
    )

    timings["total_seconds"] = round(
        time.perf_counter() - total_start,
        3
    )

    return {
        "answer": response.content,
        "sources": sources,
        "timings": timings,
        "standalone_question": standalone_question
    }

# --------------------------------------------------
# COMPARE DOCUMENTS
# --------------------------------------------------

def compare_documents(
    selected_documents,
    comparison_question=None
):
    """
    Compare two or more selected PDFs.

    Retrieve relevant chunks separately from each document
    so that the comparison can include information from
    every selected PDF.
    """

    total_start = time.perf_counter()

    if not selected_documents or len(selected_documents) < 2:

        return {
            "answer": (
                "Please select at least two indexed PDFs "
                "to compare."
            ),
            "sources": [],
            "timings": {
                "total_seconds": round(
                    time.perf_counter() - total_start,
                    3
                )
            }
        }

    vector_store, chain = create_rag_chain(
        LLM_MODEL
    )

    if comparison_question is None:

        comparison_question = (
            "Compare the selected documents. Identify their "
            "main topics, key concepts, similarities, differences, "
            "and unique information. Do not invent information "
            "that is not present in the documents."
        )

    retrieval_start = time.perf_counter()

    all_documents = []

    # Retrieve independently from every selected PDF.
    # This prevents one PDF from occupying every retrieval result.
    for filename in selected_documents:

        # Retrieve more chunks per file for comparison than normal chat.
        # Retrieval is run separately for each PDF so every selected file
        # gets its own opportunity to contribute evidence.
        retrieved = retrieve_documents(
            question=comparison_question,
            vector_store=vector_store,
            k=5,
            selected_documents=[filename]
        )

        all_documents.extend(
            retrieved
        )

    retrieval_end = time.perf_counter()

    # Remove duplicate chunks without discarding the source
    # information from different documents.
    unique_documents = []
    seen_chunks = set()

    for document in all_documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get("page")

        content = " ".join(
            document.page_content.lower().split()
        )

        chunk_key = (
            source,
            page,
            content
        )

        if chunk_key in seen_chunks:
            continue

        seen_chunks.add(
            chunk_key
        )

        unique_documents.append(
            document
        )

    if not unique_documents:

        return {
            "answer": (
                "I couldn't retrieve relevant content from "
                "the selected PDFs. Check whether the documents "
                "contain extractable text."
            ),
            "sources": [],
            "timings": {
                "retrieval_seconds": round(
                    retrieval_end - retrieval_start,
                    3
                ),
                "total_seconds": round(
                    time.perf_counter() - total_start,
                    3
                )
            }
        }

    context = build_context(
        unique_documents
    )

    comparison_prompt = ChatPromptTemplate.from_template(
        """
You are a careful document comparison assistant.
Compare the selected documents using ONLY the supplied retrieved context.
Do not assume that missing information is absent from the entire PDF;
state that it was not found in the retrieved text available for this comparison.
Never invent skills, facts, dates, qualifications, examples, or conclusions.

User's comparison request:
{question}

Retrieved document context (each passage includes its source filename and page):
{context}

Write a complete, readable comparison using these headings:

## 1. Overview of Each Document
For EVERY selected document, identify its main topic and summarize the
key information actually supported by the context. Use the exact filename.

## 2. Key Concepts or Information
Give concise bullet points for the important concepts/details found in each
file. Do not mix details between files.

## 3. Similarities
List only similarities supported by evidence in both documents. Explain
briefly what is similar. If no supported similarity can be established,
say: "No clear similarity can be established from the retrieved text."

## 4. Differences
Explain the meaningful differences, identifying which filename supports
each side of the difference. Do not create a difference just to fill space.

## 5. Unique Information in Each Document
List information found in one document's retrieved text that was not found
in the other document's retrieved text. Do not claim it is absent from the
full PDF unless the full PDF was actually examined.

## 6. Information Not Available
For each requested comparison point that cannot be supported by the
retrieved text, explicitly write: "Not found in the retrieved text."
Explain what information would be needed to make that comparison.

Formatting rules:
- Prefer headings and bullets over a wide Markdown table.
- Use a small table only if it genuinely improves clarity.
- Keep each section meaningful; avoid empty columns and empty table rows.
- Never output raw HTML such as <br>.
- Do not repeat the question or start with generic filler.
- Use clear, complete sentences and finish every section.
- Do not claim to have reviewed information that is not in the context.

Comparison:
"""
    )

    # Use a larger generation limit only for comparisons. Normal PDF chat
    # continues to use create_llm(..., num_predict=128) as before.
    comparison_llm = create_llm(
        LLM_MODEL,
        num_predict=768
    )
    comparison_chain = comparison_prompt | comparison_llm

    generation_start = time.perf_counter()

    response = comparison_chain.invoke({
        "question": comparison_question,
        "context": context
    })

    generation_end = time.perf_counter()

    sources = collect_sources(
        unique_documents
    )

    total_end = time.perf_counter()

    return {
        "answer": response.content,
        "sources": sources,
        "timings": {
            "retrieval_seconds": round(
                retrieval_end - retrieval_start,
                3
            ),
            "generation_seconds": round(
                generation_end - generation_start,
                3
            ),
            "total_seconds": round(
                total_end - total_start,
                3
            )
        }
    }


# --------------------------------------------------
# COMMAND-LINE TESTING
# --------------------------------------------------

if __name__ == "__main__":

    question = input(
        "\nEnter your question about the documents: "
    )

    result = ask_question(
        question
    )

    print("\n" + "=" * 50)
    print("ANSWER")
    print("=" * 50)

    print(
        result["answer"]
    )

    print("\n" + "=" * 50)
    print("SOURCE DOCUMENTS")
    print("=" * 50)

    for source in result["sources"]:

        print(
            f"\nDocument: {source['document']}"
        )

        print(
            f"Page: {source['page']}"
        )

        print(
            f"Content: {source['content'][:300]}..."
        )

    print("\n" + "=" * 50)
    print("PERFORMANCE")
    print("=" * 50)

    print(
        result["timings"]
    )