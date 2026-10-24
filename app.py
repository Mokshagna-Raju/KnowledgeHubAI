from pathlib import Path
import os

import streamlit as st

from ingest import (
    load_documents,
    split_documents,
    create_vector_store,
    list_documents,
    list_indexed_documents,
    delete_document,
)

from rag_pipeline import (
    ask_question,
    compare_documents,
    LLM_MODEL,
)

from chat_history import (
    initialize_database,
    create_conversation,
    list_conversations,
    get_conversation,
    rename_conversation,
    delete_conversation,
    save_message,
    get_messages,
    clear_conversation_messages,
)


# --------------------------------------------------
# PAGE CONFIGURATION AND DATABASE
# --------------------------------------------------

st.set_page_config(
    page_title="KnowledgeHub AI",
    page_icon="📚",
    layout="wide",
)

initialize_database()

st.title("📚 KnowledgeHub AI")
provider = os.getenv("LLM_PROVIDER", "ollama").lower()

if provider == "groq":
    model_display = os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile"
    )
    st.caption(f"Powered by Groq · Model: {model_display}")
else:
    st.caption(f"Powered by Ollama · Model: {LLM_MODEL}")
st.caption(
    "Your local AI assistant for searching, understanding, "
    "and comparing documents."
)


# --------------------------------------------------
# SESSION STATE AND ACTIVE CONVERSATION
# --------------------------------------------------

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "document_scope" not in st.session_state:
    st.session_state.document_scope = "All indexed PDFs"

if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Create the first conversation on the first launch.
if st.session_state.active_conversation_id is None:
    existing_conversations = list_conversations()
    if existing_conversations:
        active_id = existing_conversations[0]["id"]
    else:
        active_id = create_conversation()

    st.session_state.active_conversation_id = active_id
    st.session_state.messages = get_messages(active_id)

if (
    "rename_title_value" not in st.session_state
    or st.session_state.get("rename_title_conversation_id")
    != st.session_state.active_conversation_id
):
    initial_conversation = get_conversation(
        st.session_state.active_conversation_id
    )
    st.session_state.rename_title_value = (
        initial_conversation["title"]
        if initial_conversation
        else "New Chat"
    )
    st.session_state.rename_title_conversation_id = (
        st.session_state.active_conversation_id
    )


# --------------------------------------------------
# DOCUMENT STATUS
# --------------------------------------------------

def get_document_status():
    """Separate saved PDFs into indexed and not-indexed lists."""
    all_pdfs = set(list_documents())
    indexed_pdfs = set(list_indexed_documents())
    indexed_pdfs = all_pdfs.intersection(indexed_pdfs)
    not_indexed_pdfs = all_pdfs - indexed_pdfs
    return sorted(not_indexed_pdfs), sorted(indexed_pdfs)


def render_document_list(title, documents, section_key, indexed=False):
    """Display PDFs with confirmation before deletion."""
    st.subheader(title)

    if not documents:
        if indexed:
            st.info("No indexed PDFs found.")
        else:
            st.success("All stored PDFs are indexed.")
        return

    st.caption(f"Total: {len(documents)} PDF(s)")

    for filename in documents:
        with st.container(border=True):
            st.markdown(f"📄 **{filename}**")
            st.caption(
                "Status: Indexed in ChromaDB"
                if indexed
                else "Status: Waiting to be indexed"
            )

            confirm_key = f"confirm_{section_key}_{filename}"
            st.checkbox("Confirm deletion", key=confirm_key)

            if st.button(
                f"🗑️ Delete {filename}",
                key=f"delete_{section_key}_{filename}",
                use_container_width=True,
            ):
                if not st.session_state.get(confirm_key, False):
                    st.warning("Please confirm deletion first.")
                else:
                    try:
                        deleted_chunks = delete_document(filename)
                        st.success(
                            f"Deleted {filename}. "
                            f"Removed {deleted_chunks} indexed chunks."
                        )
                        st.rerun()
                    except Exception as error:
                        st.error(f"Deletion failed: {error}")


# --------------------------------------------------
# SIDEBAR: CHAT HISTORY
# --------------------------------------------------

with st.sidebar:
    st.header("💬 Chat History")

    if st.button("＋ New Chat", use_container_width=True):
        new_id = create_conversation()
        st.session_state.active_conversation_id = new_id
        st.session_state.messages = []
        st.session_state.rename_title_value = ""
        st.rerun()

    conversations = list_conversations()

    if conversations:
        active_id = st.session_state.active_conversation_id
        conversation_ids = [item["id"] for item in conversations]

        if active_id not in conversation_ids:
            active_id = conversations[0]["id"]
            st.session_state.active_conversation_id = active_id
            st.session_state.messages = get_messages(active_id)

        def conversation_label(conversation_id):
            item = next(
                (c for c in conversations if c["id"] == conversation_id),
                None,
            )
            return item["title"] if item else f"Chat {conversation_id}"

        selected_id = st.selectbox(
            "Open a conversation",
            options=conversation_ids,
            index=conversation_ids.index(active_id),
            format_func=conversation_label,
            key=f"conversation_selector_{active_id}",
        )

        if selected_id != st.session_state.active_conversation_id:
            st.session_state.active_conversation_id = selected_id
            st.session_state.messages = get_messages(selected_id)
            selected_conversation = get_conversation(selected_id)
            st.session_state.rename_title_value = (
                selected_conversation["title"] if selected_conversation else ""
            )
            st.session_state.rename_title_conversation_id = selected_id
            st.rerun()

        active_conversation = get_conversation(
            st.session_state.active_conversation_id
        )

        if active_conversation:
            st.caption(
                f"Last updated: {active_conversation['updated_at']}"
            )

            with st.expander("✏️ Rename current chat"):
                st.text_input(
                    "Conversation title",
                    key="rename_title_value",
                    max_chars=100,
                )

                if st.button(
                    "Save Chat Name",
                    key="save_chat_name",
                    use_container_width=True,
                ):
                    try:
                        rename_conversation(
                            st.session_state.active_conversation_id,
                            st.session_state.rename_title_value,
                        )
                        st.success("Conversation renamed.")
                        st.rerun()
                    except Exception as error:
                        st.error(f"Could not rename conversation: {error}")

            if st.button(
                "🗑️ Delete Current Chat",
                use_container_width=True,
            ):
                deleted_id = st.session_state.active_conversation_id
                delete_conversation(deleted_id)

                remaining = list_conversations()
                if remaining:
                    next_id = remaining[0]["id"]
                else:
                    next_id = create_conversation()

                st.session_state.active_conversation_id = next_id
                st.session_state.messages = get_messages(next_id)
                
                st.rerun()

            if st.button(
                "🧹 Clear Current Chat Messages",
                use_container_width=True,
            ):
                clear_conversation_messages(
                    st.session_state.active_conversation_id
                )
                st.session_state.messages = []
                st.session_state.rename_title_value = "New Chat"
                st.rerun()

    st.divider()

    # --------------------------------------------------
    # SIDEBAR: DOCUMENT MANAGEMENT
    # --------------------------------------------------

    st.header("📁 Document Management")
    st.subheader("Upload PDF")

    uploaded_files = st.file_uploader(
        "Choose PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state.uploader_key}",
    )

    if st.button("⚡ Index Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            documents_dir = (
                Path(__file__).resolve().parent / "data" / "documents"
            )
            documents_dir.mkdir(parents=True, exist_ok=True)

            try:
                with st.spinner("Saving and indexing PDFs..."):
                    uploaded_filenames = []

                    for uploaded_file in uploaded_files:
                        filename = Path(uploaded_file.name).name
                        if (
                            not filename
                            or not filename.lower().endswith(".pdf")
                        ):
                            raise ValueError("Invalid PDF filename.")

                        file_path = documents_dir / filename
                        file_path.write_bytes(uploaded_file.getvalue())
                        uploaded_filenames.append(filename)

                    documents = load_documents(uploaded_filenames)

                    if not documents:
                        st.warning("No readable PDF pages were found.")
                    else:
                        chunks = split_documents(documents)

                        if not chunks:
                            st.warning(
                                "No extractable text was found. "
                                "The PDF may be scanned or image-only."
                            )
                        else:
                            create_vector_store(chunks)
                            st.success(
                                f"Indexed {len(uploaded_filenames)} PDF(s), "
                                f"{len(documents)} page(s), and "
                                f"{len(chunks)} chunk(s)."
                            )
                            st.session_state.uploader_key += 1
                            st.rerun()

            except Exception as error:
                st.error(f"Indexing failed: {error}")

    st.divider()

    try:
        not_indexed_pdfs, indexed_pdfs = get_document_status()

        render_document_list(
            title="⏳ PDFs Not Indexed",
            documents=not_indexed_pdfs,
            section_key="not_indexed",
            indexed=False,
        )

        st.divider()

        render_document_list(
            title="✅ Indexed Documents",
            documents=indexed_pdfs,
            section_key="indexed",
            indexed=True,
        )
    except Exception as error:
        st.error(f"Could not load document status: {error}")


# --------------------------------------------------
# MAIN INTERFACE TABS
# --------------------------------------------------

chat_tab, comparison_tab = st.tabs(
    ["💬 Chat with PDFs", "📊 Document Comparison"]
)


# --------------------------------------------------
# FEATURE: CHAT WITH MULTIPLE PDFS + PERSISTENT HISTORY
# --------------------------------------------------

with chat_tab:
    active_conversation = get_conversation(
        st.session_state.active_conversation_id
    )
    chat_title = (
        active_conversation["title"]
        if active_conversation
        else "Chat"
    )

    st.header(f"💬 {chat_title}")

    try:
        _, indexed_pdfs = get_document_status()
    except Exception as error:
        indexed_pdfs = []
        st.error(f"Could not retrieve indexed documents: {error}")

    if not indexed_pdfs:
        st.info(
            "Please upload and index at least one PDF before asking questions."
        )
    else:
        st.subheader("Choose Documents")

        document_scope = st.radio(
            "Where should the AI search?",
            options=["All indexed PDFs", "Selected PDFs"],
            horizontal=True,
            key="document_scope",
        )

        if document_scope == "All indexed PDFs":
            selected_documents = None
            st.caption(
                f"Searching across all {len(indexed_pdfs)} indexed PDFs."
            )
        else:
            selected_documents = st.multiselect(
                "Select PDFs to search",
                options=indexed_pdfs,
                default=indexed_pdfs,
                key="chat_selected_documents",
            )
            if selected_documents:
                st.caption(f"Selected {len(selected_documents)} PDF(s).")
            else:
                st.warning("Select at least one PDF to ask questions.")

        st.divider()

        # Render saved messages from the selected conversation.
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message.get("mode", "chat") == "comparison":
                    st.caption("📊 Document comparison")
                st.markdown(message["content"])

                if message.get("sources"):
                    with st.expander("📚 View retrieved sources"):
                        for index, source in enumerate(
                            message["sources"], start=1
                        ):
                            st.markdown(
                                f"**Source {index}:** {source.get('document', 'Unknown')}"
                            )
                            if source.get("page") is not None:
                                st.caption(f"Page {source['page']}")
                            st.write(source.get("content", ""))

                if message.get("timings"):
                    with st.expander("⚡ Performance Details"):
                        st.json(message["timings"])

        question = st.chat_input(
            "Ask a question about your PDFs...",
            key="chat_question",
            disabled=(
                document_scope == "Selected PDFs"
                and not selected_documents
            ),
        )

        if question:
            conversation_id = st.session_state.active_conversation_id

            # Capture only previous messages for contextualization.
            previous_history = [
                {
                    "role": message["role"],
                    "content": message["content"],
                }
                for message in st.session_state.messages
                if message.get("mode", "chat") == "chat"
                and message["role"] in {"user", "assistant"}
            ]

            user_message = {
                "role": "user",
                "content": question,
                "sources": [],
                "timings": {},
                "mode": "chat",
            }

            # Save user message immediately so it survives a failed model call.
            save_message(
                conversation_id=conversation_id,
                role="user",
                content=question,
            )
            st.session_state.messages.append(user_message)

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                try:
                    with st.spinner(
                        "Searching documents and generating an answer..."
                    ):
                        result = ask_question(
                            question,
                            selected_documents=selected_documents,
                            chat_history=previous_history,
                        )

                    answer = result["answer"]
                    sources = result.get("sources", [])
                    timings = result.get("timings", {})

                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 View retrieved sources"):
                            for index, source in enumerate(sources, start=1):
                                st.markdown(
                                    f"**Source {index}:** "
                                    f"{source.get('document', 'Unknown')}"
                                )
                                if source.get("page") is not None:
                                    st.caption(f"Page {source['page']}")
                                st.write(source.get("content", ""))

                    if timings:
                        with st.expander("⚡ Performance Details"):
                            st.json(timings)

                    save_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=answer,
                        sources=sources,
                        timings=timings,
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "timings": timings,
                        "mode": "chat",
                    })

                except Exception as error:
                    error_message = (
                        f"Question answering failed: {error}"
                    )
                    st.error(error_message)

                    # Persist the error response as part of the conversation.
                    save_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=error_message,
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "timings": {},
                        "mode": "chat",
                    })


# --------------------------------------------------
# FEATURE: DOCUMENT COMPARISON
# --------------------------------------------------

with comparison_tab:
    st.header("📊 Compare Documents")
    st.write(
        "Select two or more indexed PDFs to identify their "
        "similarities, differences, key concepts, and unique information."
    )

    # Reopen and display saved document-comparison conversations.
    comparison_messages = [
        message for message in st.session_state.messages
        if message.get("mode") == "comparison"
    ]
    if comparison_messages:
        st.subheader("Saved Comparison")
        for message in comparison_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📚 Supporting Sources"):
                        for index, source in enumerate(message["sources"], start=1):
                            st.markdown(
                                f"**Source {index}:** "
                                f"{source.get('document', 'Unknown')}"
                            )
                            if source.get("page") is not None:
                                st.caption(f"Page {source['page']}")
                            st.write(source.get("content", ""))
                if message.get("timings"):
                    with st.expander("⚡ Performance Details"):
                        st.json(message["timings"])
        st.divider()

    try:
        _, indexed_pdfs = get_document_status()
    except Exception as error:
        indexed_pdfs = []
        st.error(f"Could not retrieve indexed documents: {error}")

    if len(indexed_pdfs) < 2:
        st.info(
            "You need at least two indexed PDFs to use Document Comparison."
        )
    else:
        selected_comparison_documents = st.multiselect(
            "Choose documents to compare",
            options=indexed_pdfs,
            default=[],
            key="comparison_selected_documents",
        )

        comparison_question = st.text_area(
            "What would you like to compare?",
            value=(
                "Compare the main topics, key concepts, similarities, "
                "differences, and unique information in the selected documents."
            ),
            height=100,
            key="comparison_question",
        )

        if st.button(
            "🔍 Compare Documents",
            use_container_width=True,
            type="primary",
        ):
            if len(selected_comparison_documents) < 2:
                st.warning("Please select at least two PDFs.")
            elif not comparison_question.strip():
                st.warning("Please enter a comparison question.")
            else:
                try:
                    with st.spinner(
                        "Retrieving document content and generating comparison..."
                    ):
                        result = compare_documents(
                            selected_documents=selected_comparison_documents,
                            comparison_question=comparison_question.strip(),
                        )

                    # Save the comparison in the CURRENT conversation, so a
                    # single chat can contain both PDF Q&A and comparisons.
                    conversation_id = st.session_state.active_conversation_id
                    comparison_prompt = (
                        "Compare documents: "
                        + ", ".join(selected_comparison_documents)
                        + "\n\nQuestion: "
                        + comparison_question.strip()
                    )
                    comparison_answer = result.get(
                        "answer", "No comparison answer was returned."
                    )
                    comparison_sources = result.get("sources", [])
                    comparison_timings = result.get("timings", {})

                    save_message(
                        conversation_id=conversation_id,
                        role="user",
                        content=comparison_prompt,
                        mode="comparison",
                    )
                    save_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=comparison_answer,
                        sources=comparison_sources,
                        timings=comparison_timings,
                        mode="comparison",
                    )

                    st.session_state.messages = get_messages(conversation_id)
                    st.success("Comparison saved in the current chat.")
                    st.rerun()

                except Exception as error:
                    st.error(
                        f"Document comparison failed: {error}"
                    )
