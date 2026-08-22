import os
import shutil
import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="AI Book Assistant",
    page_icon="📚"
)

st.title("📚 AI Book Assistant")
st.write("Upload a PDF and ask questions about it.")


# --------------------------------
# Load environment
# --------------------------------

from dotenv import load_dotenv
load_dotenv()


# --------------------------------
# Embedding model
# --------------------------------

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = get_embeddings()


# --------------------------------
# Mistral
# --------------------------------

@st.cache_resource
def get_llm():

    return ChatMistralAI(
        model="mistral-small-2506"
    )


llm = get_llm()


# --------------------------------
# Upload PDF
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF book",
    type=["pdf"]
)


# --------------------------------
# Process PDF
# --------------------------------

if uploaded_file is not None:

    if st.button("Process PDF"):

        # Remove old database
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")

        # Save uploaded PDF
        os.makedirs("temp", exist_ok=True)

        pdf_path = os.path.join(
            "temp",
            uploaded_file.name
        )

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing PDF..."):

            # -------------------------
            # Load PDF
            # -------------------------

            loader = PyPDFLoader(pdf_path)

            docs = loader.load()

            # -------------------------
            # Split PDF
            # -------------------------

            splitter = TokenTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(docs)

            # -------------------------
            # Remove empty chunks
            # -------------------------

            chunks = [
                chunk
                for chunk in chunks
                if isinstance(chunk.page_content, str)
                and chunk.page_content.strip()
            ]

            # -------------------------
            # Create Chroma DB
            # -------------------------

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )

        st.session_state.vectorstore = vectorstore
        st.session_state.processed = True

        st.success("PDF processed successfully!")

        st.write(f"📄 Pages: {len(docs)}")
        st.write(f"🧩 Chunks: {len(chunks)}")


# --------------------------------
# Chat
# --------------------------------

if "processed" in st.session_state and st.session_state.processed:

    st.divider()

    st.subheader("💬 Ask Questions")

    query = st.chat_input(
        "Ask something about your book..."
    )

    if query:

        # -------------------------
        # Retriever
        # -------------------------

        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        # -------------------------
        # Retrieve documents
        # -------------------------

        docs = retriever.invoke(query)

        # -------------------------
        # Context
        # -------------------------

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # -------------------------
        # Prompt
        # -------------------------

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are a helpful AI assistant.

                Use ONLY the provided context to answer
                the question.

                If the answer is not present in the context,
                say:

                "I could not find the answer in the document."

                Do not make up information.
                """
            ),
            (
                "human",
                """
                Context:
                {context}

                Question:
                {question}
                """
            )
        ])

        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        # -------------------------
        # Mistral
        # -------------------------

        with st.spinner("Thinking..."):

            response = llm.invoke(final_prompt)

        # -------------------------
        # Display
        # -------------------------

        st.chat_message("user").write(query)

        st.chat_message("assistant").write(
            response.content
        )