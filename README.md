# 📚 AI Book RAG Assistant

A **Retrieval-Augmented Generation (RAG)** application that lets users upload a PDF book and ask questions about its content through a Streamlit chat interface.

The application extracts text from the uploaded PDF, splits it into chunks, generates vector embeddings locally using **Sentence Transformers**, stores those embeddings in **ChromaDB**, retrieves relevant chunks using **MMR (Maximal Marginal Relevance)**, and uses **Mistral AI** to generate the final answer.

---

## 🚀 Features

- 📄 Upload any PDF book through the Streamlit UI
- 🔍 Extract text from PDF documents
- ✂️ Split documents into manageable chunks
- 🧠 Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- 🗄️ Store and search vectors using Chroma
- 🎯 Retrieve relevant and diverse chunks using MMR
- 🤖 Generate answers using Mistral AI
- 💬 Interactive Streamlit chat interface
- 🔐 API keys stored in `.env`
- 🛡️ `.env`, virtual environments, local vector databases, and temporary files excluded through `.gitignore`
- 🧪 Includes examples of different LangChain retrievers

---

## 🏗️ RAG Architecture

```text
                    ┌──────────────────┐
                    │   Upload PDF     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   PyPDFLoader    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Text Splitter   │
                    │ 1000 / 100 chunk │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ HuggingFace      │
                    │ Embeddings       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Chroma       │
                    │   Vector Store   │
                    └────────┬─────────┘
                             │
                       User Question
                             │
                             ▼
                    ┌──────────────────┐
                    │   MMR Retriever  │
                    │ k=4, fetch_k=10  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Retrieved Context│
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Mistral AI    │
                    │  Answer Generator│
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Streamlit UI    │
                    └──────────────────┘
```

---

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application language |
| Streamlit | Web UI and chat interface |
| LangChain | RAG orchestration |
| Mistral AI | LLM / answer generation |
| Hugging Face Sentence Transformers | Local text embeddings |
| Chroma | Vector database |
| PyPDF | PDF document loading |
| TokenTextSplitter | Document chunking |
| python-dotenv | Environment variable management |

---

## 📂 Project Structure

```text
RAG PROJECT/
│
├── app.py
├── create_db.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
│
├── document loaders/
│   ├── deep_learning.pdf
│   ├── notes.txt
│   ├── pdf.py
│   ├── test.py
│   └── web.py
│
├── retrievers/
│   ├── arixv.py
│   ├── mmr.py
│   └── multiquery.py
│
└── vector store/
    └── ...
```

> `chroma_db/`, `.venv/`, `.env`, uploaded/temporary files, and local model/cache files should remain outside Git tracking.

### Repository screenshots

![Repository structure](assets/github-repository-structure.png)

![Retrievers](assets/retrievers-folder.png)

![Document loaders](assets/document-loaders-folder.png)

---

## 🔄 How the RAG Pipeline Works

### 1. Upload PDF

The user uploads a PDF through Streamlit.

```python
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)
```

The PDF is temporarily saved locally.

### 2. Load the document

`PyPDFLoader` extracts the text and metadata from the PDF.

```python
loader = PyPDFLoader(pdf_path)
docs = loader.load()
```

### 3. Split the document

Large documents are divided into smaller chunks.

```python
splitter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)
```

Chunking makes retrieval more precise and prevents the entire book from being sent to the LLM.

### 4. Generate embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model converts each chunk into a numerical vector representing its semantic meaning.

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

The embedding model runs locally, so document embeddings do not require an OpenAI embedding API call.

### 5. Store vectors in Chroma

The chunks and their embeddings are stored in Chroma.

```python
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
```

### 6. Retrieve relevant information

The application uses MMR retrieval:

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)
```

MMR balances **relevance** and **diversity**, helping prevent the retriever from returning several nearly identical chunks.

### 7. Build the context

The retrieved chunks are combined into the context sent to the LLM.

```python
context = "\n\n".join(
    doc.page_content
    for doc in docs
)
```

### 8. Generate the answer

The context and question are passed to Mistral AI.

The system prompt instructs the model to use only the retrieved context and avoid making up information.

```text
Use ONLY the provided context to answer the question.

If the answer is not present in the context, say:
"I could not find the answer in the document."
```

---

## 💬 Example

A user can upload a deep learning book and ask:

```text
What is the Word2Vec framework?
```

The system:

```text
Question
   ↓
MMR Retriever
   ↓
Word2Vec-related chunks
   ↓
Context
   ↓
Mistral
   ↓
Answer
```

The model can then answer from the retrieved book content instead of relying only on its pretrained knowledge.

---

## 🔎 Retriever Experiments

The repository also contains examples for experimenting with different retrieval approaches.

### Similarity Search

Retrieves the documents with the highest semantic similarity to the query.

### MMR

Balances relevance and diversity.

```python
search_type="mmr"
```

### MultiQuery Retriever

Uses an LLM to generate alternative versions of a user's query before performing retrieval. This can improve retrieval when the original query is ambiguous or uses different terminology from the source documents.

### Arxiv Retriever

An example retriever for obtaining research-paper information from arXiv.

These examples are kept separately in the `retrievers/` directory so the main application remains simple.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key
```

The repository contains `.env.example` as a template.

**Never commit your real `.env` file or API keys to GitHub.**

The `.gitignore` file excludes:

```text
.env
.venv/
chroma_db/
temp/
uploads/
```

---

## ▶️ Run the Streamlit Application

Start the application with:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

Upload a PDF, process it, and start asking questions.

---

## 🗃️ Vector Database

The application creates a local Chroma database after processing the uploaded document.

The generated database is intentionally ignored by Git:

```text
chroma_db/
```

This keeps the repository lightweight and prevents generated vector data from being committed.

If you want to rebuild the database from scratch, delete:

```text
chroma_db/
```

and upload/process the PDF again.

---

## 🔐 Security

API credentials are loaded using environment variables:

```python
from dotenv import load_dotenv

load_dotenv()
```

The actual API key is never hardcoded into the Python source code.

Before pushing the project to GitHub, verify:

```bash
git status
```

and make sure `.env` is not listed as a tracked file.

---

## 📈 Possible Future Improvements

- Add conversation memory
- Add source/page citations in answers
- Support multiple uploaded books simultaneously
- Add document metadata filtering
- Add hybrid keyword + vector search
- Add reranking
- Add streaming Mistral responses
- Add authentication and user accounts
- Add PostgreSQL/pgvector for production deployment
- Add document deletion and management
- Add RAG evaluation metrics
- Deploy the application to Streamlit Cloud or another cloud platform
- Add OCR support for scanned PDFs

---

## 🎯 Learning Objectives

This project demonstrates practical understanding of:

- Retrieval-Augmented Generation
- Document ingestion
- PDF processing
- Text chunking
- Vector embeddings
- Vector databases
- Semantic search
- MMR retrieval
- LLM prompting
- Mistral AI API integration
- Streamlit application development
- Environment variable management
- Basic RAG architecture

---

## 👨‍💻 Author

**Shivam Gupta**

Built as a practical RAG project to explore document question-answering, vector search, retrieval strategies, and LLM integration.

---

## ⭐ If You Find This Project Useful

Feel free to star the repository and experiment with different embedding models, chunking strategies, retrievers, and LLMs.
