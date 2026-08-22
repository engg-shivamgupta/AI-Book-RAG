📚 AI Book RAG Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF book and ask questions about its content through an interactive Streamlit interface.

The application extracts text from the uploaded PDF, splits it into chunks, generates local vector embeddings using Sentence Transformers, stores them in Chroma, retrieves relevant information using MMR (Maximal Marginal Relevance), and uses Mistral AI to generate the final answer.

🚀 Features
📄 Upload your own PDF book
🔍 Extract text from PDF documents
✂️ Split documents into smaller chunks
🧠 Generate embeddings using sentence-transformers/all-MiniLM-L6-v2
🗄️ Store embeddings using Chroma Vector Database
🎯 MMR-based document retrieval
🤖 Mistral AI for answer generation
💬 Interactive Streamlit chat interface
🔐 API keys managed using .env
🧪 Examples of different LangChain retrievers
🚫 Prevents the LLM from answering outside the retrieved document context
🏗️ RAG Architecture
                  <img width="312" height="638" alt="image" src="https://github.com/user-attachments/assets/e855c7f1-58e0-403a-b7e1-b2ad1a4b12a4" />

🧰 Tech Stack
Technology	Purpose
Python	Core programming language
Streamlit	Web UI and chat interface
LangChain	RAG orchestration
Mistral AI	LLM / answer generation
Hugging Face Sentence Transformers	Local embeddings
Chroma	Vector database
PyPDF	PDF document loading
TokenTextSplitter	Document chunking
python-dotenv	Environment variable management
📂 Project Structure
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

.env, .venv, chroma_db, temporary files, uploaded files, and local caches are excluded from Git using .gitignore.

🔄 How the RAG Pipeline Works
1. Upload PDF

Users can upload their own PDF through the Streamlit interface.

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)
2. Load PDF

PyPDFLoader extracts the text from the PDF.

loader = PyPDFLoader(pdf_path)


docs = loader.load()
3. Split Documents

The extracted document is divided into smaller chunks.

splitter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)


chunks = splitter.split_documents(docs)

This allows the retriever to search smaller and more relevant pieces of the document.

4. Generate Embeddings

The project uses:

sentence-transformers/all-MiniLM-L6-v2

to convert text chunks into numerical vector representations.

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

The embedding model runs locally.

5. Store Vectors in Chroma

The document chunks and embeddings are stored in Chroma.

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
6. Retrieve Relevant Documents

The application uses MMR retrieval:

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

MMR balances:

Relevance
    +
Diversity

This helps prevent the retriever from returning multiple highly similar chunks.

7. Build Context

The retrieved documents are combined into a context:

context = "\n\n".join(
    doc.page_content
    for doc in docs
)
8. Generate Answer

The retrieved context and user's question are passed to Mistral AI.

The system prompt instructs the model to only use the retrieved context:

Use ONLY the provided context to answer the question.


If the answer is not present in the context, say:


"I could not find the answer in the document."

This helps reduce hallucinations and keeps the answer grounded in the uploaded document.

💬 Example

After uploading a deep learning book, a user can ask:

What is the Word2Vec framework?

The application performs:

User Question
      ↓
MMR Retriever
      ↓
Relevant Word2Vec Chunks
      ↓
Context
      ↓
Mistral AI
      ↓
Final Answer
🔎 Retriever Experiments

The repository also contains separate examples for experimenting with different retrieval techniques.

Similarity Search

Retrieves documents based on semantic similarity to the query.

MMR

Retrieves relevant documents while also promoting diversity between the selected chunks.

MultiQuery Retriever

Generates multiple variations of a user query and searches the vector database using those variations.

This can improve retrieval when the user's wording differs from the wording used in the document.

Arxiv Retriever

An example of retrieving research-paper information from arXiv.

These experiments are stored separately in the retrievers/ directory.

⚙️ Installation
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
2. Create a virtual environment
python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
🔑 Environment Variables

Create a .env file in the project root:

MISTRAL_API_KEY=your_mistral_api_key

You can use .env.example as a template.

Never upload your actual .env file or API keys to GitHub.

▶️ Run the Application

Start the Streamlit application:

streamlit run app.py

Streamlit will provide a local URL such as:

http://localhost:8501

Open it in your browser, upload a PDF, process it, and start asking questions.

🗃️ Vector Database

The application creates a local Chroma database after processing the uploaded PDF.

chroma_db/

The database is intentionally excluded from Git because it is generated locally.

To rebuild the database from scratch:

Delete chroma_db/
        ↓
Run the application
        ↓
Upload PDF
        ↓
Process PDF
        ↓
New Chroma database
🔐 Security

API keys are loaded from environment variables:

from dotenv import load_dotenv


load_dotenv()

The actual API key is never hardcoded in the source code.

The .gitignore file prevents sensitive/local files such as:

.env
.venv/
chroma_db/
temp/
uploads/

from being committed.

📸 Screenshots
GitHub Repository Structure

Retriever Examples

Document Loaders

📈 Future Improvements
 Conversation memory
 Source/page citations in answers
 Multiple PDF support
 Document metadata filtering
 Hybrid search
 Reranking
 Streaming Mistral responses
 Authentication and user accounts
 PostgreSQL / pgvector support
 Document management and deletion
 RAG evaluation metrics
 Cloud deployment
 OCR support for scanned PDFs
🎯 Learning Objectives

This project demonstrates practical implementation of:

Retrieval-Augmented Generation (RAG)
PDF document ingestion
Text chunking
Vector embeddings
Vector databases
Semantic search
MMR retrieval
Multi-query retrieval
LLM prompting
Mistral AI integration
Streamlit application development
Environment variable management
Document-grounded question answering
👨‍💻 Author

Shivam Gupta

Built as a practical RAG project to explore document question-answering, vector search, retrieval strategies, embeddings, and LLM integration.

⭐ If you find this project useful, consider giving the repository a star!
