from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# -------------------------
# 1. Load PDF
# -------------------------

loader = PyPDFLoader(
    "document loaders/deep learning.pdf"
)

docs = loader.load()

print("PDF pages:", len(docs))


# -------------------------
# 2. Split documents
# -------------------------

splitter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print("Total chunks:", len(chunks))


# -------------------------
# 3. Clean chunks
# -------------------------

valid_chunks = []

for chunk in chunks:

    text = chunk.page_content

    if isinstance(text, str) and text.strip():

        chunk.page_content = text.strip()

        valid_chunks.append(chunk)

chunks = valid_chunks

print("Valid chunks:", len(chunks))


# -------------------------
# 4. Debug
# -------------------------

for i, chunk in enumerate(chunks[:5]):

    print("\nChunk:", i)
    print("Type:", type(chunk.page_content))
    print("Length:", len(chunk.page_content))
    print("Preview:", repr(chunk.page_content[:100]))


# -------------------------
# 5. Embedding model
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------
# 6. Create vector DB
# -------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("\nVector database created successfully!")