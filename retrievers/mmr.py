from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# -------------------------
# 1. Create 3 documents
# -------------------------

docs = [
    Document(
        page_content="""
        Artificial Intelligence allows computers to perform tasks
        that normally require human intelligence and reasoning.
        """
    ),

    Document(
        page_content="""
        Machine Learning is a branch of AI where computers learn
        patterns from data and use them to make predictions.
        """
    ),

    Document(
        page_content="""
        Deep Learning uses neural networks with multiple layers.
        It is widely used for computer vision and natural language processing.
        """
    )
]


# -------------------------
# 2. Create embeddings
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------
# 3. Create vector database
# -------------------------

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)


# -------------------------
# 4. Similarity Search
# -------------------------

query = "What is machine learning?"

results = vectorstore.similarity_search(
    query,
    k=2
)

print("\n========== SIMILARITY SEARCH ==========\n")

for i, doc in enumerate(results):
    print(f"Document {i + 1}:")
    print(doc.page_content.strip())
    print()


# -------------------------
# 5. MMR Search
# -------------------------

mmr_results = vectorstore.max_marginal_relevance_search(
    query,
    k=2,
    fetch_k=3,
    lambda_mult=0.5
)

print("\n========== MMR SEARCH ==========\n")

for i, doc in enumerate(mmr_results):
    print(f"Document {i + 1}:")
    print(doc.page_content.strip())
    print()