from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

load_dotenv()


# -------------------------
# 1. Create documents
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
    ),
]


# -------------------------
# 2. Create embeddings
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------
# 3. Create Chroma DB
# -------------------------

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)


# -------------------------
# 4. Create base retriever
# -------------------------

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)


# -------------------------
# 5. Create Mistral LLM
# -------------------------

llm = ChatMistralAI(
    model="mistral-small-2506"
)


# -------------------------
# 6. MultiQuery Retriever
# -------------------------

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)


# -------------------------
# 7. Query
# -------------------------

query = "What is machine learning?"


# -------------------------
# 8. Retrieve documents
# -------------------------

results = multi_query_retriever.invoke(query)


# -------------------------
# 9. Print results
# -------------------------

print("\n========== MULTI QUERY RESULTS ==========\n")

for i, doc in enumerate(results):
    print(f"Document {i + 1}")
    print(doc.page_content.strip())
    print()