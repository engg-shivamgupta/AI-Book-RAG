from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


# -------------------------
# Embedding model
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------
# Load existing vector DB
# -------------------------

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# -------------------------
# Retriever
# -------------------------

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)


# -------------------------
# Mistral LLM
# -------------------------

llm = ChatMistralAI(
    model="mistral-small-2506"
)


# -------------------------
# Prompt
# -------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.

        Use ONLY the provided context to answer the question.

        If the answer is not present in the context, say:
        "I could not find the answer in the document."
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


print("RAG system created")
print("Press 0 to exit")


# -------------------------
# RAG loop
# -------------------------

while True:

    query = input("\nYou: ")

    if query == "0":
        break

    # Retrieve relevant documents
    docs = retriever.invoke(query)
    # print("\n===== RETRIEVED DOCUMENTS =====")

    # for i, doc in enumerate(docs):
    #  print(f"\n--- Document {i+1} ---")
    #  print(doc.page_content[:1000])
    #  print(f"\nRetrieved {len(docs)} documents")

    # Create context
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )
    # print("\n===== CONTEXT SENT TO MISTRAL =====")
    # print(context)

    # Create prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # Generate answer
    response = llm.invoke(final_prompt)

    # print("\nResponse object:")
    # print(response)
    print("AI:")
    print("\nResponse content:")
    print(response.content)