from langchain_community.retrievers import ArxivRetriever

# create the retirever
retriever = ArxivRetriever(
  load_max_docs=2,
  load_all_available_meta=True
)

#query arxiv

docs = retriever.invoke("large language model")

#print result

for i , doc in enumerate(docs):
   print(f"\nResult {i+1}")
   print("Title:", doc.metadata.get("Title"))
  
   print("Authors :", doc.metadata.get("Author"))
  
   print("Summary :", doc.page_content[:1000])
  