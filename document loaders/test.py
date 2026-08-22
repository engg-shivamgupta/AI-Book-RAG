from langchain_core.documents import Document

#Splitting by character - Text splitter
#link docs -  https://docs.langchain.com/oss/python/integrations/splitters/character_text_splitter

from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
  separator="",
  chunk_size = 10,
  chunk_overlap = 1
)
with open("document loaders/notes.txt", "r", encoding="utf-8") as f:
    documents = [Document(page_content=f.read())]


chunks = splitter.split_documents(documents)

for i in chunks:
  print(i.page_content)
  print()
  print()
  