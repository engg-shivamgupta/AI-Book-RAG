from langchain_community.document_loaders import WebBaseLoader

url = "https://en.wikipedia.org/wiki/MacBook"

data = WebBaseLoader(url)
docs = data.load()
print(docs[0].page_content)