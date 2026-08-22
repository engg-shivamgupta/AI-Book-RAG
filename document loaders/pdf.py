# to load pdf , pdfloader
from langchain_community.document_loaders import PyPDFLoader


#Tokentextsplitter
from langchain_text_splitters import TokenTextSplitter


data = PyPDFLoader("document loaders/TourEase PRD.pdf")

docs = data.load()

splitter = TokenTextSplitter(
  chunk_size = 1000,
  chunk_overlap = 10
)

chunks = splitter.split_documents(docs)
print(len(chunks))

print(chunks[0])

print(chunks[0].page_content)

#jst go to main.py import the 1 to 3 lines and change notes.txt to tourease.pdf  and then the mistral ai will use this pdf for answers.

#---------------------------------------

# Splitting recursively - Text splitter
# ----------------------------------------
# This text splitter is the recommended one for generic text. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.
# How the text is split: by list of characters.
# How the chunk size is measured: by number of characters.


# from langchain_text_splitters import CharacterTextSplitter

# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#     encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
# )
# texts = text_splitter.split_text(document)