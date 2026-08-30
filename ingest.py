import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

print("Reading PDF...")
loader = PyPDFLoader("data/Automation_Testing_Master_Blueprint.pdf")
documents = loader.load()
print(f"{len(documents)} pages mile.")

# DEBUG: dekhte hain pehle page mein text hai ya nahi
if len(documents) > 0:
    print("Pehle page ka text (pehle 300 characters):")
    print(repr(documents[0].page_content[:300]))
else:
    print("⚠️ Koi page hi nahi mila!")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print(f"{len(chunks)} chunks bane.")

# Agar chunks 0 hain, to yahi ruk jao
if len(chunks) == 0:
    print("⚠️ PDF se text nahi nikla — ye scanned/image PDF ho sakti hai.")
    exit()

print("Embeddings ban rahe hain... (pehli baar thoda time lagega)")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("✅ Done! Notes vector database mein save ho gaye.")