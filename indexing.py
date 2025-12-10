from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pathlib import Path

load_dotenv()

file_path = Path(__file__).parent/"Manuscript_lirias.pdf"

## PDF loading
loader = PyPDFLoader(file_path)

docs = loader.load()
# print(docs[1])

#splitting docs into chunks 

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
chunks = text_splitter.split_documents(documents=docs)

# creating vector embedding of the chunks and storing it in the vector database

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    url="http://localhost:6333",
    collection_name="chat-pdf"
)

print("Document indexing done")