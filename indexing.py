from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader # PDF loader
from langchain_text_splitters import RecursiveCharacterTextSplitter # for chinking the documents
from langchain_google_genai import GoogleGenerativeAIEmbeddings #Embedding model
from langchain_qdrant import QdrantVectorStore      # Vector database    
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


# create embedding model
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# create and store the embeddings in the vector database
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    url="http://localhost:6333",
    collection_name="chat-pdf"
)

print("Document indexing done")