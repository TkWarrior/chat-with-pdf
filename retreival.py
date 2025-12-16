from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from openai import OpenAI
import os

load_dotenv()

# create embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/")

# load the existing vector database
vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="chat-pdf"
)

# take user query as input
user_query = input("Ask from Pdf📄 :")

# retrieve relevant documents from the vector database
search_res = vector_db.similarity_search(query=user_query)

# prepare the context from the search results
context = "\n\n\n".join([f"page content : {res.page_content}\npage number : {res.metadata['page_label']}\nfile location : {res.metadata['source']}" 
           for res in search_res])


SYSTEM_PROMPT = """
    You're an helpful AI assistant who answer query based on available context retreived from the pdf
    file along with page_content and page_number.
    You should only answer the user query based on the follpwing context and navigating the user to open
    the right page number to know more
    
    Context:{context}
"""

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role":"system" , "content":SYSTEM_PROMPT},
        {"role":"user" , "content":user_query}   
    ]
)

print(f"🤖:{response.choices[0].message.content}")