import argparse
import time
import uuid
import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import the necessary fetch functions from retrieval.py
# The hybrid_retrieve function is not used directly for ingestion, 
# but its component functions (fetch_wikipedia/fetch_google) are.
from retrieval import fetch_wikipedia, fetch_google

load_dotenv()

# -------------------------------
# Pinecone setup
# -------------------------------
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = os.environ.get("PINECONE_INDEX_NAME") 
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
existing_indexes = pc.list_indexes().names

if existing_indexes:
    print(f"Found existing indexes: {existing_indexes}")
    
    # 2. Iterate through the list and delete each one 
    for index_name in existing_indexes:
        print(f"Deleting index: {index_name}...")
        pc.delete_index(name=index_name)
        print(f"Index {index_name} deleted.")
else:
    print("No existing indexes found. Ready to create new ones.")

if index_name not in existing_indexes:
    print(f"Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# -------------------------------
# Embeddings + vector store
# -------------------------------
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# -------------------------------
# Main Execution Block (for terminal use only)
# -------------------------------
if __name__ == "__main__":
    
    # 1. Command-line arguments
    parser = argparse.ArgumentParser(description='Ingest web pages into Pinecone.')
    parser.add_argument('titles', type=str, nargs='+', help='List of web queries or titles to ingest')
    args = parser.parse_args()
    titles = args.titles 

    print(f"Starting ingestion for queries: {titles}")

    # 2. Load and prepare documents
    raw_documents = []
    
    for title in titles:
        print(f"\n--- Fetching data for: {title} ---")
        
        # --- Fetch Wikipedia ---
        # Assuming fetch_wikipedia now returns a Document object (with URL source)
        wiki_doc = fetch_wikipedia(title)
        if wiki_doc:
            raw_documents.append(wiki_doc)
            print("Fetched Wikipedia.")

        # --- Fetch Google/Tavily ---
        # Assuming fetch_google now returns a list of dictionaries [{'content': ..., 'source': ...}]
        google_results = fetch_google(title)
        
        if google_results and isinstance(google_results, list):
            print(f"Fetched {len(google_results)} Google/Tavily snippets.")
            for result in google_results:
                raw_documents.append(
                    Document(
                        page_content=result.get('content', ''),
                        metadata={
                            "title": title, 
                            "source": result.get('source', 'Google/Tavily Link')
                        }
                    )
                )

        # 3. Chunk documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=400, 
            length_function=len
        )
        documents = text_splitter.split_documents(raw_documents)
        
        # 4. Generate unique IDs and Add to Pinecone
        uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=uuids)
        
        print(f"\nIngestion Complete! Added {len(documents)} total chunks to Pinecone.")