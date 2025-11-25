# import basics
import os
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
import wikipediaapi
from tavily import TavilyClient
import os

load_dotenv()

# initialize pinecone database
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
# set the pinecone index

index_name = os.environ.get("PINECONE_INDEX_NAME") 
index = pc.Index(index_name)

# initialize embeddings model + vector store

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

USER_AGENT_STRING = 'RAGChatbot/1.0 (farah@example.com)'

# 1. Use the positional 'user_agent' first (as the library demands)
# 2. Use the keyword 'language' second for clarity
wiki = wikipediaapi.Wikipedia(USER_AGENT_STRING, language='en')

# wiki._sUSER_AGENT_STRING = 'RAGChatbot/1.0 (farah@example.com)'

# wiki = wikipediaapi.Wikipedia(
#     language='en', 
#     user_agent=USER_AGENT_STRING 
# ) 


def fetch_wikipedia(query):
    page = wiki.page(query)
    
    # Check if the page exists
    if page.exists():
        # *** CRITICAL FIX: Return a Document with the actual URL in metadata ***
        return Document(
            page_content=page.text,
            metadata={
                "source": page.fullurl, # <-- This is the Wikipedia page URL
                "title": page.title
            }
        )
    return None 

def fetch_google(query, max_chars=2000): 
    try:
        response = tavily_client.search(query=query, search_depth="basic")
        
        google_docs = [] # Will hold Document objects
        
        for result in response.get('results', []):
            content = result.get('content', '')
            url = result.get('url', 'Google/Tavily Link') # Get the source URL
            
            # Create a Document for each snippet
            google_docs.append(
                Document(
                    page_content=content[:max_chars], # Truncate content
                    metadata={"source": url}
                )
            )
            
        return google_docs # Return a list of Documents
    
    except Exception as e:
        print(f"Tavily Error: {e}")
        return [] # Return empty list on failure

from langchain_core.documents import Document

def hybrid_retrieve(query, retriever):
    # Step 1: Pinecone retrieval
    pinecone_docs = retriever.invoke(query)
    # Ensure metadata includes 'source'
    for doc in pinecone_docs:
        if "source" not in doc.metadata:
            doc.metadata["source"] = "Pinecone (Internal DB)" # Clarified source

    # Step 2: Wikipedia + Google (Web Search)
    web_docs = []
    
    # 2a: Retrieve Wikipedia Document (Expected to return a Document object with URL in metadata)
    wiki_doc = fetch_wikipedia(query) 
    if wiki_doc:
        web_docs.append(wiki_doc)
    
    # 2b: Retrieve Google/Tavily Results (Expected to return a list of structured results)
    google_results = fetch_google(query)
    if google_results: 
        web_docs.extend(google_results) # Use .extend() to add all Documents from the list
    
    # # Iterate through each structured result from the web search
    # if google_results and isinstance(google_results, list):
    #     for result in google_results:
    #         # We assume result is a dict: {'content': '...', 'source': 'http://link.com'}
    #         web_docs.append(
    #             Document(
    #                 page_content=result.get('content', ''),
    #                 metadata={"source": result.get('source', 'Google/Tavily Link')} # <-- Stores the actual URL
    #             )
    #         )

    return pinecone_docs + web_docs

def main():
    # 1. Setup Argument Parser
    import argparse
    parser = argparse.ArgumentParser(
        description="Hybrid RAG Retrieval Script. Executes the retrieval process based on a command-line query."
    )
    # Define the required query argument
    parser.add_argument(
        "query", 
        type=str, 
        help="The query to search for (e.g., 'what is retrieval augmented generation?')"
    )
    
    # Parse the arguments
    args = parser.parse_args()
    user_query = args.query

    print(f"Running retrieval for query: **'{user_query}'**")
    
    # 2. Initialize Pinecone Retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.5},
    )

    # 3. Perform Hybrid Retrieval
    results = hybrid_retrieve(user_query, retriever)

    # 4. Show Results
    print("\n--- RESULTS ---")
    
    if not results:
        print("No results found from Pinecone, Wikipedia, or Google.")
        return

    for i, res in enumerate(results):
        print(f"\n#{i+1}:")
        print(f"  Source: **{res.metadata.get('source', 'N/A')}**")
        print(f"  Content Snippet (First 200 chars):")
        # Print a snippet of the content for readability
        print(f"    > {res.page_content.strip()[:200]}...")


# This block ensures that 'main()' only runs when the script is executed directly 
# from the command line (e.g., `python retrieval.py "my query"`), 
# and not when it's imported into another file (like a Streamlit app).
if __name__ == "__main__":
    main()