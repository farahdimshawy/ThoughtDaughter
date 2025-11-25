import streamlit as st
import os
from dotenv import load_dotenv

# Import the core retrieval function
from retrieval import hybrid_retrieve 

# Database and LLM related imports
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


load_dotenv()

st.title("₊✩‧₊Thought Daughter₊✩‧₊")

# --- INITIALIZATION ---

# Initialize Pinecone and Embeddings
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME") 

# Check if index exists before proceeding (Fix #1 for robustness)
if index_name not in [idx['name'] for idx in pc.list_indexes()]:
    st.error(f"Pinecone index '{index_name}' not found. Please run the ingestion script first.")
    st.stop()
    
index = pc.Index(index_name)

# Initialize embeddings model + vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# --- CHAT HISTORY ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add a clean, permanent base system message (This is the first message)
    st.session_state.messages.append(SystemMessage("You are an assistant for question-answering tasks. Use a professional and concise tone."))

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# --- PROMPT PROCESSING ---

prompt = st.chat_input("Ask me anything")

if prompt:

    # Add the user's new message to the history first
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    # Initialize the LLM for this turn
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1 # Lower temperature for factual Q&A
    )

    # 1. Initialize the Retriever (search_kwargs set to 0.75 is correct)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.75}, 
    )

    # 2. Perform Hybrid Retrieval
    docs = hybrid_retrieve(prompt, retriever)
    docs_text = "\n---\n".join(d.page_content for d in docs)

    # 3. Construct the Turn-Specific Context Prompt
    context_template = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If the context does not contain the answer, state that you don't know based on the provided information. 
    Use three sentences maximum and keep the answer concise.
    
    Context:
    ---
    {context}
    ---
    
    Answer the question based on the context: {question}
    """
    
    # Populate the context injection prompt with the retrieved documents and user query
    context_injection_message = context_template.format(context=docs_text, question=prompt)

    # 4. Prepare the final message list for LLM invocation
    # Fix #2: Use the existing, initial SystemMessage for consistency.
    messages_for_llm = [
        st.session_state.messages[0], # The original system message
        HumanMessage(context_injection_message) # The query + context
    ]

    # 5. Invoke the LLM with the clean, focused prompt
    result = llm.invoke(messages_for_llm).content

    # 6. Add the final response to the screen (and history)
    with st.chat_message("assistant"):
        st.markdown(result)
    st.session_state.messages.append(AIMessage(result))

    # 7. Displaying the sources (Sources will now show Pinecone, Wikipedia URL, or Tavily URL)
    with st.expander("Sources"):
        if docs:
            for doc in docs:
                # Use markdown for sources to make URLs clickable
                source_link = doc.metadata.get('source', 'Pinecone (Source Missing)')
                st.markdown(f"- Source: **{source_link}**")
        else:
            st.markdown("No context retrieved for this query.")