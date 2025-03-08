import streamlit as st
import os
import numpy as np
import faiss
import json
from mistralai import Mistral, UserMessage

# Page configuration
st.set_page_config(
    page_title="UDST Policy RAG Assistant", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .policy-header {
        margin-bottom: 1.5rem;
    }
    .answer-container {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.title("UDST Policy Q&A Assistant")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("This chatbot answers questions about UDST policies using RAG (Retrieval Augmented Generation).")
    
    st.subheader("How it works")
    st.write("1. Select a policy from the dropdown")
    st.write("2. Enter your question about the policy")
    st.write("3. View the answer and source references")
    
    st.markdown("---")
    
    api_key = "LoXIODO6VkldB64uwva76l1zDpIz6cfu"
    
    # Check if preprocessing is done
    if not os.path.exists("indexes") or len(os.listdir("indexes")) == 0:
        st.error("Policy data not found. Please run the preprocess.py script first.")
    else:
        st.success(f"Found data for {len(os.listdir('indexes'))} policies")

# Policy URLs list
POLICY_URLS = {
    "Sport and Wellness Facilities": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and",
    "Credit Hour Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
    "Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy",
    "Student Appeals Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy",
    "Student Attendance Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy",
    "Student Counselling Services": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-counselling-services-policy",
    "Library Space Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/use-library-space-policy",
    "Transfer Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/transfer-policy",
    "Academic Schedule Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/academic-schedule-policy",
    "Student Conduct Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy"
}

# Function to get safe filename from policy name
def get_safe_filename(policy_name):
    return policy_name.lower().replace(" ", "_")

# Function to get text embeddings
def get_text_embedding(list_txt_chunks, api_key):
    client = Mistral(api_key=api_key)
    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=list_txt_chunks
    )
    return embeddings_batch_response.data

# Function to query Mistral
def query_mistral(prompt, api_key):
    client = Mistral(api_key=api_key)
    messages = [UserMessage(content=prompt)]
    try:
        chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=messages,
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        # Fallback to small model if large is unavailable
        try:
            chat_response = client.chat.complete(
                model="mistral-small-latest",
                messages=messages,
            )
            return chat_response.choices[0].message.content + "\n\n(Note: Used fallback model due to unavailability of primary model)"
        except:
            return f"Error: Unable to generate response. {str(e)}"

# Get list of available policies (those with processed data)
def get_available_policies():
    if not os.path.exists("indexes"):
        return []
    
    available = []
    for policy, url in POLICY_URLS.items():
        safe_name = get_safe_filename(policy)
        if os.path.exists(f"indexes/{safe_name}.index") and os.path.exists(f"chunks/{safe_name}_chunks.json"):
            available.append(policy)
    
    return available

# Main interface
st.subheader("Ask a question about UDST policies")

# Policy selector with available policies
available_policies = get_available_policies()
if not available_policies:
    st.warning("No policy data available. Please run the preprocessing script first.")
    selected_policy = None
else:
    selected_policy = st.selectbox(
        "Select a policy:",
        available_policies
    )

# Query input
query = st.text_input("Your question about this policy:")

# Generate answer when query is submitted
if query and api_key and selected_policy:
    with st.spinner("Searching policy and generating answer..."):
        try:
            # Get safe filename
            safe_name = get_safe_filename(selected_policy)
            
            # Check if required files exist
            chunks_path = f"chunks/{safe_name}_chunks.json"
            index_path = f"indexes/{safe_name}.index"
            
            if not os.path.exists(chunks_path) or not os.path.exists(index_path):
                st.error(f"Data for {selected_policy} not found. Please run the preprocess.py script.")
            else:
                # Load chunks
                with open(chunks_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                
                # Load index
                index = faiss.read_index(index_path)
                
                # Generate query embedding
                query_embedding_data = get_text_embedding([query], api_key)
                query_embedding = np.array([query_embedding_data[0].embedding])
                
                # Search for relevant chunks
                k = min(3, len(chunks))  # Get top 3 or fewer if not enough chunks
                D, I = index.search(query_embedding, k)
                
                # Get retrieved chunks
                retrieved_chunks = [chunks[i] for i in I.tolist()[0]]
                context = "\n".join(retrieved_chunks)
                
                # Create prompt
                prompt = f"""
                Context information about {selected_policy} is below:
                ---------------------
                {context}
                ---------------------
                
                Based ONLY on the context information provided and not prior knowledge, 
                answer the following query about {selected_policy}:
                
                Query: {query}
                
                Answer:
                """
                
                # Get response from Mistral
                response = query_mistral(prompt, api_key)
                
                # Display response
                st.markdown(f"<div class='answer-container'>", unsafe_allow_html=True)
                st.subheader("Answer:")
                st.write(response)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show sources in expandable section
                with st.expander("View source context"):
                    for i, chunk in enumerate(retrieved_chunks):
                        st.markdown(f"**Source {i+1}:**")
                        st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                
                # Relevance scores
                with st.expander("Relevance metrics"):
                    st.markdown("**Similarity scores:** (Lower is better)")
                    for i, (score, idx) in enumerate(zip(D.tolist()[0], I.tolist()[0])):
                        st.markdown(f"Source {i+1}: {score:.2f}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    if not api_key:
        st.info("Please enter your Mistral API key in the sidebar.")
    elif not selected_policy:
        st.info("Please select a policy.")
    else:
        st.info("Enter your question above.")

# Footer
st.markdown("---")
st.caption("UDST Policy RAG Assistant")
