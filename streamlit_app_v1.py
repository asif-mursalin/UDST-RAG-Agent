# streamlit_app.py
import streamlit as st
import requests
import numpy as np
import faiss
from bs4 import BeautifulSoup
import os
import time
import json
import hashlib
from mistralai import Mistral, UserMessage
from mistralai.models.sdkerror import SDKError

# Configure the page
st.set_page_config(
    page_title="UDST Policy Chatbot",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("UDST Policy Chatbot")
st.markdown("Ask questions about University of Doha for Science and Technology policies.")

# Mistral API key input
api_key = st.sidebar.text_input("Enter Mistral API Key", type="password")
if not api_key:
    st.warning("Please enter your Mistral API key to continue")
    st.stop()
os.environ["MISTRAL_API_KEY"] = api_key

# Define the list of policy URLs and names
policy_data = {
    "Sport and Wellness Facilities": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and",
    "Student Engagement Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-engagement-policy",
    "Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy",
    "Student Appeals Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy",
    "Student Attendance Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy",
    "Student Counselling Services Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-counselling-services-policy",
    "Use Library Space Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/use-library-space-policy",
    "Transfer Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/transfer-policy",
    "Academic Schedule Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/academic-schedule-policy",
    "Student Conduct Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy"
}

# Sidebar for policy selection
st.sidebar.header("Select Policies")
selected_policies = st.sidebar.multiselect(
    "Choose policies to query:",
    list(policy_data.keys()),
    default=list(policy_data.keys())[:3]  # Default select first 3
)

# Cache directory
cache_dir = "cache"
os.makedirs(cache_dir, exist_ok=True)

# Helper function to generate cache key
def get_cache_key(data):
    return hashlib.md5(str(data).encode()).hexdigest()

# Cached embedding retrieval
def get_text_embedding_cached(list_txt_chunks):
    """Get embeddings with caching to reduce API calls"""
    cache_key = get_cache_key(list_txt_chunks)
    cache_file = os.path.join(cache_dir, f"embed_{cache_key}.json")
    
    # Check if cache exists
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        return [type('obj', (object,), {'embedding': item}) for item in cached_data]
    
    # No cache, call the API with retry logic
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            client = Mistral(api_key=api_key)
            # Process in smaller batches to avoid rate limits
            batch_size = 5
            all_embeddings = []
            
            for i in range(0, len(list_txt_chunks), batch_size):
                batch = list_txt_chunks[i:i+batch_size]
                embeddings_batch_response = client.embeddings.create(
                    model="mistral-embed",
                    inputs=batch
                )
                all_embeddings.extend(embeddings_batch_response.data)
                # Small delay between batches
                time.sleep(1)
            
            # Cache the results
            embeddings_to_cache = [item.embedding for item in all_embeddings]
            with open(cache_file, 'w') as f:
                json.dump(embeddings_to_cache, f)
            
            return all_embeddings
            
        except SDKError as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                st.sidebar.warning(f"Rate limit exceeded. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                raise e
    
    raise Exception("Failed to get embeddings after multiple retries")

def fetch_and_process_policy(url, policy_name):
    """Fetch and chunk a policy document with caching"""
    cache_key = get_cache_key(url)
    cache_file = os.path.join(cache_dir, f"policy_{cache_key}.json")
    
    # Check if cache exists
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    with st.spinner(f"Processing {policy_name}..."):
        # Fetch the content
        response = requests.get(url)
        html_doc = response.text
        soup = BeautifulSoup(html_doc, "html.parser")
        
        # Extract the main content (adjust selector as needed)
        # tag = soup.find("div", {"class": "content"})  
        # if not tag:
        tag = soup.find("div")
        
        text = tag.text
        
        # Create chunks
        chunk_size = 512
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Add metadata to each chunk
        chunks_with_metadata = [[chunk, policy_name, url] for chunk in chunks]
        
        # Cache the results
        with open(cache_file, 'w') as f:
            json.dump(chunks_with_metadata, f)
        
        st.sidebar.info(f"Created {len(chunks)} chunks for {policy_name}")
        return chunks_with_metadata

def mistral_completion(prompt):
    """Get completion from Mistral API with retry logic"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            client = Mistral(api_key=api_key)
            messages = [
                UserMessage(content=prompt),
            ]
            
            chat_response = client.chat.complete(
                model="mistral-small-latest",  # Using small model to reduce rate limits
                messages=messages,
            )
            
            return chat_response.choices[0].message.content
            
        except SDKError as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                st.warning(f"Rate limit exceeded. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                st.error(f"API Error: {str(e)}")
                return "Sorry, I encountered an error with the Mistral API. Please try again later or with a different API key."
    
    return "Failed to get a response after multiple retries due to rate limiting. Please try again in a few minutes."

# Initialize session state for storing data between interactions
if 'index' not in st.session_state:
    st.session_state.index = None
    st.session_state.all_chunks = []
    st.session_state.all_chunks_with_metadata = []
    st.session_state.processed_policies = set()
    st.session_state.embeddings_loaded = False

# Process selected policies button
process_button = st.sidebar.button("Process Selected Policies")
if process_button and selected_policies:
    # Reset if new policies are selected
    new_policies = set(selected_policies) - st.session_state.processed_policies
    if new_policies:
        st.session_state.all_chunks = []
        st.session_state.all_chunks_with_metadata = []
        st.session_state.processed_policies = set()
        st.session_state.embeddings_loaded = False
    
    # Process each selected policy
    with st.sidebar.status("Processing policies...") as status:
        for policy_name in selected_policies:
            if policy_name not in st.session_state.processed_policies:
                url = policy_data[policy_name]
                try:
                    chunks = fetch_and_process_policy(url, policy_name)
                    st.session_state.all_chunks_with_metadata.extend(chunks)
                    st.session_state.processed_policies.add(policy_name)
                except Exception as e:
                    st.sidebar.error(f"Error processing {policy_name}: {str(e)}")
        
        # Extract just the text chunks for embedding
        st.session_state.all_chunks = [item[0] for item in st.session_state.all_chunks_with_metadata]
        
        try:
            # Get embeddings and build index
            st.sidebar.text("Getting embeddings...")
            text_embeddings = get_text_embedding_cached(st.session_state.all_chunks)
            embeddings = np.array([text_embeddings[i].embedding for i in range(len(text_embeddings))])
            
            # Store in FAISS
            d = len(text_embeddings[0].embedding)
            st.session_state.index = faiss.IndexFlatL2(d)
            st.session_state.index.add(embeddings)
            st.session_state.embeddings_loaded = True
            
            status.update(label="Processing complete!", state="complete")
        except Exception as e:
            status.update(label=f"Error: {str(e)}", state="error")
            st.sidebar.error(f"Error building index: {str(e)}")

# Main query section
st.header("Ask a Question")
query = st.text_input("Enter your question about the selected policies:")

# Display the answer area
answer_container = st.container(height=400)

if query and st.session_state.index is not None and st.session_state.embeddings_loaded:
    with st.spinner("Searching for answer..."):
        try:
            # Embed the question
            question_embeddings = np.array([get_text_embedding_cached([query])[0].embedding])
            
            # Search for similar chunks
            k = min(3, len(st.session_state.all_chunks))
            D, I = st.session_state.index.search(question_embeddings, k=k)
            
            # Retrieve the chunks with their metadata
            retrieved_chunks = [st.session_state.all_chunks[i] for i in I.tolist()[0]]
            retrieved_metadata = [st.session_state.all_chunks_with_metadata[i][1:] for i in I.tolist()[0]]
            
            # Create a context with source information
            context = ""
            sources = []
            for i in range(len(retrieved_chunks)):
                chunk = retrieved_chunks[i]
                policy_name, url = retrieved_metadata[i]
                context += f"Source: {policy_name}\n{chunk}\n\n"
                sources.append(policy_name)
            
            # Create the prompt
            prompt = f"""
            Context information is below.
            ---------------------
            {context}
            ---------------------
            Given the context information and not prior knowledge, answer the query.
            If the information is not found in the context, say "I don't have enough information to answer this question."
            
            Query: {query}
            Answer:
            """
            
            # Get response from Mistral
            response = mistral_completion(prompt)
            
            # Display the answer
            with answer_container:
                st.subheader("Answer")
                st.write(response)
                
                # Show sources
                st.subheader("Sources")
                st.write("Information retrieved from the following policies:")
                for source in set(sources):
                    st.write(f"- {source}")
        except Exception as e:
            with answer_container:
                st.error(f"Error processing your question: {str(e)}")
                st.info("Try simplifying your question or selecting fewer policies to reduce API calls.")
elif query:
    with answer_container:
        st.warning("Please process the selected policies first by clicking the 'Process Selected Policies' button in the sidebar.")

# Add some helpful information at the bottom
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Tips to avoid rate limits:
1. Process fewer policies at once
2. Wait a few minutes between queries
3. Use a paid Mistral API key with higher limits
""")