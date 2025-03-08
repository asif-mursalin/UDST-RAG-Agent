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

# Add debug mode toggle - with a unique key
debug_mode = st.sidebar.checkbox("Enable Debug Mode", key="debug_mode_toggle")

# Global force refresh - outside of the policy loop, with unique key
force_refresh_global = st.sidebar.checkbox("Force Refresh All Content", value=False, key="force_refresh_global")

# Use large model toggle - with a unique key
use_large_model = st.sidebar.checkbox("Use large model (higher quality but more rate limits)", value=False, key="use_large_model_toggle")

# Force reprocess toggle - with a unique key
force_reprocess = st.sidebar.checkbox("Force Reprocess", value=False, key="force_reprocess_toggle")

# Helper function to generate cache key
def get_cache_key(data):
    return hashlib.md5(str(data).encode()).hexdigest()

# Cached embedding retrieval
def get_text_embedding_cached(list_txt_chunks):
    """Get embeddings with caching to reduce API calls"""
    cache_key = get_cache_key(str(list_txt_chunks))
    cache_file = os.path.join(cache_dir, f"embed_{cache_key}.json")
    
    # Check if cache exists
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            return [type('obj', (object,), {'embedding': item}) for item in cached_data]
        except Exception as e:
            if debug_mode:
                st.sidebar.error(f"Cache error: {str(e)}")
    
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
                if debug_mode:
                    st.sidebar.error(f"API Error: {str(e)}")
                raise e
    
    raise Exception("Failed to get embeddings after multiple retries")

def fetch_and_process_policy(url, policy_name):
    """Fetch and chunk a policy document with improved content extraction"""
    cache_key = get_cache_key(url)
    cache_file = os.path.join(cache_dir, f"policy_{cache_key}.json")
    
    # Use the global force refresh setting - no need for individual checkboxes
    if force_refresh_global and os.path.exists(cache_file):
        os.remove(cache_file)
        if debug_mode:
            st.sidebar.info(f"Cache cleared for {policy_name}")
    
    # Check if cache exists
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            return cached_data
        except Exception as e:
            if debug_mode:
                st.sidebar.error(f"Cache read error: {str(e)}")
    
    with st.spinner(f"Processing {policy_name}..."):
        # Fetch the content with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        response = requests.get(url, headers=headers, timeout=30)
        html_doc = response.text
        
        # Debug information
        if debug_mode:
            st.sidebar.text(f"Status code: {response.status_code}")
            st.sidebar.text(f"Content size: {len(html_doc)} bytes")
        
        soup = BeautifulSoup(html_doc, "html.parser")
        
        # Try different selectors for better content extraction
        content = None
        
        # Try specific selectors first
        selectors = [
            "div.content", 
            "div.policy-content",
            "article",
            "main",
            "div.main-content",
            "div.post-content",
            "div#content"
        ]
        
        for selector in selectors:
            content_element = soup.select_one(selector)
            if content_element:
                content = content_element.get_text(strip=True, separator=" ")
                if debug_mode:
                    st.sidebar.success(f"Found content using selector: {selector}")
                break
        
        # Fallback to general div if no specific selector worked
        if not content:
            # Get all divs and find the one with the most text
            divs = soup.find_all("div")
            longest_text = ""
            for div in divs:
                div_text = div.get_text(strip=True)
                if len(div_text) > len(longest_text):
                    longest_text = div_text
            
            content = longest_text
            if debug_mode:
                st.sidebar.info("Used longest div as fallback")
        
        # If still no content, try getting the whole body
        if not content or len(content) < 100:  # If content is too short, it's probably not what we want
            body = soup.find("body")
            if body:
                content = body.get_text(strip=True, separator=" ")
                if debug_mode:
                    st.sidebar.info("Used body content as fallback")
        
        # Final check
        if not content or len(content) < 100:
            # If we still can't get content, raise error
            raise ValueError(f"Could not extract meaningful content from {policy_name}")
            
        # Debug sample
        if debug_mode:
            st.sidebar.text(f"Sample content: {content[:200]}...")
        
        # Create chunks with better overlap for context
        chunk_size = 512
        overlap = 100
        chunks = []
        
        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i:i + chunk_size]
            chunks.append(chunk)
        
        # Add metadata to each chunk
        chunks_with_metadata = [[chunk, policy_name, url] for chunk in chunks]
        
        # Cache the results
        with open(cache_file, 'w') as f:
            json.dump(chunks_with_metadata, f)
        
        st.sidebar.info(f"Created {len(chunks)} chunks for {policy_name}")
        return chunks_with_metadata

def mistral_completion(prompt):
    """Get completion from Mistral API with retry logic"""
    # Show the prompt in debug mode
    if debug_mode:
        with st.expander("Debug: Prompt Sent to API"):
            st.markdown(f"```\n{prompt}\n```")
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            client = Mistral(api_key=api_key)
            messages = [
                UserMessage(content=prompt),
            ]
            
            model = "mistral-small-latest"  # Default to small model
            if use_large_model:  # Use the global toggle
                model = "mistral-large-latest"
            
            chat_response = client.chat.complete(
                model=model,
                messages=messages,
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=1024,
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
    if new_policies or force_reprocess:  # Use the global toggle
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

# Increase number of returned chunks
k_results = st.sidebar.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=5, key="k_results_slider")

# Display the answer area
answer_container = st.container(height=400)

if query and st.session_state.index is not None and st.session_state.embeddings_loaded:
    with st.spinner("Searching for answer..."):
        try:
            # Embed the question
            question_embeddings = np.array([get_text_embedding_cached([query])[0].embedding])
            
            # Search for similar chunks
            k = min(k_results, len(st.session_state.all_chunks))
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
            
            # Show retrieved context in debug mode
            if debug_mode:
                with st.expander("Debug: Retrieved Context"):
                    st.markdown(context)
            
            # Create an improved prompt
            prompt = f"""
            You are a helpful assistant that can answer questions about University of Doha for Science and Technology (UDST) policies.
            
            Context information from policy documents is provided below:
            ---------------------
            {context}
            ---------------------
            
            Using ONLY the context information (not prior knowledge), answer the following query about UDST policies.
            If the information needed to answer the query is not present in the context, say "I don't have enough information to answer this question completely, but here's what I know:" and then provide any relevant information you can find.
            
            For questions about policy purpose, look for statements about the policy's objectives, goals, or mission.
            For questions about procedures, look for step-by-step instructions.
            For questions about requirements, look for mandatory conditions or criteria.
            
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
### Instructions:
1. Enter your Mistral API key
2. Select policies to include in your queries
3. Click "Process Selected Policies"
4. Enter your questions and get answers

### Tips to avoid rate limits:
1. Process fewer policies at once
2. Wait a few minutes between queries
3. Use a paid Mistral API key with higher limits
""")