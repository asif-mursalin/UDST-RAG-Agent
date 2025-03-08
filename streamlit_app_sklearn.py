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
import re
from sklearn.feature_extraction.text import TfidfVectorizer
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

# Cache directory
cache_dir = "cache"
os.makedirs(cache_dir, exist_ok=True)

# Define the list of policy URLs and names
policy_data = {
    "Sport and Wellness Facilities": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and",
    "Student Engagement Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-engagement-policy",
    "Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy",
    "Student Appeals Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy",
    "Student Attendance Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy"
}

# Advanced settings in sidebar
st.sidebar.header("Settings")
use_mistral_api = st.sidebar.checkbox("Use Mistral API (if unchecked, will use basic TF-IDF)", value=False, key="use_mistral_api")

if use_mistral_api:
    # Mistral API key input
    api_key = st.sidebar.text_input("Enter Mistral API Key", type="password")
    if not api_key:
        st.warning("Please enter your Mistral API key to continue")
        st.stop()
    os.environ["MISTRAL_API_KEY"] = api_key

# Helper function to generate cache key
def get_cache_key(data):
    return hashlib.md5(str(data).encode()).hexdigest()

# Cached text processing functions
def fetch_and_process_policy(url, policy_name, force_refresh=False):
    """Fetch and chunk a policy document with improved content extraction"""
    cache_key = get_cache_key(url)
    cache_file = os.path.join(cache_dir, f"policy_{cache_key}.json")
    
    # Check if cache exists and not forcing refresh
    if os.path.exists(cache_file) and not force_refresh:
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.sidebar.error(f"Cache read error: {str(e)}")
    
    with st.spinner(f"Processing {policy_name}..."):
        # Fetch the content with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        response = requests.get(url, headers=headers, timeout=30)
        html_doc = response.text
        
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
        
        # If still no content, try getting the whole body
        if not content or len(content) < 100:  # If content is too short, it's probably not what we want
            body = soup.find("body")
            if body:
                content = body.get_text(strip=True, separator=" ")
        
        # Final check
        if not content or len(content) < 100:
            # If we still can't get content, raise error
            raise ValueError(f"Could not extract meaningful content from {policy_name}")
        
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

class TfidfSearchEngine:
    """Simple TF-IDF based search engine as fallback when API calls are limited"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.chunks = []
        
    def fit(self, chunks):
        """Build the TF-IDF matrix from chunks"""
        self.chunks = chunks
        self.tfidf_matrix = self.vectorizer.fit_transform(chunks)
        return self
        
    def search(self, query, top_k=5):
        """Search for similar chunks"""
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top k results
        top_indices = similarities.argsort()[-top_k:][::-1]
        return top_indices, similarities[top_indices]

def mistral_completion(prompt):
    """Get completion from Mistral API with retry logic"""
    max_retries = 3
    retry_delay = 5  # Increased initial delay
    
    for attempt in range(max_retries):
        try:
            client = Mistral(api_key=api_key)
            messages = [
                UserMessage(content=prompt),
            ]
            
            chat_response = client.chat.complete(
                model="mistral-small-latest",  # Using small model to reduce rate limits
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            
            return chat_response.choices[0].message.content
            
        except SDKError as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                st.warning(f"Rate limit exceeded. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                st.error(f"API Error: {str(e)}")
                return f"Sorry, I encountered an error with the Mistral API. Error: {str(e)}"
    
    return "Failed to get a response after multiple retries due to rate limiting. Please try again in a few minutes."

def get_policy_text(policy_name):
    """Get cached policy text if available"""
    url = policy_data.get(policy_name)
    if not url:
        return None
    
    cache_key = get_cache_key(url)
    cache_file = os.path.join(cache_dir, f"policy_{cache_key}.json")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                chunks_with_metadata = json.load(f)
            return "\n\n".join([chunk[0] for chunk in chunks_with_metadata])
        except:
            return None
    return None

def summarize_text(text, max_length=2000):
    """Basic text summarization by keeping first parts and important sentences"""
    if len(text) <= max_length:
        return text
    
    # First, keep the beginning which often has important context
    beginning = text[:int(max_length * 0.3)]
    
    # Then find sentences that might contain important information
    # Look for sentences with key terms that might indicate policy purpose
    important_patterns = [
        r"[^.]*\bpurpose\b[^.]*\.",
        r"[^.]*\bobjective\b[^.]*\.",
        r"[^.]*\bpolicy\s+statement\b[^.]*\.",
        r"[^.]*\bscope\b[^.]*\.",
        r"[^.]*\bdefinition\b[^.]*\."
    ]
    
    important_sentences = []
    for pattern in important_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        important_sentences.extend(matches)
    
    # Join the beginning and important sentences
    summary = beginning + "\n\n" + "\n".join(important_sentences)
    
    # If still too long, truncate
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
    
    return summary

# Sidebar for policy selection
st.sidebar.header("Select Policies")
selected_policies = st.sidebar.multiselect(
    "Choose policies to query:",
    list(policy_data.keys()),
    default=["Sport and Wellness Facilities"]  # Default to just one policy
)

# Initialize session state
if 'processed_policies' not in st.session_state:
    st.session_state.processed_policies = set()
    st.session_state.tfidf_engine = None
    st.session_state.all_chunks = []
    st.session_state.all_chunks_with_metadata = []

# Process selected policies button
process_button = st.sidebar.button("Process Selected Policies")
if process_button and selected_policies:
    force_refresh = st.sidebar.checkbox("Force Refresh Content", value=False, key="force_refresh_toggle")
    
    # Reset if force refresh
    if force_refresh:
        st.session_state.processed_policies = set()
        st.session_state.all_chunks = []
        st.session_state.all_chunks_with_metadata = []
    
    # Process each selected policy
    with st.status("Processing policies...") as status:
        for policy_name in selected_policies:
            if policy_name not in st.session_state.processed_policies:
                url = policy_data[policy_name]
                try:
                    chunks = fetch_and_process_policy(url, policy_name, force_refresh)
                    st.session_state.all_chunks_with_metadata.extend(chunks)
                    st.session_state.processed_policies.add(policy_name)
                except Exception as e:
                    st.error(f"Error processing {policy_name}: {str(e)}")
        
        # Extract just the text chunks for TF-IDF
        st.session_state.all_chunks = [item[0] for item in st.session_state.all_chunks_with_metadata]
        
        # Build TF-IDF search engine (no API needed)
        st.session_state.tfidf_engine = TfidfSearchEngine().fit(st.session_state.all_chunks)
        
        status.update(label="Processing complete!", state="complete")

# Main query section
st.header("Ask a Question")
query = st.text_input("Enter your question about the selected policies:")

# Display the answer area
answer_container = st.container(height=400)

if query and len(st.session_state.all_chunks) > 0:
    with st.spinner("Searching for answer..."):
        try:
            # Use TF-IDF search (no API call)
            top_k = min(3, len(st.session_state.all_chunks))
            indices, scores = st.session_state.tfidf_engine.search(query, top_k=top_k)
            
            # Retrieve the chunks with their metadata
            retrieved_chunks = [st.session_state.all_chunks[i] for i in indices]
            retrieved_metadata = [st.session_state.all_chunks_with_metadata[i][1:] for i in indices]
            
            # Create a context with source information
            context = ""
            sources = []
            for i in range(len(retrieved_chunks)):
                chunk = retrieved_chunks[i]
                policy_name, url = retrieved_metadata[i]
                context += f"Source: {policy_name}\n{chunk}\n\n"
                sources.append(policy_name)
            
            # If using Mistral API for the completion
            if use_mistral_api:
                # Create the prompt
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
            else:
                # BACKUP APPROACH: Generate a simple answer without API calls
                # This is a basic approach but it will work without hitting rate limits
                st.info("Using basic text matching (no API calls)")
                
                # Look for relevant sentences from the context
                query_terms = query.lower().split()
                
                # Check if it's a purpose question
                is_purpose_question = any(term in query.lower() for term in ['purpose', 'objective', 'aim', 'goal'])
                
                relevant_sentences = []
                for chunk in retrieved_chunks:
                    sentences = re.split(r'(?<=[.!?])\s+', chunk)
                    for sentence in sentences:
                        # For purpose questions, look for sentences containing purpose-related words
                        if is_purpose_question and any(term in sentence.lower() for term in ['purpose', 'objective', 'aim', 'policy statement']):
                            relevant_sentences.append(sentence)
                        # For other questions, look for sentences with multiple query terms
                        elif sum(1 for term in query_terms if term in sentence.lower()) >= 2:
                            relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    response = "Based on the selected policies, I found the following information:\n\n"
                    # Add top relevant sentences (up to 5)
                    response += " ".join(relevant_sentences[:5])
                else:
                    # If no specific sentences found, just include the highest-scoring chunk
                    response = "I couldn't find specific information about your query, but here's the most relevant policy content:\n\n"
                    response += retrieved_chunks[0]
                
                # Add source attribution
                response += f"\n\nThis information comes from the {sources[0]} policy."
            
            # Display the answer
            with answer_container:
                st.subheader("Answer")
                st.write(response)
                
                # Show sources
                st.subheader("Sources")
                st.write("Information retrieved from the following policies:")
                for source in set(sources):
                    st.write(f"- {source}")
                    
                    # Display button to view full policy
                    if st.button(f"View Full {source} Policy", key=f"view_{source}"):
                        policy_text = get_policy_text(source)
                        if policy_text:
                            with st.expander(f"Full Content: {source}"):
                                summarized = summarize_text(policy_text)
                                st.markdown(summarized)
                        else:
                            st.error(f"Cannot retrieve full text for {source}")
                            
        except Exception as e:
            with answer_container:
                st.error(f"Error processing your question: {str(e)}")
elif query:
    with answer_container:
        st.warning("Please process the selected policies first by clicking the 'Process Selected Policies' button in the sidebar.")

# Add some helpful information at the bottom
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Instructions:
1. Select a SINGLE policy to work with first
2. Click "Process Selected Policies"
3. Enter your questions about the policy
4. Check "Use Mistral API" only if you have a premium API key

### Note:
The basic mode uses TF-IDF text matching which doesn't need API calls.
This is recommended since you're experiencing rate limiting issues.
""")