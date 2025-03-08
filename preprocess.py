import requests
from bs4 import BeautifulSoup
import os
import numpy as np
import faiss
import json
import time
from mistralai import Mistral
import re

# Policy URLs
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

# Function to get safe filename
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

# Improved scraping function with multiple fallback methods
def scrape_webpage(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Method 1: Try to find the main content by class
        content = ""
        
        # Try common content containers
        content_candidates = [
            soup.find("div", class_="content"),
            soup.find("div", class_="main-content"),
            soup.find("article"),
            soup.find("main"),
            soup.find("div", id=lambda x: x and ('content' in x.lower() if x else False)),
            soup.find("div", class_=lambda x: x and ('content' in x.lower() if x else False))
        ]
        
        # Try to find policy text specifically
        policy_elements = soup.find_all(string=re.compile(r"Policy Statement|Purpose|Scope|Policy|Procedure"))
        
        # Check candidates in order
        for candidate in content_candidates:
            if candidate and len(candidate.get_text(strip=True)) > 200:  # Reasonable content length
                content = candidate.get_text(separator="\n", strip=True)
                print(f"  Found content using selector: {candidate.name}{'.'+candidate.get('class', [''])[0] if candidate.get('class') else ''}")
                break
        
        # If no content found, try policy elements
        if not content and policy_elements:
            for element in policy_elements:
                parent = element.parent
                if parent and len(parent.get_text(strip=True)) > 200:
                    content = parent.get_text(separator="\n", strip=True)
                    print(f"  Found content using policy element: {parent.name}")
                    break
        
        # Last resort: take the body text
        if not content or len(content) < 500:  # If content is too small
            body = soup.find("body")
            if body:
                # Filter out navigation, headers, footers
                for tag in body.find_all(["nav", "header", "footer", "script", "style"]):
                    tag.decompose()
                content = body.get_text(separator="\n", strip=True)
                print("  Used body text as fallback")
        
        # Clean up content
        content = re.sub(r'\n{3,}', '\n\n', content)  # Remove excessive newlines
        
        if len(content.strip()) < 100:
            print("  Warning: Content seems too short, might be incomplete")
            
        return content
    
    except Exception as e:
        print(f"  Error scraping {url}: {str(e)}")
        return ""

# Function to create dummy data for testing if scraping fails
def create_dummy_policy_data(policy_name):
    return f"""
    {policy_name} Policy
    
    Purpose:
    This policy establishes guidelines for {policy_name.lower()} at UDST.
    
    Scope:
    This policy applies to all students, faculty, and staff at UDST.
    
    Policy Statement:
    UDST is committed to providing clear guidelines regarding {policy_name.lower()}.
    
    Procedures:
    1. Follow established procedures for {policy_name.lower()}.
    2. Consult with appropriate departments as needed.
    3. Documentation must be submitted in a timely manner.
    
    Responsibilities:
    Students and faculty are responsible for adhering to this policy.
    
    Note: This is dummy data created because the actual policy could not be scraped.
    """

def main():
    # Get API key
    api_key = "LoXIODO6VkldB64uwva76l1zDpIz6cfu"
    
    # Create directories
    os.makedirs("policy_texts", exist_ok=True)
    os.makedirs("chunks", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)
    os.makedirs("indexes", exist_ok=True)
    
    # Process each policy
    for policy_name, url in POLICY_URLS.items():
        print(f"\nProcessing {policy_name}...")
        safe_name = get_safe_filename(policy_name)
        
        try:
            # Step 1: Scrape website and save text
            print(f"  Scraping {url}...")
            text = scrape_webpage(url)
            
            # If scraping fails completely, use dummy data for testing
            if not text or len(text.strip()) < 200:
                print(f"  Warning: Insufficient content scraped. Using dummy data for {policy_name}")
                text = create_dummy_policy_data(policy_name)
            
            # Save raw text
            print(f"  Saving raw text to policy_texts/{safe_name}.txt")
            with open(f"policy_texts/{safe_name}.txt", "w", encoding="utf-8") as f:
                f.write(text)
            
            # Print text length for debugging
            print(f"  Text length: {len(text)} characters")
            
            # Step 2: Chunk the text
            print(f"  Chunking text...")
            chunk_size = 512
            # Handle overlapping chunks for better context
            overlap = 100
            chunks = []
            
            # Create overlapping chunks
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if len(chunk.strip()) > 50:  # Only include substantial chunks
                    chunks.append(chunk)
            
            # If we still have no chunks (should be rare), create one with the entire text
            if not chunks and text:
                chunks = [text]
            
            print(f"  Created {len(chunks)} chunks")
            
            # Save chunks
            print(f"  Saving chunks to chunks/{safe_name}_chunks.json")
            with open(f"chunks/{safe_name}_chunks.json", "w", encoding="utf-8") as f:
                json.dump(chunks, f)
            
            if not chunks:
                raise ValueError("No chunks were created from the text")
                
            # Step 3: Generate embeddings
            print(f"  Generating embeddings for {len(chunks)} chunks...")
            text_embeddings = get_text_embedding(chunks, api_key)
            embeddings = np.array([emb.embedding for emb in text_embeddings])
            
            # Save embeddings
            print(f"  Saving embeddings to embeddings/{safe_name}_embeddings.npy")
            np.save(f"embeddings/{safe_name}_embeddings.npy", embeddings)
            
            # Step 4: Create and save FAISS index
            print(f"  Creating FAISS index...")
            d = len(embeddings[0])  # Dimensionality
            index = faiss.IndexFlatL2(d)
            index.add(embeddings)
            
            print(f"  Saving index to indexes/{safe_name}.index")
            faiss.write_index(index, f"indexes/{safe_name}.index")
            
            print(f"✅ {policy_name} processed successfully")
            
        except Exception as e:
            print(f"❌ Error processing {policy_name}: {str(e)}")
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    print("\nAll policies processed. You can now run the Streamlit app.")

if __name__ == "__main__":
    main()