import requests
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import logging
import pickle
import atexit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model
device = torch.device("cpu")
model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True).to(device)

# In-memory query cache
query_cache = {}

# Function to load cache from file at startup
def load_cache():
    global query_cache
    try:
        with open("query_cache.pkl", "rb") as f:
            query_cache = pickle.load(f)
            logger.info("Cache loaded successfully.")
    except FileNotFoundError:
        query_cache = {}
        logger.info("No cache file found. Starting with an empty cache.")

# Function to save cache to file on exit
def save_cache():
    with open("query_cache.pkl", "wb") as f:
        pickle.dump(query_cache, f)
        logger.info("Cache saved successfully.")

# Load cache on startup
load_cache()

# Register save_cache to run on program exit
atexit.register(save_cache)

# Function to retrieve data from URLs based on domain
def fetch_data_from_url(domain):
    try:
        if domain == "technology":
            urls = [
                "https://techcrunch.com/",
                "https://www.wired.com/category/tech/"
            ]
        elif domain == "health":
            urls = [
                "https://www.cdc.gov/health/index.html",
                "https://www.healthline.com/"
            ]
        else:  # Wikipedia
            urls = ["https://en.wikipedia.org/wiki/Main_Page"]

        content = []
        for url in urls:
            response = requests.get(url)
            if response.status_code == 200:
                logger.info(f"Successfully fetched data from {url}")
                content.append(response.text[:1000])
            else:
                logger.warning(f"Failed to retrieve data from {url}")

        if not content:
            logger.warning(f"No content retrieved for domain '{domain}'. Falling back to general knowledge.")

        return "\n\n".join(content) if content else "Error: No data available."

    except Exception as e:
        logger.error(f"Error fetching data for domain '{domain}': {e}")
        return "Error: Unable to retrieve data."

# Function to dynamically retrieve context based on domain and threshold
def retrieve_context(query, domain, top_k=5, similarity_threshold=0.5):
    context = fetch_data_from_url(domain)
    paragraphs = context.split("\n\n")

    docs_embed = model.encode(paragraphs, convert_to_tensor=True, device=device)
    query_embed = model.encode(query, convert_to_tensor=True, device=device)

    # Compute similarities only if docs_embed and query_embed have the right dimensions
    if docs_embed.shape[0] == 0 or query_embed.shape[0] == 0:
        logger.warning("No valid embeddings generated for the query or documents.")
        return "Error: No valid content to retrieve context."

    similarities = torch.matmul(query_embed, docs_embed.T).squeeze(0).cpu().numpy()

    # Check if similarities is a 1D array as expected
    if similarities.ndim != 1:
        logger.error("Unexpected similarity array dimension.")
        return "Error: Failed to compute similarities properly."

    relevant_indices = [i for i, score in enumerate(similarities) if score >= similarity_threshold]
    similarities = similarities[relevant_indices]
    paragraphs = [paragraphs[i] for i in relevant_indices]

    if len(similarities) == 0:
        logger.warning("No relevant paragraphs found above the similarity threshold.")
        return "No relevant context found for the query."

    top_indices = np.argsort(similarities)[-top_k:]
    selected_content = "\n\n".join([paragraphs[i] for i in top_indices])
    return selected_content

# Enhanced RAG system function
def rag_system(query, domain="general_knowledge", top_k=5, similarity_threshold=0.5):
    cache_key = f"{query}_{domain}_{top_k}_{similarity_threshold}"
    
    if cache_key in query_cache:
        logger.info("Returning cached response")
        return query_cache[cache_key]

    context = retrieve_context(query, domain=domain, top_k=top_k, similarity_threshold=similarity_threshold)
    
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    llm_response = call_llm("llama3", prompt)

    query_cache[cache_key] = llm_response
    return llm_response

# Function to interact with the LLM API
def call_llm(model_name, user_input, temperature=0.7, max_tokens=100):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_input.strip()}]
    }
    response = requests.post(url, json=payload, stream=True)
    full_response = ""
    for chunk in response.iter_lines():
        if chunk:
            chunk_data = json.loads(chunk.decode("utf-8"))
            full_response += chunk_data.get("message", {}).get("content", "")
            if chunk_data.get("done", False):
                break
    return full_response

# Main function for getting a response
def get_llm_response(model_name, user_query, use_rag=False, domain="general_knowledge", top_k=5, similarity_threshold=0.5):
    if use_rag:
        response = rag_system(user_query, domain=domain, top_k=top_k, similarity_threshold=similarity_threshold)
        if "I couldn't find any relevant information" in response:
            return call_llm(model_name, user_query)
        return response
    else:
        return call_llm(model_name, user_query)





















