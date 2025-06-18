#!/usr/bin/env python3
"""
Diagnostic script to test environment variable loading and OpenAI API connectivity
"""
import os
import logging

# Configure logging to show debug messages
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_environment_loading():
    """Test if environment variables are being loaded correctly"""
    print("=== Environment Variable Diagnosis ===")
    
    # Test without dotenv
    print("1. Without dotenv:")
    api_key_no_dotenv = os.getenv("OPENAI_API_KEY")
    base_url_no_dotenv = os.getenv("OPENAI_BASE_URL")
    print(f"   OPENAI_API_KEY present: {api_key_no_dotenv is not None}")
    print(f"   OPENAI_API_KEY length: {len(api_key_no_dotenv) if api_key_no_dotenv else 0}")
    print(f"   OPENAI_BASE_URL: {base_url_no_dotenv}")
    
    # Test with dotenv
    print("\n2. With dotenv:")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key_with_dotenv = os.getenv("OPENAI_API_KEY")
        base_url_with_dotenv = os.getenv("OPENAI_BASE_URL")
        print(f"   OPENAI_API_KEY present: {api_key_with_dotenv is not None}")
        print(f"   OPENAI_API_KEY length: {len(api_key_with_dotenv) if api_key_with_dotenv else 0}")
        print(f"   OPENAI_BASE_URL: {base_url_with_dotenv}")
        
        if api_key_with_dotenv and api_key_with_dotenv.count('.') == 2:
            print("   API key format: aipipe (JWT-like)")
        elif api_key_with_dotenv:
            print("   API key format: standard OpenAI")
            
    except ImportError:
        print("   python-dotenv not available")

def test_openai_connectivity():
    """Test OpenAI API connectivity with aipipe proxy"""
    print("\n=== OpenAI API Connectivity Test ===")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    if not api_key:
        print("❌ No API key available for testing")
        return
    
    try:
        import openai
        
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # Test with a simple embedding request
        print(f"Testing embedding request to: {base_url}")
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="test embedding",
            encoding_format="float"
        )
        
        print(f"✅ Success! Embedding dimension: {len(response.data[0].embedding)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        if hasattr(e, 'response'):
            print(f"   HTTP status: {getattr(e.response, 'status_code', 'unknown')}")

def test_hybrid_search_initialization():
    """Test hybrid search engine initialization"""
    print("\n=== Hybrid Search Engine Test ===")
    
    try:
        from hybrid_search_engine import HybridEmbeddingSearchEngine
        
        # Initialize with enhanced logging
        print("Initializing hybrid search engine...")
        engine = HybridEmbeddingSearchEngine()
        
        print(f"✅ Initialization complete:")
        print(f"   Posts loaded: {len(engine.posts)}")
        print(f"   Subthreads extracted: {len(engine.subthreads)}")
        print(f"   OpenAI client ready: {engine.openai_client is not None}")
        print(f"   Local model ready: {engine.local_model is not None}")
        print(f"   ChromaDB ready: {engine.collection is not None}")
        
        # Test a simple search
        print("\nTesting search functionality...")
        results = engine.search("docker containerization", top_k=2)
        print(f"✅ Search completed, found {len(results)} results")
        
        if results:
            print("   Sample result:")
            result = results[0]
            print(f"     Title: {result.get('topic_title', 'N/A')[:50]}...")
            print(f"     Score type: {'hybrid' if 'hybrid_similarity' in result else 'tfidf'}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_environment_loading()
    test_openai_connectivity()
    test_hybrid_search_initialization()