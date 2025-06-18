#!/usr/bin/env python3
"""
Test script for the hybrid search engine upgrade
"""
import os
import sys
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported"""
    try:
        from hybrid_search_engine import HybridEmbeddingSearchEngine
        logger.info("✓ HybridEmbeddingSearchEngine imported successfully")
        
        from main import app, QuestionAnswerer, search_engine
        logger.info("✓ Main application components imported successfully")
        
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_search_engine_initialization():
    """Test search engine initialization"""
    try:
        from hybrid_search_engine import HybridEmbeddingSearchEngine
        
        # Test initialization (this will use the actual discourse_posts.json)
        engine = HybridEmbeddingSearchEngine()
        
        logger.info(f"✓ Search engine initialized with {len(engine.posts)} posts")
        logger.info(f"✓ Extracted {len(engine.subthreads)} subthreads")
        logger.info(f"✓ OpenAI client ready: {engine.openai_client is not None}")
        logger.info(f"✓ Local model ready: {engine.local_model is not None}")
        logger.info(f"✓ ChromaDB ready: {engine.collection is not None}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Search engine initialization failed: {e}")
        return False

def test_search_functionality():
    """Test basic search functionality"""
    try:
        from hybrid_search_engine import HybridEmbeddingSearchEngine
        
        engine = HybridEmbeddingSearchEngine()
        
        # Test search with a simple query
        test_query = "docker containerization"
        results = engine.search(test_query, top_k=3)
        
        logger.info(f"✓ Search completed for query: '{test_query}'")
        logger.info(f"✓ Found {len(results)} results")
        
        if results:
            for i, result in enumerate(results):
                logger.info(f"  Result {i+1}: {result.get('topic_title', 'No title')[:50]}...")
        
        return True
    except Exception as e:
        logger.error(f"✗ Search functionality test failed: {e}")
        return False

def test_api_compatibility():
    """Test API compatibility"""
    try:
        from main import search_engine, question_answerer
        
        # Test that the API components work together
        test_query = "TDS course information"
        search_results = search_engine.search(test_query, top_k=5)
        answer = question_answerer.generate_answer(test_query, search_results)
        
        logger.info(f"✓ API compatibility test passed")
        logger.info(f"✓ Generated answer length: {len(answer)} characters")
        
        return True
    except Exception as e:
        logger.error(f"✗ API compatibility test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting hybrid search engine tests...")
    
    tests = [
        ("Import Test", test_imports),
        ("Search Engine Initialization", test_search_engine_initialization),
        ("Search Functionality", test_search_functionality),
        ("API Compatibility", test_api_compatibility)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        if test_func():
            passed += 1
        else:
            logger.error(f"Failed: {test_name}")
    
    logger.info(f"\n--- Test Results ---")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Hybrid search engine is ready.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())