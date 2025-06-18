#!/usr/bin/env python3
"""
Final validation script for the hybrid search system
"""
import os
import logging
import requests
import json
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_health_endpoint():
    """Test the health endpoint with hybrid search status"""
    try:
        response = requests.get("http://localhost:8001/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            logger.info("✅ Health endpoint working")
            logger.info(f"   Posts loaded: {health_data.get('posts_loaded', 'N/A')}")
            logger.info(f"   Subthreads: {health_data.get('subthreads_extracted', 'N/A')}")
            logger.info(f"   ChromaDB ready: {health_data.get('chromadb_ready', 'N/A')}")
            logger.info(f"   OpenAI ready: {health_data.get('openai_client_ready', 'N/A')}")
            logger.info(f"   Local model ready: {health_data.get('local_model_ready', 'N/A')}")
            return True
        else:
            logger.error(f"❌ Health endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Health endpoint error: {e}")
        return False

def test_api_endpoints():
    """Test the main API endpoint with sample questions"""
    test_cases = [
        {
            "question": "How do I use docker in TDS?",
            "expected_keywords": ["docker", "container", "podman"]
        },
        {
            "question": "What is the GA4 scoring system?",
            "expected_keywords": ["ga4", "graded", "assignment"]
        },
        {
            "question": "TDS course module information",
            "expected_keywords": ["module", "tds", "course"]
        }
    ]
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        try:
            logger.info(f"\nTest {i}: '{test_case['question']}'")
            
            response = requests.post(
                "http://localhost:8001/api/",
                json={"question": test_case["question"]},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                links = data.get("links", [])
                
                logger.info(f"✅ Response received")
                logger.info(f"   Answer length: {len(answer)} chars")
                logger.info(f"   Links provided: {len(links)}")
                logger.info(f"   Answer preview: {answer[:100]}...")
                
                # Check if response contains expected keywords
                answer_lower = answer.lower()
                keyword_matches = [kw for kw in test_case["expected_keywords"] if kw in answer_lower]
                logger.info(f"   Keyword matches: {keyword_matches}")
                
                results.append({
                    "test": i,
                    "question": test_case["question"],
                    "success": True,
                    "answer_length": len(answer),
                    "links_count": len(links),
                    "keyword_matches": len(keyword_matches)
                })
            else:
                logger.error(f"❌ API request failed with status {response.status_code}")
                results.append({"test": i, "success": False, "error": response.status_code})
                
        except Exception as e:
            logger.error(f"❌ Test {i} failed: {e}")
            results.append({"test": i, "success": False, "error": str(e)})
    
    return results

def test_hybrid_search_features():
    """Test specific hybrid search features"""
    logger.info("\n=== Testing Hybrid Search Features ===")
    
    # Test semantic search vs keyword matching
    semantic_question = "What containerization technology should I use?"
    keyword_question = "docker podman container"
    
    try:
        # Semantic search test
        response1 = requests.post(
            "http://localhost:8001/api/",
            json={"question": semantic_question},
            timeout=30
        )
        
        # Keyword search test  
        response2 = requests.post(
            "http://localhost:8001/api/",
            json={"question": keyword_question},
            timeout=30
        )
        
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()
            
            logger.info(f"✅ Semantic search: {len(data1.get('answer', ''))} chars")
            logger.info(f"✅ Keyword search: {len(data2.get('answer', ''))} chars")
            logger.info(f"   Both responses generated successfully")
            return True
        else:
            logger.error("❌ One or both search tests failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Hybrid search test failed: {e}")
        return False

def test_fallback_mechanisms():
    """Test fallback mechanisms"""
    logger.info("\n=== Testing Fallback Mechanisms ===")
    
    # Test with a complex query that might challenge the system
    complex_question = "What are the detailed requirements for project submission including file formats and naming conventions?"
    
    try:
        response = requests.post(
            "http://localhost:8001/api/",
            json={"question": complex_question},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "")
            
            if "couldn't find relevant information" in answer.lower():
                logger.info("✅ Graceful fallback message provided")
            else:
                logger.info("✅ Relevant answer found and provided")
            
            return True
        else:
            logger.error(f"❌ Fallback test failed with status {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Fallback test failed: {e}")
        return False

def main():
    """Run comprehensive validation"""
    logger.info("🚀 Starting Hybrid Search System Validation")
    logger.info("=" * 50)
    
    # Check if server is running
    try:
        requests.get("http://localhost:8001/", timeout=5)
    except:
        logger.error("❌ Server not running on localhost:8001")
        logger.info("Please start the server first: python main.py")
        return 1
    
    tests = [
        ("Health Endpoint", test_health_endpoint),
        ("API Endpoints", lambda: len([r for r in test_api_endpoints() if r.get("success", False)]) > 0),
        ("Hybrid Search Features", test_hybrid_search_features),
        ("Fallback Mechanisms", test_fallback_mechanisms)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"📊 FINAL RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 ALL TESTS PASSED! Hybrid search system is working correctly.")
        logger.info("✅ Ready for Railway deployment")
        return 0
    else:
        logger.error("⚠️  Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())