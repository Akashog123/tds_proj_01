#!/usr/bin/env python3
"""
Test script for the TDS Virtual Teaching Assistant API
"""
import requests
import json
import time

# API endpoint
API_URL = "http://localhost:8000/api/"
HEALTH_URL = "http://localhost:8000/health"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(HEALTH_URL)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed: {data}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_question(question, description):
    """Test a specific question"""
    print(f"\nTesting: {description}")
    print(f"Question: {question}")
    
    try:
        response = requests.post(
            API_URL,
            json={"question": question},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Answer: {data['answer'][:200]}...")
            print(f"✓ Links found: {len(data['links'])}")
            for i, link in enumerate(data['links'][:2]):  # Show first 2 links
                print(f"  Link {i+1}: {link['url']}")
                print(f"  Text: {link['text'][:100]}...")
            return True
        else:
            print(f"✗ Request failed: {response.status_code}")
            print(f"✗ Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("TDS Virtual Teaching Assistant API Tests")
    print("=" * 50)
    
    # Wait a moment for server to be ready
    time.sleep(1)
    
    # Test health check
    if not test_health_check():
        print("Health check failed. Make sure the server is running.")
        return
    
    # Test cases based on promptfoo config
    test_cases = [
        (
            "The question asks to use gpt-3.5-turbo-0125 model but the ai-proxy provided by Anand sir only supports gpt-4o-mini. So should we just use gpt-4o-mini or use the OpenAI API for gpt3.5 turbo?",
            "GPT model question"
        ),
        (
            "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?",
            "GA4 dashboard scoring question"
        ),
        (
            "I know Docker but have not used Podman before. Should I use Docker for this course?",
            "Docker vs Podman question"
        ),
        (
            "When is the TDS Sep 2025 end-term exam?",
            "Unknown information question"
        ),
        (
            "How do I submit my assignment?",
            "General course question"
        )
    ]
    
    # Run tests
    passed = 0
    total = len(test_cases)
    
    for question, description in test_cases:
        if test_question(question, description):
            passed += 1
        time.sleep(0.5)  # Small delay between requests
    
    print(f"\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. Check the output above.")

if __name__ == "__main__":
    main()