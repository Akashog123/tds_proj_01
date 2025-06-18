#!/usr/bin/env python3
"""
Final comprehensive validation for TDS Virtual Teaching Assistant API
Tests all requirements from the task specification
"""
import requests
import json
import base64

def test_server_health():
    """Test server health and basic functionality"""
    print("1. Testing Server Health")
    print("-" * 30)
    
    try:
        # Health check
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Server healthy: {data['status']}")
            print(f"✓ Posts loaded: {data['posts_loaded']}")
            print(f"✓ Search index ready: {data['search_index_ready']}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_json_schema():
    """Test API response JSON schema compliance"""
    print("\n2. Testing JSON Schema Compliance")
    print("-" * 40)
    
    try:
        response = requests.post(
            "http://localhost:8000/api/",
            json={"question": "Test question"}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Check required fields
            has_answer = 'answer' in data and isinstance(data['answer'], str)
            has_links = 'links' in data and isinstance(data['links'], list)
            
            print(f"✓ Status code 200: {response.status_code == 200}")
            print(f"✓ Has 'answer' field (string): {has_answer}")
            print(f"✓ Has 'links' field (array): {has_links}")
            
            # Check links structure
            links_valid = True
            if data.get('links'):
                for link in data['links']:
                    if not (isinstance(link, dict) and 'url' in link and 'text' in link):
                        links_valid = False
                        break
            
            print(f"✓ Links have correct structure: {links_valid}")
            return has_answer and has_links and links_valid
        else:
            print(f"✗ Request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Schema test error: {e}")
        return False

def test_promptfoo_cases():
    """Test all specific promptfoo test cases"""
    print("\n3. Testing Promptfoo Test Cases")
    print("-" * 35)
    
    test_cases = [
        {
            'name': 'GPT model question',
            'question': 'The question asks to use gpt-3.5-turbo-0125 model but the ai-proxy provided by Anand sir only supports gpt-4o-mini. So should we just use gpt-4o-mini or use the OpenAI API for gpt3.5 turbo?',
            'expected_content': 'gpt-4o-mini'
        },
        {
            'name': 'GA4 dashboard scoring',
            'question': 'If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?',
            'expected_content': '110'
        },
        {
            'name': 'Docker vs Podman',
            'question': 'I know Docker but have not used Podman before. Should I use Docker for this course?',
            'expected_content': 'Podman'
        },
        {
            'name': 'Unknown information',
            'question': 'When is the TDS Sep 2025 end-term exam?',
            'expected_content': "don't have information"
        }
    ]
    
    passed = 0
    for case in test_cases:
        try:
            response = requests.post(
                "http://localhost:8000/api/",
                json={"question": case['question']}
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data['answer'].lower()
                expected = case['expected_content'].lower()
                has_content = expected in answer
                has_links = len(data.get('links', [])) > 0
                
                print(f"✓ {case['name']}: Content match={has_content}, Links={has_links}")
                if has_content and has_links:
                    passed += 1
                else:
                    print(f"  Expected: {case['expected_content']}")
                    print(f"  Answer preview: {data['answer'][:100]}...")
            else:
                print(f"✗ {case['name']}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"✗ {case['name']}: Error {e}")
    
    print(f"\nPromptfoo cases passed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)

def test_image_handling():
    """Test image attachment handling"""
    print("\n4. Testing Image Handling")
    print("-" * 25)
    
    try:
        # Test with valid base64 image
        test_image = base64.b64encode(b'fake_image_data').decode('utf-8')
        response = requests.post(
            "http://localhost:8000/api/",
            json={
                "question": "What is wrong with this code?",
                "image": test_image
            }
        )
        
        valid_image_test = response.status_code == 200
        print(f"✓ Valid base64 image accepted: {valid_image_test}")
        
        # Test with invalid base64
        response2 = requests.post(
            "http://localhost:8000/api/",
            json={
                "question": "What is wrong with this code?",
                "image": "invalid_base64!!!"
            }
        )
        
        # Should still work but log warning
        invalid_image_test = response2.status_code == 200
        print(f"✓ Invalid image handled gracefully: {invalid_image_test}")
        
        return valid_image_test and invalid_image_test
        
    except Exception as e:
        print(f"✗ Image handling error: {e}")
        return False

def test_error_handling():
    """Test error handling for malformed requests"""
    print("\n5. Testing Error Handling")
    print("-" * 25)
    
    try:
        # Test missing question field
        response1 = requests.post("http://localhost:8000/api/", json={})
        missing_field_test = response1.status_code == 422
        print(f"✓ Missing required field returns 422: {missing_field_test}")
        
        # Test invalid field
        response2 = requests.post("http://localhost:8000/api/", json={"invalid": "field"})
        invalid_field_test = response2.status_code == 422
        print(f"✓ Invalid field returns 422: {invalid_field_test}")
        
        # Test empty question (should work)
        response3 = requests.post("http://localhost:8000/api/", json={"question": ""})
        empty_question_test = response3.status_code == 200
        print(f"✓ Empty question handled: {empty_question_test}")
        
        return missing_field_test and invalid_field_test and empty_question_test
        
    except Exception as e:
        print(f"✗ Error handling test error: {e}")
        return False

def test_discourse_search():
    """Test general discourse search functionality"""
    print("\n6. Testing Discourse Search")
    print("-" * 30)
    
    try:
        response = requests.post(
            "http://localhost:8000/api/",
            json={"question": "How do I submit my assignment?"}
        )
        
        if response.status_code == 200:
            data = response.json()
            has_answer = len(data['answer']) > 0
            has_links = len(data.get('links', [])) > 0
            links_have_discourse = any('discourse.onlinedegree.iitm.ac.in' in link.get('url', '') 
                                    for link in data.get('links', []))
            
            print(f"✓ Returns answer: {has_answer}")
            print(f"✓ Returns links: {has_links}")
            print(f"✓ Links contain discourse URLs: {links_have_discourse}")
            
            return has_answer and has_links
        else:
            print(f"✗ Search failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Search test error: {e}")
        return False

def main():
    """Run comprehensive validation"""
    print("TDS Virtual Teaching Assistant API - Final Validation")
    print("=" * 60)
    
    tests = [
        ("Server Health", test_server_health),
        ("JSON Schema", test_json_schema),
        ("Promptfoo Cases", test_promptfoo_cases),
        ("Image Handling", test_image_handling),
        ("Error Handling", test_error_handling),
        ("Discourse Search", test_discourse_search)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"FINAL VALIDATION RESULTS: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! API is ready for promptfoo evaluation.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
        
    print(f"\nServer Status: Running on http://localhost:8000")
    print(f"API Endpoint: http://localhost:8000/api/")
    print(f"Health Check: http://localhost:8000/health")
    print(f"Promptfoo Config: Updated for localhost:8000")

if __name__ == "__main__":
    main()