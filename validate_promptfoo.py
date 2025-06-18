#!/usr/bin/env python3
"""
Validate promptfoo test cases for TDS Virtual Teaching Assistant API
"""
import requests
import json

def test_promptfoo_cases():
    """Test specific promptfoo cases"""
    print("Validating Promptfoo Test Cases")
    print("=" * 50)
    
    test_cases = [
        {
            'question': 'The question asks to use gpt-3.5-turbo-0125 model but the ai-proxy provided by Anand sir only supports gpt-4o-mini. So should we just use gpt-4o-mini or use the OpenAI API for gpt3.5 turbo?', 
            'expected_keyword': 'gpt-4o-mini',
            'description': 'GPT model question'
        },
        {
            'question': 'If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?', 
            'expected_keyword': '110',
            'description': 'GA4 dashboard scoring'
        },
        {
            'question': 'I know Docker but have not used Podman before. Should I use Docker for this course?', 
            'expected_keyword': 'Podman',
            'description': 'Docker vs Podman'
        },
        {
            'question': 'When is the TDS Sep 2025 end-term exam?', 
            'expected_keyword': "don't have information",
            'description': 'Unknown information handling'
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {case['description']}")
        try:
            r = requests.post('http://localhost:8000/api/', json={'question': case['question']})
            if r.status_code == 200:
                response = r.json()
                answer = response['answer'].lower()
                keyword = case['expected_keyword'].lower()
                match = keyword in answer
                
                print(f"✓ Status: 200 OK")
                print(f"✓ Response format: {'answer' in response and 'links' in response}")
                print(f"✓ Links count: {len(response.get('links', []))}")
                print(f"{'✓' if match else '✗'} Expected keyword '{case['expected_keyword']}' found: {match}")
                
                if match:
                    passed += 1
                else:
                    print(f"  Actual answer: {response['answer'][:150]}...")
                    
            else:
                print(f"✗ HTTP Error: {r.status_code}")
                print(f"  Response: {r.text}")
                
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\n" + "=" * 50)
    print(f"Promptfoo Validation Results: {passed}/{total} tests passed")
    return passed == total

if __name__ == "__main__":
    test_promptfoo_cases()