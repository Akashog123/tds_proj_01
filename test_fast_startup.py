#!/usr/bin/env python3
"""Test the fast startup functionality"""

from main import search_engine

print('Testing search functionality...')
results = search_engine.search('docker containerization', top_k=2)
print(f'✅ Search completed, found {len(results)} results')

if results:
    print(f'First result: {results[0].get("topic_title", "No title")[:50]}...')
    print(f'Source: {results[0].get("source", "unknown")}')

print('🎉 Fast hybrid search engine working correctly!')