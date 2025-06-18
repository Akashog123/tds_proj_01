#!/usr/bin/env python3
"""
Pre-compute embeddings and save them to files for fast Railway deployment startup.
This script generates embeddings once and saves them as JSON files that can be
quickly loaded during deployment, avoiding the 2-minute startup time.
"""

import json
import logging
import os
from typing import Dict, Any
import pickle
from pathlib import Path

# Import the search engine to reuse its logic
from hybrid_search_engine import HybridEmbeddingSearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def precompute_and_save_embeddings():
    """Pre-compute all embeddings and save them to files"""
    logger.info("🚀 Starting embedding pre-computation...")
    
    # Initialize the search engine (this will take time)
    engine = HybridEmbeddingSearchEngine()
    
    # Check if embeddings were generated
    if not engine.collection:
        logger.error("❌ Failed to initialize ChromaDB collection")
        return False
    
    # If local embeddings cache doesn't exist, we need to regenerate them
    if not hasattr(engine, 'local_embeddings_cache') or not engine.local_embeddings_cache:
        logger.info("🔄 Regenerating local embeddings cache...")
        engine.build_local_embeddings_cache()
    
    logger.info("💾 Saving embeddings to files...")
    
    # Create embeddings directory
    embeddings_dir = Path("precomputed_embeddings")
    embeddings_dir.mkdir(exist_ok=True)
    
    # Save local embeddings cache
    local_embeddings_file = embeddings_dir / "local_embeddings.json"
    with open(local_embeddings_file, 'w') as f:
        json.dump(engine.local_embeddings_cache, f)
    logger.info(f"✅ Saved local embeddings to {local_embeddings_file}")
    
    # Save ChromaDB data as JSON for fast loading
    chromadb_data = {
        'documents': [],
        'metadatas': [],
        'ids': [],
        'embeddings': []
    }
    
    try:
        # Get all data from ChromaDB - skip embeddings to avoid array issues
        results = engine.collection.get(include=['documents', 'metadatas'])
        
        chromadb_data = {
            'documents': results.get('documents', []),
            'metadatas': results.get('metadatas', []),
            'ids': results.get('ids', []),
            'embeddings': []  # We'll regenerate these from local cache if needed
        }
        
        chromadb_file = embeddings_dir / "chromadb_data.json"
        with open(chromadb_file, 'w') as f:
            json.dump(chromadb_data, f)
        logger.info(f"✅ Saved ChromaDB metadata to {chromadb_file}")
        
    except Exception as e:
        logger.warning(f"⚠️ Could not save ChromaDB data: {e}")
        # Continue anyway, local embeddings are more important
    
    # Save TF-IDF data
    if engine.tfidf_matrix is not None and engine.vectorizer is not None:
        tfidf_file = embeddings_dir / "tfidf_data.pkl"
        tfidf_data = {
            'matrix': engine.tfidf_matrix,
            'vectorizer': engine.vectorizer,
            'documents': [engine.subthreads[topic_id] for topic_id in engine.subthreads.keys()]
        }
        with open(tfidf_file, 'wb') as f:
            pickle.dump(tfidf_data, f)
        logger.info(f"✅ Saved TF-IDF data to {tfidf_file}")
    
    # Save subthreads data
    subthreads_file = embeddings_dir / "subthreads.json"
    with open(subthreads_file, 'w') as f:
        json.dump(engine.subthreads, f)
    logger.info(f"✅ Saved subthreads to {subthreads_file}")
    
    # Create metadata file
    metadata = {
        'total_embeddings': len(chromadb_data['ids']),
        'local_embeddings_count': len(engine.local_embeddings_cache) if hasattr(engine, 'local_embeddings_cache') else 0,
        'subthreads_count': len(engine.subthreads),
        'posts_count': len(engine.posts),
        'created_timestamp': str(os.path.getctime('discourse_posts.json')) if os.path.exists('discourse_posts.json') else None
    }
    
    metadata_file = embeddings_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Saved metadata to {metadata_file}")
    
    logger.info("🎉 Embedding pre-computation completed successfully!")
    logger.info(f"📊 Generated {metadata['total_embeddings']} embeddings for {metadata['subthreads_count']} subthreads")
    
    return True

if __name__ == "__main__":
    success = precompute_and_save_embeddings()
    if not success:
        exit(1)