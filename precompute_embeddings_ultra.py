#!/usr/bin/env python3
"""
Ultra-lightweight precomputation script for Railway deployment.
This script generates all needed data for deployment with minimal dependencies.

Features:
- Precomputes OpenAI embeddings for all discourse posts/subthreads
- Generates TF-IDF vectors and vocabulary
- Creates search indices and metadata
- Optimizes data storage (float32 -> float16)
- Saves all precomputed data to lightweight format
"""

import json
import logging
import os
import pickle
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
import gzip
import struct

# Import the search engine to reuse its logic
from hybrid_search_engine import HybridEmbeddingSearchEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UltraLightweightPrecomputer:
    """Ultra-lightweight precomputation system for Railway deployment"""
    
    def __init__(self):
        self.output_dir = Path("precomputed_ultra")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize the hybrid search engine for full computation
        logger.info("🔄 Initializing hybrid search engine for precomputation...")
        self.engine = HybridEmbeddingSearchEngine()
        
    def compress_embeddings(self, embeddings: List[List[float]]) -> bytes:
        """Compress embeddings from float32 to float16 and serialize efficiently"""
        # Convert to numpy array and compress to float16
        embeddings_array = np.array(embeddings, dtype=np.float32)
        embeddings_compressed = embeddings_array.astype(np.float16)
        
        # Serialize with shape information
        shape = embeddings_compressed.shape
        data = struct.pack('II', *shape)  # Pack shape as two unsigned integers
        data += embeddings_compressed.tobytes()
        
        # Compress with gzip
        return gzip.compress(data)
    
    def decompress_embeddings(self, compressed_data: bytes) -> np.ndarray:
        """Decompress embeddings back to float32 numpy array"""
        # Decompress
        data = gzip.decompress(compressed_data)
        
        # Unpack shape
        rows, cols = struct.unpack('II', data[:8])
        
        # Reconstruct array
        embeddings_bytes = data[8:]
        embeddings_compressed = np.frombuffer(embeddings_bytes, dtype=np.float16)
        embeddings_array = embeddings_compressed.reshape(rows, cols)
        
        # Convert back to float32 for computation
        return embeddings_array.astype(np.float32)
    
    def precompute_openai_embeddings(self) -> Dict[str, Any]:
        """Extract and compress OpenAI embeddings from ChromaDB"""
        logger.info("📊 Extracting OpenAI embeddings from ChromaDB...")
        
        if not self.engine.collection:
            logger.error("❌ ChromaDB collection not available")
            return {}
        
        try:
            # Get all data from ChromaDB
            results = self.engine.collection.get(
                include=['documents', 'metadatas', 'embeddings']
            )
            
            if not results or not results.get('embeddings'):
                logger.error("❌ No embeddings found in ChromaDB")
                return {}
            
            embeddings = results['embeddings']
            ids = results['ids'] if results.get('ids') else []
            metadatas = results['metadatas'] if results.get('metadatas') else []
            documents = results['documents'] if results.get('documents') else []
            
            # Validate embeddings data
            if not embeddings or len(embeddings) == 0:
                logger.error("❌ Empty embeddings list from ChromaDB")
                return {}
            
            logger.info(f"✅ Extracted {len(embeddings)} OpenAI embeddings")
            
            # Compress embeddings
            compressed_embeddings = self.compress_embeddings(embeddings)
            
            # Calculate compression ratio
            original_size = len(embeddings) * len(embeddings[0]) * 4  # float32
            compressed_size = len(compressed_embeddings)
            compression_ratio = original_size / compressed_size
            
            logger.info(f"📦 Compressed embeddings: {original_size/1024:.1f}KB -> {compressed_size/1024:.1f}KB (ratio: {compression_ratio:.2f}x)")
            
            return {
                'embeddings_compressed': compressed_embeddings,
                'ids': ids,
                'metadatas': metadatas,
                'documents': documents,
                'embedding_dim': len(embeddings[0]),
                'count': len(embeddings),
                'compression_info': {
                    'original_size_kb': original_size / 1024,
                    'compressed_size_kb': compressed_size / 1024,
                    'compression_ratio': compression_ratio
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting OpenAI embeddings: {e}")
            return {}
    
    def precompute_local_embeddings(self) -> Dict[str, Any]:
        """Extract and compress local embeddings"""
        logger.info("📊 Processing local embeddings...")
        
        if not hasattr(self.engine, 'local_embeddings_cache') or not self.engine.local_embeddings_cache:
            logger.warning("⚠️ No local embeddings cache found, rebuilding...")
            self.engine.build_local_embeddings_cache()
        
        if not self.engine.local_embeddings_cache:
            logger.error("❌ Failed to build local embeddings cache")
            return {}
        
        # Convert to lists for compression
        ids = list(self.engine.local_embeddings_cache.keys())
        embeddings = list(self.engine.local_embeddings_cache.values())
        
        logger.info(f"✅ Processed {len(embeddings)} local embeddings")
        
        # Compress embeddings
        compressed_embeddings = self.compress_embeddings(embeddings)
        
        # Calculate compression ratio
        original_size = len(embeddings) * len(embeddings[0]) * 4  # float32
        compressed_size = len(compressed_embeddings)
        compression_ratio = original_size / compressed_size
        
        logger.info(f"📦 Compressed local embeddings: {original_size/1024:.1f}KB -> {compressed_size/1024:.1f}KB (ratio: {compression_ratio:.2f}x)")
        
        return {
            'embeddings_compressed': compressed_embeddings,
            'ids': ids,
            'embedding_dim': len(embeddings[0]),
            'count': len(embeddings),
            'compression_info': {
                'original_size_kb': original_size / 1024,
                'compressed_size_kb': compressed_size / 1024,
                'compression_ratio': compression_ratio
            }
        }
    
    def precompute_tfidf_data(self) -> Dict[str, Any]:
        """Extract and compress TF-IDF data"""
        logger.info("📊 Processing TF-IDF data...")
        
        if self.engine.tfidf_matrix is None or self.engine.vectorizer is None:
            logger.warning("⚠️ TF-IDF not built, building now...")
            self.engine.build_tfidf_fallback()
        
        if self.engine.tfidf_matrix is None:
            logger.error("❌ Failed to build TF-IDF matrix")
            return {}
        
        # Convert sparse matrix to dense for compression
        tfidf_dense = self.engine.tfidf_matrix.toarray().astype(np.float32)
        
        # Compress TF-IDF matrix
        compressed_tfidf = gzip.compress(tfidf_dense.tobytes())
        
        # Calculate compression ratio
        original_size = tfidf_dense.nbytes
        compressed_size = len(compressed_tfidf)
        compression_ratio = original_size / compressed_size
        
        logger.info(f"📦 Compressed TF-IDF matrix: {original_size/1024:.1f}KB -> {compressed_size/1024:.1f}KB (ratio: {compression_ratio:.2f}x)")
        
        # Extract vocabulary and feature names
        vocabulary = self.engine.vectorizer.vocabulary_
        feature_names = self.engine.vectorizer.get_feature_names_out().tolist()
        
        return {
            'matrix_compressed': compressed_tfidf,
            'matrix_shape': tfidf_dense.shape,
            'vocabulary': vocabulary,
            'feature_names': feature_names,
            'vectorizer_params': {
                'max_features': self.engine.vectorizer.max_features,
                'stop_words': 'english',  # Can't serialize the actual stop words set
                'ngram_range': self.engine.vectorizer.ngram_range,
                'min_df': self.engine.vectorizer.min_df,
                'max_df': self.engine.vectorizer.max_df
            },
            'compression_info': {
                'original_size_kb': original_size / 1024,
                'compressed_size_kb': compressed_size / 1024,
                'compression_ratio': compression_ratio
            }
        }
    
    def precompute_search_indices(self) -> Dict[str, Any]:
        """Create optimized search indices"""
        logger.info("📊 Creating search indices...")
        
        # Create lookup structures for fast retrieval
        topic_id_to_index = {}
        index_to_topic_id = {}
        
        subthread_lookup = {}
        
        for i, (topic_id, subthread) in enumerate(self.engine.subthreads.items()):
            topic_id_to_index[topic_id] = i
            index_to_topic_id[i] = topic_id
            
            # Store essential metadata only
            subthread_lookup[topic_id] = {
                'topic_title': subthread.get('topic_title', ''),
                'url': subthread.get('url', ''),
                'author': subthread['root_post'].get('author', ''),
                'reply_count': len(subthread.get('replies', [])),
                'is_staff_answer': subthread['root_post'].get('author', '') in self.engine.staff_authors,
                'like_count': subthread['root_post'].get('like_count', 0),
                'content_length': len(subthread.get('combined_content', '')),
                'has_accepted_answer': subthread['root_post'].get('is_accepted_answer', False)
            }
        
        # Create content index for fast text access
        content_index = {}
        for topic_id, subthread in self.engine.subthreads.items():
            content_index[topic_id] = {
                'combined_content': subthread.get('combined_content', ''),
                'topic_title': subthread.get('topic_title', '')
            }
        
        logger.info(f"✅ Created search indices for {len(subthread_lookup)} subthreads")
        
        return {
            'topic_id_to_index': topic_id_to_index,
            'index_to_topic_id': index_to_topic_id,
            'subthread_lookup': subthread_lookup,
            'content_index': content_index,
            'staff_authors': list(self.engine.staff_authors)
        }
    
    def save_precomputed_data(self):
        """Save all precomputed data to files"""
        logger.info("💾 Saving precomputed data...")
        
        # 1. OpenAI embeddings
        openai_data = self.precompute_openai_embeddings()
        if openai_data:
            openai_file = self.output_dir / "openai_embeddings.pkl"
            with open(openai_file, 'wb') as f:
                pickle.dump(openai_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"✅ Saved OpenAI embeddings to {openai_file}")
        
        # 2. Local embeddings
        local_data = self.precompute_local_embeddings()
        if local_data:
            local_file = self.output_dir / "local_embeddings.pkl"
            with open(local_file, 'wb') as f:
                pickle.dump(local_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"✅ Saved local embeddings to {local_file}")
        
        # 3. TF-IDF data
        tfidf_data = self.precompute_tfidf_data()
        if tfidf_data:
            tfidf_file = self.output_dir / "tfidf_data.pkl"
            with open(tfidf_file, 'wb') as f:
                pickle.dump(tfidf_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"✅ Saved TF-IDF data to {tfidf_file}")
        
        # 4. Search indices
        search_data = self.precompute_search_indices()
        search_file = self.output_dir / "search_indices.json"
        with open(search_file, 'w') as f:
            json.dump(search_data, f, indent=2)
        logger.info(f"✅ Saved search indices to {search_file}")
        
        # 5. Subthreads data (lightweight version)
        subthreads_light = {}
        for topic_id, subthread in self.engine.subthreads.items():
            subthreads_light[topic_id] = {
                'topic_title': subthread.get('topic_title', ''),
                'url': subthread.get('url', ''),
                'combined_content': subthread.get('combined_content', ''),
                'root_post': {
                    'author': subthread['root_post'].get('author', ''),
                    'like_count': subthread['root_post'].get('like_count', 0),
                    'is_accepted_answer': subthread['root_post'].get('is_accepted_answer', False)
                },
                'reply_count': len(subthread.get('replies', []))
            }
        
        subthreads_file = self.output_dir / "subthreads_light.json"
        with open(subthreads_file, 'w') as f:
            json.dump(subthreads_light, f, indent=2)
        logger.info(f"✅ Saved lightweight subthreads to {subthreads_file}")
        
        # 6. Create metadata summary
        total_size = sum(f.stat().st_size for f in self.output_dir.glob('*') if f.is_file())
        
        metadata = {
            'created_timestamp': time.time(),
            'total_size_mb': total_size / (1024 * 1024),
            'files_created': [f.name for f in self.output_dir.glob('*') if f.is_file()],
            'openai_embeddings': openai_data.get('count', 0) if openai_data else 0,
            'local_embeddings': local_data.get('count', 0) if local_data else 0,
            'tfidf_features': len(tfidf_data.get('feature_names', [])) if tfidf_data else 0,
            'subthreads_count': len(self.engine.subthreads),
            'compression_ratios': {
                'openai': openai_data.get('compression_info', {}).get('compression_ratio', 0) if openai_data else 0,
                'local': local_data.get('compression_info', {}).get('compression_ratio', 0) if local_data else 0,
                'tfidf': tfidf_data.get('compression_info', {}).get('compression_ratio', 0) if tfidf_data else 0
            }
        }
        
        metadata_file = self.output_dir / "ultra_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ Saved metadata to {metadata_file}")
        
        return metadata
    
    def run_precomputation(self):
        """Run the complete precomputation process"""
        logger.info("🚀 Starting ultra-lightweight precomputation...")
        start_time = time.time()
        
        try:
            metadata = self.save_precomputed_data()
            
            duration = time.time() - start_time
            
            logger.info("🎉 Ultra-lightweight precomputation completed successfully!")
            logger.info(f"⏱️  Total time: {duration:.1f} seconds")
            logger.info(f"📦 Total size: {metadata['total_size_mb']:.2f} MB")
            logger.info(f"📊 OpenAI embeddings: {metadata['openai_embeddings']}")
            logger.info(f"📊 Local embeddings: {metadata['local_embeddings']}")
            logger.info(f"📊 TF-IDF features: {metadata['tfidf_features']}")
            logger.info(f"📊 Subthreads: {metadata['subthreads_count']}")
            
            compression_ratios = metadata['compression_ratios']
            logger.info(f"🗜️  Compression ratios - OpenAI: {compression_ratios['openai']:.2f}x, Local: {compression_ratios['local']:.2f}x, TF-IDF: {compression_ratios['tfidf']:.2f}x")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Precomputation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

def main():
    """Main function"""
    precomputer = UltraLightweightPrecomputer()
    success = precomputer.run_precomputation()
    
    if success:
        logger.info("✅ Ultra-lightweight precomputation completed successfully!")
        logger.info("🚀 Ready for Railway deployment with minimal dependencies!")
    else:
        logger.error("❌ Precomputation failed!")
        exit(1)

if __name__ == "__main__":
    main()