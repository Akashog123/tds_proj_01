#!/usr/bin/env python3
"""
Ultra-lightweight search engine for Railway deployment.
Uses only precomputed data with minimal dependencies (numpy, scipy only).

Features:
- Loads precomputed OpenAI and local embeddings
- Fast vector similarity search using only numpy
- No heavy ML dependencies (no chromadb, openai, sentence-transformers)
- Compressed data storage for minimal memory usage
- TF-IDF fallback for robust search
"""

import json
import logging
import pickle
import gzip
import struct
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

class UltraLightweightSearchEngine:
    """Ultra-lightweight search engine using only precomputed data"""
    
    def __init__(self, precomputed_dir: str = "precomputed_ultra"):
        self.precomputed_dir = Path(precomputed_dir)
        self.staff_authors = set()
        
        # Precomputed data containers
        self.openai_embeddings = None
        self.openai_ids = []
        self.openai_metadatas = []
        
        self.local_embeddings = None
        self.local_ids = []
        
        self.tfidf_matrix = None
        self.tfidf_vocabulary = {}
        self.tfidf_feature_names = []
        
        self.search_indices = {}
        self.subthreads_light = {}
        
        # Load all precomputed data
        self._load_precomputed_data()
        
        logger.info("✅ Ultra-lightweight search engine initialized")
    
    def _decompress_embeddings(self, compressed_data: bytes) -> np.ndarray:
        """Decompress embeddings from compressed format"""
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
    
    def _load_openai_embeddings(self) -> bool:
        """Load precomputed OpenAI embeddings"""
        openai_file = self.precomputed_dir / "openai_embeddings.pkl"
        
        if not openai_file.exists():
            logger.warning("⚠️ OpenAI embeddings file not found")
            return False
        
        try:
            with open(openai_file, 'rb') as f:
                data = pickle.load(f)
            
            self.openai_embeddings = self._decompress_embeddings(data['embeddings_compressed'])
            self.openai_ids = data['ids']
            self.openai_metadatas = data['metadatas']
            
            logger.info(f"✅ Loaded {len(self.openai_ids)} OpenAI embeddings")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading OpenAI embeddings: {e}")
            return False
    
    def _load_local_embeddings(self) -> bool:
        """Load precomputed local embeddings"""
        local_file = self.precomputed_dir / "local_embeddings.pkl"
        
        if not local_file.exists():
            logger.warning("⚠️ Local embeddings file not found")
            return False
        
        try:
            with open(local_file, 'rb') as f:
                data = pickle.load(f)
            
            self.local_embeddings = self._decompress_embeddings(data['embeddings_compressed'])
            self.local_ids = data['ids']
            
            logger.info(f"✅ Loaded {len(self.local_ids)} local embeddings")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading local embeddings: {e}")
            return False
    
    def _load_tfidf_data(self) -> bool:
        """Load precomputed TF-IDF data"""
        tfidf_file = self.precomputed_dir / "tfidf_data.pkl"
        
        if not tfidf_file.exists():
            logger.warning("⚠️ TF-IDF data file not found")
            return False
        
        try:
            with open(tfidf_file, 'rb') as f:
                data = pickle.load(f)
            
            # Decompress TF-IDF matrix
            compressed_matrix = data['matrix_compressed']
            matrix_shape = data['matrix_shape']
            
            decompressed_data = gzip.decompress(compressed_matrix)
            matrix_flat = np.frombuffer(decompressed_data, dtype=np.float32)
            self.tfidf_matrix = matrix_flat.reshape(matrix_shape)
            
            self.tfidf_vocabulary = data['vocabulary']
            self.tfidf_feature_names = data['feature_names']
            
            logger.info(f"✅ Loaded TF-IDF matrix: {self.tfidf_matrix.shape}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading TF-IDF data: {e}")
            return False
    
    def _load_search_indices(self) -> bool:
        """Load precomputed search indices"""
        indices_file = self.precomputed_dir / "search_indices.json"
        
        if not indices_file.exists():
            logger.warning("⚠️ Search indices file not found")
            return False
        
        try:
            with open(indices_file, 'r') as f:
                self.search_indices = json.load(f)
            
            self.staff_authors = set(self.search_indices.get('staff_authors', []))
            
            logger.info(f"✅ Loaded search indices for {len(self.search_indices.get('subthread_lookup', {}))} subthreads")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading search indices: {e}")
            return False
    
    def _load_subthreads(self) -> bool:
        """Load lightweight subthreads data"""
        subthreads_file = self.precomputed_dir / "subthreads_light.json"
        
        if not subthreads_file.exists():
            logger.warning("⚠️ Subthreads file not found")
            return False
        
        try:
            with open(subthreads_file, 'r') as f:
                self.subthreads_light = json.load(f)
            
            logger.info(f"✅ Loaded {len(self.subthreads_light)} lightweight subthreads")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading subthreads: {e}")
            return False
    
    def _load_precomputed_data(self):
        """Load all precomputed data"""
        logger.info("📂 Loading precomputed data...")
        
        # Check if precomputed directory exists
        if not self.precomputed_dir.exists():
            raise FileNotFoundError(f"Precomputed directory not found: {self.precomputed_dir}")
        
        # Load all components
        openai_loaded = self._load_openai_embeddings()
        local_loaded = self._load_local_embeddings()
        tfidf_loaded = self._load_tfidf_data()
        indices_loaded = self._load_search_indices()
        subthreads_loaded = self._load_subthreads()
        
        # Check if critical components are loaded
        if not (indices_loaded and subthreads_loaded):
            raise RuntimeError("Failed to load critical search components")
        
        if not (openai_loaded or local_loaded or tfidf_loaded):
            raise RuntimeError("No search algorithms available (no embeddings or TF-IDF)")
        
        logger.info("✅ All precomputed data loaded successfully")
    
    def _cosine_similarity_numpy(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity using pure numpy"""
        # Normalize vectors
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        
        # Calculate dot product
        return np.dot(a_norm, b_norm)
    
    def _cosine_similarity_matrix(self, query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query vector and matrix rows"""
        # Normalize query vector
        query_norm = query_vector / np.linalg.norm(query_vector)
        
        # Normalize matrix rows
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_normalized = matrix / (matrix_norms + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Calculate similarities
        similarities = np.dot(matrix_normalized, query_norm)
        
        return similarities
    
    def _dummy_embedding_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Dummy embedding search that returns TF-IDF results when embeddings are not available"""
        logger.info("Using TF-IDF fallback for embedding search")
        return self._tfidf_search(query, top_k)
    
    def _openai_embedding_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using precomputed OpenAI embeddings"""
        if self.openai_embeddings is None:
            logger.warning("OpenAI embeddings not available")
            return []
        
        try:
            # Calculate similarities
            similarities = self._cosine_similarity_matrix(query_embedding, self.openai_embeddings)
            
            # Get top results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    doc_id = self.openai_ids[idx]
                    metadata = self.openai_metadatas[idx]
                    topic_id = metadata['topic_id']
                    
                    if topic_id in self.subthreads_light:
                        subthread = self.subthreads_light[topic_id]
                        
                        # Calculate final score with engagement factors
                        final_score = self._calculate_subthread_score(subthread, similarities[idx])
                        
                        result = {
                            'topic_id': topic_id,
                            'topic_title': metadata['topic_title'],
                            'url': metadata['url'],
                            'content': subthread.get('combined_content', ''),
                            'author': metadata['root_author'],
                            'score': final_score,
                            'similarity': float(similarities[idx]),
                            'source': 'openai_precomputed'
                        }
                        results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in OpenAI embedding search: {e}")
            return []
    
    def _local_embedding_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using precomputed local embeddings"""
        if self.local_embeddings is None:
            logger.warning("Local embeddings not available")
            return []
        
        try:
            # Calculate similarities
            similarities = self._cosine_similarity_matrix(query_embedding, self.local_embeddings)
            
            # Get top results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    doc_id = self.local_ids[idx]
                    topic_id = doc_id.replace('topic_', '')
                    
                    if topic_id in self.subthreads_light:
                        subthread = self.subthreads_light[topic_id]
                        
                        # Calculate final score with engagement factors
                        final_score = self._calculate_subthread_score(subthread, similarities[idx])
                        
                        result = {
                            'topic_id': topic_id,
                            'topic_title': subthread.get('topic_title', ''),
                            'url': subthread.get('url', ''),
                            'content': subthread.get('combined_content', ''),
                            'author': subthread.get('root_post', {}).get('author', ''),
                            'score': final_score,
                            'similarity': float(similarities[idx]),
                            'source': 'local_precomputed'
                        }
                        results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in local embedding search: {e}")
            return []
    
    def _tfidf_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using precomputed TF-IDF data"""
        if self.tfidf_matrix is None:
            logger.warning("TF-IDF data not available")
            return []
        
        try:
            # Simple query vectorization using precomputed vocabulary
            query_vector = self._vectorize_query(query)
            
            if query_vector is None:
                return []
            
            # Calculate similarities
            similarities = self._cosine_similarity_matrix(query_vector, self.tfidf_matrix)
            
            # Get top results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            subthread_list = list(self.subthreads_light.keys())
            
            for idx in top_indices:
                if idx < len(subthread_list) and similarities[idx] > 0.01:
                    topic_id = subthread_list[idx]
                    subthread = self.subthreads_light[topic_id]
                    
                    # Calculate final score with engagement factors
                    final_score = self._calculate_subthread_score(subthread, similarities[idx])
                    
                    result = {
                        'topic_id': topic_id,
                        'topic_title': subthread.get('topic_title', ''),
                        'url': subthread.get('url', ''),
                        'content': subthread.get('combined_content', ''),
                        'author': subthread.get('root_post', {}).get('author', ''),
                        'score': final_score,
                        'similarity': float(similarities[idx]),
                        'source': 'tfidf_precomputed'
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in TF-IDF search: {e}")
            return []
    
    def _vectorize_query(self, query: str) -> Optional[np.ndarray]:
        """Vectorize query using precomputed TF-IDF vocabulary"""
        try:
            # Simple tokenization and vectorization
            query_lower = query.lower()
            
            # Create query vector
            query_vector = np.zeros(len(self.tfidf_feature_names))
            
            # Split query into terms
            query_terms = query_lower.split()
            
            # Map terms to features
            for term in query_terms:
                if term in self.tfidf_vocabulary:
                    feature_idx = self.tfidf_vocabulary[term]
                    query_vector[feature_idx] = 1.0
            
            # Normalize
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
                return query_vector
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error vectorizing query: {e}")
            return None
    
    def _calculate_subthread_score(self, subthread: Dict[str, Any], similarity_score: float) -> float:
        """Calculate overall score for a subthread based on multiple factors"""
        score = similarity_score
        
        # Boost staff answers
        root_post = subthread.get('root_post', {})
        author = root_post.get('author', '')
        if author in self.staff_authors:
            score *= 2.0
        
        # Boost based on engagement
        like_count = root_post.get('like_count', 0)
        reply_count = subthread.get('reply_count', 0)
        
        # Add engagement bonus
        engagement_bonus = (like_count * 0.1) + (reply_count * 0.05)
        score += engagement_bonus
        
        # Boost accepted answers
        if root_post.get('is_accepted_answer', False):
            score *= 1.5
        
        return score
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Main search interface - pure lookup-based search"""
        if not query.strip():
            return []
        
        try:
            # For ultra-lightweight deployment, we'll use TF-IDF as primary search
            # since it doesn't require external API calls for query embedding
            results = self._tfidf_search(query, top_k)
            
            # Sort by final score
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return {
            'total_subthreads': len(self.subthreads_light),
            'has_openai_embeddings': self.openai_embeddings is not None,
            'has_local_embeddings': self.local_embeddings is not None,
            'has_tfidf': self.tfidf_matrix is not None,
            'openai_embeddings_count': len(self.openai_ids) if self.openai_embeddings is not None else 0,
            'local_embeddings_count': len(self.local_ids) if self.local_embeddings is not None else 0,
            'tfidf_features': len(self.tfidf_feature_names) if self.tfidf_matrix is not None else 0,
            'staff_authors_count': len(self.staff_authors),
            'precomputed_dir': str(self.precomputed_dir)
        }