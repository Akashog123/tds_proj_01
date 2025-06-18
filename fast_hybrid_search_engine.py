import json
import logging
import os
from typing import List, Optional, Dict, Any, Tuple
import pickle
from pathlib import Path
from collections import defaultdict

import chromadb
from chromadb.config import Settings
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class FastHybridEmbeddingSearchEngine:
    """Fast-loading hybrid search engine that uses pre-computed embeddings"""
    
    def __init__(self, posts_file: str = "discourse_posts.json", use_precomputed: bool = True):
        self.posts = []
        self.subthreads = {}
        self.staff_authors = {"s.anand", "carlton", "Jivraj"}
        
        # Initialize embedding models
        self.local_model = None
        self.openai_client = None
        
        # Initialize ChromaDB
        self.chroma_client = None
        self.collection = None
        
        # TF-IDF fallback
        self.vectorizer = None
        self.tfidf_matrix = None
        
        # Local embeddings cache
        self.local_embeddings_cache = {}
        
        # Load data
        self.load_posts(posts_file)
        self.extract_subthreads()
        
        # Fast loading path if precomputed embeddings exist
        if use_precomputed and self.load_precomputed_embeddings():
            logger.info("✅ Loaded pre-computed embeddings - fast startup!")
            self.setup_openai_client()  # Still need client for real-time queries
            self.setup_local_model()    # Still need local model for real-time queries
        else:
            logger.info("🐌 Pre-computed embeddings not found, falling back to full initialization...")
            self.setup_openai_client()
            self.setup_local_model()
            self.setup_chromadb()
            self.build_vector_index()
            self.build_tfidf_fallback()
    
    def load_precomputed_embeddings(self) -> bool:
        """Load pre-computed embeddings for fast startup"""
        embeddings_dir = Path("precomputed_embeddings")
        
        try:
            # Check if all required files exist
            required_files = [
                "local_embeddings.json",
                "chromadb_data.json", 
                "subthreads.json",
                "metadata.json"
            ]
            
            for file in required_files:
                if not (embeddings_dir / file).exists():
                    logger.warning(f"Missing precomputed file: {file}")
                    return False
            
            logger.info("📂 Loading pre-computed embeddings...")
            
            # Load metadata
            with open(embeddings_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
            logger.info(f"📊 Loading {metadata['total_embeddings']} embeddings for {metadata['subthreads_count']} subthreads")
            
            # Load local embeddings cache
            with open(embeddings_dir / "local_embeddings.json", 'r') as f:
                self.local_embeddings_cache = json.load(f)
            logger.info(f"✅ Loaded {len(self.local_embeddings_cache)} local embeddings")
            
            # Load subthreads (override the ones we extracted)
            with open(embeddings_dir / "subthreads.json", 'r') as f:
                self.subthreads = json.load(f)
                # Convert string keys back to integers
                self.subthreads = {int(k): v for k, v in self.subthreads.items()}
            logger.info(f"✅ Loaded {len(self.subthreads)} subthreads")
            
            # Skip ChromaDB setup for now since embeddings are empty
            # We'll rely on TF-IDF and local embeddings
            logger.info("⚡ Skipping ChromaDB setup, using TF-IDF + local embeddings for fast startup")
            
            # Load TF-IDF data if available
            tfidf_file = embeddings_dir / "tfidf_data.pkl"
            if tfidf_file.exists():
                with open(tfidf_file, 'rb') as f:
                    tfidf_data = pickle.load(f)
                    self.tfidf_matrix = tfidf_data['matrix']
                    self.vectorizer = tfidf_data['vectorizer']
                logger.info("✅ Loaded TF-IDF fallback index")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading pre-computed embeddings: {e}")
            return False
    
    def setup_inmemory_chromadb(self):
        """Setup in-memory ChromaDB for fast loading"""
        try:
            # Use in-memory client for speed
            self.chroma_client = chromadb.Client()
            
            # Create collection
            collection_name = "discourse_embeddings"
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
            except Exception:
                self.collection = self.chroma_client.create_collection(collection_name)
                
        except Exception as e:
            logger.error(f"Error setting up in-memory ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None
    
    def load_posts(self, posts_file: str):
        """Load discourse posts from JSON file"""
        try:
            with open(posts_file, 'r', encoding='utf-8') as f:
                self.posts = json.load(f)
            logger.info(f"Loaded {len(self.posts)} discourse posts")
        except Exception as e:
            logger.error(f"Error loading posts: {e}")
            self.posts = []
    
    def extract_subthreads(self):
        """Extract subthreads from posts (group by topic_id)"""
        # Group posts by topic_id
        topics = defaultdict(list)
        for post in self.posts:
            topic_id = post.get('topic_id')
            if topic_id:
                topics[topic_id].append(post)
        
        # Process each topic
        for topic_id, topic_posts in topics.items():
            # Sort by post_number to get chronological order
            topic_posts.sort(key=lambda x: x.get('post_number', 0))
            
            # First post is the root post
            if topic_posts:
                root_post = topic_posts[0]
                replies = topic_posts[1:] if len(topic_posts) > 1 else []
                
                # Combine content for embedding
                subthread_content = f"Root: {root_post.get('content', '')}"
                if replies:
                    reply_contents = [f"Reply {i+1}: {reply.get('content', '')}" for i, reply in enumerate(replies)]
                    subthread_content += " " + " ".join(reply_contents)
                
                self.subthreads[topic_id] = {
                    'root_post': root_post,
                    'replies': replies,
                    'combined_content': subthread_content,
                    'topic_title': root_post.get('topic_title', ''),
                    'url': root_post.get('url', '')
                }
        
        logger.info(f"Extracted {len(self.subthreads)} subthreads from {len(self.posts)} posts")
    
    def setup_openai_client(self):
        """Setup OpenAI client with aipipe proxy configuration"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            
            if not api_key:
                logger.warning("OPENAI_API_KEY not found in environment variables")
                return
            
            self.openai_client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            logger.info(f"OpenAI client configured successfully")
        except Exception as e:
            logger.error(f"Error setting up OpenAI client: {e}")
            self.openai_client = None
    
    def setup_local_model(self):
        """Setup local sentence transformer model"""
        try:
            self.local_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Local sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Error setting up local model: {e}")
            self.local_model = None
    
    def setup_chromadb(self):
        """Setup ChromaDB client and collection (fallback)"""
        try:
            self.chroma_client = chromadb.Client()
            collection_name = "discourse_embeddings"
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
            except Exception:
                self.collection = self.chroma_client.create_collection(collection_name)
        except Exception as e:
            logger.error(f"Error setting up ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None
    
    def build_vector_index(self):
        """Build vector index (fallback method)"""
        # This is the slow method, only used if precomputed embeddings fail
        from hybrid_search_engine import HybridEmbeddingSearchEngine
        logger.warning("Using slow embedding generation - this will take ~2 minutes")
        fallback_engine = HybridEmbeddingSearchEngine()
        
        # Copy the results
        if hasattr(fallback_engine, 'local_embeddings_cache'):
            self.local_embeddings_cache = fallback_engine.local_embeddings_cache
        if fallback_engine.collection:
            self.collection = fallback_engine.collection
    
    def build_tfidf_fallback(self):
        """Build TF-IDF index as fallback"""
        if not self.subthreads:
            return
        
        documents = []
        for subthread in self.subthreads.values():
            title = subthread.get('topic_title', '')
            content = subthread.get('combined_content', '')
            combined_text = f"{title} {content}"
            documents.append(combined_text)
        
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)
            logger.info(f"Built TF-IDF fallback index with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error building TF-IDF index: {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using hybrid approach"""
        if not query.strip():
            return []
        
        results = []
        
        # Try ChromaDB search first
        if self.collection:
            try:
                chroma_results = self.collection.query(
                    query_texts=[query],
                    n_results=min(top_k * 2, 20)
                )
                
                for i, doc_id in enumerate(chroma_results['ids'][0]):
                    topic_id = int(doc_id.split('_')[1])
                    if topic_id in self.subthreads:
                        subthread = self.subthreads[topic_id]
                        result = {
                            'topic_id': topic_id,
                            'topic_title': subthread.get('topic_title', ''),
                            'content': subthread.get('combined_content', ''),
                            'url': subthread.get('url', ''),
                            'author': subthread['root_post'].get('author', ''),
                            'score': 1.0 - (i * 0.05),  # Decreasing score
                            'source': 'chromadb'
                        }
                        results.append(result)
            except Exception as e:
                logger.warning(f"ChromaDB search failed: {e}")
        
        # Fallback to TF-IDF if ChromaDB failed or no results
        if not results and self.tfidf_matrix is not None:
            try:
                query_vector = self.vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                top_indices = similarities.argsort()[-top_k:][::-1]
                
                subthread_list = list(self.subthreads.values())
                for idx in top_indices:
                    if similarities[idx] > 0:
                        subthread = subthread_list[idx]
                        result = {
                            'topic_id': list(self.subthreads.keys())[idx],
                            'topic_title': subthread.get('topic_title', ''),
                            'content': subthread.get('combined_content', ''),
                            'url': subthread.get('url', ''),
                            'author': subthread['root_post'].get('author', ''),
                            'score': float(similarities[idx]),
                            'source': 'tfidf'
                        }
                        results.append(result)
            except Exception as e:
                logger.warning(f"TF-IDF search failed: {e}")
        
        return results[:top_k]