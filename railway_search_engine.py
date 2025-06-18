import json
import logging
import os
from typing import List, Optional, Dict, Any
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class RailwaySearchEngine:
    """Lightweight search engine for Railway deployment using only TF-IDF"""
    
    def __init__(self, posts_file: str = "discourse_posts.json"):
        self.posts = []
        self.subthreads = {}
        self.staff_authors = {"s.anand", "carlton", "Jivraj"}
        
        # TF-IDF only (no heavy ML dependencies)
        self.vectorizer = None
        self.tfidf_matrix = None
        
        # Load data
        self.load_posts(posts_file)
        self.extract_subthreads()
        
        # Try to load pre-computed TF-IDF data first.
        if not self.load_precomputed_tfidf():
            # Then try to load precomputed embeddings from Chroma DB.
            if not self.load_precomputed_chroma():
                logger.info("Building TF-IDF index from scratch...")
                self.build_tfidf_index()
                self.save_precomputed_tfidf()
        
        logger.info("✅ Railway search engine initialized successfully")
    
    def load_precomputed_tfidf(self) -> bool:
        """Load pre-computed TF-IDF data if available"""
        tfidf_file = Path("precomputed_embeddings") / "tfidf_data.pkl"
        
        if not tfidf_file.exists():
            return False
        
        try:
            with open(tfidf_file, 'rb') as f:
                tfidf_data = pickle.load(f)
                self.tfidf_matrix = tfidf_data['matrix']
                self.vectorizer = tfidf_data['vectorizer']
            logger.info("✅ Loaded pre-computed TF-IDF index")
            return True
        except Exception as e:
            logger.error(f"Error loading pre-computed TF-IDF: {e}")
            return False
    def load_precomputed_chroma(self) -> bool:
        """Load precomputed embeddings from Chroma DB if available.
        
        Expects file 'precomputed_embeddings/chroma_data.pkl' containing a dictionary
        with keys 'matrix' and 'vectorizer'.
        """
        chroma_file = Path("precomputed_embeddings") / "chroma_data.pkl"
        if not chroma_file.exists():
            logger.info("Precomputed Chroma DB embeddings not found.")
            return False
        try:
            with open(chroma_file, "rb") as f:
                chroma_data = pickle.load(f)
            self.tfidf_matrix = chroma_data.get("matrix", None)
            self.vectorizer = chroma_data.get("vectorizer", None)
            if self.tfidf_matrix is not None and self.vectorizer is not None:
                logger.info("✅ Loaded precomputed embeddings from Chroma DB")
                return True
            else:
                logger.error("Chroma DB file is invalid: missing required data.")
                return False
        except Exception as e:
            logger.error(f"Error loading precomputed Chroma DB data: {e}")
            return False
            logger.error(f"Error loading pre-computed TF-IDF: {e}")
            return False
    
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
    
    def build_tfidf_index(self):
        """Build TF-IDF index"""
        if not self.subthreads:
            logger.warning("No subthreads available for TF-IDF")
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
            logger.info(f"Built TF-IDF index with {len(documents)} documents")
            self.save_precomputed_tfidf()
        except Exception as e:
            logger.error(f"Error building TF-IDF index: {e}")
    def save_precomputed_tfidf(self):
        """Save computed TF-IDF index for future use"""
        tfidf_file = Path("precomputed_embeddings") / "tfidf_data.pkl"
        try:
            with open(tfidf_file, "wb") as f:
                pickle.dump({
                    "matrix": self.tfidf_matrix,
                    "vectorizer": self.vectorizer
                }, f)
            logger.info("✅ Saved TF-IDF index to precomputed file")
        except Exception as e:
            logger.error(f"Error saving precomputed TF-IDF: {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using TF-IDF"""
        if not query.strip() or self.tfidf_matrix is None:
            return []
        
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            subthread_list = list(self.subthreads.values())
            subthread_ids = list(self.subthreads.keys())
            
            for idx in top_indices:
                if similarities[idx] > 0:
                    subthread = subthread_list[idx]
                    result = {
                        'topic_id': subthread_ids[idx],
                        'topic_title': subthread.get('topic_title', ''),
                        'content': subthread.get('combined_content', ''),
                        'url': subthread.get('url', ''),
                        'author': subthread['root_post'].get('author', ''),
                        'score': float(similarities[idx]),
                        'source': 'railway_tfidf'
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"TF-IDF search failed: {e}")
            return []