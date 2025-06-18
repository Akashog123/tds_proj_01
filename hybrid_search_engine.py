import json
import logging
import os
from typing import List, Optional, Dict, Any, Tuple
import re
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

class HybridEmbeddingSearchEngine:
    """Hybrid search engine using OpenAI embeddings, local embeddings, and ChromaDB"""
    
    def __init__(self, posts_file: str = "discourse_posts.json", persist_directory: str = "./chroma_db"):
        self.posts = []
        self.subthreads = {}
        self.staff_authors = {"s.anand", "carlton", "Jivraj"}
        
        # Initialize embedding models
        self.local_model = None
        self.openai_client = None
        
        # Initialize ChromaDB
        self.chroma_client = None
        self.collection = None
        self.persist_directory = persist_directory
        
        # TF-IDF fallback
        self.vectorizer = None
        self.tfidf_matrix = None
        
        # Initialize all components
        self.load_posts(posts_file)
        self.extract_subthreads()
        self.setup_openai_client()
        self.setup_local_model()
        self.setup_chromadb()
        self.build_vector_index()
        self.build_tfidf_fallback()
    
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
        """Extract subthreads by grouping posts by topic_id"""
        topic_groups = defaultdict(list)
        
        # Group posts by topic_id
        for post in self.posts:
            topic_id = post.get('topic_id')
            if topic_id:
                topic_groups[topic_id].append(post)
        
        # Create subthreads for each topic
        for topic_id, posts_in_topic in topic_groups.items():
            # Sort posts by post_number to maintain thread order
            posts_in_topic.sort(key=lambda x: x.get('post_number', 0))
            
            # Find the root post (post_number = 1)
            root_post = None
            replies = []
            
            for post in posts_in_topic:
                if post.get('post_number') == 1:
                    root_post = post
                else:
                    replies.append(post)
            
            if root_post:
                # Create subthread with root post and all replies
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
            
            # Enhanced diagnostic logging
            logger.info(f"Environment variable check:")
            logger.info(f"  OPENAI_API_KEY present: {api_key is not None}")
            logger.info(f"  OPENAI_API_KEY length: {len(api_key) if api_key else 0}")
            logger.info(f"  OPENAI_BASE_URL: {base_url}")
            
            if not api_key:
                logger.warning("OPENAI_API_KEY not found in environment variables")
                return
            
            # Test if key looks like aipipe format (JWT-like)
            if api_key.count('.') == 2:
                logger.info("API key appears to be aipipe format (JWT-like)")
            else:
                logger.info("API key appears to be standard OpenAI format")
            
            self.openai_client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            logger.info(f"OpenAI client configured successfully with base_url: {base_url}")
        except Exception as e:
            logger.error(f"Error setting up OpenAI client: {e}")
            self.openai_client = None
    
    def setup_local_model(self):
        """Setup local sentence transformer model"""
        try:
            # Use a lightweight model for local embeddings
            self.local_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Local sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Error setting up local model: {e}")
            self.local_model = None
    
    def setup_chromadb(self):
        """Setup ChromaDB client and collection"""
        try:
            # Create persistent client
            self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Create or get collection
            collection_name = "discourse_embeddings"
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
                logger.info(f"Found existing ChromaDB collection: {collection_name}")
            except Exception:
                self.collection = self.chroma_client.create_collection(collection_name)
                logger.info(f"Created new ChromaDB collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"Error setting up ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None
    
    def get_openai_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from OpenAI API"""
        if not self.openai_client:
            logger.debug("OpenAI client not available, skipping embedding")
            return None
        
        try:
            logger.debug(f"Attempting to get OpenAI embedding for text: {text[:50]}...")
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                encoding_format="float"
            )
            logger.debug(f"Successfully got OpenAI embedding, dimension: {len(response.data[0].embedding)}")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            if hasattr(e, 'response'):
                logger.error(f"HTTP status: {getattr(e.response, 'status_code', 'unknown')}")
                logger.error(f"Response text: {getattr(e.response, 'text', 'no response text')}")
            return None
    
    def get_local_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from local model"""
        if not self.local_model:
            return None
        
        try:
            embedding = self.local_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error getting local embedding: {e}")
            return None
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better matching"""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def build_vector_index(self):
        """Build vector index using both OpenAI and local embeddings"""
        if not self.collection:
            logger.warning("ChromaDB collection not available")
            return
        
        # Check if we already have embeddings
        try:
            existing_count = self.collection.count()
            if existing_count > 0:
                logger.info(f"Found {existing_count} existing embeddings in ChromaDB")
                return
        except Exception as e:
            logger.warning(f"Error checking existing embeddings: {e}")
        
        logger.info("Building vector index with embeddings...")
        
        documents = []
        metadatas = []
        ids = []
        openai_embeddings = []
        local_embeddings = []
        
        for topic_id, subthread in self.subthreads.items():
            # Combine title and content for embedding
            title = subthread.get('topic_title', '')
            content = subthread.get('combined_content', '')
            combined_text = f"{title} {content}"
            processed_text = self.preprocess_text(combined_text)
            
            if not processed_text:
                continue
            
            # Get embeddings
            openai_emb = self.get_openai_embedding(processed_text)
            local_emb = self.get_local_embedding(processed_text)
            
            if openai_emb and local_emb:
                documents.append(processed_text)
                ids.append(f"topic_{topic_id}")
                
                # Store metadata (ChromaDB doesn't allow list/array values in metadata)
                metadata = {
                    'topic_id': str(topic_id),
                    'topic_title': title,
                    'url': subthread.get('url', ''),
                    'root_author': subthread['root_post'].get('author', ''),
                    'reply_count': len(subthread.get('replies', [])),
                    'is_staff_answer': subthread['root_post'].get('author', '') in self.staff_authors
                }
                metadatas.append(metadata)
                
                # Store embeddings separately for retrieval
                # We'll store local embeddings in a separate structure
                if not hasattr(self, 'local_embeddings_cache'):
                    self.local_embeddings_cache = {}
                self.local_embeddings_cache[f"topic_{topic_id}"] = local_emb
                
                # Use OpenAI embeddings as primary for ChromaDB storage
                openai_embeddings.append(openai_emb)
        
        if documents:
            try:
                # Add to ChromaDB using OpenAI embeddings
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=openai_embeddings
                )
                logger.info(f"Added {len(documents)} embeddings to ChromaDB")
            except Exception as e:
                logger.error(f"Error adding embeddings to ChromaDB: {e}")
    
    def build_local_embeddings_cache(self):
        """Build local embeddings cache from existing data"""
        if not self.local_model or not self.subthreads:
            logger.warning("Local model or subthreads not available")
            return
        
        logger.info("Building local embeddings cache...")
        self.local_embeddings_cache = {}
        
        for topic_id, subthread in self.subthreads.items():
            title = subthread.get('topic_title', '')
            content = subthread.get('combined_content', '')
            combined_text = f"{title} {content}"
            processed_text = self.preprocess_text(combined_text)
            
            if processed_text:
                local_emb = self.get_local_embedding(processed_text)
                if local_emb:
                    self.local_embeddings_cache[f"topic_{topic_id}"] = local_emb
        
        logger.info(f"Built local embeddings cache with {len(self.local_embeddings_cache)} embeddings")
    
    def build_tfidf_fallback(self):
        """Build TF-IDF index as fallback"""
        if not self.subthreads:
            logger.warning("No subthreads available for TF-IDF fallback")
            return
        
        documents = []
        for subthread in self.subthreads.values():
            title = subthread.get('topic_title', '')
            content = subthread.get('combined_content', '')
            combined_text = f"{title} {content}"
            processed_text = self.preprocess_text(combined_text)
            documents.append(processed_text)
        
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)
            logger.info(f"Built TF-IDF fallback index with {self.tfidf_matrix.shape[0]} documents")
        except Exception as e:
            logger.error(f"Error building TF-IDF fallback: {e}")
    
    def calculate_subthread_score(self, subthread: Dict[str, Any], similarity_score: float) -> float:
        """Calculate overall score for a subthread based on multiple factors"""
        score = similarity_score
        
        # Boost staff answers
        root_post = subthread.get('root_post', {})
        if root_post.get('author', '') in self.staff_authors:
            score *= 2.0
        
        # Boost based on engagement (from root post and replies)
        total_likes = root_post.get('like_count', 0)
        reply_count = len(subthread.get('replies', []))
        
        # Add engagement from replies
        for reply in subthread.get('replies', []):
            total_likes += reply.get('like_count', 0)
        
        # Add engagement bonus
        engagement_bonus = (total_likes * 0.1) + (reply_count * 0.05)
        score += engagement_bonus
        
        # Boost accepted answers
        if root_post.get('is_accepted_answer', False):
            score *= 1.5
        
        return score
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search using both OpenAI and local embeddings"""
        if not self.collection:
            logger.warning("ChromaDB not available, falling back to TF-IDF")
            return self.tfidf_fallback_search(query, top_k)
        
        try:
            # Get query embeddings
            query_openai_emb = self.get_openai_embedding(query)
            query_local_emb = self.get_local_embedding(query)
            
            if not query_openai_emb:
                logger.warning("OpenAI embedding failed, falling back to TF-IDF")
                return self.tfidf_fallback_search(query, top_k)
            
            # Search with OpenAI embeddings (primary)
            openai_results = self.collection.query(
                query_embeddings=[query_openai_emb],
                n_results=min(top_k * 2, 20)  # Get more results for hybrid scoring
            )
            
            # Process results and calculate hybrid scores
            hybrid_results = []
            
            for i, doc_id in enumerate(openai_results['ids'][0]):
                metadata = openai_results['metadatas'][0][i]
                openai_distance = openai_results['distances'][0][i]
                openai_similarity = 1 - openai_distance  # Convert distance to similarity
                
                # Calculate local similarity if available
                local_similarity = 0.0
                if query_local_emb and hasattr(self, 'local_embeddings_cache'):
                    try:
                        doc_id_key = doc_id  # doc_id is already in format "topic_{topic_id}"
                        if doc_id_key in self.local_embeddings_cache:
                            local_emb = self.local_embeddings_cache[doc_id_key]
                            local_similarity = np.dot(query_local_emb, local_emb) / (
                                np.linalg.norm(query_local_emb) * np.linalg.norm(local_emb)
                            )
                    except Exception as e:
                        logger.warning(f"Error calculating local similarity: {e}")
                
                # Hybrid scoring: 70% OpenAI + 30% local
                hybrid_similarity = 0.7 * openai_similarity + 0.3 * local_similarity
                
                # Get subthread info
                topic_id = metadata['topic_id']
                if topic_id in self.subthreads:
                    subthread = self.subthreads[topic_id]
                    final_score = self.calculate_subthread_score(subthread, hybrid_similarity)
                    
                    result = {
                        'topic_id': topic_id,
                        'topic_title': metadata['topic_title'],
                        'url': metadata['url'],
                        'content': subthread['combined_content'],
                        'author': metadata['root_author'],
                        'reply_count': metadata['reply_count'],
                        'openai_similarity': openai_similarity,
                        'local_similarity': local_similarity,
                        'hybrid_similarity': hybrid_similarity,
                        'final_score': final_score,
                        'is_staff_answer': metadata.get('is_staff_answer', False)
                    }
                    hybrid_results.append(result)
            
            # Sort by final score and return top results
            hybrid_results.sort(key=lambda x: x['final_score'], reverse=True)
            return hybrid_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return self.tfidf_fallback_search(query, top_k)
    
    def tfidf_fallback_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Fallback to TF-IDF search if embeddings fail"""
        if not self.vectorizer or self.tfidf_matrix is None:
            logger.warning("TF-IDF fallback not available")
            return []
        
        try:
            # Process query
            processed_query = self.preprocess_text(query)
            query_vector = self.vectorizer.transform([processed_query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top results
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            subthread_list = list(self.subthreads.values())
            
            for idx in top_indices:
                if idx < len(subthread_list) and similarities[idx] > 0.01:
                    subthread = subthread_list[idx]
                    final_score = self.calculate_subthread_score(subthread, similarities[idx])
                    
                    result = {
                        'topic_id': subthread['root_post'].get('topic_id'),
                        'topic_title': subthread.get('topic_title', ''),
                        'url': subthread.get('url', ''),
                        'content': subthread.get('combined_content', ''),
                        'author': subthread['root_post'].get('author', ''),
                        'reply_count': len(subthread.get('replies', [])),
                        'similarity_score': similarities[idx],
                        'final_score': final_score,
                        'is_staff_answer': subthread['root_post'].get('author', '') in self.staff_authors
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in TF-IDF fallback search: {e}")
            return []
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Main search interface"""
        return self.hybrid_search(query, top_k)