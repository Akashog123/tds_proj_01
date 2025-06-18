import os
import sys
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel

# Import the fast hybrid search engine for better answers
from fast_hybrid_search_engine import FastHybridEmbeddingSearchEngine

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add startup diagnostics
logger.info("=== TDS Virtual TA (Railway) Startup Diagnostics ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"PORT environment variable: {os.getenv('PORT', 'Not set')}")
logger.info(f"OPENAI_API_KEY set: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
logger.info(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', 'Not set')}")

# Check critical files
discourse_file = "discourse_posts.json"
logger.info(f"discourse_posts.json exists: {os.path.exists(discourse_file)}")
if os.path.exists(discourse_file):
    logger.info(f"discourse_posts.json size: {os.path.getsize(discourse_file) / 1024:.1f} KB")

# Create FastAPI app
app = FastAPI(
    title="TDS Virtual TA API (Railway)",
    description="Lightweight Teaching Assistant API for TDS course using Railway deployment",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://exam.sanand.workers.dev",
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost",
        "*"  # Allow all origins for development/testing
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

# Initialize fast hybrid search engine for Railway
logger.info("Initializing FastHybridEmbeddingSearchEngine...")
try:
    search_engine = FastHybridEmbeddingSearchEngine()
    logger.info("✅ FastHybridEmbeddingSearchEngine initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize FastHybridEmbeddingSearchEngine: {e}")
    logger.error(f"Exception type: {type(e).__name__}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class QuestionResponse(BaseModel):
    answer: str
    links: List[Dict[str, str]]

class SearchResult(BaseModel):
    topic_id: int
    topic_title: str
    content: str
    url: str
    author: str
    score: float
    source: str

class QueryResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    search_engine: str = "hybrid_openai_embeddings"

class QuestionAnswerer:
    def __init__(self, search_engine: FastHybridEmbeddingSearchEngine):
        self.search_engine = search_engine

    def extract_relevant_excerpt(self, post: Dict[str, Any], query: str, max_length: int = 200) -> str:
        """Extract relevant excerpt from post content"""
        content = post.get('content', '')
        if not content:
            return post.get('topic_title', '')[:max_length]
        
        # Simple approach: take first part of content
        if len(content) <= max_length:
            return content
        
        # Try to find query terms in content for better excerpt
        query_words = query.lower().split()
        content_lower = content.lower()
        
        best_start = 0
        max_matches = 0
        
        # Find the section with most query word matches
        for i in range(0, len(content) - max_length, 50):
            section = content_lower[i:i + max_length]
            matches = sum(1 for word in query_words if word in section)
            if matches > max_matches:
                max_matches = matches
                best_start = i
        
        excerpt = content[best_start:best_start + max_length]
        if best_start > 0:
            excerpt = "..." + excerpt
        if best_start + max_length < len(content):
            excerpt = excerpt + "..."
            
        return excerpt

    def generate_answer_from_hybrid_search(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Generate answer using hybrid search results without direct GPT prompting"""
        if not search_results:
            return "I couldn't find relevant information in the discourse posts to answer your question. Please try rephrasing your question or check the course materials directly."
        
        # Get the best result from hybrid search
        best_result = search_results[0]
        
        # Check if it's from staff
        author = best_result.get('author', '')
        is_staff_answer = author in self.search_engine.staff_authors
        
        # Generate answer based on content
        content = best_result.get('content', '')
        title = best_result.get('topic_title', '')
        score = best_result.get('score', 0)
        
        # Create answer prefix based on author and relevance
        if is_staff_answer:
            answer_prefix = f"According to {author} (course staff), "
        elif score > 0.8:  # High confidence from hybrid search
            answer_prefix = "Based on the most relevant discussion in the course forum, "
        else:
            answer_prefix = "Based on the discussions in the course forum, "
        
        # Extract the most relevant part of the content
        if content:
            relevant_excerpt = self.extract_relevant_excerpt(best_result, query, 400)
            answer = f"{answer_prefix}{relevant_excerpt}"
            
            # Add context about additional results if available
            if len(search_results) > 1:
                high_score_results = [r for r in search_results[1:3] if r.get('score', 0) > 0.6]
                if high_score_results:
                    answer += f"\n\nAdditional relevant information was found in {len(high_score_results)} other discussions that might also help."
        else:
            answer = f"{answer_prefix}there was a discussion about '{title}' but detailed content is not available."
        
        return answer

    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        # Get relevant context from hybrid search (OpenAI + local embeddings)
        search_results = self.search_engine.search(question, top_k=top_k)
        
        # Generate intelligent answer using hybrid search results
        answer = self.generate_answer_from_hybrid_search(question, search_results)
        
        # Format response with search results
        sources = []
        for result in search_results:
            sources.append({
                "title": result['topic_title'],
                "url": result['url'],
                "relevance_score": result['score'],
                "author": result['author']
            })
        
        # Determine confidence based on search quality
        confidence = "low"
        if search_results:
            best_score = search_results[0].get('score', 0)
            if best_score > 0.8:
                confidence = "high"
            elif best_score > 0.6:
                confidence = "medium"
        
        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "search_method": "hybrid_openai_local_embeddings"
        }

# Initialize question answerer
qa_system = QuestionAnswerer(search_engine)

# API Routes
@app.get("/")
async def root():
    return {
        "message": "TDS Virtual TA API (Railway)",
        "version": "2.0.0",
        "status": "running",
        "search_engine": "hybrid_openai_embeddings",
        "features": ["hybrid_search", "openai_embeddings", "chromadb", "railway_optimized"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "search_engine": "hybrid_openai_embeddings",
        "total_subthreads": len(search_engine.subthreads),
        "has_chromadb": search_engine.collection is not None,
        "has_openai_client": search_engine.openai_client is not None,
        "has_local_model": search_engine.local_model is not None,
        "has_tfidf_fallback": search_engine.tfidf_matrix is not None,
        "deployment": "railway"
    }

@app.post("/search", response_model=QueryResponse)
async def search_posts(request: QueryRequest):
    try:
        results = search_engine.search(request.query, top_k=request.top_k)
        
        search_results = [
            SearchResult(
                topic_id=result['topic_id'],
                topic_title=result['topic_title'],
                content=result['content'][:500] + "...",  # Truncate for API response
                url=result['url'],
                author=result['author'],
                score=result['score'],
                source=result['source']
            )
            for result in results
        ]
        
        return QueryResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results)
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        answer_data = qa_system.answer_question(request.query, top_k=request.top_k)
        return answer_data
    except Exception as e:
        logger.error(f"Question answering error: {e}")
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")

@app.post("/api/", response_model=QuestionResponse)
async def answer_question_api(request: QuestionRequest):
    """
    Answer student questions based on discourse posts (compatible with main.py API)
    
    - **question**: The student's question (required)
    - **image**: Optional base64 encoded image attachment
    """
    try:
        logger.info(f"Received question via /api/: {request.question[:100]}...")
        
        # Handle image if provided (basic processing)
        if request.image:
            try:
                import base64
                # Validate base64 image (basic check)
                base64.b64decode(request.image, validate=True)
                logger.info("Image attachment received and validated")
                # For now, we'll acknowledge the image but not process it fully
            except Exception as e:
                logger.warning(f"Invalid image data: {e}")
        
        # Get answer using the existing QA system
        answer_data = qa_system.answer_question(request.question, top_k=5)
        
        # Convert to the expected response format
        links = []
        for source in answer_data.get('sources', []):
            links.append({
                "url": source.get('url', ''),
                "text": source.get('title', '')[:150] + "..." if len(source.get('title', '')) > 150 else source.get('title', '')
            })
        
        response = QuestionResponse(
            answer=answer_data.get('answer', 'No answer could be generated.'),
            links=links
        )
        
        logger.info(f"Generated response with {len(links)} links via /api/")
        return response
        
    except Exception as e:
        logger.error(f"Error processing question via /api/: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while processing question")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)