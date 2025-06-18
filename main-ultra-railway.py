import os
import sys
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel

# Import the ultra-lightweight search engine
from ultra_lightweight_engine import UltraLightweightSearchEngine

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Startup diagnostics for ultra-lightweight deployment
logger.info("=== TDS Virtual TA (Ultra-Lightweight Railway) Startup ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"PORT environment variable: {os.getenv('PORT', 'Not set')}")
logger.info("🚀 Ultra-lightweight mode: NO external API dependencies")
logger.info("📦 Using precomputed embeddings only")

# Check for precomputed data
precomputed_dir = "precomputed_ultra"
logger.info(f"Precomputed directory exists: {os.path.exists(precomputed_dir)}")
if os.path.exists(precomputed_dir):
    import glob
    files = glob.glob(f"{precomputed_dir}/*")
    logger.info(f"Precomputed files: {[os.path.basename(f) for f in files]}")

# Create FastAPI app
app = FastAPI(
    title="TDS Virtual TA API (Ultra-Lightweight Railway)",
    description="Ultra-lightweight Teaching Assistant API using precomputed embeddings only",
    version="3.0.0"
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

# Initialize ultra-lightweight search engine
logger.info("🔧 Initializing UltraLightweightSearchEngine...")
try:
    search_engine = UltraLightweightSearchEngine()
    stats = search_engine.get_stats()
    logger.info("✅ UltraLightweightSearchEngine initialized successfully")
    logger.info(f"📊 Engine stats: {stats}")
except Exception as e:
    logger.error(f"❌ Failed to initialize UltraLightweightSearchEngine: {e}")
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
    topic_id: str
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
    search_engine: str = "ultra_lightweight_precomputed"

class UltraLightweightQuestionAnswerer:
    """Ultra-lightweight question answerer using precomputed data only"""
    
    def __init__(self, search_engine: UltraLightweightSearchEngine):
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

    def generate_answer_from_precomputed_search(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Generate answer using precomputed search results"""
        if not search_results:
            return "I couldn't find relevant information in the precomputed discourse posts to answer your question. Please try rephrasing your question or check the course materials directly."
        
        # Get the best result from precomputed search
        best_result = search_results[0]
        
        # Check if it's from staff
        author = best_result.get('author', '')
        staff_authors = self.search_engine.staff_authors
        is_staff_answer = author in staff_authors
        
        # Generate answer based on content
        content = best_result.get('content', '')
        title = best_result.get('topic_title', '')
        score = best_result.get('score', 0)
        
        # Create answer prefix based on author and relevance
        if is_staff_answer:
            answer_prefix = f"According to {author} (course staff), "
        elif score > 0.8:  # High confidence from search
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
        """Answer question using ultra-lightweight precomputed search"""
        # Get relevant context from precomputed search
        search_results = self.search_engine.search(question, top_k=top_k)
        
        # Generate answer using precomputed search results
        answer = self.generate_answer_from_precomputed_search(question, search_results)
        
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
            "search_method": "ultra_lightweight_precomputed"
        }

# Initialize question answerer
qa_system = UltraLightweightQuestionAnswerer(search_engine)

# API Routes
@app.get("/")
async def root():
    stats = search_engine.get_stats()
    return {
        "message": "TDS Virtual TA API (Ultra-Lightweight Railway)",
        "version": "3.0.0",
        "status": "running",
        "search_engine": "ultra_lightweight_precomputed",
        "features": ["precomputed_embeddings", "ultra_fast_startup", "no_external_apis"],
        "stats": stats
    }

@app.get("/health")
async def health_check():
    stats = search_engine.get_stats()
    return {
        "status": "healthy",
        "search_engine": "ultra_lightweight_precomputed",
        "total_subthreads": stats["total_subthreads"],
        "has_openai_embeddings": stats["has_openai_embeddings"],
        "has_local_embeddings": stats["has_local_embeddings"],
        "has_tfidf": stats["has_tfidf"],
        "deployment": "ultra_lightweight_railway",
        "external_apis": "none",
        "startup_mode": "precomputed_only"
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
    Answer student questions based on precomputed discourse data (compatible with main.py API)
    
    - **question**: The student's question (required)
    - **image**: Optional base64 encoded image attachment (acknowledged but not processed)
    """
    try:
        logger.info(f"Received question via /api/: {request.question[:100]}...")
        
        # Handle image if provided (basic acknowledgment)
        if request.image:
            try:
                import base64
                # Validate base64 image (basic check)
                base64.b64decode(request.image, validate=True)
                logger.info("Image attachment received and acknowledged (not processed in ultra-lightweight mode)")
            except Exception as e:
                logger.warning(f"Invalid image data: {e}")
        
        # Get answer using the ultra-lightweight QA system
        answer_data = qa_system.answer_question(request.question, top_k=5)
        
        # Convert to the expected response format
        links = []
        for source in answer_data.get('sources', []):
            links.append({
                "url": source.get('url', ''),
                "text": source.get('title', '')[:150] + "..." if len(source.get('title', '')) > 150 else source.get('title', '')
            })
        
        response = QuestionResponse(
            answer=answer_data.get('answer', 'No answer could be generated from precomputed data.'),
            links=links
        )
        
        logger.info(f"Generated response with {len(links)} links via /api/ (ultra-lightweight mode)")
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