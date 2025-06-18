import os
import sys
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel

# Import the lightweight Railway search engine
from railway_search_engine import RailwaySearchEngine

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize lightweight search engine for Railway
logger.info("Initializing RailwaySearchEngine...")
try:
    search_engine = RailwaySearchEngine()
    logger.info("✅ RailwaySearchEngine initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize RailwaySearchEngine: {e}")
    logger.error(f"Exception type: {type(e).__name__}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

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
    search_engine: str = "railway_tfidf"

class QuestionAnswerer:
    def __init__(self, search_engine: RailwaySearchEngine):
        self.search_engine = search_engine

    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        # Get relevant context from search
        search_results = self.search_engine.search(question, top_k=top_k)
        
        if not search_results:
            return {
                "answer": "I couldn't find relevant information for your question. Could you try rephrasing it?",
                "confidence": "low",
                "sources": [],
                "search_method": "railway_tfidf"
            }
        
        # Format response with search results
        sources = []
        for result in search_results:
            sources.append({
                "title": result['topic_title'],
                "url": result['url'],
                "relevance_score": result['score'],
                "author": result['author']
            })
        
        # Create answer based on top result
        top_result = search_results[0]
        answer = f"Based on the discussion '{top_result['topic_title']}', here's what I found:\n\n"
        answer += f"{top_result['content'][:500]}..."
        
        if len(search_results) > 1:
            answer += f"\n\nI also found {len(search_results) - 1} other related discussions that might help."
        
        return {
            "answer": answer,
            "confidence": "medium",
            "sources": sources,
            "search_method": "railway_tfidf"
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
        "search_engine": "railway_tfidf",
        "features": ["lightweight", "tfidf_search", "railway_optimized"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "search_engine": "railway_tfidf",
        "total_subthreads": len(search_engine.subthreads),
        "has_tfidf_index": search_engine.tfidf_matrix is not None,
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