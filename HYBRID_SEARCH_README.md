# Hybrid Embedding Search System

This document describes the upgraded Virtual Teaching Assistant API that now uses a hybrid search system combining OpenAI embeddings with ChromaDB for semantic search.

## Overview

The system has been upgraded from TF-IDF search to a sophisticated hybrid search system that combines:

- **OpenAI text-embedding-3-small** embeddings (via aipipe proxy)
- **Local sentence-transformers** embeddings (all-MiniLM-L6-v2)
- **ChromaDB** vector database for efficient storage and retrieval
- **Subthread extraction** for better context understanding
- **TF-IDF fallback** for reliability

## Architecture

### HybridEmbeddingSearchEngine

The new search engine (`hybrid_search_engine.py`) provides:

1. **Subthread Extraction**: Groups discourse posts by topic_id to create conversation threads with root posts and all replies
2. **Dual Embeddings**: Generates both OpenAI and local embeddings for each subthread
3. **Vector Storage**: Stores embeddings in ChromaDB for fast similarity search
4. **Hybrid Scoring**: Combines OpenAI (70%) and local (30%) similarity scores
5. **Enhanced Ranking**: Considers staff answers, engagement metrics, and accepted answers

### Key Features

- **Semantic Search**: Uses state-of-the-art embeddings for better understanding
- **Context Preservation**: Maintains conversation threads for better context
- **Graceful Degradation**: Falls back to TF-IDF if embeddings fail
- **Railway Compatible**: Designed for cloud deployment

## Environment Configuration

Required environment variables:

```bash
# Required for OpenAI embeddings
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom OpenAI base URL for aipipe proxy
OPENAI_BASE_URL=https://your-aipipe-proxy-url/v1

# Railway automatically sets PORT
PORT=8000
```

## Dependencies

New dependencies added to `requirements.txt`:

```
chromadb>=0.4.18
openai>=1.12.0
sentence-transformers>=2.2.2
```

## API Compatibility

The API maintains **complete backward compatibility**:

- Same endpoints: `POST /api/`, `GET /health`, `GET /`
- Same request/response formats
- Same specific question handlers
- All promptfoo test cases continue to pass

## Search Process

1. **Query Processing**: Preprocesses and cleans the input query
2. **Embedding Generation**: Creates embeddings using both OpenAI and local models
3. **Vector Search**: Queries ChromaDB using OpenAI embeddings
4. **Hybrid Scoring**: Combines similarities with weighted average (70% OpenAI + 30% local)
5. **Ranking Enhancement**: Applies staff author boost, engagement metrics, and answer acceptance
6. **Result Formatting**: Returns results in the same format as before

## Data Structure

### Subthread Format

Each subthread contains:
- Root post (post_number = 1)
- All replies in chronological order
- Combined content for embedding
- Metadata (topic_title, URL, author info)

### Search Results

Results include:
- Topic information and URLs
- Author details and staff indicators
- Engagement metrics (likes, replies)
- Multiple similarity scores
- Enhanced ranking scores

## Performance Optimizations

- **Persistent Storage**: ChromaDB stores embeddings locally
- **Efficient Querying**: Uses vector similarity for fast retrieval
- **Batch Processing**: Optimized embedding generation
- **Memory Management**: Efficient data structures

## Fallback Mechanisms

1. **TF-IDF Fallback**: If OpenAI embeddings fail
2. **Local-Only Search**: If OpenAI API is unavailable
3. **Error Handling**: Graceful degradation with logging

## Deployment Notes

- ChromaDB creates a local `chroma_db/` directory for persistence
- The directory is excluded from Railway deployment via `.railwayignore`
- Embeddings are regenerated on each deployment for fresh data
- Environment variables must be configured in Railway dashboard

## Monitoring

Enhanced health check endpoint provides:
- Posts loaded count
- Subthreads extracted count
- ChromaDB status
- OpenAI client status
- Local model status
- TF-IDF fallback status

## Migration Notes

The system automatically:
- Extracts subthreads from existing discourse posts
- Generates embeddings for all content
- Builds vector index in ChromaDB
- Creates TF-IDF fallback index
- Maintains all existing functionality

No manual migration steps are required.