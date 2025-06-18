# TDS Virtual Teaching Assistant API

A FastAPI-based Virtual Teaching Assistant that answers student questions by searching through discourse posts and providing relevant answers with supporting links.

## Features

- **Question Answering**: Processes student questions and provides relevant answers
- **Discourse Search**: Searches through ~17,515 discourse posts using TF-IDF and cosine similarity
- **Staff Answer Prioritization**: Prioritizes answers from staff members (s.anand, carlton, Jivraj)
- **Engagement-based Ranking**: Considers like counts and reply counts in ranking
- **Image Support**: Accepts base64 encoded image attachments (basic validation)
- **Specific Question Handling**: Pre-programmed responses for known questions
- **RESTful API**: Clean JSON API compatible with promptfoo testing

## API Specification

### POST `/api/`

**Request Body:**
```json
{
  "question": "string (required) - Student question",
  "image": "string (optional) - Base64 encoded image attachment"
}
```

**Response:**
```json
{
  "answer": "string - Answer to the question",
  "links": [
    {
      "url": "string - Discourse URL",
      "text": "string - Relevant excerpt"
    }
  ]
}
```

### GET `/health`

Health check endpoint returning system status and loaded post count.

## Installation

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Ensure Data File:**
Make sure `discourse_posts.json` is in the same directory as `main.py`.

3. **Run the Server:**
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Testing

### Manual Testing
```bash
python test_api.py
```

### Using curl
```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/api/ \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I use Docker in this course?"}'
```

### Promptfoo Testing

The API is designed to work with the provided promptfoo configuration. Update the URL in `project-tds-virtual-ta-promptfoo.yaml`:

```yaml
providers:
  - id: https
    config:
      url: http://localhost:8000/api/  # Update this URL
```

Then run:
```bash
promptfoo eval -c project-tds-virtual-ta-promptfoo.yaml
```

## Architecture

### Core Components

1. **DiscourseSearchEngine**: 
   - Loads and indexes discourse posts
   - Uses TF-IDF vectorization for text similarity
   - Implements scoring algorithm prioritizing staff answers

2. **QuestionAnswerer**:
   - Generates answers based on search results
   - Handles specific predefined questions
   - Extracts relevant excerpts for links

3. **FastAPI Application**:
   - Provides REST endpoints
   - Validates input/output with Pydantic models
   - Handles errors and logging

### Search Algorithm

1. **Text Preprocessing**: Cleans URLs, HTML tags, normalizes whitespace
2. **TF-IDF Vectorization**: Creates feature vectors with bigrams and unigrams
3. **Cosine Similarity**: Computes similarity between query and all posts
4. **Scoring**: Combines similarity with engagement metrics and staff bonus
5. **Ranking**: Returns top-k results sorted by final score

### Scoring Formula

```
final_score = similarity_score * staff_multiplier + engagement_bonus + accepted_answer_bonus

Where:
- staff_multiplier = 2.0 for staff authors, 1.0 for others
- engagement_bonus = (like_count * 0.1) + (reply_count * 0.05)
- accepted_answer_bonus = similarity_score * 0.5 if is_accepted_answer
```

## Specific Question Handling

The system provides predefined answers for specific course questions:

- **GPT Model Questions**: Recommends using gpt-4o-mini from ai-proxy
- **GA4 Dashboard Scoring**: Explains bonus scoring display (shows "110")
- **Docker vs Podman**: Recommends Podman but accepts Docker
- **Unknown Information**: Gracefully handles questions about unavailable data

## File Structure

```
.
├── main.py                 # Main FastAPI application
├── requirements.txt        # Python dependencies
├── test_api.py            # Test script
├── README.md              # This file
├── discourse_posts.json   # Discourse data (required)
└── project-tds-virtual-ta-promptfoo.yaml  # Promptfoo config
```

## Dependencies

- **FastAPI**: Web framework for the API
- **scikit-learn**: TF-IDF vectorization and cosine similarity
- **pydantic**: Request/response validation
- **uvicorn**: ASGI server
- **numpy**: Numerical operations

## Configuration

### Environment Variables

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

### Tunable Parameters

In `DiscourseSearchEngine.__init__()`:
- `max_features`: Maximum TF-IDF features (default: 5000)
- `ngram_range`: N-gram range for tokenization (default: (1, 2))
- `min_df`: Minimum document frequency (default: 1)
- `max_df`: Maximum document frequency (default: 0.95)

## Deployment

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations

1. **Scaling**: Use multiple worker processes with Gunicorn
2. **Caching**: Implement Redis caching for frequent queries
3. **Monitoring**: Add health checks and metrics
4. **Security**: Add authentication and rate limiting
5. **Performance**: Pre-compute embeddings for faster search

## Troubleshooting

### Common Issues

1. **"No posts available"**: Ensure `discourse_posts.json` exists and is valid JSON
2. **Import errors**: Install all requirements with `pip install -r requirements.txt`
3. **Port conflicts**: Change port in uvicorn command or main.py
4. **Memory issues**: Reduce `max_features` parameter for large datasets

### Logs

The application logs important events:
- Post loading status
- Search index building
- Query processing
- Error conditions

Check console output for debugging information.

## API Examples

### Basic Question
```bash
curl -X POST http://localhost:8000/api/ \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I submit my assignment?"}'
```

### Question with Image
```bash
curl -X POST http://localhost:8000/api/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What is wrong with this code?", "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="}'
```

## License

This project is part of the TDS course curriculum.