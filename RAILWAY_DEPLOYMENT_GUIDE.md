# Railway Deployment Guide

A comprehensive guide for deploying the TDS Virtual Teaching Assistant FastAPI application to Railway platform.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Step-by-Step Deployment](#step-by-step-deployment)
- [Testing the Deployment](#testing-the-deployment)
- [Troubleshooting](#troubleshooting)
- [Alternative Platforms](#alternative-platforms-flyio-render)
- [Performance Monitoring](#performance-monitoring)

## Prerequisites

Before deploying to Railway, ensure you have:

1. **Railway Account**: Sign up at [railway.app](https://railway.app) (free tier available)
2. **GitHub Repository**: Your code must be in a Git repository (GitHub, GitLab, or Bitbucket)
3. **Project Structure**: Ensure your project contains:
   - [`main.py`](main.py:1) - FastAPI application entry point
   - [`requirements.txt`](requirements.txt:1) - Python dependencies
   - [`railway.toml`](railway.toml:1) - Railway configuration
   - [`.railwayignore`](.railwayignore:1) - Files to exclude from deployment
   - `discourse_posts.json` - Data file for the search engine

### Railway Free Tier Limits
- **Memory**: 512MB RAM
- **Storage**: 1GB disk space
- **Bandwidth**: 100GB/month
- **Execution time**: 500 hours/month
- **Sleep after 30 minutes** of inactivity

### Application Resource Requirements
- **Estimated Memory Usage**: 210-340MB (well within free tier)
- **Dependencies Size**: ~50MB
- **Data File**: discourse_posts.json (~5-10MB)

## Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Verify Configuration Files**:
   ```bash
   # Check that these files exist in your repository root:
   ls -la railway.toml .railwayignore main.py requirements.txt
   ```

2. **Review Railway Configuration** ([`railway.toml`](railway.toml:1)):
   ```toml
   [build]
   builder = "NIXPACKS"
   
   [deploy]
   startCommand = "uvicorn main:app --host 0.0.0.0 --port $PORT"
   restartPolicyType = "ON_FAILURE"
   restartPolicyMaxRetries = 10
   
   [deploy.healthcheck]
   httpPath = "/health"
   timeoutSeconds = 60
   ```

3. **Commit and Push** all changes to your repository:
   ```bash
   git add .
   git commit -m "Prepare for Railway deployment"
   git push origin main
   ```

### Step 2: Create Railway Project

1. **Login to Railway**:
   - Visit [railway.app](https://railway.app)
   - Sign in with GitHub (recommended for easy repository access)

2. **Create New Project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository from the list
   - Select the branch (usually `main` or `master`)

3. **Configure Deployment**:
   - Railway will automatically detect it's a Python project
   - It will use NIXPACKS builder (specified in railway.toml)
   - The build process will install dependencies from requirements.txt

### Step 3: Environment Configuration

Railway will automatically set the `PORT` environment variable. No additional environment variables are required for basic deployment.

**Optional Environment Variables** (if needed later):
- `LOG_LEVEL`: Set to "INFO" or "DEBUG"
- `WORKERS`: Number of Uvicorn workers (default: 1 for free tier)

### Step 4: Monitor Deployment

1. **Build Process**:
   - Watch the build logs in Railway dashboard
   - Build typically takes 2-5 minutes
   - NIXPACKS will automatically detect Python and install dependencies

2. **Deployment Logs**:
   ```
   [NIXPACKS] Installing Python dependencies...
   [NIXPACKS] Starting with uvicorn main:app --host 0.0.0.0 --port $PORT
   INFO: Started server process
   INFO: Waiting for application startup.
   INFO: Application startup complete.
   INFO: Uvicorn running on http://0.0.0.0:XXXX
   ```

3. **Generate Domain**:
   - Railway automatically assigns a domain: `https://[project-name].railway.app`
   - You can customize the domain name in project settings

## Testing the Deployment

### Step 1: Health Check

Test the basic health endpoint:
```bash
curl https://[your-app-name].railway.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "TDS Virtual Teaching Assistant",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Step 2: API Documentation

Access the interactive API documentation:
- Swagger UI: `https://[your-app-name].railway.app/docs`
- ReDoc: `https://[your-app-name].railway.app/redoc`

### Step 3: Test Question Endpoint

Test the main functionality:
```bash
curl -X POST "https://[your-app-name].railway.app/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is machine learning?"
  }'
```

Expected response format:
```json
{
  "answer": "Machine learning is...",
  "links": [
    {
      "url": "https://discourse.example.com/t/topic/123",
      "text": "Relevant excerpt from the post..."
    }
  ]
}
```

### Step 4: Load Testing (Optional)

For production readiness:
```bash
# Simple load test
for i in {1..10}; do
  curl -s -X POST "https://[your-app-name].railway.app/ask" \
    -H "Content-Type: application/json" \
    -d '{"question": "test question '$i'"}' &
done
wait
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Deployment Fails During Build

**Problem**: Dependencies installation fails
```
ERROR: Could not find a version that satisfies the requirement
```

**Solution**:
- Check [`requirements.txt`](requirements.txt:1) for version conflicts
- Update package versions:
  ```txt
  fastapi>=0.104.0
  uvicorn[standard]>=0.24.0
  scikit-learn>=1.3.0
  ```

#### 2. Application Starts But Health Check Fails

**Problem**: Health endpoint returns 404 or 500

**Solution**:
- Verify health endpoint in [`main.py`](main.py:1)
- Check if the route is properly defined:
  ```python
  @app.get("/health")
  async def health_check():
      return {"status": "healthy"}
  ```

#### 3. Memory Usage Too High

**Problem**: Application exceeds 512MB memory limit
```
Application exceeded memory limit and was terminated
```

**Solutions**:
- Reduce scikit-learn memory usage:
  ```python
  # In main.py, optimize TF-IDF parameters
  vectorizer = TfidfVectorizer(
      max_features=5000,  # Reduce from default
      stop_words='english',
      lowercase=True
  )
  ```
- Implement lazy loading for large data files

#### 4. Port Configuration Issues

**Problem**: Application not responding on correct port
```
Application failed to bind to port
```

**Solution**:
- Verify PORT environment variable usage in [`main.py`](main.py:1):
  ```python
  if __name__ == "__main__":
      port = int(os.environ.get("PORT", 8000))
      uvicorn.run(app, host="0.0.0.0", port=port)
  ```

#### 5. Data File Not Found

**Problem**: discourse_posts.json not loading
```
FileNotFoundError: discourse_posts.json
```

**Solution**:
- Ensure data file is not in [`.railwayignore`](.railwayignore:1)
- Check file path in DiscourseSearchEngine initialization
- Verify file is committed to repository

### Debugging Steps

1. **Check Deployment Logs**:
   - Go to Railway dashboard → Your project → Deployments
   - Click on latest deployment to view logs

2. **Monitor Resource Usage**:
   - Railway dashboard shows CPU and memory usage
   - Watch for spikes during traffic

3. **Test Locally First**:
   ```bash
   # Test the exact same configuration locally
   PORT=8000 uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

## Alternative Platforms (Fly.io, Render)

If Railway doesn't meet your needs, consider these alternatives:

### Fly.io Deployment

1. **Install Fly CLI**:
   ```bash
   # Windows (PowerShell)
   iwr https://fly.io/install.ps1 -useb | iex
   
   # macOS/Linux
   curl -L https://fly.io/install.sh | sh
   ```

2. **Initialize Fly App**:
   ```bash
   fly auth login
   fly launch
   ```

3. **Create Dockerfile** (Fly.io requires Docker):
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8000
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

4. **Deploy**:
   ```bash
   fly deploy
   ```

**Fly.io Free Tier**: 256MB RAM, 1GB storage, 160GB bandwidth

### Render Deployment

1. **Connect Repository**:
   - Go to [render.com](https://render.com)
   - Create new "Web Service"
   - Connect GitHub repository

2. **Configure Service**:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **Deploy**:
   - Render automatically deploys on git push
   - Free tier includes SSL certificate

**Render Free Tier**: 512MB RAM, 1GB storage, 750 hours/month

### Platform Comparison

| Feature | Railway | Fly.io | Render |
|---------|---------|--------|--------|
| Free RAM | 512MB | 256MB | 512MB |
| Free Storage | 1GB | 1GB | 1GB |
| Auto-sleep | Yes (30min) | Yes | Yes (15min) |
| Custom domains | Yes | Yes | Yes |
| Build time | Fast | Medium | Medium |
| Complexity | Low | Medium | Low |

## Performance Monitoring

### Railway Built-in Monitoring

1. **Metrics Dashboard**:
   - CPU usage over time
   - Memory consumption
   - Request count and response times
   - Error rates

2. **Logs Monitoring**:
   - Real-time application logs
   - Build and deployment logs
   - System events

### Application-Level Monitoring

Add performance logging to your FastAPI app:

```python
import time
from fastapi import Request

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.2f}s")
    return response
```

### Health Monitoring

Set up external monitoring:

1. **UptimeRobot** (free):
   - Monitor `/health` endpoint
   - Email alerts on downtime
   - 5-minute check intervals

2. **Custom Health Checks**:
   ```python
   @app.get("/health")
   async def enhanced_health():
       return {
           "status": "healthy",
           "service": "TDS Virtual Teaching Assistant",
           "timestamp": datetime.utcnow().isoformat(),
           "memory_usage": psutil.virtual_memory().percent,
           "disk_usage": psutil.disk_usage('/').percent
       }
   ```

### Performance Optimization Tips

1. **Memory Management**:
   ```python
   # Implement caching for frequently accessed data
   from functools import lru_cache
   
   @lru_cache(maxsize=100)
   def search_similar_posts(query: str):
       # Search implementation
       pass
   ```

2. **Response Optimization**:
   ```python
   # Enable gzip compression
   from fastapi.middleware.gzip import GZipMiddleware
   app.add_middleware(GZipMiddleware, minimum_size=1000)
   ```

3. **Database Connection Pooling** (if using database):
   ```python
   # Use connection pooling for better performance
   # Implement proper connection management
   ```

## Conclusion

Your TDS Virtual Teaching Assistant is now ready for Railway deployment. The configuration files are optimized for the platform, and the application should run smoothly within the free tier limits.

**Quick Deployment Checklist**:
- ✅ Repository contains all required files
- ✅ [`railway.toml`](railway.toml:1) configured with correct start command
- ✅ [`.railwayignore`](.railwayignore:1) excludes unnecessary files
- ✅ [`requirements.txt`](requirements.txt:1) has all dependencies
- ✅ Health check endpoint implemented
- ✅ PORT environment variable properly handled

**Next Steps**:
1. Deploy to Railway using the steps above
2. Test all endpoints thoroughly
3. Set up monitoring and alerts
4. Consider implementing caching for better performance
5. Plan for scaling if usage grows beyond free tier limits

For issues not covered in this guide, consult:
- [Railway Documentation](https://docs.railway.app)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Uvicorn Configuration](https://www.uvicorn.org/deployment/)