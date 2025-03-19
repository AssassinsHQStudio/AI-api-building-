from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import openai
from datetime import datetime
import os
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="AI Message API",
    description="API for creating messages with OpenAI and retrieving previous jobs",
    version="1.0.0"
)

# In-memory storage for jobs (in a real application, you'd use a database)
jobs = []

@app.get("/")
async def root():
    return {
        "message": "Welcome to the AI Message API",
        "endpoints": {
            "create_message": "POST /messages",
            "get_all_jobs": "GET /jobs",
            "get_job_by_id": "GET /jobs/{job_id}",
            "get_models": "GET /models"
        },
        "documentation": "/docs"
    }

class MessageRequest(BaseModel):
    content: str
    model: str = "gpt-3.5-turbo"  # Default model

class Job(BaseModel):
    id: str
    content: str
    model: str
    response: str
    created_at: datetime
    status: str

@app.post("/messages", response_model=Job)
async def create_message(message: MessageRequest):
    try:
        # Create a chat completion
        response = openai.chat.completions.create(
            model=message.model,
            messages=[{"role": "user", "content": message.content}]
        )
        
        # Create a new job with UUID
        job = Job(
            id=str("llmjobid:" + str(uuid.uuid4())[:5].lower()),
            content=message.content,
            model=message.model,
            response=response.choices[0].message.content,
            created_at=datetime.now(),
            status="completed"
        )
        
        # Store the job
        jobs.append(job)
        
        return job
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs", response_model=List[Job])
async def list_jobs():
    return jobs

@app.get("/jobs/{job_id}", response_model=Job)
async def get_job(job_id: str):
    job = next((job for job in jobs if job.id == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/models")
async def get_models():
    try:
        # Fetch available models from OpenAI
        models = openai.models.list()
        
        # Format the response to include relevant information
        formatted_models = []
        for model in models:
            formatted_models.append({
                "id": model.id,
                "object": model.object,
                "created": model.created,
                "owned_by": model.owned_by,
            })
        
        return formatted_models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 