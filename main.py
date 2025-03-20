from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
import openai
from datetime import datetime
import os
from dotenv import load_dotenv
import uuid
import base64
from io import BytesIO
import json
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
JOBS_FILE = data_dir / "jobs.json"

app = FastAPI(
    title="Jaydens Message API",
    description="API for creating messages with OpenAI and retrieving previous jobs",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Messages",
            "description": "Operations with messages and AI responses"
        },
        {
            "name": "Jobs",
            "description": "Operations with job history and status"
        },
        {
            "name": "Models",
            "description": "Information about available AI models"
        }
    ]
)

class Job(BaseModel):
    id: str
    content: str
    model: str
    response: str
    created_at: datetime
    status: str
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

def load_jobs() -> List[Job]:
    if not JOBS_FILE.exists():
        return []
    try:
        with open(JOBS_FILE, 'r') as f:
            jobs_data = json.load(f)
            return [Job(**{**job, 'created_at': datetime.fromisoformat(job['created_at'])}) for job in jobs_data]
    except Exception as e:
        print(f"Error loading jobs: {e}")
        return []

def save_jobs(jobs: List[Job]):
    try:
        with open(JOBS_FILE, 'w') as f:
            json.dump([json.loads(job.json()) for job in jobs], f, indent=2)
    except Exception as e:
        print(f"Error saving jobs: {e}")

# Load jobs from file at startup
jobs = load_jobs()

@app.get("/", tags=["Messages"])
async def root():
    return {
        "message": "Welcome to the Jaydens Message API",
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
    model: str = "gpt-4-vision-preview"  # Updated default model
    images: Optional[List[str]] = None  # Changed to support multiple images

@app.post("/job", response_model=Job, tags=["Jobs"])
async def create_job(
    prompt: str = Form(...),
    model: str = Form("gpt-3.5-turbo"),
    images: Optional[List[UploadFile]] = None
):
    try:
        messages = []
        
        if images:
            # Process multiple images
            message_content = [{"type": "text", "text": prompt}]
            
            # Read and encode each image
            for image in images:
                image_content = await image.read()
                base64_image = base64.b64encode(image_content).decode('utf-8')
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            
            model = "gpt-4o-2024-11-20"  # Use vision model when images are present
            messages = [{"role": "user", "content": message_content}]
        else:
            # Create message with text only
            messages = [{"role": "user", "content": prompt}]
        
        # Create a chat completion
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000
        )
        
        # Create a new job with UUID
        job = Job(
            id=str("llmjobid:" + str(uuid.uuid4())[:5].lower()),
            content=prompt,
            model=model,
            response=response.choices[0].message.content,
            created_at=datetime.now(),
            status="completed"
        )
        
        # Store the job and save to file
        jobs.append(job)
        save_jobs(jobs)
        
        return job
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs", response_model=List[Job], tags=["Jobs"])
async def list_jobs():
    return jobs

@app.get("/jobs/{job_id}", response_model=Job, tags=["Jobs"])
async def get_job(job_id: str):
    job = next((job for job in jobs if job.id == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/models", tags=["Models"])
async def list_models():
    try:
        # Fetch available models from OpenAI using the newer method
        models = openai.models.list()
        
        # Define model capabilities
        model_capabilities = {
            "gpt-4-vision-preview": {
                "text_input": True,
                "text_output": True,
                "image_input": True,
                "image_output": False,
                "reasoning": 5,
                "speed": 3,
                "pricing": {
                    "input": 0.01,  # $0.01 per 1K tokens
                    "output": 0.03,  # $0.03 per 1K tokens
                    "image": 0.02  # $0.02 per image
                }
            },
            "gpt-4": {
                "text_input": True,
                "text_output": True,
                "image_input": False,
                "image_output": False,
                "reasoning": 5,
                "speed": 3,
                "pricing": {
                    "input": 0.01,  # $0.01 per 1K tokens
                    "output": 0.03  # $0.03 per 1K tokens
                }
            },
            "gpt-3.5-turbo": {
                "text_input": True,
                "text_output": True,
                "image_input": False,
                "image_output": False,
                "reasoning": 3,
                "speed": 5,
                "pricing": {
                    "input": 0.0005,  # $0.0005 per 1K tokens
                    "output": 0.0015  # $0.0015 per 1K tokens
                }
            },
            "dall-e-3": {
                "text_input": True,
                "text_output": False,
                "image_input": False,
                "image_output": True,
                "reasoning": 4,
                "speed": 2,
                "pricing": {
                    "input": 0.04,  # $0.04 per prompt
                    "output": 0.08  # $0.08 per image
                }
            },
            "dall-e-2": {
                "text_input": True,
                "text_output": False,
                "image_input": False,
                "image_output": True,
                "reasoning": 3,
                "speed": 4,
                "pricing": {
                    "input": 0.02,  # $0.02 per prompt
                    "output": 0.02  # $0.02 per image
                }
            }
        }
        
        # Format the response to include relevant information
        formatted_models = []
        for model in models:
            # Find matching capabilities by checking if model name starts with any known base model name
            matching_capabilities = None
            for base_model, capabilities in model_capabilities.items():
                if model.id.startswith(base_model):
                    matching_capabilities = capabilities
                    break
            
            if not matching_capabilities:
                matching_capabilities = {
                    "text_input": "Not found",
                    "text_output": "Not found",
                    "image_input": "Not found",
                    "image_output": "Not found",
                    "reasoning": "Not found",
                    "speed": "Not found",
                    "pricing": {
                        "input": "Not found",
                        "output": "Not found"
                    }
                }
            
            formatted_models.append({
                "id": model.id,
                "object": model.object,
                "created": model.created,
                "owned_by": model.owned_by,
                "capabilities": matching_capabilities,
                "metrics": {
                    "reasoning": matching_capabilities["reasoning"],
                    "speed": matching_capabilities["speed"]
                },
                "pricing": matching_capabilities["pricing"]
            })
        
        return formatted_models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 