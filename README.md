# AI Message API

A FastAPI-based backend service that allows you to create messages using OpenAI's API and retrieve previous LLM jobs.

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the API

Start the server with:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access the Swagger documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Available Endpoints

### POST /messages
Create a new message and get an AI response.

Request body:
```json
{
    "content": "Your message here",
    "model": "gpt-3.5-turbo"  // Optional, defaults to gpt-3.5-turbo
}
```

### GET /jobs
Retrieve all previous jobs.

### GET /jobs/{job_id}
Retrieve a specific job by ID.
