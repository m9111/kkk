from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import json
import os
import logging
from dotenv import load_dotenv
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store the text state
class TextState:
    def __init__(self):
        self.current_text = "Hello, I am Ubix AI. Please feel free to ask me if you have any questions!"
        self.last_sent_text = None
        self.last_updated = datetime.datetime.now()

text_state = TextState()

class TTSRequest(BaseModel):
    input: dict
    voice: dict
    audioConfig: dict

class ReceiveText(BaseModel):
    text: str

async def get_access_token():
    """Get access token from API key"""
    api_key = os.getenv("GOOGLE_TTS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Google API key not configured")
    
    url = f"https://eu-texttospeech.googleapis.com/v1beta1/text:synthesize?key={api_key}"
    return url

@app.post("/receive")
async def receive_text(text_data: ReceiveText):
    """Receive text from external source and update current_text"""
    if text_data.text != text_state.current_text:
        text_state.current_text = text_data.text
        text_state.last_sent_text = None  # Reset last_sent_text to ensure new text will be sent
        text_state.last_updated = datetime.datetime.now()
        logger.info(f"Received new text: {text_state.current_text}")
        return {"status": "success", "text": text_state.current_text}
    return {"status": "skipped", "message": "Text unchanged"}

@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    try:
        # Get URL with API key
        url = await get_access_token()
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Prepare request payload
        payload = {
            "input": request.input,
            "voice": request.voice,
            "audioConfig": request.audioConfig
        }

        # Make request to Google TTS API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=30.0
            )
            
            # Check for errors
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Google TTS API error: {response.text}"
                )
            
            return response.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to Google TTS API timed out")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"HTTP error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/get_text")
async def get_text():
    """Get the current text to be spoken by the avatar"""
    # Check if there's new text to send
    if text_state.current_text == text_state.last_sent_text:
        raise HTTPException(
            status_code=204,  # No Content
            detail="No new text available"
        )
    
    # Return new text and update last_sent_text
    text_state.last_sent_text = text_state.current_text
    return {
        "text": text_state.current_text,
        "status": "new",
        "timestamp": text_state.last_updated.isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "current_text": text_state.current_text,
        "last_sent_text": text_state.last_sent_text,
        "last_updated": text_state.last_updated.isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or default to 8000
    port = int(os.getenv("PORT", 10000))
    
    logger.info(f"Starting server on port {port}...")
    uvicorn.run(
        "server4:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )