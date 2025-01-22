import os
import toml
import httpx
import ssl
from PIL import Image
import imagehash
from io import BytesIO
import base64
import logging
from logging.config import dictConfig
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
from cachetools import LRUCache
from contextlib import asynccontextmanager

# Configure logging
dictConfig({
    "version": 1,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "default"}
    },
    "root": {"level": "INFO", "handlers": ["console"]}
})

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan with modern context manager"""
    try:
        # Initialize application state
        app.state.config = await read_config()
        app.state.cache = LRUCache(maxsize=config["cache"]["cache_size"])
        app.state.cache_lock = asyncio.Lock()
        yield
    except Exception as e:
        logger.critical(f"Startup failed: {str(e)}")
        raise RuntimeError("Application initialization failed")

app = FastAPI(lifespan=lifespan)

class ImageRequest(BaseModel):
    url: str

async def read_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.toml')
    try:
        with open(config_path, 'r') as f:
            config = toml.load(f)
            return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except toml.TomlDecodeError:
        logger.error(f"Invalid TOML in config file: {config_path}")
        raise
    except Exception as e:
        logger.error(f"Config read error: {str(e)}")
        raise

async def get_image_from_url(url: str):
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    ssl_context.set_ciphers("AES128-GCM-SHA256")
    ssl_context.set_alpn_protocols(["http/1.1"])
    
    try:
        async with httpx.AsyncClient(verify=ssl_context, timeout=10) as client:
            response = await client.get(url)
            response.raise_for_status()
            return {
                "type": response.headers.get("content-type", "application/octet-stream"),
                "data": response.content
            }
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error {e.response.status_code} fetching image: {e.response.text}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, 
                          detail=f"Image fetch failed: {e.response.status_code}")
    except Exception as e:
        logger.error(f"Error fetching image: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, 
                          detail="Failed to download image")

async def generate_image_hash(image_data: bytes, hash_size: int):
    try:
        loop = asyncio.get_running_loop()
        image = await loop.run_in_executor(None, Image.open, BytesIO(image_data))
        image_hash = await loop.run_in_executor(None, imagehash.average_hash, image, hash_size)
        return str(image_hash)
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                          detail="Invalid image format")

def generate_data_uri(image_data: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_data).decode('utf-8')
    return f"data:{mime_type};base64,{encoded}"

def build_prompt(system_prompt: str, data_uri: str) -> list:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": data_uri}}]
        }
    ]

async def request_llm(url: str, api_key: str, prompt: list, request_body: dict):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                url,
                json={"messages": prompt, **request_body},
                headers=headers
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        logger.error(f"LLM API error {e.response.status_code}: {e.response.text}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                          detail="AI service unavailable")
    except Exception as e:
        logger.error(f"LLM request failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail="AI processing failed")

@app.post("/")
async def analyze_image(request: Request, image_req: ImageRequest):
    """Analyze image endpoint"""
    config = request.app.state.config
    cache = request.app.state.cache
    cache_lock = request.app.state.cache_lock
    
    try:
        # Fetch and validate image
        image = await get_image_from_url(image_req.url)
        image_hash = await generate_image_hash(image["data"], config["cache"]["hash_size"])
        
        # Cache check
        async with cache_lock:
            if image_hash in cache:
                return JSONResponse(content={"data": cache[image_hash]})
        
        # Process image
        data_uri = generate_data_uri(image["data"], image["type"])
        prompt = build_prompt(config["api"]["system_prompt"], data_uri)
        
        # Call LLM API
        result = await request_llm(
            config["api"]["url"],
            config["api"]["key"],
            prompt,
            config["api"]["request_body"]
        )
        
        # Update cache
        async with cache_lock:
            cache[image_hash] = result
        
        return JSONResponse(content={"data": result})
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    try:
        # Read config directly for server settings
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, 'config.toml')) as f:
            config = toml.load(f)
        host, port = config["server"]["listen_address"].split(":")
        uvicorn.run(app, host=host, port=int(port))
    except Exception as e:
        logger.critical(f"Server startup failed: {str(e)}")
        exit(1)