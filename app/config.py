import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    # OpenRouter Free Models
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "google/gemini-1.5-flash-exp:free")
    ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    
    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))

config = Config()