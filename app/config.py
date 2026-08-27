import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Qdrant in-memory fallback for standalone Docker runs without host resolution
# Force default QDRANT_HOST to "memory" if host is localhost/empty to avoid DNS errors in local Docker
qdrant_host = os.getenv("QDRANT_HOST", "memory")
if qdrant_host.strip() in ["localhost", "127.0.0.1", ""]:
    qdrant_host = "memory"
qdrant_port = int(os.getenv("QDRANT_PORT", "6333")) if os.getenv("QDRANT_PORT") else 6333

class Config:
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    # OpenRouter Free Models
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "google/gemini-1.5-flash-exp:free")
    ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    
    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))

    # Qdrant Settings - hardcoded memory fallback for standalone docker runs
    QDRANT_HOST: str = qdrant_host
    QDRANT_PORT: int = qdrant_port

config = Config()