import uvicorn
from .config import get_settings

def main():
    settings = get_settings()
    uvicorn.run("prodml.api.main:app", host="0.0.0.0", port=settings.PORT, reload=True)