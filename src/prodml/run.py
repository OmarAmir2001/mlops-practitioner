import uvicorn
import mlflow
from prodml.config import get_settings

def main():
    settings = get_settings()
    uvicorn.run("prodml.api.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
    
if __name__ == "__main__":
    main()