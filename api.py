import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import embed
import query

# Global state for the FAISS index
app_state = {
    "index": None,
    "chunks": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load index on startup if it exists
    print("[api] Starting up... checking for existing FAISS index.")
    try:
        app_state["index"], app_state["chunks"] = embed.load_index()
        print("[api] Successfully loaded FAISS index.")
    except FileNotFoundError:
        print("[api] Warning: No existing FAISS index found. Run ingest first.")
        app_state["index"] = None
        app_state["chunks"] = None
    
    yield
    
    # Clean up (if needed) at shutdown
    print("[api] Shutting down...")

app = FastAPI(title="VidyaVault API", lifespan=lifespan)

# Allow requests from the local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo purposes, allow all. Reduce in prod!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    message: str

@app.post("/api/query")
async def handle_query(req: QueryRequest):
    if app_state["index"] is None or app_state["chunks"] is None:
        return {
            "output": "The document index has not been built yet. Please ingest a document first.",
            "sources": []
        }
    
    try:
        result = query.answer(
            query=req.message,
            index=app_state["index"],
            chunks=app_state["chunks"],
            top_k=4
        )
        return {
            "output": result["answer"],
            "sources": [s[:80] + "..." for s in result["sources"]]
        }
    except Exception as e:
        print(f"[api] Error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest")
async def handle_ingest(file: UploadFile = File(...)):
    # Create the files/ directory if it doesn't exist
    os.makedirs("files", exist_ok=True)
    file_path = f"files/{file.filename}"
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Ingest the file to update the index
        print(f"[api] Ingesting newly uploaded file: {file_path}")
        index, chunks = embed.ingest(file_path)
        
        # Update app state
        app_state["index"] = index
        app_state["chunks"] = chunks
        
        return {
            "status": "success",
            "message": f"Successfully ingested {file.filename}. Index now has {index.ntotal} vectors."
        }
    except Exception as e:
        print(f"[api] Error during ingest: {e}")
        raise HTTPException(status_code=500, detail=str(e))
