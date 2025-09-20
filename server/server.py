from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os
from dotenv import load_dotenv
from pipeline import ingest_file_bytes

load_dotenv()
API_KEY = os.getenv("API_KEY", "").strip()
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

app = FastAPI(title="Motion Uploader API")

# CORS（預設 *，可用環境變數限制）
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/api/health")
async def health():
    return {"ok": True}

@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...), x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and (x_api_key or "") != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    results = []
    for uf in files:
        data = await uf.read()
        res = ingest_file_bytes(uf.filename, data)
        results.append(res)
    return {"results": results}

# 本機開發： python -m uvicorn server:app --reload --port 8787
