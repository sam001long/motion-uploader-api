from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Optional
import os, io
import pandas as pd
from sqlalchemy.sql import text
from dotenv import load_dotenv
from pipeline import ingest_file_bytes, engine

load_dotenv()
API_KEY = os.getenv("API_KEY", "").strip()
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

app = FastAPI(title="Motion Uploader API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def require_key(x_api_key: Optional[str]):
    if API_KEY and (x_api_key or "") != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/api/health")
async def health():
    return {"ok": True}

@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...), x_api_key: Optional[str] = Header(default=None)):
    require_key(x_api_key)
    results = []
    for uf in files:
        data = await uf.read()
        res = ingest_file_bytes(uf.filename, data)
        results.append(res)
    return {"results": results}

# ---- 匯出端點：下載整理後 CSV ----

def _df_to_csv_response(df: pd.DataFrame, filename: str) -> StreamingResponse:
    bio = io.StringIO()
    df.to_csv(bio, index=False)
    bio.seek(0)
    return StreamingResponse(
        iter([bio.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

@app.get("/api/export/measurements.csv")
async def export_measurements(
    x_api_key: Optional[str] = Header(default=None),
    limit: Optional[int] = Query(None, ge=1, le=500000),
    since: Optional[float] = Query(None),
    session_id: Optional[str] = Query(None)
):
    require_key(x_api_key)
    base_sql = "SELECT timestamp,joint,metric,value,unit,confidence,session_id,subject_id,activity FROM measurements"
    conds = []
    params = {}
    if since is not None:
        conds.append("timestamp >= :since")
        params["since"] = since
    if session_id:
        conds.append("session_id = :sid")
        params["sid"] = session_id
    if conds:
        base_sql += " WHERE " + " AND ".join(conds)
    base_sql += " ORDER BY timestamp ASC"
    if limit:
        base_sql += " LIMIT :lim"
        params["lim"] = int(limit)
    with engine.connect() as conn:
        df = pd.read_sql_query(text(base_sql), conn, params=params)
    return _df_to_csv_response(df, "measurements.csv")

@app.get("/api/export/sources.csv")
async def export_sources(x_api_key: Optional[str] = Header(default=None)):
    require_key(x_api_key)
    sql = "SELECT id,provider,filename,file_hash,status,message,created_at FROM source_files ORDER BY id DESC"
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn)
    return _df_to_csv_response(df, "sources.csv")

# 本機： uvicorn server:app --reload --port 8787
