
<!--
Motion Data Uploader — 前端＋後端（FastAPI）最小可用專案

專案結構建議：
- web/index.html           ← 前端：大量上傳頁面（可放 GitHub P<!-- =============================
= FILE: server/server.py
============================== -->
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
ages）
- server/server.py         ← 後端 API（/api/upload）
- server/pipeline.py       ← 後端資料清洗與入庫（含資料庫 Schema）
- server/ingest_config.yaml← 規則設定（遇到新格式只要新增規則）
- server/requirements.txt  ← 後端套件
- server/.env.example      ← 後端環境變數範例（DB、API_KEY）
- server/Dockerfile        ← 部署用

使用方式（開發測試）：
1) 在本機啟動後端：
   cd server
   cp .env.example .env   # 視需要修改
   pip install -r requirements.txt
   uvicorn server:app --reload --port 8787

2) 打開前端：
   直接用瀏覽器開 web/index.html（或推到 GitHub Pages）。
   若後端網址不同，請在 index.html 上方把 API_BASE 換成你的後端網址。

部署建議：
- 前端：放在 GitHub Pages。
- 後端：Render/Railway/Fly.io 皆可免管機器部署；或 Docker 任意雲主機。
-->

<!-- =============================
= FILE: web/index.html
============================== -->
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Motion Data Uploader</title>
  <style>
    :root { --bg:#0b1020; --card:#11162a; --muted:#8fa3c7; --accent:#7cc7ff; }
    body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, "Noto Sans TC", sans-serif; background:var(--bg); color:#e6eefc; }
    .wrap { max-width:980px; margin:40px auto; padding:0 16px; }
    .card { background:var(--card); border:1px solid #24304f; border-radius:18px; padding:20px; box-shadow:0 10px 30px rgba(0,0,0,.25); }
    h1 { margin:0 0 8px; font-size:28px; letter-spacing:.3px }
    p.muted { color: var(--muted); margin-top:6px }
    .row { display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
    .drop { flex:1; min-height:140px; border:2px dashed #2e3d64; border-radius:16px; display:flex; align-items:center; justify-content:center; text-align:center; padding:18px; cursor:pointer; transition:.2s border-color,.2s background; }
    .drop.hover { border-color: var(--accent); background: rgba(124,199,255,.06); }
    .btn { background:linear-gradient(180deg,#1e3a8a,#0f245c); color:white; border:1px solid #224185; padding:10px 14px; border-radius:10px; cursor:pointer; font-weight:600; }
    .btn:disabled { opacity:.5; cursor:not-allowed }
    .grid { margin-top:16px; display:grid; grid-template-columns: 1fr; gap:10px; }
    .item { background:#0b1329; border:1px solid #1f2a4a; border-radius:12px; padding:10px 12px; }
    .item .name { font-weight:600 }
    .item .bar { height:8px; background:#1c2544; border-radius:6px; margin-top:8px; overflow:hidden }
    .item .bar > i { display:block; height:100%; width:0%; background: var(--accent); transition: width .3s }
    .small { font-size:12px; color: var(--muted) }
    .ok { color:#9fffb2 }
    .err { color:#ff9f9f }
    .cfg { margin-top:10px; }
    .cfg input { background:#091126; color:#e6eefc; border:1px solid #223058; border-radius:8px; padding:8px 10px; width:100% }
    .footer { margin-top:16px; color: var(--muted); font-size:12px }
    a { color: var(--accent); text-decoration: none }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Motion Data Uploader</h1>
      <p class="muted">大量上傳 CSV / XLSX / JSON，後端自動清洗與入庫。遇到新格式只要在伺服器 <code>ingest_config.yaml</code> 加規則即可。</p>

      <div class="cfg">
        <label class="small">API 位址（通常是你的後端，例如 https://your-app.onrender.com）</label>
        <input id="apiBase" placeholder="https://localhost:8787" />
      </div>

      <div class="row" style="margin-top:14px">
        <div id="drop" class="drop">
          <div>
            <div style="font-weight:700; font-size:16px">拖放檔案到這裡，或點擊選擇</div>
            <div class="small">支援 .csv、.tsv、.xlsx、.xls、.json（可多選）</div>
          </div>
          <input id="file" type="file" multiple style="display:none" accept=".csv,.tsv,.xlsx,.xls,.json" />
        </div>
        <button id="send" class="btn" disabled>上傳並匯入</button>
      </div>

      <div id="list" class="grid"></div>

      <div class="footer">安全性：預設開放測試。上線前請在伺服器設定 <code>API_KEY</code> 後，於此頁加入自訂標頭（程式碼已預留）。</div>
    </div>
  </div>

  <script>
    // === 可調整：預設 API（初次載入會寫到輸入框） ===
    const DEFAULT_API_BASE = "http://localhost:8787"; // 本機開發時用；上線請改成你的後端網址
    const apiBaseInput = document.getElementById('apiBase');
    apiBaseInput.value = localStorage.getItem('apiBase') || DEFAULT_API_BASE;
    apiBaseInput.addEventListener('change', () => localStorage.setItem('apiBase', apiBaseInput.value.trim()));

    const drop = document.getElementById('drop');
    const file = document.getElementById('file');
    const send = document.getElementById('send');
    const list = document.getElementById('list');

    let files = [];

    const renderList = () => {
      list.innerHTML = '';
      files.forEach(f => {
        const el = document.createElement('div');
        el.className = 'item';
        el.innerHTML = `
          <div class="name">${f.name} <span class="small">(${(f.size/1024).toFixed(1)} KB)</span></div>
          <div class="bar"><i style="width:${f._p||0}%"></i></div>
          <div class="small" id="m_${f._id}">待上傳</div>
        `;
        list.appendChild(el);
      });
      send.disabled = files.length === 0;
    };

    const pick = () => file.click();
    drop.addEventListener('click', pick);
    drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('hover'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('hover'));
    drop.addEventListener('drop', e => {
      e.preventDefault(); drop.classList.remove('hover');
      files = [...files, ...Array.from(e.dataTransfer.files)];
      files = files.map((f,i)=>{ f._id = crypto.randomUUID(); f._p = 0; return f; });
      renderList();
    });
    file.addEventListener('change', e => {
      files = [...files, ...Array.from(e.target.files)].map(f=>{ f._id=crypto.randomUUID(); f._p=0; return f; });
      renderList();
    });

    async function upload() {
      send.disabled = true;
      const api = (apiBaseInput.value || DEFAULT_API_BASE).replace(/\/$/,'');

      // 一次性多檔上傳
      const fd = new FormData();
      files.forEach(f => fd.append('files', f, f.name));

      try {
        const res = await fetch(api + '/api/upload', {
          method: 'POST', body: fd,
          // 上線時若伺服器要求 API_KEY，解開下方設定：
          // headers: { 'X-API-Key': 'YOUR_API_KEY_HERE' },
        });
        const data = await res.json();

        // 顯示結果
        (data.results || []).forEach(r => {
          const target = files.find(f => f.name === r.filename);
          if (target) {
            target._p = 100;
            const m = document.getElementById('m_' + target._id);
            if (r.status === 'ingested') {
              m.innerHTML = `<span class=ok>已匯入</span>：新增 <b>${r.rows_added}</b> 列`;
            } else if (r.status === 'skipped') {
              m.innerHTML = `<span class=small>略過</span>：${r.message || '重複檔案'}`;
            } else {
              m.innerHTML = `<span class=err>失敗</span>：${r.message || ''}`;
            }
          }
        });
      } catch (e) {
        alert('上傳失敗：' + e.message);
      } finally {
        // 更新進度條
        files = files.map(f => (f._p=100, f));
        renderList();
      }
    }

    send.addEventListener('click', upload);
  </script>
</body>
</html>


<!-- =============================
= FILE: server/requirements.txt
============================== -->
# 後端（FastAPI + 清洗與入庫）
fastapi
uvicorn
pandas
SQLAlchemy
pydantic
pyyaml
python-dotenv
openpyxl


<!-- =============================
= FILE: server/.env.example
============================== -->
# 預設用 SQLite，一個檔案就能跑；要換 Postgres 請覆蓋 DATABASE_URL。
DATABASE_URL=sqlite:///motion.sqlite
# 可選：設定後伺服器會要求前端帶 X-API-Key
API_KEY=
# 允許的 CORS 來源（逗號分隔；留空則為 * ）
CORS_ORIGINS=


<!-- =============================
= FILE: server/ingest_config.yaml
============================== -->
canonical:
  required: [timestamp, joint, metric, value]
  defaults:
    unit: deg
    confidence: 1.0

providers:
  provider_a:
    detect_any_header: ["time", "joint", "angle_deg"]
    rename:
      time: timestamp
      joint: joint
      angle_deg: value
    set:
      metric: angle
      unit: deg

  provider_b:
    detect_any_header: ["frame", "joint_name", "theta"]
    rename:
      frame: frame
      joint_name: joint
      theta: value
    set:
      metric: angle
      unit: rad
    derived:
      timestamp: "frame / 30.0"

  wide_example:
    detect_any_header: ["hip_deg", "knee_deg"]
    wide_to_long:
      id_vars: ["time"]
      var_name: "feature"
      value_name: "value"
    rename:
      time: timestamp
    parse_feature:
      pattern: "(?P<joint>\\w+?)_(?P<unit>deg|rad)"
      set:
        metric: angle


<!-- =============================
= FILE: server/pipeline.py
============================== -->
from __future__ import annotations
import os, hashlib, json, math, io
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
from pydantic import BaseModel, validator
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
import yaml
from dotenv import load_dotenv

# --- DB 連線設定 ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///motion.sqlite")
engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
Base = declarative_base()

class SourceFile(Base):
    __tablename__ = "source_files"
    id = Column(Integer, primary_key=True)
    provider = Column(String, index=True)
    filename = Column(String, nullable=False)
    file_hash = Column(String, unique=True, index=True)
    status = Column(String, default="ingested")  # ingested/failed/skipped
    message = Column(Text)
    meta = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())
    measurements = relationship("Measurement", back_populates="source")

class Measurement(Base):
    __tablename__ = "measurements"
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("source_files.id"), index=True)
    session_id = Column(String, index=True, nullable=True)
    subject_id = Column(String, nullable=True)
    activity = Column(String, nullable=True)
    timestamp = Column(Float, index=True)
    joint = Column(String, index=True)
    metric = Column(String, index=True)
    value = Column(Float)
    unit = Column(String, default="deg")
    confidence = Column(Float, default=1.0)
    source = relationship("SourceFile", back_populates="measurements")

Base.metadata.create_all(engine)

class CanonicalRow(BaseModel):
    timestamp: float
    joint: str
    metric: str
    value: float
    unit: Optional[str] = "deg"
    confidence: Optional[float] = 1.0
    session_id: Optional[str] = None
    subject_id: Optional[str] = None
    activity: Optional[str] = None

    @validator("timestamp", pre=True)
    def to_float(cls, v):
        return float(v)

# --- 設定檔 ---
CFG_CACHE: Optional[Dict[str, Any]] = None

def get_cfg(path: str = "ingest_config.yaml") -> Dict[str, Any]:
    global CFG_CACHE
    if CFG_CACHE is None:
        with open(path, "r", encoding="utf-8") as f:
            CFG_CACHE = yaml.safe_load(f)
    return CFG_CACHE

# --- 工具 ---

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def detect_provider(df: pd.DataFrame, cfg) -> Optional[str]:
    headers = set(map(str, df.columns))
    for name, rule in cfg.get("providers", {}).items():
        keys = set(map(str, rule.get("detect_any_header", [])))
        if headers & keys:
            return name
    return None


def wide_to_long(df: pd.DataFrame, rule: Dict[str, Any]) -> pd.DataFrame:
    id_vars = rule["wide_to_long"]["id_vars"]
    var_name = rule["wide_to_long"]["var_name"]
    value_name = rule["wide_to_long"]["value_name"]
    long_df = pd.melt(df, id_vars=id_vars, var_name=var_name, value_name=value_name)
    if "parse_feature" in rule:
        pat = rule["parse_feature"]["pattern"]
        extracted = long_df[var_name].str.extract(pat, expand=True)
        long_df = pd.concat([long_df, extracted], axis=1)
        for k, v in (rule["parse_feature"].get("set") or {}).items():
            long_df[k] = v
    return long_df


def normalize_df(df: pd.DataFrame, provider: str, cfg) -> pd.DataFrame:
    rule = cfg["providers"][provider]

    if "wide_to_long" in rule:
        df = wide_to_long(df, rule)

    rename = rule.get("rename", {})
    df = df.rename(columns=rename)

    for k, v in (rule.get("set") or {}).items():
        df[k] = v

    if "derived" in rule:
        for new_col, expr in rule["derived"].items():
            safe_locals = {c: df[c] for c in df.columns if c in expr}
            df[new_col] = eval(expr, {"__builtins__": {}}, safe_locals)

    for k, v in (get_cfg()["canonical"].get("defaults") or {}).items():
        if k not in df.columns:
            df[k] = v

    missing = [c for c in get_cfg()["canonical"]["required"] if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要欄位: {missing}")

    keep = list(set(get_cfg()["canonical"]["required"] + list((get_cfg()["canonical"].get("defaults") or {}).keys()) + ["session_id", "subject_id", "activity"]))
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def read_any_bytes(filename: str, data: bytes) -> List[pd.DataFrame]:
    ext = Path(filename).suffix.lower()
    bio = io.BytesIO(data)
    if ext in [".csv", ".tsv"]:
        sep = "," if ext == ".csv" else "\t"
        return [pd.read_csv(bio, sep=sep)]
    if ext in [".xlsx", ".xls"]:
        xls = pd.ExcelFile(bio)
        return [xls.parse(sheet_name) for sheet_name in xls.sheet_names]
    if ext in [".json"]:
        text = data.decode("utf-8")
        obj = json.loads(text)
        if isinstance(obj, list):
            return [pd.DataFrame(obj)]
        elif isinstance(obj, dict) and "data" in obj:
            return [pd.DataFrame(obj["data"])]
        else:
            return [pd.json_normalize(obj)]
    raise ValueError(f"不支援的格式: {ext}")


# --- 對外：單檔匯入（bytes） ---

def ingest_file_bytes(filename: str, data: bytes) -> Dict[str, Any]:
    session = SessionLocal()
    try:
        file_hash = sha256_bytes(data)
        # 防重：若已處理過同樣內容，直接略過
        existed = session.query(SourceFile).filter_by(file_hash=file_hash).first()
        if existed:
            return {"status": "skipped", "filename": filename, "message": "duplicate file", "rows_added": 0}

        dfs = read_any_bytes(filename, data)
        cfg = get_cfg()

        all_rows = []
        provider_used = None
        for df in dfs:
            provider = detect_provider(df, cfg)
            if provider is None:
                raise ValueError("無法偵測提供者（請在 ingest_config.yaml 加規則）")
            provider_used = provider
            norm = normalize_df(df, provider, cfg)
            for _, row in norm.iterrows():
                all_rows.append(CanonicalRow(**row.to_dict()).dict())

        sf = SourceFile(provider=provider_used or "unknown", filename=filename, file_hash=file_hash, status="ingested", meta={"size": len(data)})
        session.add(sf)
        session.flush()

        objs = [Measurement(source_id=sf.id, **r) for r in all_rows]
        session.bulk_save_objects(objs)
        session.commit()

        return {"status": "ingested", "filename": filename, "rows_added": len(objs)}

    except Exception as e:
        session.rollback()
        # 仍記錄來源與錯誤（以 hash 做唯一性）
        try:
            sf = SourceFile(provider="unknown", filename=filename, file_hash=sha256_bytes(data), status="failed", message=str(e))
            session.add(sf)
            session.commit()
        except Exception:
            session.rollback()
        return {"status": "failed", "filename": filename, "message": str(e), "rows_added": 0}
    finally:
        session.close()


<!-- =============================
= FILE: server/server.py
============================== -->
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


<!-- =============================
= FILE: server/Dockerfile
============================== -->
# 簡單 Docker 部署：
#   docker build -t motion-uploader .
#   docker run -it --rm -p 8787:8787 --env-file .env -v $PWD:/app motion-uploader
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8787
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8787"]
