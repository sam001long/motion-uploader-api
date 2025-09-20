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

