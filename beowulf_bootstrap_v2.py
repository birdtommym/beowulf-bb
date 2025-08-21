# beowulf_bootstrap_v2.py
# Writes a complete Beowulf Biostatistics app (incl. Video→Spreadsheet) into ./beowulf_app + deps.

from pathlib import Path
import textwrap

root = Path.cwd()
app = root / "beowulf_app"
utils = app / "utils"
pages = app / "pages"
stcfg = app / ".streamlit"
for p in [app, utils, pages, stcfg]:
    p.mkdir(parents=True, exist_ok=True)

def w(path: Path, s: str): path.write_text(textwrap.dedent(s).lstrip("\n"), encoding="utf-8")

# -------- requirements --------
w(root/"requirements.txt", """
streamlit>=1.36
pandas>=2.2
numpy>=1.26
pyarrow>=16.1
scipy>=1.11
statsmodels>=0.14
pyreadstat>=1.2
openpyxl>=3.1
Pillow>=10.0
pytesseract>=0.3.10
opencv-python-headless>=4.10
rapidfuzz>=3.6
SQLAlchemy>=2.0
""")

# -------- app shell --------
w(app/"app.py", """
import os, streamlit as st
from beowulf_app.state import ensure_state
from beowulf_app.config import UPLOAD_MB, IS_MOBILE, profile_badge
from beowulf_app.ui_theme import inject_ios_theme

st.set_page_config(page_title="Beowulf Biostatistics", layout="wide")
st.set_option("server.maxUploadSize", UPLOAD_MB)
inject_ios_theme()

def gate():
    code = os.getenv("BB_PASSCODE", "").strip()
    if not code: return True
    if st.session_state.get("_auth_ok"): return True
    st.title("🔒 Beowulf Beta")
    st.text_input("Access code", type="password", key="_access_code")
    if st.button("Enter"):
        if st.session_state.get("_access_code") == code:
            st.session_state["_auth_ok"] = True; st.rerun()
        else: st.error("Incorrect code.")
    st.stop()

gate()
ensure_state()
st.title("Beowulf Biostatistics")
st.caption("Grendel · Hrunting · Naegling · Wiglaf · Wealhtheow")

st.sidebar.header("Navigation")
st.sidebar.markdown(profile_badge())
st.sidebar.info("Mobile profile (1 GB cap)" if IS_MOBILE else "Desktop profile (32 GB cap)")
st.write("Use the **Pages** menu (top-left) to open each suite.")
""")

w(app/"state.py", """
import streamlit as st
def ensure_state():
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("df_name", None)
    st.session_state.setdefault("recipe", [])
def set_df(df, name="dataset", record_original=False):
    st.session_state.df, st.session_state.df_name = df, name
    if record_original and "original_df" not in st.session_state:
        st.session_state.original_df = df.copy()
def add_recipe_step(step: dict): st.session_state.recipe.append(step)
""")

w(app/"config.py", """
import os
PROFILE = os.environ.get("BEOWULF_PROFILE", "mobile").strip().lower()  # default mobile feel
IS_MOBILE = PROFILE == "mobile"
UPLOAD_MB = 1024 if IS_MOBILE else 32768
def profile_badge() -> str:
    cap = f"{UPLOAD_MB:,} MB"
    return f"**Profile:** {PROFILE.title()} · **Upload cap:** {cap}"
""")

w(app/"ui_theme.py", """
import streamlit as st
def inject_ios_theme():
    st.markdown(\"\"\"
    <style>
    :root { --bb:#10b981; --fg:#111827; --muted:#6b7280; }
    html, body, [class^="main"] { font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue",Arial,sans-serif; }
    h1,h2 { letter-spacing:-.01em; font-weight:800; }
    .stButton>button { border-radius:14px; padding:10px 16px; background:var(--bb); color:#fff; border:none; }
    .stTabs [data-baseweb="tab"]{ border-radius:12px; background:#f4f6f8; padding:10px 12px;}
    .stTabs [aria-selected="true"]{ background:#e9fbf4; border:1px solid var(--bb); color:#0b7e5e;}
    footer{display:none !important;}
    </style>
    \"\"\", unsafe_allow_html=True)
""")

# -------- utils --------
w(utils/"io_fs.py", """
from pathlib import Path
import pandas as pd
def read_local(path: str, columns=None) -> pd.DataFrame:
    p = Path(path); s = p.suffix.lower()
    if s in {".parquet",".pq"}: return pd.read_parquet(p, columns=columns)
    if s in {".xlsx",".xls"}:   return pd.read_excel(p)
    return pd.read_csv(p)
""")

w(utils/"readstat_io.py", """
from __future__ import annotations
import io, pandas as pd, pyreadstat
from typing import Tuple, Optional
def read_with_labels(file_or_path, file_name: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
    name = (file_name or getattr(file_or_path,"name","") or str(file_or_path)).lower()
    ext = ".sav" if name.endswith(".sav") else ".dta" if name.endswith(".dta") else ".sas7bdat" if name.endswith(".sas7bdat") else None
    if not ext: raise ValueError("Unsupported labeled format (.sav/.dta/.sas7bdat)")
    data = file_or_path
    df, meta = (pyreadstat.read_sav if ext==".sav" else pyreadstat.read_dta if ext==".dta" else pyreadstat.read_sas7bdat)(data, apply_value_formats=False)
    return df, {"variable_value_labels": getattr(meta,"variable_value_labels",{}), "variable_labels": getattr(meta,"column_names_to_labels",{})}
""")

w(utils/"readers.py", """
from __future__ import annotations
import io, pandas as pd
from typing import Optional
from .readstat_io import read_with_labels
def _bytes_like(obj) -> bool:
    return hasattr(obj,"read") or isinstance(obj,(bytes,bytearray,io.BytesIO))
def read_any(file_or_path, name: Optional[str]=None, columns=None, nrows=None) -> pd.DataFrame:
    name = (name or getattr(file_or_path,"name","") or str(file_or_path)).lower()
    if name.endswith((".sav",".dta",".sas7bdat")):
        df,_ = read_with_labels(file_or_path, name)
        if columns: df=df[columns]
        if nrows: df=df.head(nrows)
        return df
    if name.endswith((".xlsx",".xls")):
        bio = io.BytesIO(file_or_path.read()) if _bytes_like(file_or_path) else None
        return pd.read_excel(bio or file_or_path, nrows=nrows, usecols=columns)
    if name.endswith((".parquet",".pq")):
        bio = io.BytesIO(file_or_path.read()) if _bytes_like(file_or_path) else None
        return pd.read_parquet(bio or file_or_path, columns=columns)
    if name.endswith(".csv"):
        bio = io.TextIOWrapper(io.BytesIO(file_or_path.read()), encoding="utf-8", errors="ignore") if _bytes_like(file_or_path) else None
        if nrows: return pd.read_csv(bio or file_or_path, nrows=nrows, usecols=columns)
        it = pd.read_csv(bio or file_or_path, chunksize=200_000, usecols=columns)
        return next(it)
    raise ValueError(f"Unsupported file type: {name}")
""")

w(utils/"ocr_table.py", """
from __future__ import annotations
import io, re, pandas as pd
from PIL import Image, ImageOps, ImageFilter
import pytesseract
def ocr_available() -> bool:
    try: pytesseract.get_tesseract_version(); return True
    except Exception: return False
def _pre(img):
    g = ImageOps.grayscale(img); g = g.filter(ImageFilter.MedianFilter(size=3)); return ImageOps.autocontrast(g)
def _split(text: str) -> pd.DataFrame:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines: return pd.DataFrame()
    splitter = "\\t+" if any("\\t" in ln for ln in lines) else "\\s*,\\s*" if sum(ln.count(",") for ln in lines)>=len(lines) else "\\s{2,}"
    rows = [re.split(splitter, ln) for ln in lines]; n = max(len(r) for r in rows)
    rows = [r+['']*(n-len(r)) for r in rows]; header = rows[0]; return pd.DataFrame(rows[1:], columns=header)
def image_to_dataframe(file_or_bytes) -> pd.DataFrame:
    if hasattr(file_or_bytes,"read"): b=file_or_bytes.read()
    elif isinstance(file_or_bytes,(bytes,bytearray)): b=bytes(file_or_bytes)
    else: b=open(file_or_bytes,"rb").read()
    img = Image.open(io.BytesIO(b)).convert("RGB"); img = _pre(img)
    text = pytesseract.image_to_string(img, config="--psm 6 -c preserve_interword_spaces=1")
    return _split(text)
""")

w(utils/"video_to_table.py", """
from __future__ import annotations
import os, io, tempfile
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import cv2, numpy as np, pandas as pd
from PIL import Image, ImageOps, ImageFilter
import pytesseract
from rapidfuzz import fuzz

def _ensure_tesseract():
    cmd = os.environ.get("TESSERACT_CMD")
    if cmd: pytesseract.pytesseract.tesseract_cmd = cmd

def _pre(img: Image.Image, target_w=1600) -> Image.Image:
    w,h = img.size
    if w < target_w: img = img.resize((target_w, int(h*target_w/w)))
    g = ImageOps.grayscale(img)
    g = g.filter(ImageFilter.MedianFilter(size=3))
    return ImageOps.autocontrast(g)

def _split_lines(text: str) -> List[List[str]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines: return []
    if any("\\t" in ln for ln in lines): split=lambda s:s.split("\\t")
    elif sum(ln.count(",") for ln in lines) >= len(lines): split=lambda s:s.split(",")
    else:
        import re; rx=re.compile(r"\\s{2,}"); split=lambda s:[c for c in rx.split(s) if c]
    rows=[ [c.strip() for c in split(ln)] for ln in lines ]
    n=max((len(r) for r in rows), default=0)
    return [ r+['']*(n-len(r)) for r in rows ]

def _norm(s: str) -> str:
    import re, string
    s=s.lower().translate(str.maketrans("","",string.punctuation))
    return re.sub(r"\\s+"," ",s).strip()

@dataclass
class RowHit:
    t: float; raw: str; conf: float; cells: List[str]

def auto_detect_roi(pil: Image.Image) -> Optional[Tuple[int,int,int,int]]:
    img = np.array(pil.convert("L"))
    thr = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,15)
    hsize=max(10,img.shape[1]//30); vsize=max(10,img.shape[0]//30)
    hor=cv2.morphologyEx(thr,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(hsize,1)))
    ver=cv2.morphologyEx(thr,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,vsize)))
    grid=cv2.add(hor,ver)
    cnts,_=cv2.findContours(grid,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    x,y,w,h=cv2.boundingRect(max(cnts,key=cv2.contourArea))
    if w*h < img.shape[0]*img.shape[1]*0.05: return None
    return int(x),int(y),int(w),int(h)

@dataclass
class VideoOCRResult:
    df: pd.DataFrame; header: List[str]; meta: Dict[str, float]

def _ocr_data(pil: Image.Image):
    data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DATAFRAME, config="--psm 6")
    data = data.dropna(subset=["text"])
    txt  = pytesseract.image_to_string(pil, config="--psm 6 -c preserve_interword_spaces=1")
    return txt, data

def extract_table_from_video_bytes(file_like, fps_sample=2, roi=None, max_seconds=None, progress_cb=None) -> VideoOCRResult:
    _ensure_tesseract()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        b = file_like.read() if hasattr(file_like,"read") else file_like
        tmp.write(b if isinstance(b,(bytes,bytearray)) else bytes(b))
        path = tmp.name
    try:
        return extract_table_from_video_file(path, fps_sample=fps_sample, roi=roi, max_seconds=max_seconds, progress_cb=progress_cb)
    finally:
        try: os.remove(path)
        except Exception: pass

def extract_table_from_video_file(path, fps_sample=2, roi=None, max_seconds=None, progress_cb=None) -> VideoOCRResult:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): raise RuntimeError("Cannot open video.")
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur     = total / fps_src if total else None
    step    = max(int(round(fps_src / float(fps_sample))), 1)

    rows_by_key: Dict[str,List[RowHit]] = {}
    header_candidates: Dict[str,int] = {}
    frame_idx=0; processed=0
    determined_roi=roi; did_roi=False

    while True:
        ok = cap.grab()
        if not ok: break
        if frame_idx % step != 0: frame_idx+=1; continue
        ok, frame = cap.retrieve()
        if not ok: break
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not did_roi:
            did_roi=True
            if determined_roi is None:
                try: determined_roi = auto_detect_roi(pil)
                except Exception: determined_roi=None
        if determined_roi:
            x,y,w,h = determined_roi; pil = pil.crop((x,y,x+w,y+h))
        pil = _pre(pil)
        txt, data = _ocr_data(pil)
        confs = []
        if not data.empty and "conf" in data:
            confs = data.groupby(["page_num","block_num","par_num","line_num"])["conf"].mean().tolist()
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        cells = _split_lines(txt)
        t = frame_idx / fps_src
        for i, raw in enumerate(lines):
            rowcells = cells[i] if i < len(cells) else [raw]
            key = _norm(raw)
            if key not in rows_by_key:
                best=None; score=0
                for k in rows_by_key.keys():
                    from rapidfuzz import fuzz
                    s=fuzz.partial_ratio(key,k)
                    if s>score: best,score=k,s
                if best and score>=92: key=best
            rows_by_key.setdefault(key,[]).append(RowHit(t=t, raw=raw, conf=float(confs[i]) if i < len(confs) else 70.0, cells=rowcells))
            if frame_idx < step*4: header_candidates[key] = header_candidates.get(key,0)+1
        processed += 1; frame_idx += 1
        if progress_cb and dur: progress_cb(min(frame_idx/(dur*fps_src),1.0))
        if max_seconds and processed >= int(max_seconds * max(1,fps_src/step)): break

    cap.release()
    if not rows_by_key:
        return VideoOCRResult(pd.DataFrame(), [], {"frames_used":processed,"duration":float(dur or 0)})

    import re
    def alpha_ratio(s): return len(re.findall(r"[A-Za-z]", s)) / max(len(s),1)
    header_key = max(header_candidates.keys(), key=lambda k:(header_candidates[k], alpha_ratio(k))) if header_candidates else None
    header_cells = []
    if header_key and rows_by_key[header_key]:
        best_hdr = max(rows_by_key[header_key], key=lambda r:(r.conf, len(r.raw)))
        header_cells = best_hdr.cells

    rows=[]
    for k,hits in rows_by_key.items():
        if k==header_key: continue
        rows.append(max(hits, key=lambda r:(r.conf,len(r.raw))))
    rows.sort(key=lambda r:r.t)

    if header_cells:
        n=len(header_cells)
        for r in rows:
            if len(r.cells)<n: r.cells += [""]*(n-len(r.cells))
            elif len(r.cells)>n: r.cells = r.cells[:n]
        df = pd.DataFrame([r.cells for r in rows], columns=header_cells)
    else:
        n=max((len(r.cells) for r in rows), default=0)
        df = pd.DataFrame([r.cells+['']*(n-len(r.cells)) for r in rows], columns=[f"col{i+1}" for i in range(n)])
    df.insert(0,"time_sec",[round(r.t,2) for r in rows])

    meta={"frames_used":processed,"duration":float(dur or 0.0),"fps_sample":float(fps_sample),"roi_used":1.0 if determined_roi else 0.0,"rows":float(len(df)),"cols":float(len(df.columns))}
    return VideoOCRResult(df=df, header=header_cells, meta=meta)
""")

w(utils/"db_connectors.py", """
from __future__ import annotations
import pandas as pd
from sqlalchemy import create_engine, text, inspect
def make_engine(url: str): return create_engine(url, pool_pre_ping=True)
def list_tables(url: str):
    eng = make_engine(url)
    with eng.connect() as con: return inspect(con).get_table_names()
def read_table(url: str, table: str, limit: int = 5000) -> pd.DataFrame:
    eng = make_engine(url)
    with eng.connect() as con:
        q = text(f"SELECT * FROM {table} LIMIT :lim")
        return pd.read_sql(q, con, params={"lim": limit})
""")

w(utils/"reports.py", """
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone
def append_feedback(record: dict, path: str = "reports/feedback.csv"):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    rec=dict(record); rec["ts_utc"]=datetime.now(timezone.utc).isoformat()
    df_new=pd.DataFrame([rec])
    if p.exists(): df_all=pd.concat([pd.read_csv(p), df_new], ignore_index=True)
    else: df_all=df_new
    df_all.to_csv(p, index=False)
""")

# -------- pages --------
w(pages/"1_Grendel_Data_Prep.py", """
from pathlib import Path
import streamlit as st, pandas as pd
from beowulf_app.state import ensure_state, set_df, add_recipe_step
from beowulf_app.config import IS_MOBILE, UPLOAD_MB
from beowulf_app.utils.io_fs import read_local
from beowulf_app.utils.readers import read_any
from beowulf_app.utils.ocr_table import image_to_dataframe, ocr_available
from beowulf_app.utils.db_connectors import list_tables, read_table
from beowulf_app.utils.video_to_table import extract_table_from_video_bytes

ensure_state()
st.title("Grendel — Data Ingestion & Cleaning")
tabs = st.tabs(["Files","Image → Spreadsheet (OCR)","Database","Video → Table (beta)"])

with tabs[0]:
    left,right = st.columns([2,1])
    with left:
        st.subheader("Load from file")
        if IS_MOBILE:
            up = st.file_uploader("Upload (≤ 1 GB): CSV/Parquet/Excel/SPSS/Stata/SAS",
                                  type=["csv","parquet","pq","xlsx","xls","sav","dta","sas7bdat"])
            if up is not None:
                if up.size > UPLOAD_MB*1024*1024: st.error(f"File exceeds mobile cap of {UPLOAD_MB} MB.")
                else:
                    try: df = read_any(up, name=up.name)
                    except Exception as e: st.error(f"Read error: {e}")
                    else:
                        set_df(df, up.name, record_original=True)
                        add_recipe_step({"action":"upload","name":up.name,"size":up.size})
                        st.success(f"Loaded {len(df):,} rows (preview)."); st.dataframe(df.head(20), use_container_width=True)
        else:
            src = st.text_input("Local file path (.csv/.parquet/.xlsx/.sav/.dta/.sas7bdat)")
            if st.button("Load local") and src:
                try:
                    df = read_any(src, name=Path(src).name) if Path(src).suffix.lower() not in {".csv",".parquet",".pq"} else read_local(src)
                except Exception as e: st.error(f"Read error: {e}")
                else:
                    set_df(df, Path(src).name, record_original=True)
                    add_recipe_step({"action":"load","path":src})
                    st.success(f"Loaded {len(df):,} rows from {src}")
    with right:
        if st.session_state.df is not None:
            st.subheader("Preview"); st.dataframe(st.session_state.df.head(20), use_container_width=True)
    st.markdown("—"); st.subheader("Cleaning")
    if st.session_state.df is not None and st.button("Standardize column names"):
        df = st.session_state.df
        df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
        set_df(df, st.session_state.df_name); add_recipe_step({"action":"standardize_columns"})
        st.success("Column names standardized.")

with tabs[1]:
    st.subheader("Photo / Camera → Table")
    if not ocr_available(): st.warning("Install the Tesseract binary to enable OCR.")
    img_up = st.file_uploader("Upload table photo (PNG/JPG)", type=["png","jpg","jpeg"])
    cam_img = st.camera_input("Take a photo (mobile friendly)")
    src = cam_img or img_up
    if src and st.button("Extract table", disabled=not ocr_available()):
        with st.spinner("Running OCR…"):
            try: df_ocr = image_to_dataframe(src)
            except Exception as e: st.error(f"OCR error: {e}"); df_ocr = pd.DataFrame()
        if df_ocr.empty: st.warning("No table-like text detected.")
        else:
            set_df(df_ocr, "ocr_table", record_original=True)
            add_recipe_step({"action":"ocr"})
            st.success(f"OCR extracted {df_ocr.shape[0]}×{df_ocr.shape[1]}"); st.dataframe(df_ocr.head(50), use_container_width=True)

with tabs[2]:
    st.subheader("Connect to database")
    url = st.text_input("SQLAlchemy URL (e.g., postgresql+psycopg2://user:pass@host:5432/db)")
    if url and st.button("List tables"):
        try: names = list_tables(url)
        except Exception as e: st.error(f"DB error: {e}"); names=[]
        if names:
            st.success(f"Found {len(names)} tables.")
            choice = st.selectbox("Pick a table", names)
            limit = st.number_input("Preview rows", min_value=100, max_value=100_000, value=5000, step=100)
            if st.button("Load table preview"):
                try: df = read_table(url, choice, limit=int(limit))
                except Exception as e: st.error(f"DB read error: {e}")
                else:
                    set_df(df, f"{choice}@db", record_original=True)
                    add_recipe_step({"action":"db_read","table":choice,"limit":int(limit)})
                    st.success(f"Loaded preview: {len(df):,} rows"); st.dataframe(df.head(50), use_container_width=True)
    st.caption("Install DB driver packages as needed (psycopg2, pyodbc, pymysql, etc.).")

with tabs[3]:
    st.subheader("Video → Table (beta)")
    st.caption("Upload a screen recording of a scrolling dataset. We sample frames, OCR them, and fuse near‑duplicate lines into rows.")
    vid = st.file_uploader("Upload video (MP4/MOV/WebM)", type=["mp4","mov","webm","m4v"])
    fps = st.slider("Sampling FPS", 1, 5, 2)
    col1,col2 = st.columns([1,1])
    with col1:
        use_auto = st.checkbox("Auto-detect table region (ROI)", value=True)
    with col2:
        max_sec = st.number_input("Max seconds (0 = full)", min_value=0, max_value=3600, value=0, step=5)
    roi = None
    r1,r2,r3,r4 = st.columns(4)
    with r1: x = st.number_input("x", min_value=0, value=0, step=10, disabled=use_auto)
    with r2: y = st.number_input("y", min_value=0, value=0, step=10, disabled=use_auto)
    with r3: w = st.number_input("w", min_value=0, value=0, step=10, disabled=use_auto)
    with r4: h = st.number_input("h", min_value=0, value=0, step=10, disabled=use_auto)
    if not use_auto and w>0 and h>0: roi = (int(x),int(y),int(w),int(h))
    if vid is not None and st.button("Extract"):
        prog = st.progress(0.0)
        cb = lambda p: prog.progress(min(max(float(p),0.0),1.0))
        with st.spinner("Sampling frames and running OCR…"):
            try:
                res = extract_table_from_video_bytes(vid, fps_sample=fps, roi=None if use_auto else roi, max_seconds=None if max_sec==0 else max_sec, progress_cb=cb)
            except Exception as e:
                st.error(f"Video OCR error: {e}")
            else:
                if res.df.empty: st.warning("No table-like content detected. Try lower FPS or manual ROI.")
                else:
                    set_df(res.df, "video_table", record_original=True)
                    add_recipe_step({"action":"video_ocr","fps":fps,"rows":int(res.meta.get('rows',0))})
                    st.success(f"Extracted {res.df.shape[0]}×{res.df.shape[1]} from {int(res.meta['frames_used'])} frames.")
                    st.caption(f"Header: {res.header or '[inferred]'} • ROI auto: {'yes' if res.meta.get('roi_used') else 'no'}")
                    st.dataframe(res.df.head(100), use_container_width=True)
                    c1,c2 = st.columns(2)
                    with c1:
                        st.download_button("Download CSV", res.df.to_csv(index=False).encode("utf-8"), "video_table.csv", "text/csv")
                    with c2:
                        try:
                            import pyarrow as pa, pyarrow.parquet as pq, io
                            buf = io.BytesIO(); pq.write_table(pa.Table.from_pandas(res.df), buf, compression="zstd")
                            st.download_button("Download Parquet", buf.getvalue(), "video_table.parquet")
                        except Exception: st.info("Install pyarrow for Parquet export.")
""")

w(pages/"2_Hrunting_Epidemiology.py", """
import streamlit as st, numpy as np, pandas as pd
from scipy import stats
from beowulf_app.state import ensure_state
ensure_state()
st.title("Hrunting — Epidemiology & Classical Stats")
df = st.session_state.get("df")
tabs = st.tabs(["2×2 measures","Crosstabs & χ²/Fisher","Nonparametrics"])

with tabs[0]:
    st.subheader("2×2 Measures (RR/OR/RD)")
    a=st.number_input("a (exp+, out+)",0,step=1); b=st.number_input("b (exp+, out-)",0,step=1)
    c=st.number_input("c (exp-, out+)",0,step=1); d=st.number_input("d (exp-, out-)",0,step=1)
    if st.button("Compute 2×2"):
        a,b,c,d=[x+0.5 if x==0 else x for x in (a,b,c,d)]
        z=1.959963984540054; p1,p0=a/(a+b), c/(c+d)
        rr=p1/p0; se_rr=np.sqrt(1/a-1/(a+b)+1/c-1/(c+d))
        or_=(a*d)/(b*c); se_or=np.sqrt(1/a+1/b+1/c+1/d)
        rd=p1-p0; se_rd=np.sqrt(p1*(1-p1)/(a+b)+p0*(1-p0)/(c+d))
        st.json({"RR":rr,"RR 95% CI":[np.exp(np.log(rr)-z*se_rr),np.exp(np.log(rr)+z*se_rr)],
                 "OR":or_,"OR 95% CI":[np.exp(np.log(or_)-z*se_or),np.exp(np.log(or_)+z*se_or)],
                 "RD":rd,"RD 95% CI":[rd - z*se_rd, rd + z*se_rd]})
    st.caption("Associational only; causal methods live in Naegling.")

with tabs[1]:
    st.subheader("Crosstabs")
    if df is None: st.info("Load a dataset in Grendel.")
    else:
        cat1 = st.selectbox("Rows (categorical)", df.columns)
        cat2 = st.selectbox("Cols (categorical)", df.columns, index=min(1,len(df.columns)-1))
        if st.button("Compute χ² & Fisher"):
            tbl = pd.crosstab(df[cat1], df[cat2])
            chi2,p,dof,exp = stats.chi2_contingency(tbl)
            st.write("Observed"); st.dataframe(tbl, use_container_width=True)
            st.write("Expected"); st.dataframe(pd.DataFrame(exp,index=tbl.index,columns=tbl.columns), use_container_width=True)
            st.json({"chi2":float(chi2),"df":int(dof),"p_value":float(p)})
            if tbl.shape==(2,2):
                a,b,c,d = tbl.values.ravel()
                p_fisher = stats.fisher_exact([[a,b],[c,d]])[1]
                st.json({"Fisher_exact_p": float(p_fisher)})

with tabs[2]:
    st.subheader("Nonparametric tests")
    if df is None: st.info("Load a dataset in Grendel.")
    else:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols)==0: st.warning("No numeric columns detected.")
        else:
            y = st.selectbox("Numeric outcome", num_cols)
            g = st.selectbox("Grouping", df.columns)
            if st.button("Run tests"):
                groups = [grp[y].dropna().values for _, grp in df.groupby(g)]
                out={}
                if len(groups)==2:
                    u,pv=stats.mannwhitneyu(groups[0],groups[1],alternative="two-sided"); out["Mann-Whitney U"]={"U":float(u),"p":float(pv)}
                if len(groups)>2:
                    kw,pv=stats.kruskal(*groups); out["Kruskal-Wallis"]={"H":float(kw),"p":float(pv)}
                st.json(out)
""")

w(pages/"3_Naegling_Causal_Inference.py", 'import streamlit as st\nst.title("Naegling — Causal Inference")\nst.info("Scaffolds for PSM, IPTW + Love plot, DiD, IV/2SLS coming next.")\n')
w(pages/"4_Wiglaf_ML.py", 'import streamlit as st\nst.title("Wiglaf — Machine Learning")\nst.info("Model training, metrics, and explainability scaffolds.")\n')
w(pages/"5_Wealhtheow_Visualization.py", 'import streamlit as st\nst.title("Wealhtheow — Visualization")\nst.info("Exploratory plots and report export.")\n')
w(pages/"6_Beta_Feedback.py", """
import streamlit as st
from beowulf_app.utils.reports import append_feedback
st.title("Beta Feedback")
with st.form("feedback"):
    role = st.selectbox("Type", ["Bug","Idea","Question","Praise"])
    email = st.text_input("Contact (optional)")
    msg = st.text_area("Your feedback")
    submitted = st.form_submit_button("Submit")
    if submitted:
        if not msg.strip(): st.error("Please enter some feedback.")
        else: append_feedback({"type":role,"email":email,"message":msg}); st.success("Thanks!")
""")

# streamlit config
w(stcfg/"config.toml", """
[server]
maxUploadSize = 32768
fileWatcherType = "auto"
[theme]
primaryColor = "#10b981"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f6f7f9"
textColor = "#111827"
""")

print("OK")
