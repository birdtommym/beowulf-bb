w(app/"app.py", """
import os, streamlit as st
from beowulf_app.state import ensure_state
from beowulf_app.config import UPLOAD_MB, IS_MOBILE, profile_badge
from beowulf_app.ui_theme import inject_ios_theme

# Optional shim so absolute imports work even if Streamlit runs inside the package dir
import sys, os as _os
ROOT = _os.path.dirname(_os.path.dirname(__file__))  # /app
if ROOT not in sys.path: sys.path.insert(0, ROOT)

st.set_page_config(page_title="Beowulf Biostatistics", layout="wide")
# DO NOT call st.set_option('server.maxUploadSize') here (Streamlit forbids runtime change).
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
