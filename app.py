"""
Machine Health Intelligence: Web Application Launcher.
NASA C-MAPSS FD001 Turbofan Engine Degradation & RUL Prognostics.

Replaces Streamlit with high-performance FastAPI + Uvicorn server for Render.
Serves interactive REST APIs and the Portfolio Web UI.
"""

import os
import uvicorn
from server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[START] Starting Machine Health Intelligence Web Service on http://0.0.0.0:{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
