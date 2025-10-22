# backend/app/main.py (или где у тебя /analyze)
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io

from backend.app.core.chain import analyze_question
from backend.utils.io import read_any_table

app = FastAPI(title="Diven LC")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok"}

def build_summary(df: pd.DataFrame, result: dict) -> dict:

    rows = int(len(df))
    cols = list(map(str, df.columns.tolist()))
    base = {
        "dataset": {"rows": rows, "cols_count": len(cols), "columns": cols},
        "kpis": [],
        "highlights": [],
    }

    gcol = result.get("groupby_col") or (result.get("used_columns") or [None, None])[0]
    vcol = result.get("value_col") or (result.get("used_columns") or [None, None])[-1]
    agg  = (result.get("agg") or "sum").lower()

    if gcol and vcol and gcol in df.columns and vcol in df.columns and pd.api.types.is_numeric_dtype(df[vcol]):
        grouped = df.groupby(gcol, dropna=False)[vcol]
        if agg == "mean":
            s = grouped.mean().sort_values(ascending=False)
        else:
            s = grouped.sum().sort_values(ascending=False)

        if not s.empty:
            top_label, top_value = s.index[0], float(s.iloc[0])
            bottom_label, bottom_value = s.index[-1], float(s.iloc[-1])
            total_value = float(s.sum())
            avg_value = float(df[vcol].mean())

            base["kpis"] = [
                {"label": "Top", "value": top_label, "sub": f"{vcol}: {top_value:,.2f}"},
                {"label": "Low", "value": bottom_label, "sub": f"{vcol}: {bottom_value:,.2f}"},
                {"label": "Average", "value": f"{avg_value:,.2f}", "sub": vcol},
                {"label": "Total", "value": f"{total_value:,.2f}", "sub": vcol},
            ]

            base["highlights"] = [
                f"Dominant: {top_label} ({top_value:,.2f})",
                f"Weakest: {bottom_label} ({bottom_value:,.2f})",
            ]
    return base

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), question: str = Form("")):
    try:
        raw = await file.read()
        df = read_any_table(raw, file.filename or "")
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"Erorr reading file: {e}"})

    if not question.strip():
        head = df.head(5).to_dict(orient="records")
        return {"ok": True, "message": "No questions.", "rows": len(df), "cols": list(df.columns), "head": head}

    try:
        result = analyze_question(df, question)  
        summary = build_summary(df, result)
        return {"ok": True, **result, "summary": summary}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": f"Pipeline died: {e}"})