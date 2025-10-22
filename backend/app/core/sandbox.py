import io, base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ast

def render_chart_png(result: object, chart: dict | None) -> str:
    if isinstance(chart, str) or chart is None:
        chart = {"type": chart or "bar"}
    ctype = (chart.get("type") or "bar").lower()
    if chart is None:
        chart = {}
    ctype = (chart.get("type") or "bar").lower()
    title = chart.get("title") or ""

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)

    def _finalize_and_dump():
        ax.set_title(title, pad=8)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ----- HEATMAP -----
    if ctype == "heatmap":
        if not isinstance(result, pd.DataFrame):
            raise ValueError("Heatmap requires a pandas DataFrame result (pivot).")
        data = result.apply(pd.to_numeric, errors="coerce").fillna(0.0).values

        im = ax.imshow(
            data,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest"
        )
        ax.set_xticks(np.arange(result.shape[1]))
        ax.set_xticklabels([str(c) for c in result.columns], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(result.shape[0]))
        ax.set_yticklabels([str(i) for i in result.index], fontsize=9)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel(chart.get("colorbar_label", "Value"), rotation=90, labelpad=10)

        return _finalize_and_dump()

    # ----- TABLE -----
    if ctype == "table":
        if isinstance(result, (pd.Series, pd.DataFrame)):
            df = result if isinstance(result, pd.DataFrame) else result.to_frame()
        else:
            df = pd.DataFrame({"result": [result]})

        ax.axis("off")
        tbl = ax.table(
            cellText=df.round(3).astype(str).values,
            colLabels=[str(c) for c in df.columns],
            rowLabels=[str(i) for i in df.index],
            loc="center",
            cellLoc="right",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.2)
        return _finalize_and_dump()

    # -----  pandas.plot -----
    if isinstance(result, pd.Series):
        result = result.to_frame()

    if isinstance(result, pd.DataFrame):
        kind_map = {"bar": "bar", "line": "line", "hist": "hist", "pie": "pie"}
        kind = kind_map.get(ctype, "bar")
        if kind == "pie":
            num_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
            if not num_cols:
                raise ValueError("No numeric columns for pie chart.")
            s = result[num_cols[0]].squeeze()
            s.plot(kind="pie", ax=ax, ylabel="")
        else:
            result.plot(kind=kind, ax=ax)
        return _finalize_and_dump()

    ax.axis("off")
    ax.text(0.02, 0.98, str(result)[:2000], va="top", ha="left", fontsize=9, family="monospace")
    return _finalize_and_dump()
def validate_pandas_code(code: str):
  
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in generated code: {e}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Disallowed syntax: import")
        if isinstance(node, ast.Call):
            if hasattr(node.func, "id") and node.func.id in ["exec", "eval", "open", "system"]:
                raise ValueError("Disallowed syntax: unsafe call")
    return True


def exec_pandas(code: str, df):
    validate_pandas_code(code)
    safe_locals = {"df": df.copy(), "pd": pd, "np": np}
    try:
        exec(code, {}, safe_locals)
    except Exception as e:
        raise RuntimeError(f"Execution failed: {e}")
    if "result" not in safe_locals:
        raise ValueError("Generated code did not set 'result'")
    return safe_locals["result"]