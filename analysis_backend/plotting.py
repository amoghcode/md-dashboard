import os
import subprocess
from typing import List, Optional
from .analysis_utils import AnalysisResult

# Optional Plotly import
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def generate_plotly_html(results: List[AnalysisResult], selected_name: Optional[str] = None, output_html: str = "analysis_dashboard.html") -> str:
    """
    Generates an interactive Plotly HTML plot for all results, one selected analysis, or empty dashboard.
    Gracefully handles missing plotly dependency.
    """
    if not PLOTLY_AVAILABLE:
        print("[PLOT] Warning: plotly is not installed. Skipping HTML dashboard generation.")
        return ""
        
    if selected_name:
        filtered_results = [r for r in results if r.analysis_name.lower() == selected_name.lower()]
    else:
        filtered_results = results
        
    if not filtered_results:
        fig = go.Figure()
        fig.add_annotation(text="No analysis results available.", showarrow=False, font=dict(size=16))
        fig.update_layout(title="MD Analysis Plot")
        fig.write_html(output_html)
        return output_html

    n_plots = len(filtered_results)
    fig = make_subplots(rows=n_plots, cols=1, subplot_titles=[r.analysis_name for r in filtered_results])

    for idx, res in enumerate(filtered_results, start=1):
        df = res.data
        if df.shape[1] >= 2:
            x_col = df.columns[0]
            for col in df.columns[1:]:
                fig.add_trace(
                    go.Scatter(x=df[x_col], y=df[col], mode='lines', name=f"{res.analysis_name}: {col}"),
                    row=idx, col=1
                )
            fig.update_xaxes(title_text=x_col, row=idx, col=1)
            fig.update_yaxes(title_text="Value", row=idx, col=1)

    fig.update_layout(height=400 * n_plots, width=900, title_text="GROMACS Trajectory Analysis", template="plotly_white")
    fig.write_html(output_html)
    print(f"[PLOT] Plotly graph successfully saved to: {output_html}")
    return output_html

def get_xmgrace_command(xvg_path: str) -> str:
    """Returns shell command to open XVG in Xmgrace if available."""
    return f"xmgrace {xvg_path}"

def launch_xmgrace(xvg_path: str) -> bool:
    """Callable hook to launch Xmgrace as an asynchronous subprocess. Never invalidates calculation on failure."""
    if not os.path.exists(xvg_path):
        print(f"[XMGRACE] Error: File not found: {xvg_path}")
        return False
    try:
        subprocess.Popen(["xmgrace", xvg_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[XMGRACE] Launched xmgrace for {xvg_path}")
        return True
    except FileNotFoundError:
        print("[XMGRACE] xmgrace executable not found on PATH.")
        return False
    except Exception as e:
        print(f"[XMGRACE] Failed to launch xmgrace: {e}")
        return False
