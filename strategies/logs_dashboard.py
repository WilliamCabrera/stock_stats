"""
Logs Dashboard  –  Streamlit
Run:  streamlit run strategies/logs_dashboard.py
"""
import json
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Logs Dashboard",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOGS_DIR = Path(__file__).parent.parent / "logs"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


@st.cache_data(show_spinner=False)
def find_log_files() -> list[dict]:
    files = []
    for path in sorted(LOGS_DIR.rglob("*.log")):
        rel = path.relative_to(LOGS_DIR)
        mtime = path.stat().st_mtime
        size_kb = path.stat().st_size / 1024
        files.append({"path": path, "rel": str(rel), "mtime": mtime, "size_kb": size_kb})
    files.sort(key=lambda x: x["mtime"], reverse=True)
    return files


def is_json_log(path: Path) -> bool:
    try:
        with open(path, "r", errors="replace") as f:
            first = f.readline().strip()
        if first:
            json.loads(first)
            return True
    except (json.JSONDecodeError, OSError):
        pass
    return False


@st.cache_data(show_spinner=False)
def load_json_log(path_str: str) -> pd.DataFrame:
    rows = []
    with open(path_str, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append({
                    "timestamp": obj.get("timestamp") or obj.get("asctime", ""),
                    "level": strip_ansi(str(obj.get("level", ""))).strip(),
                    "logger": obj.get("logger", ""),
                    "module": obj.get("module", ""),
                    "line": obj.get("line", ""),
                    "message": str(obj.get("message", "")),
                    "exception": str(obj.get("exception", "")) if obj.get("exception") else "",
                })
            except json.JSONDecodeError:
                rows.append({
                    "timestamp": "", "level": "", "logger": "", "module": "",
                    "line": "", "message": line, "exception": "",
                })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_plain_log(path_str: str) -> pd.DataFrame:
    rows = []
    with open(path_str, "r", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            rows.append({"line_no": i, "text": line.rstrip()})
    return pd.DataFrame(rows)


def load_file(path: Path) -> tuple[pd.DataFrame, bool]:
    json_fmt = is_json_log(path)
    if json_fmt:
        return load_json_log(str(path)), True
    return load_plain_log(str(path)), False


def search_df(df: pd.DataFrame, query: str, json_fmt: bool) -> pd.DataFrame:
    q = query.lower()
    if json_fmt:
        mask = (
            df["message"].str.lower().str.contains(q, na=False)
            | df["level"].str.lower().str.contains(q, na=False)
            | df["logger"].str.lower().str.contains(q, na=False)
            | df["module"].str.lower().str.contains(q, na=False)
            | df["exception"].str.lower().str.contains(q, na=False)
        )
    else:
        mask = df["text"].str.lower().str.contains(q, na=False)
    return df[mask]


def search_all_files(log_files: list[dict], query: str) -> pd.DataFrame:
    results = []
    progress = st.progress(0, text="Buscando en todos los ficheros...")
    total = len(log_files)
    for i, f in enumerate(log_files):
        df, json_fmt = load_file(f["path"])
        hits = search_df(df, query, json_fmt)
        if not hits.empty:
            hits = hits.copy()
            hits.insert(0, "file", f["rel"])
            results.append(hits)
        progress.progress((i + 1) / total, text=f"Buscando... {i+1}/{total}")
    progress.empty()
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


def render_level_badge(level: str) -> str:
    colors = {
        "DEBUG": "#42a5f5",
        "INFO": "#26a69a",
        "WARNING": "#ffca28",
        "WARN": "#ffca28",
        "ERROR": "#ef5350",
        "CRITICAL": "#ab47bc",
    }
    key = level.upper().strip()
    color = colors.get(key, "#888")
    return f'<span style="background:{color};color:#000;padding:1px 6px;border-radius:3px;font-size:11px;font-weight:bold">{level}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📋 Logs Dashboard")
st.sidebar.markdown("---")

log_files = find_log_files()
if st.sidebar.button("Refrescar lista", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# Group by subdirectory
groups: dict[str, list[dict]] = {}
for f in log_files:
    folder = str(Path(f["rel"]).parent)
    groups.setdefault(folder, []).append(f)

st.sidebar.markdown("### Ficheros de log")
label_map = {f["rel"]: f for f in log_files}
labels = [f["rel"] for f in log_files]

# Build display labels with size info
display_labels = []
for f in log_files:
    size = f["size_kb"]
    size_str = f"{size:.0f} KB" if size < 1024 else f"{size/1024:.1f} MB"
    display_labels.append(f"{f['rel']}  ({size_str})")

selected_idx = st.sidebar.selectbox(
    "Seleccionar fichero",
    range(len(log_files)),
    format_func=lambda i: display_labels[i],
)

selected_file = log_files[selected_idx]
st.sidebar.caption(f"Última modificación: {pd.Timestamp(selected_file['mtime'], unit='s').strftime('%Y-%m-%d %H:%M:%S')}")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title(f"📄 {selected_file['rel']}")

# Load selected file
with st.spinner("Cargando logs..."):
    df, json_fmt = load_file(selected_file["path"])

total_rows = len(df)
st.caption(f"{total_rows:,} líneas  •  formato: {'JSON estructurado' if json_fmt else 'texto plano'}")

# ── Filters ───────────────────────────────────────────────────────────────────
col_search, col_scope, col_level = st.columns([3, 1, 2])

with col_search:
    query = st.text_input("Buscar", placeholder="Escribe para filtrar...", label_visibility="collapsed")

with col_scope:
    search_all = st.checkbox("Buscar en todos los ficheros", value=False)
    if search_all:
        st.caption("⚠️ Puede ser lento")

with col_level:
    if json_fmt and not search_all:
        levels = ["Todos"] + sorted(df["level"].dropna().unique().tolist())
        selected_level = st.selectbox("Nivel", levels, label_visibility="collapsed")
    else:
        selected_level = "Todos"

st.markdown("---")

# ── Search logic ──────────────────────────────────────────────────────────────
if query and search_all:
    with st.spinner(f'Buscando "{query}" en {len(log_files)} ficheros...'):
        result_df = search_all_files(log_files, query)

    if result_df.empty:
        st.warning(f"No se encontraron resultados para **{query}**")
    else:
        st.success(f"{len(result_df):,} resultados en todos los ficheros")

        # Show grouped by file
        for file_rel in result_df["file"].unique():
            file_hits = result_df[result_df["file"] == file_rel].drop(columns=["file"])
            with st.expander(f"📄 {file_rel}  ({len(file_hits):,} resultados)", expanded=True):
                cols_to_show = [c for c in file_hits.columns if c != "exception"]
                st.dataframe(file_hits[cols_to_show], use_container_width=True, height=300)

else:
    display_df = df.copy()

    if query:
        display_df = search_df(display_df, query, json_fmt)

    if json_fmt and selected_level != "Todos":
        display_df = display_df[display_df["level"] == selected_level]

    if query or selected_level != "Todos":
        st.info(f"{len(display_df):,} de {total_rows:,} líneas coinciden")

    if display_df.empty:
        st.warning("No hay resultados con los filtros actuales.")
    else:
        if json_fmt:
            # Show exception details in expander when clicking
            cols_main = ["timestamp", "level", "module", "line", "message"]
            available = [c for c in cols_main if c in display_df.columns]

            show_exc = st.checkbox("Mostrar columna exception/traceback", value=False)
            if show_exc:
                available = available + ["exception"]

            st.dataframe(
                display_df[available].reset_index(drop=True),
                use_container_width=True,
                height=600,
                column_config={
                    "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                    "level": st.column_config.TextColumn("Level", width="small"),
                    "module": st.column_config.TextColumn("Module", width="small"),
                    "line": st.column_config.NumberColumn("Line", width="small"),
                    "message": st.column_config.TextColumn("Message", width="large"),
                    "exception": st.column_config.TextColumn("Exception", width="large"),
                },
            )

            # Exception viewer
            st.markdown("---")
            with st.expander("Ver excepciones del resultado actual"):
                exc_df = display_df[display_df["exception"].str.len() > 0][["timestamp", "level", "message", "exception"]]
                if exc_df.empty:
                    st.info("No hay excepciones en los logs filtrados.")
                else:
                    st.caption(f"{len(exc_df)} entradas con excepción")
                    for _, row in exc_df.iterrows():
                        st.markdown(f"**{row['timestamp']}** — `{row['level']}`")
                        st.markdown(f"_{row['message']}_")
                        st.code(row["exception"], language="python")
                        st.markdown("---")
        else:
            st.dataframe(
                display_df.reset_index(drop=True),
                use_container_width=True,
                height=600,
                column_config={
                    "line_no": st.column_config.NumberColumn("#", width="small"),
                    "text": st.column_config.TextColumn("Contenido", width="large"),
                },
            )
