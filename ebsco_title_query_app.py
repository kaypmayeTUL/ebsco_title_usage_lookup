"""
EBSCO Title Query — Batch Lookup

Upload a COUNTER 5 Title Report (TR) CSV plus a list of titles you want details
for, and see per-title metrics, monthly breakdowns, and bibliographic info for
just those titles. Also supports ad-hoc single-title search.

Design: Howard-Tilton Memorial Library / Tulane (green #285C4D, blue #71C5E8,
Source Serif 4 / DM Sans).
"""

from __future__ import annotations

import csv
import io
import re
from io import BytesIO, StringIO

import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# PAGE CONFIG + TULANE DESIGN SYSTEM
# ============================================================

st.set_page_config(
    page_title="EBSCO Title Query",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    :root {
        --tulane-green: #285C4D;
        --tulane-blue: #71C5E8;
        --tulane-blue-light: #B8E2F2;
        --soft-bg: #F7F9F8;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'Source Serif 4', serif !important;
        color: var(--tulane-green) !important;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .stButton > button {
        background-color: var(--tulane-green);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.25rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #1d443a;
        color: white;
    }
    .stDownloadButton > button {
        background-color: var(--tulane-blue);
        color: var(--tulane-green);
        font-weight: 600;
        border: none;
        border-radius: 6px;
    }
    .metric-card {
        background: linear-gradient(135deg, #e8f5f1, #e1f4fb);
        border-left: 5px solid var(--tulane-green);
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.75rem;
    }
    .metric-card h4 {
        margin: 0 0 0.25rem 0 !important;
        font-size: 1.05rem !important;
    }
    .title-hit {
        background: white;
        border: 1px solid #e0e4e2;
        border-left: 5px solid var(--tulane-green);
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    .title-hit-missing {
        border-left-color: #c97b63;
        background: #fdf5f2;
    }
    .small-muted {
        color: #6b7370;
        font-size: 0.9rem;
    }
    .tool-card {
        background: white;
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        border-top: 4px solid var(--tulane-green);
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        height: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

HEADER_MARKER = "Title"
GROUPING_FIELDS = ["Title", "ISBN", "Print_ISSN", "Online_ISSN", "Platform"]
META_FIELDS = [
    "Title", "Publisher", "Platform", "DOI", "Proprietary_ID",
    "ISBN", "Print_ISSN", "Online_ISSN", "URI", "Data Type",
]
PREFERRED_METRICS = [
    "Total_Item_Requests",
    "Unique_Item_Requests",
    "Unique_Title_Requests",
    "Unique_Title_Investigations",
    "Total_Item_Investigations",
    "Unique_Item_Investigations",
]


@st.cache_data(show_spinner=False)
def load_counter_tr(file_bytes: bytes) -> tuple[list[str], list[dict], list[str]]:
    """Parse a COUNTER 5 TR CSV. Returns (header, rows, month_columns).

    Dynamically finds the data header row so this works even if EBSCO
    adds or removes preamble lines in future exports.
    """
    text = file_bytes.decode("utf-8-sig")
    reader = csv.reader(StringIO(text))
    header = None
    rows: list[dict] = []
    for row in reader:
        if not row:
            continue
        first = (row[0] or "").lstrip("\ufeff").strip()
        if header is None:
            if first == HEADER_MARKER:
                header = [(c or "").lstrip("\ufeff").strip() for c in row]
            continue
        if not any((c or "").strip() for c in row):
            continue
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[: len(header)]
        rec = {header[i]: (row[i] or "").lstrip("\ufeff").strip() for i in range(len(header))}
        rows.append(rec)

    if header is None:
        raise ValueError(
            "Could not find the data header row. Is this a COUNTER 5 TR report?"
        )

    try:
        rp_idx = header.index("Reporting Period_Total")
    except ValueError:
        rp_idx = len(header) - 1
    month_cols = header[rp_idx + 1:]
    return header, rows, month_cols


def _to_int(s) -> int:
    if s is None or s == "":
        return 0
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return 0


@st.cache_data(show_spinner=False)
def group_by_title(rows: list[dict], month_cols: list[str]) -> list[dict]:
    """Pivot per-metric rows into one record per unique title."""
    grouped: dict[tuple, dict] = {}
    for r in rows:
        key = tuple(r.get(f, "") for f in GROUPING_FIELDS)
        if key not in grouped:
            grouped[key] = {
                "meta": {f: r.get(f, "") for f in META_FIELDS},
                "metrics": {},
            }
        metric = r.get("Metric_Type", "").strip()
        if not metric:
            continue
        total = _to_int(r.get("Reporting Period_Total", "0"))
        months = {m: _to_int(r.get(m, "0")) for m in month_cols}
        grouped[key]["metrics"][metric] = {"total": total, "months": months}
    return list(grouped.values())


# ============================================================
# TITLE NORMALIZATION AND MATCHING
# ============================================================

def normalize_title(s: str) -> str:
    """Lowercase, strip punctuation and extra whitespace for fuzzy matching."""
    if not s:
        return ""
    s = s.lower()
    # Drop common leading articles so "The X" matches "X"
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@st.cache_data(show_spinner=False)
def build_title_index(_records_key: int, records: list[dict]) -> dict[str, list[int]]:
    """Map normalized title -> list of record indices. Enables fast lookup."""
    idx: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        norm = normalize_title(rec["meta"].get("Title", ""))
        if not norm:
            continue
        idx.setdefault(norm, []).append(i)
    return idx


def find_matches(
    query: str,
    records: list[dict],
    index: dict[str, list[int]],
    mode: str = "contains",
) -> list[int]:
    """Return record indices matching ``query``.

    mode: 'exact' (normalized equality), 'contains' (normalized substring),
    'starts_with' (normalized prefix).
    """
    q = normalize_title(query)
    if not q:
        return []
    if mode == "exact":
        return list(index.get(q, []))
    hits: list[int] = []
    for norm, recs in index.items():
        if mode == "starts_with":
            if norm.startswith(q):
                hits.extend(recs)
        else:  # contains
            if q in norm:
                hits.extend(recs)
    return hits


# ============================================================
# TITLE LIST PARSING
# ============================================================

def parse_title_list(uploaded_file) -> list[str]:
    """Read an uploaded title list. Accepts .txt (one per line) or .csv.

    For CSVs, looks for a 'Title' column (case-insensitive), falling back to
    the first column if none found.
    """
    if uploaded_file is None:
        return []
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith((".txt", ".tsv")):
        text = data.decode("utf-8-sig", errors="replace")
        return [line.strip() for line in text.splitlines() if line.strip()]

    # CSV path
    try:
        df = pd.read_csv(BytesIO(data), encoding="utf-8-sig", dtype=str)
    except UnicodeDecodeError:
        df = pd.read_csv(BytesIO(data), encoding="latin-1", dtype=str)
    if df.empty:
        return []
    df.columns = [c.strip() for c in df.columns]
    # Prefer an explicit Title column
    title_col = None
    for c in df.columns:
        if c.lower() == "title":
            title_col = c
            break
    if title_col is None:
        title_col = df.columns[0]
    titles = (
        df[title_col]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .tolist()
    )
    return titles


def parse_pasted_titles(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


# ============================================================
# DISPLAY HELPERS
# ============================================================

def render_record(rec: dict, month_cols: list[str], key_prefix: str = "") -> None:
    """Render a single title record. ``key_prefix`` must be unique per call
    within a page render — Streamlit requires unique element IDs for charts."""
    meta = rec["meta"]
    st.markdown(
        f"""
        <div class="title-hit">
            <h4 style="margin-top:0;">{meta.get('Title', '(no title)')}</h4>
            <div class="small-muted">
                {meta.get('Publisher', '') or '—'} &nbsp;•&nbsp;
                {meta.get('Platform', '') or '—'} &nbsp;•&nbsp;
                {meta.get('Data Type', '') or '—'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # IDs in a small expander to reduce visual noise
    id_bits = []
    if meta.get("ISBN"): id_bits.append(f"**ISBN:** {meta['ISBN']}")
    if meta.get("Print_ISSN"): id_bits.append(f"**Print ISSN:** {meta['Print_ISSN']}")
    if meta.get("Online_ISSN"): id_bits.append(f"**Online ISSN:** {meta['Online_ISSN']}")
    if meta.get("DOI"): id_bits.append(f"**DOI:** {meta['DOI']}")
    if meta.get("Proprietary_ID"): id_bits.append(f"**ID:** {meta['Proprietary_ID']}")
    if meta.get("URI"): id_bits.append(f"**URI:** {meta['URI']}")
    if id_bits:
        with st.expander("Identifiers", expanded=False):
            st.markdown("  \n".join(id_bits))

    # Metric totals as tiles
    metrics = rec.get("metrics", {})
    if metrics:
        ordered = [m for m in PREFERRED_METRICS if m in metrics] + \
                  sorted(m for m in metrics if m not in PREFERRED_METRICS)
        cols = st.columns(min(len(ordered), 6))
        for i, m in enumerate(ordered):
            with cols[i % len(cols)]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h4>{metrics[m]['total']}</h4>
                        <div class="small-muted">{m.replace('_', ' ')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Monthly chart — show the most informative metric that has non-zero data
    for m in PREFERRED_METRICS:
        if m in metrics:
            monthly = metrics[m]["months"]
            if any(monthly.get(c, 0) for c in month_cols):
                df = pd.DataFrame({
                    "Month": month_cols,
                    "Count": [monthly.get(c, 0) for c in month_cols],
                })
                fig = px.bar(
                    df, x="Month", y="Count",
                    title=f"{m.replace('_', ' ')} by month",
                    color_discrete_sequence=["#285C4D"],
                )
                fig.update_layout(
                    height=260,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=False,
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"chart_{key_prefix}_{m}" if key_prefix else f"chart_{id(rec)}_{m}",
                )
                break


def build_summary_dataframe(
    matched: list[tuple[str, list[int]]],
    records: list[dict],
    month_cols: list[str],
) -> pd.DataFrame:
    """Flatten match results into a spreadsheet-friendly DataFrame.

    Each row is (query_title, matched_record) with meta + metric totals + months.
    Unmatched queries get one row with blank metrics so users can see gaps.
    """
    rows = []
    for query, idxs in matched:
        if not idxs:
            rows.append({
                "Query": query,
                "Match Status": "Not found",
                "Title": "", "Publisher": "", "Platform": "", "Data Type": "",
                "ISBN": "", "Print_ISSN": "", "Online_ISSN": "",
                **{m: 0 for m in PREFERRED_METRICS},
                **{f"month_{c}": 0 for c in month_cols},
            })
            continue
        for i in idxs:
            rec = records[i]
            meta = rec["meta"]
            metrics = rec["metrics"]
            row = {
                "Query": query,
                "Match Status": "Matched",
                "Title": meta.get("Title", ""),
                "Publisher": meta.get("Publisher", ""),
                "Platform": meta.get("Platform", ""),
                "Data Type": meta.get("Data Type", ""),
                "ISBN": meta.get("ISBN", ""),
                "Print_ISSN": meta.get("Print_ISSN", ""),
                "Online_ISSN": meta.get("Online_ISSN", ""),
            }
            for m in PREFERRED_METRICS:
                row[m] = metrics.get(m, {}).get("total", 0)
            # Monthly columns use the primary metric
            primary = next((m for m in PREFERRED_METRICS if m in metrics), None)
            if primary:
                primary_months = metrics[primary]["months"]
                for c in month_cols:
                    row[f"{primary}__{c}"] = primary_months.get(c, 0)
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# SIDEBAR — FILE UPLOAD (shared across tabs)
# ============================================================

def _sidebar() -> tuple[list[dict] | None, list[str] | None, dict | None]:
    with st.sidebar:
        st.title("🔎 EBSCO Title Query")
        st.markdown("*Howard-Tilton Memorial Library*")
        st.markdown("---")
        st.subheader("1. Upload COUNTER TR report")
        tr_file = st.file_uploader(
            "COUNTER 5 Title Report (CSV)",
            type=["csv"],
            key="tr_upload",
            help="Standard EBSCOhost TR export. Preamble rows are handled automatically.",
        )

    if tr_file is None:
        return None, None, None

    try:
        with st.spinner("Parsing COUNTER report…"):
            header, rows, month_cols = load_counter_tr(tr_file.getvalue())
            records = group_by_title(rows, month_cols)
            index = build_title_index(len(records), records)
    except Exception as e:
        with st.sidebar:
            st.error(f"Couldn't parse file: {e}")
        return None, None, None

    with st.sidebar:
        st.success(f"Loaded **{len(records):,}** unique titles from **{len(rows):,}** metric rows.")
        if month_cols:
            st.caption(f"Reporting range: {month_cols[0]} → {month_cols[-1]}")

    return records, month_cols, index


# ============================================================
# PAGES
# ============================================================

def page_home():
    st.title("EBSCO Title Query")
    st.markdown(
        "Look up metrics and bibliographic details for specific titles in a "
        "COUNTER 5 Title Report. Search one title at a time or upload a list of titles "
        "to batch-check dozens or hundreds at once."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="tool-card">
                <h3>📋 Batch lookup</h3>
                <p><em>Check a list of titles against the report.</em></p>
                <p>Paste titles or upload a <code>.txt</code> / <code>.csv</code> file.
                Get metrics for every title found, plus a list of titles that weren't in the report.
                Download the combined result as a spreadsheet.</p>
                <hr>
                <p><strong>Good for:</strong></p>
                <ul>
                    <li>Cancellation/renewal review against a package list</li>
                    <li>Comparing a reading list against e-book usage</li>
                    <li>Faculty request triage</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="tool-card">
                <h3>🔍 Single title search</h3>
                <p><em>Ad-hoc lookup for one title at a time.</em></p>
                <p>Type any title or substring. Supports exact, contains, and starts-with matching.
                See bibliographic metadata, metric totals, and month-by-month usage.</p>
                <hr>
                <p><strong>Good for:</strong></p>
                <ul>
                    <li>Quick questions at the desk</li>
                    <li>Spot-checking a specific vendor claim</li>
                    <li>Investigating a usage spike</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="tool-card">
                <h3>🏆 Top titles</h3>
                <p><em>Auto-ranked report of your most-used titles.</em></p>
                <p>See the highest-use titles by any COUNTER metric, filtered
                by platform, publisher, or data type. Download the ranking as a CSV.</p>
                <hr>
                <p><strong>Good for:</strong></p>
                <ul>
                    <li>Collection reports and annual summaries</li>
                    <li>Demonstrating e-resource ROI</li>
                    <li>Spotting unexpected heavy hitters</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.info(
        "👈 Upload your COUNTER TR CSV in the sidebar to get started. "
        "Then pick a tool from the sidebar."
    )


def page_batch_lookup(records, month_cols, index):
    st.title("📋 Batch title lookup")
    st.markdown(
        "Upload or paste a list of titles, and this tool will find every match "
        "in the COUNTER report. Unmatched titles are flagged so you can see gaps."
    )

    with st.container(border=True):
        st.subheader("Provide your title list")
        source = st.radio(
            "Input method:",
            ["Upload file", "Paste titles"],
            horizontal=True,
            key="batch_source",
        )

        titles: list[str] = []
        if source == "Upload file":
            upload = st.file_uploader(
                "Title list file (`.txt` one-per-line, or `.csv` with a `Title` column)",
                type=["txt", "csv", "tsv"],
                key="title_list_upload",
            )
            if upload is not None:
                try:
                    titles = parse_title_list(upload)
                except Exception as e:
                    st.error(f"Couldn't parse file: {e}")
        else:
            text = st.text_area(
                "Paste titles (one per line):",
                height=180,
                placeholder="The Caribbean Front in World War II\nThe History of Jazz\n…",
                key="paste_titles",
            )
            titles = parse_pasted_titles(text)

        match_mode = st.radio(
            "Matching mode:",
            ["Contains (loose)", "Starts with", "Exact"],
            horizontal=True,
            index=0,
            key="batch_mode",
            help=(
                "All modes ignore case, punctuation, and a leading 'The/A/An'. "
                "'Contains' is most forgiving; 'Exact' requires the whole title."
            ),
        )
        mode_key = {
            "Contains (loose)": "contains",
            "Starts with": "starts_with",
            "Exact": "exact",
        }[match_mode]

    if not titles:
        st.info("Provide at least one title above to run the lookup.")
        return

    st.markdown(f"**{len(titles)}** title(s) in your list. Running lookup…")

    matched: list[tuple[str, list[int]]] = [
        (t, find_matches(t, records, index, mode=mode_key)) for t in titles
    ]

    # Summary metrics
    found_queries = [(q, idxs) for q, idxs in matched if idxs]
    missing_queries = [q for q, idxs in matched if not idxs]
    total_records = sum(len(idxs) for _, idxs in found_queries)

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Titles in list", len(titles))
    with mc2:
        st.metric("Matched queries", len(found_queries))
    with mc3:
        st.metric("Not found", len(missing_queries))

    st.markdown("---")

    # Download
    df = build_summary_dataframe(matched, records, month_cols)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download full results as CSV",
        data=csv_bytes,
        file_name="title_query_results.csv",
        mime="text/csv",
    )

    tab_found, tab_missing, tab_table = st.tabs(
        [f"✅ Found ({len(found_queries)})",
         f"⚠️ Not found ({len(missing_queries)})",
         "📄 Full table"]
    )

    with tab_found:
        if not found_queries:
            st.info("No matches for any of the titles in your list.")
        else:
            # For each input title, show matches (multiple platforms = multiple records)
            for q_idx, (query, idxs) in enumerate(found_queries):
                with st.expander(
                    f"**{query}** — {len(idxs)} match(es)",
                    expanded=(len(found_queries) <= 3),
                ):
                    for i in idxs:
                        render_record(records[i], month_cols, key_prefix=f"batch_q{q_idx}_r{i}")

    with tab_missing:
        if not missing_queries:
            st.success("Every title in your list matched something in the report.")
        else:
            st.markdown(
                "These titles from your list had no match. Common causes: "
                "different edition/subtitle, punctuation or wording differences, "
                "or the title genuinely isn't in this reporting period."
            )
            miss_df = pd.DataFrame({"Title (not found)": missing_queries})
            st.dataframe(miss_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download unmatched list",
                data=miss_df.to_csv(index=False).encode("utf-8"),
                file_name="title_query_unmatched.csv",
                mime="text/csv",
            )

    with tab_table:
        st.dataframe(df, use_container_width=True, hide_index=True)


def page_single_search(records, month_cols, index):
    st.title("🔍 Single title search")
    st.markdown(
        "Look up one title at a time. Useful for quick questions or investigating "
        "a specific record."
    )

    with st.container(border=True):
        query = st.text_input("Title or substring:", key="single_query")
        col1, col2 = st.columns([1, 3])
        with col1:
            mode = st.selectbox(
                "Match mode:",
                ["Contains", "Starts with", "Exact"],
                index=0,
                key="single_mode",
            )
        mode_key = {"Contains": "contains", "Starts with": "starts_with", "Exact": "exact"}[mode]

    if not query.strip():
        st.info("Type a title or substring above to search.")
        return

    idxs = find_matches(query, records, index, mode=mode_key)
    if not idxs:
        st.warning(f"No matches for **{query}**. Try a shorter substring or switch to *Contains*.")
        return

    st.success(f"Found **{len(idxs)}** match(es).")

    if len(idxs) > 12:
        # Too many to render fully — show a brief table sorted by Total_Item_Requests
        st.info(
            f"{len(idxs)} results is a lot — showing a ranked summary. "
            "Narrow your search or switch to *Starts with* / *Exact* for full detail per record."
        )
        summary = []
        for i in idxs:
            rec = records[i]
            meta = rec["meta"]
            metrics = rec["metrics"]
            summary.append({
                "Title": meta.get("Title", ""),
                "Publisher": meta.get("Publisher", ""),
                "Platform": meta.get("Platform", ""),
                "Total Item Requests": metrics.get("Total_Item_Requests", {}).get("total", 0),
                "Unique Item Requests": metrics.get("Unique_Item_Requests", {}).get("total", 0),
                "Unique Title Requests": metrics.get("Unique_Title_Requests", {}).get("total", 0),
            })
        df = pd.DataFrame(summary).sort_values("Total Item Requests", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
        return

    for i in idxs:
        render_record(records[i], month_cols, key_prefix=f"single_r{i}")


# ============================================================
# TOP TITLES — auto-ranked report of most-used titles
# ============================================================

@st.cache_data(show_spinner=False)
def _top_titles_dataframe(_records_key: int, records: list[dict]) -> pd.DataFrame:
    """Flatten every record into a ranked DataFrame of metric totals + meta.

    Cached on the records-list length so switching tools doesn't recompute.
    """
    rows = []
    for rec in records:
        meta = rec["meta"]
        metrics = rec.get("metrics", {})
        row = {
            "Title": meta.get("Title", ""),
            "Publisher": meta.get("Publisher", ""),
            "Platform": meta.get("Platform", ""),
            "Data Type": meta.get("Data Type", ""),
            "ISBN": meta.get("ISBN", ""),
            "Print_ISSN": meta.get("Print_ISSN", ""),
            "Online_ISSN": meta.get("Online_ISSN", ""),
        }
        for m in PREFERRED_METRICS:
            row[m] = metrics.get(m, {}).get("total", 0)
        rows.append(row)
    return pd.DataFrame(rows)


def page_top_titles(records, month_cols, index):
    st.title("🏆 Top titles")
    st.markdown(
        "See the most-used titles in the report, ranked by any COUNTER metric. "
        "Filter by platform, publisher, or data type to focus on a specific slice."
    )

    df = _top_titles_dataframe(len(records), records)

    # ---- Controls
    with st.container(border=True):
        st.subheader("Filters and ranking")

        # Pick ranking metric — only offer metrics that actually have data
        available_metrics = [m for m in PREFERRED_METRICS if df[m].sum() > 0]
        if not available_metrics:
            st.warning("This report has no usage data in any of the standard COUNTER metrics.")
            return

        c1, c2 = st.columns([2, 1])
        with c1:
            metric_label = st.selectbox(
                "Rank by metric:",
                available_metrics,
                index=0,
                format_func=lambda m: m.replace("_", " "),
                key="top_metric",
                help=(
                    "**Total Item Requests** is the most inclusive measure (counts every "
                    "download, even repeats by the same user). **Unique Item Requests** "
                    "is the COUNTER 5 standard for comparing across platforms."
                ),
            )
        with c2:
            top_n = st.number_input(
                "Number of titles to show:",
                min_value=5,
                max_value=min(1000, len(df)),
                value=min(50, len(df)),
                step=5,
                key="top_n",
            )

        f1, f2, f3, f4 = st.columns(4)
        with f1:
            platforms = sorted(p for p in df["Platform"].unique() if p)
            sel_platforms = st.multiselect(
                "Platform",
                platforms,
                default=[],
                key="top_platform",
                placeholder="All platforms",
            )
        with f2:
            data_types = sorted(d for d in df["Data Type"].unique() if d)
            sel_data_types = st.multiselect(
                "Data type",
                data_types,
                default=[],
                key="top_data_type",
                placeholder="All data types",
            )
        with f3:
            publishers = sorted(p for p in df["Publisher"].unique() if p)
            sel_publishers = st.multiselect(
                "Publisher",
                publishers,
                default=[],
                key="top_publisher",
                placeholder="All publishers",
                help=f"{len(publishers)} publishers in this report.",
            )
        with f4:
            min_threshold = st.number_input(
                f"Min. {metric_label.replace('_', ' ')}",
                min_value=0,
                value=1,
                step=1,
                key="top_min",
                help="Exclude titles below this value. Default excludes zero-use titles.",
            )

    # ---- Apply filters
    filtered = df.copy()
    if sel_platforms:
        filtered = filtered[filtered["Platform"].isin(sel_platforms)]
    if sel_data_types:
        filtered = filtered[filtered["Data Type"].isin(sel_data_types)]
    if sel_publishers:
        filtered = filtered[filtered["Publisher"].isin(sel_publishers)]
    if min_threshold > 0:
        filtered = filtered[filtered[metric_label] >= min_threshold]

    filtered = filtered.sort_values(metric_label, ascending=False).reset_index(drop=True)

    # ---- Summary metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Titles in report", f"{len(df):,}")
    with mc2:
        st.metric("After filters", f"{len(filtered):,}")
    with mc3:
        total_use = int(filtered[metric_label].sum())
        st.metric(f"Total {metric_label.replace('_', ' ')}", f"{total_use:,}")
    with mc4:
        if len(filtered):
            top_share = filtered.head(int(top_n))[metric_label].sum()
            share_pct = (top_share / total_use * 100) if total_use else 0
            st.metric(
                f"Top {int(top_n)} share",
                f"{share_pct:.1f}%",
                help="Percentage of total usage (after filters) concentrated in the top N titles.",
            )

    if len(filtered) == 0:
        st.warning("No titles match your current filters.")
        return

    st.markdown("---")

    # ---- Ranked table + chart
    tab_chart, tab_table = st.tabs(["📊 Chart", "📄 Ranked table"])

    top = filtered.head(int(top_n)).copy()
    top.insert(0, "Rank", range(1, len(top) + 1))

    with tab_chart:
        # Horizontal bar chart for readability of long titles.
        # Truncate very long titles to keep the chart legible.
        chart_df = top.copy()
        chart_df["Display Title"] = chart_df["Title"].apply(
            lambda t: (t[:75] + "…") if len(t) > 78 else t
        )
        # Add platform suffix when multiple platforms present, for disambiguation
        if chart_df["Platform"].nunique() > 1:
            chart_df["Display Title"] = (
                chart_df["Display Title"] + "  [" + chart_df["Platform"] + "]"
            )

        # Reverse for horizontal bar so #1 appears at top
        chart_df = chart_df.iloc[::-1]
        fig_height = max(400, 22 * len(chart_df))
        fig = px.bar(
            chart_df,
            x=metric_label,
            y="Display Title",
            orientation="h",
            color_discrete_sequence=["#285C4D"],
            title=f"Top {len(top)} titles by {metric_label.replace('_', ' ')}",
            hover_data={
                "Publisher": True,
                "Platform": True,
                "Data Type": True,
                metric_label: True,
                "Display Title": False,
            },
        )
        fig.update_layout(
            height=fig_height,
            margin=dict(l=20, r=20, t=60, b=20),
            yaxis_title="",
            xaxis_title=metric_label.replace("_", " "),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key="top_titles_chart")

    with tab_table:
        # Reorder columns so ranking metric is prominent
        display_cols = (
            ["Rank", "Title", metric_label]
            + [m for m in PREFERRED_METRICS if m != metric_label and m in top.columns]
            + ["Publisher", "Platform", "Data Type", "ISBN", "Print_ISSN", "Online_ISSN"]
        )
        display_cols = [c for c in display_cols if c in top.columns]
        st.dataframe(
            top[display_cols],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # ---- Download
    dc1, dc2 = st.columns(2)
    with dc1:
        st.download_button(
            f"⬇️ Download top {len(top)} as CSV",
            data=top.to_csv(index=False).encode("utf-8"),
            file_name=f"top_titles_by_{metric_label.lower()}.csv",
            mime="text/csv",
        )
    with dc2:
        st.download_button(
            f"⬇️ Download full filtered list ({len(filtered):,} titles)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"all_filtered_titles_by_{metric_label.lower()}.csv",
            mime="text/csv",
        )


# ============================================================
# MAIN
# ============================================================

def main():
    records, month_cols, index = _sidebar()

    if records is None:
        with st.sidebar:
            st.markdown("---")
            st.caption("Upload a TR report to enable the lookup tools.")
        page_home()
        return

    with st.sidebar:
        st.markdown("---")
        st.subheader("2. Choose a tool")
        page = st.radio(
            "Tool:",
            ["🏠 Home", "🏆 Top titles", "📋 Batch lookup", "🔍 Single title search"],
            index=1,  # default to Top titles — it's the zero-input overview
            key="nav",
            label_visibility="collapsed",
        )

    if page == "🏠 Home":
        page_home()
    elif page == "🏆 Top titles":
        page_top_titles(records, month_cols, index)
    elif page == "📋 Batch lookup":
        page_batch_lookup(records, month_cols, index)
    else:
        page_single_search(records, month_cols, index)

    # Footer
    st.markdown("---")
    st.caption(
        "Howard-Tilton Memorial Library · Tulane University · "
        "Reads COUNTER 5 Title Reports (TR). "
        "Groups titles by Title + ISBN + ISSN + Platform."
    )


if __name__ == "__main__":
    main()
