import os
from typing import List

import math
import numpy as np
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI
from psycopg2.extras import RealDictCursor
import streamlit as st

load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CANONICAL_PATH = "canonical_top_k400.txt"  # hardcoded to k400 canonical set
LLM_API_KEY = os.getenv("LLM_API_KEY")


@st.cache_resource(show_spinner=False)
def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("Set DATABASE_URL for the Postgres connection.")
    return psycopg2.connect(DATABASE_URL, sslmode="require")


@st.cache_resource(show_spinner=False)
def get_embedder():
    if not LLM_API_KEY:
        raise RuntimeError("Set LLM_API_KEY for embeddings.")
    client = OpenAI(api_key=LLM_API_KEY)
    return client


@st.cache_resource(show_spinner=False)
def load_canonical_tags() -> List[str]:
    path = CANONICAL_PATH
    tags: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if "\t" in line:
                    tag = line.split("\t", 1)[0].strip()
                else:
                    tag = line.strip()
                if tag:
                    tags.append(tag)
    except FileNotFoundError:
        raise RuntimeError(f"Canonical tag file not found: {path}")
    return tags


@st.cache_resource(show_spinner=False)
def embed_canonicals():
    tags = load_canonical_tags()
    client = get_embedder()
    resp = client.embeddings.create(model=EMBED_MODEL, input=tags)
    vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return tags, vectors


def map_query_to_tags(query: str) -> List[str]:
    tokens = [t.strip() for t in query.split(",") if t.strip()]
    if not tokens:
        return []
    tags, canon_vectors = embed_canonicals()
    client = get_embedder()
    resp = client.embeddings.create(model=EMBED_MODEL, input=tokens)
    token_vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
    norms = np.linalg.norm(canon_vectors, axis=1) + 1e-8

    chosen: List[str] = []
    seen = set()
    for vec in token_vectors:
        qnorm = np.linalg.norm(vec) + 1e-8
        sims = (canon_vectors @ vec) / (norms * qnorm)
        idx = int(sims.argmax())
        tag = tags[idx]
        if tag not in seen:
            seen.add(tag)
            chosen.append(tag)
    return chosen


@st.cache_data(show_spinner=False)
def load_languages() -> List[str]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT language FROM movies WHERE language IS NOT NULL ORDER BY language;")
        return [row[0] for row in cur.fetchall()]


LANG_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "pt": "Portuguese",
    "ru": "Russian",
    "hi": "Hindi",
}


def fetch_movies(language: str, tags: List[str], limit: int = 50):
    clauses = []
    params = []

    has_tags = bool(tags)

    match_expr = "0"
    if has_tags:
        match_expr = (
            "cardinality((SELECT array(SELECT unnest(keywords_topics) INTERSECT SELECT unnest(%s::text[]))))"
        )
        params.append(tags)  # match_expr param first

    if language and language != "Any":
        clauses.append("language = %s")
        params.append(language)

    if has_tags:
        clauses.append("keywords_topics && %s::text[]")
        params.append(tags)

    where = "WHERE " + " AND ".join(clauses) if clauses else ""

    sql = f"""
        SELECT title, release_date, rating, runtime, popularity, genres, poster_path, keywords_topics,
               {match_expr} AS match_count
        FROM movies
        {where}
        ORDER BY match_count DESC, popularity DESC NULLS LAST
        LIMIT %s;
    """
    params.append(limit)

    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


@st.cache_data(show_spinner=False)
def fetch_featured(limit: int = 8):
    sql = """
        SELECT title, release_date, rating, runtime, popularity, genres, poster_path, keywords_topics
        FROM movies
        ORDER BY popularity DESC NULLS LAST, release_date DESC NULLS LAST
        LIMIT %s;
    """
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (limit,))
        return cur.fetchall()


def main():

    st.markdown(
        """
        <style>
        .stApp {background: linear-gradient(135deg, #0f172a 0%, #0b3b5a 40%, #0b172a 100%); color: #e2e8f0;}
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        div[data-testid="stToolbar"], div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] {display: none !important;}
        .hero {background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 16px 18px; margin-bottom: 16px;}
        .hero h1 {margin: 0 0 6px 0; color: #e2e8f0;}
        .hero p {margin: 0; color: #cbd5e1; font-size: 14px;}
        .hero-card {background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 16px 18px; margin-bottom: 16px;}
        .hero-card h1 {margin: 0 0 8px 0; color: #e2e8f0;}
        .hero-card p {margin: 0 0 10px 0; color: #cbd5e1; font-size: 14px;}
        label {color: #f8fafc !important;}
        .input-card {background: #f8fafc; border-radius: 14px; padding: 14px 16px; border: 1px solid #e2e8f0; color: #0f172a;}
        .input-card label {color: #f8fafc !important; font-weight: 600;}
        .stTextArea textarea {background: #f1f5f9; border-radius: 10px; border: 1px solid #cbd5e1;}
        .stSelectbox [data-baseweb="select"] {background: #f1f5f9; border-radius: 10px; border: 1px solid #cbd5e1;}
        .stButton>button {background: linear-gradient(135deg, #22c55e, #16a34a); color: white; border: none; border-radius: 10px; padding: 0.6rem 1.1rem; font-weight: 600;}
        .stButton>button:hover {transform: translateY(-1px); box-shadow: 0 8px 18px rgba(22,163,74,0.25);}
        .result-card {border:1px solid #e5e7eb;border-radius:12px;padding:12px 14px;margin-bottom:12px;background:#f9fafb;}
        .result-card strong {color: #0f1222;}
        .pill {display:inline-block;padding:2px 10px;border-radius:999px;background:#e0f2fe;color:#0f172a;margin-right:6px;margin-bottom:4px;font-size:12px;}
        .meta {color:#64748b;font-size:13px;margin:4px 0;}
        .matched {color:#0ea5e9;font-weight:600;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    has_search = st.session_state.get("has_search", False)
    # Hero with input and action
    st.markdown(
        """
        <div class="hero-card">
          <h1>Find A Movie</h1>
          <p>Describe vibes or themes, and we’ll match them to curated, tagged films.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
     # Language selector below featured
    codes = load_languages()
    lang_options = ["Any"] + [f"{LANG_NAMES.get(code, code)} ({code})" for code in codes]
    default_idx = 0
    if "English (en)" in lang_options:
        default_idx = lang_options.index("English (en)")
    language_display = st.selectbox("Language", options=lang_options, index=default_idx)
    language = None if language_display == "Any" else language_display.split("(")[-1].rstrip(")")
    st.markdown(
        "Try keywords like: `robots`, `war`, `aliens`, `superheroes`, `time travel`, `apocalypse`, `heist`."
    )
    overview_query = st.text_area(
        "Describe what you want using keywords separated by commas",
        placeholder="dystopian, robot, war",
        height=80,
        help="Comma-separated keywords; examples above.",
    )
    find_clicked = st.button("Find movies")
    if find_clicked:
        has_search = True
        st.session_state["has_search"] = True


    # Featured strip under hero
    featured = []
    if not has_search:
        with st.spinner("Loading featured..."):
            featured = fetch_featured(limit=8)
    if not has_search and featured:
        st.markdown("### Featured")
        IMG_BASE = "https://image.tmdb.org/t/p/w342"
        cols = st.columns(4)
        for idx, row in enumerate(featured):
            col = cols[idx % 4]
            with col:
                if row.get("poster_path"):
                    st.image(IMG_BASE + row["poster_path"], width=150)
                title_text = row.get("title", "") or ""
                meta_parts = []
                if row.get("release_date"):
                    meta_parts.append(str(row["release_date"]))
                if row.get("rating"):
                    meta_parts.append(row["rating"])
                if row.get("runtime"):
                    meta_parts.append(f"{row['runtime']} min")
                meta = " • ".join(meta_parts)
                genres = ", ".join(row.get("genres") or [])
                st.markdown(
                    f"<div class='result-card'><strong>{title_text}</strong>"
                    f"<div class='meta'>{meta}</div>"
                    f"<div class='meta'>{genres}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                if row.get("keywords_topics"):
                    pills = " ".join(f"<span class='pill'>{t}</span>" for t in row["keywords_topics"][:5])
                    st.markdown(pills, unsafe_allow_html=True)

   

    results = []
    chosen_tags: List[str] = []
    if find_clicked:
        st.session_state["has_search"] = True
        with st.spinner("Finding movies..."):
            chosen_tags = map_query_to_tags(overview_query)
            results = fetch_movies(language, chosen_tags, limit=50)

    if chosen_tags:
        st.write("Matched tags:")
        st.write(" ".join([f"`{t}`" for t in chosen_tags]))

    IMG_BASE = "https://image.tmdb.org/t/p/w342"

    def match_label(match_count: int, tag_count: int):
        if tag_count <= 0:
            return ""
        if match_count <= 0:
            return "No Match"
        high = max(1, math.floor(0.8 * tag_count))
        moderate = max(1, math.floor(0.6 * tag_count))
        low = max(1, math.floor(0.4 * tag_count))
        very_low = max(1, math.floor(0.2 * tag_count))
        if match_count >= tag_count:
            return "Very High Match"
        if match_count >= high:
            return "High Match"
        if match_count >= moderate:
            return "Moderate Match"
        if match_count >= low:
            return "Low Match"
        if match_count >= very_low:
            return "Very Low Match"
        return "No Match"

    tag_count = len(chosen_tags)

    if find_clicked:
        st.write(f"Showing {len(results)} result(s).")

        # Group results by match label and render sections
        grouped = {}
        for row in results:
            label = match_label(int(row.get("match_count") or 0), tag_count)
            grouped.setdefault(label, []).append(row)

        label_order = ["Very High Match", "High Match", "Moderate Match", "Low Match", "Very Low Match", "No Match"]
        for label in label_order:
            bucket = grouped.get(label, [])
            if not bucket:
                continue
            st.markdown(f"### {label} ({len(bucket)})")
            for row in bucket:
                with st.container():
                    cols = st.columns([1, 3])
                    if row.get("poster_path"):
                        cols[0].image(IMG_BASE + row["poster_path"], width=140)
                    title_text = row.get("title", "") or ""
                    meta_parts = []
                    if row.get("release_date"):
                        meta_parts.append(str(row["release_date"]))
                    if row.get("rating"):
                        meta_parts.append(row["rating"])
                    if row.get("runtime"):
                        meta_parts.append(f"{row['runtime']} min")
                    meta = " • ".join(meta_parts)
                    genres = ", ".join(row.get("genres") or [])
                    matched = ", ".join(
                        sorted(set(row.get("keywords_topics") or []) & set(chosen_tags))
                    )
                    cols[1].markdown(
                        f"<div class='result-card'><strong>{title_text}</strong>"
                        f"<div class='meta'>{meta}</div>"
                        f"<div class='meta'>{genres}</div>"
                        + (f"<div class='meta matched'>Matches: {matched}</div>" if matched else "")
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                    if row.get("keywords_topics"):
                        pills = " ".join(f"<span class='pill'>{t}</span>" for t in row["keywords_topics"][:10])
                        cols[1].markdown(pills, unsafe_allow_html=True)



if __name__ == "__main__":
    main()
