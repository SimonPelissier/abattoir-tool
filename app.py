import streamlit as st
import requests
import json
import re
import time
import io
import os
import tempfile
import pandas as pd
import fitz
from bs4 import BeautifulSoup
from google import genai

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="Abattoir Intelligence Tool",
    page_icon="🥩",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }
  .stApp {
    background-color: #0f0f0f;
    color: #e8e8e8;
  }
  h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #f0f0f0 !important;
  }
  .block-container {
    padding-top: 2rem;
    max-width: 1100px;
  }
  .header-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #c8a96e;
    margin-bottom: 4px;
  }
  .main-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #f5f5f5;
    line-height: 1.1;
    margin-bottom: 0.3rem;
  }
  .subtitle {
    font-size: 0.95rem;
    color: #888;
    margin-bottom: 2rem;
  }
  .step-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #0f0f0f;
    background: #c8a96e;
    padding: 2px 8px;
    border-radius: 2px;
    margin-bottom: 8px;
  }
  .metric-box {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 16px;
    text-align: center;
  }
  .metric-number {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #c8a96e;
  }
  .metric-label {
    font-size: 0.75rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }
  .log-box {
    background: #111;
    border: 1px solid #222;
    border-radius: 4px;
    padding: 12px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #aaa;
    max-height: 200px;
    overflow-y: auto;
  }
  .stButton > button {
    background: #c8a96e !important;
    color: #0f0f0f !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 10px 28px !important;
    width: 100% !important;
  }
  .stButton > button:hover {
    background: #d4b87a !important;
  }
  .stTextInput > div > div > input,
  .stSelectbox > div > div {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: #e8e8e8 !important;
    border-radius: 4px !important;
  }
  .stDataFrame {
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
  }
  .warning-box {
    background: #1a1500;
    border: 1px solid #c8a96e44;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 12px;
    color: #c8a96e;
    font-family: 'IBM Plex Mono', monospace;
  }
  .divider {
    border: none;
    border-top: 1px solid #222;
    margin: 24px 0;
  }
</style>
""", unsafe_allow_html=True)


# ── API Setup ──────────────────────────────────────────
@st.cache_resource
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    return genai.Client(api_key=api_key)

def get_serpapi_key():
    return st.secrets.get("SERPAPI_API_KEY", os.environ.get("SERPAPI_API_KEY", ""))

GEMINI_MODEL = "gemini-2.5-flash-lite"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.google.com/'
}

EXCLUDED_DOMAINS = [
    'yelp.com', 'maps.apple.com', 'mapquest.com', 'facebook.com',
    'instagram.com', 'twitter.com', 'linkedin.com', 'tripadvisor.com'
]


# ── Helper : chunk text ────────────────────────────────
def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 100) -> list:
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        cutoff = text.rfind('\n', start, end)
        if cutoff <= start:
            cutoff = end
        chunks.append(text[start:cutoff])
        start = cutoff - overlap
    return chunks


# ── Extractors ─────────────────────────────────────────
def extract_html(url: str, max_chars: int = 6000) -> list:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        lines = [l for l in soup.get_text('\n', strip=True).split('\n') if l.strip()]
        return chunk_text('\n'.join(lines)[:max_chars])
    except Exception as e:
        return []

def extract_pdf_bytes(content: bytes, max_pages: int = 15, max_chars: int = 15000) -> list:
    try:
        pdf  = fitz.open(stream=content, filetype='pdf')
        text = ''
        for i in range(min(len(pdf), max_pages)):
            page_text = pdf[i].get_text()
            if page_text.strip():
                text += f'\n--- Page {i+1} ---\n{page_text}'
        pdf.close()
        return chunk_text(text[:max_chars])
    except Exception:
        return []

def extract_pdf_url(url: str, max_pages: int = 15, max_chars: int = 15000) -> list:
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 403:
            r = requests.get(url, timeout=30)
        r.raise_for_status()
        return extract_pdf_bytes(r.content, max_pages, max_chars)
    except Exception:
        return []

def extract_content(url: str, local_bytes: bytes = None) -> tuple:
    if local_bytes:
        chunks = extract_pdf_bytes(local_bytes)
        return chunks, 'PDF'
    url_lower = url.lower()
    if url_lower.endswith('.pdf') or '/pdf/' in url_lower:
        return extract_pdf_url(url), 'PDF'
    chunks = extract_html(url)
    if not chunks:
        chunks = extract_pdf_url(url)
        return chunks, 'PDF'
    return chunks, 'HTML'


# ── Prompts ─────────────────────────────────────────────
QUERY_PROMPT = """
You are a specialist researcher in the global beef meatpacking industry.

Generate EXACTLY 2 search queries about SLAUGHTERHOUSES operated
by {company} in {country}. A slaughterhouse = facility where LIVE ANIMALS ARE KILLED.

Query 1 — Target the OFFICIAL COMPANY WEBSITE or annual report:
  Look for locations/plants page, SEC filing, sustainability report.
  Keywords: "slaughterhouse", "abattoir", "harvest facility", "rastro TIF"

Query 2 — Target a GOVERNMENT AGRICULTURAL REGISTER:
  - Mexico    → SENASICA directorio TIF rastros
  - USA       → USDA FSIS slaughter establishment list
  - Australia → DAFF export registered abattoir
  - France    → DGAL abattoirs agrees CE
  - Japan     → NLBC と畜場リスト
  - Other     → [country] official slaughterhouse register government

Return ONLY a valid JSON array of 2 strings.
"""

SCORING_PROMPT = """
You receive a list of URLs about slaughterhouses operated by {company}.
Select the MOST RELEVANT links (max 6) for finding slaughterhouse data.

SCORING (0-10):
  10 : Official government register (SENASICA, FSIS, DAFF, DGAL...)
  8-9: SEC/annual report listing plants
  7-8: Company website with plant/locations page
  5-6: Trade press with specific plant data
  0-4: Irrelevant

Only keep links with score >= 5.
Keep at least 1 government source, 1 company source.

Return ONLY a JSON array with: url, title, score, source_type (government/company/media/other), expected_format (PDF/HTML/CSV).
No markdown.
"""

EXTRACTION_PROMPT = """
You are a specialist in global beef slaughter industry data extraction.
Company: {company} | Source format: {fmt}

Extract every SLAUGHTERHOUSE (facility where LIVE ANIMALS ARE KILLED).

FIELD DEFINITIONS:
- facility_name     : name of the facility (NOT address, NOT number)
- address           : street address only
- city              : city name only
- country           : country name only
- establishment_number : official number ONLY (TIF/FSIS/DAFF) — NOT capacity
- capacity_day      : integer, animals per DAY only (null otherwise)
- capacity_week     : integer, animals per WEEK only (null otherwise)
- capacity_year     : integer, animals per YEAR only (null otherwise)
- capacity_reference_year : year the capacity refers to (e.g. 2023)

EXCLUDE: processing factories, offices, laboratories, cold storage, retail.

Return ONLY valid JSON:
{{
  "slaughterhouses": [{{
    "facility_name": "string or null",
    "address": "string or null",
    "city": "string or null",
    "country": "string or null",
    "species": ["cattle","pigs","sheep","poultry"],
    "capacity_day": null,
    "capacity_week": null,
    "capacity_year": null,
    "capacity_reference_year": null,
    "operational_status": "active" or "closed" or "unknown",
    "establishment_number": "string or null",
    "export_certified": true or false,
    "confidence_score": 0.0-1.0
  }}],
  "excluded": [{{"facility_name": "string", "reason": "string"}}],
  "source_quality": "high" or "medium" or "low"
}}
No markdown. No explanation outside JSON.
"""


# ── Gemini helpers ─────────────────────────────────────
def gemini_call(client, prompt: str) -> str:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={'temperature': 0.0}
    )
    return response.text.strip()

def parse_json_response(raw: str, array: bool = False) -> any:
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'^```\s*', '', raw)
    raw = re.sub(r'```\s*$', '', raw).strip()
    pattern = r'\[.*\]' if array else r'\{.*\}'
    match = re.search(pattern, raw, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(raw)


# ── Pipeline ───────────────────────────────────────────
def run_pipeline(company, country, gl, hl, location,
                 uploaded_files, log_container, progress_bar):
    client      = get_gemini_client()
    serpapi_key = get_serpapi_key()
    logs        = []

    def log(msg):
        logs.append(msg)
        log_container.markdown(
            '<div class="log-box">' +
            '<br>'.join(logs[-15:]) +
            '</div>', unsafe_allow_html=True
        )

    # ── Step 1 : Generate queries ──────────────────────
    log("🧠 [1/4] Generating search queries...")
    progress_bar.progress(10)

    query_content = QUERY_PROMPT.replace('{company}', company).replace('{country}', country)
    queries = parse_json_response(gemini_call(client, query_content), array=True)
    for i, q in enumerate(queries, 1):
        log(f"   Query {i}: {q[:70]}")

    # ── Step 2 : SerpAPI ──────────────────────────────
    log(f"\n🔍 [2/4] Fetching links via SerpAPI...")
    progress_bar.progress(25)

    all_results, seen_urls = [], set()
    for query in queries:
        resp    = requests.get('https://serpapi.com/search', params={
            'q': query, 'api_key': serpapi_key,
            'num': 3, 'gl': gl, 'hl': hl, 'location': location, 'filter': '0'
        })
        results = resp.json().get('organic_results', [])[:3]
        for r in results:
            url = r.get('link', '')
            if url not in seen_urls and not any(d in url for d in EXCLUDED_DOMAINS):
                seen_urls.add(url)
                all_results.append({
                    'title': r.get('title', ''), 'url': url,
                    'snippet': r.get('snippet', '')
                })
        time.sleep(1)
    log(f"   {len(all_results)} unique links collected")

    # ── Step 3 : Score links ──────────────────────────
    log(f"\n⭐ [3/4] Scoring {len(all_results)} links...")
    progress_bar.progress(45)

    links_input   = [{'title': r['title'][:80], 'url': r['url'],
                      'snippet': r['snippet'][:100]} for r in all_results]
    scoring_content = (
        SCORING_PROMPT.replace('{company}', company)
        + f"\n\nCompany: {company} | Country: {country}\n"
        + f"Links: {json.dumps(links_input, ensure_ascii=False)}"
    )
    scored = parse_json_response(gemini_call(client, scoring_content), array=True)
    test_links = [l for l in scored if l.get('score', 0) >= 5]

    # Ajoute les PDFs uploadés manuellement
    for uf in uploaded_files:
        test_links.append({
            'url': f"[uploaded] {uf.name}", 'title': uf.name,
            'source_type': 'government', 'score': 9.0,
            'expected_format': 'PDF', '_bytes': uf.read()
        })

    log(f"   {len(test_links)} relevant links selected")
    for l in test_links:
        icon = {'government':'🏛️','company':'🏢','media':'📰'}.get(l.get('source_type',''),'🔗')
        log(f"   {icon} [{l.get('score',0)}/10] {l.get('title','')[:55]}")

    # ── Step 4 : Extract + Gemini structure ───────────
    log(f"\n📥 [4/4] Extracting & structuring data...")
    progress_bar.progress(60)

    all_abattoirs, seen_keys, all_exclusions = [], set(), []

    for i, link in enumerate(test_links):
        url        = link['url']
        title      = link.get('title', '')
        local_bytes= link.get('_bytes', None)
        fmt        = link.get('expected_format', 'HTML')

        log(f"   [{i+1}/{len(test_links)}] {title[:50]}...")

        chunks, actual_fmt = extract_content(url, local_bytes)
        if not chunks:
            log(f"   ⚠️  Empty content — skipped")
            continue

        for j, chunk in enumerate(chunks):
            chunk_item = {'text': chunk, 'format': actual_fmt, 'url': url}
            prompt = (EXTRACTION_PROMPT
                      .replace('{company}', company)
                      .replace('{fmt}', actual_fmt)
                      + f"\n\nSOURCE TEXT:\n{chunk}")
            try:
                raw    = gemini_call(client, prompt)
                result = parse_json_response(raw, array=False)
                found  = result.get('slaughterhouses', [])
                excl   = result.get('excluded', [])

                all_exclusions.extend([{
                    'facility': (e.get('facility_name') if isinstance(e, dict) else str(e)),
                    'reason'  : (e.get('reason', '') if isinstance(e, dict) else ''),
                    'source'  : url[:60]
                } for e in excl])

                for a in found:
                    key = (
                        (a.get('facility_name') or '').lower().strip(),
                        (a.get('city') or '').lower().strip()
                    )
                    if key not in seen_keys and key != ('', ''):
                        seen_keys.add(key)
                        a['source_url'] = url
                        all_abattoirs.append(a)
                        log(f"   ✅ Found: {a.get('facility_name','?')} — {a.get('city','?')}")

            except Exception as e:
                log(f"   ❌ Error: {str(e)[:60]}")

            if j < len(chunks) - 1:
                time.sleep(20)

        if i < len(test_links) - 1:
            time.sleep(20)

    progress_bar.progress(100)
    log(f"\n✅ Done — {len(all_abattoirs)} unique slaughterhouses found")

    return all_abattoirs, all_exclusions


# ── Build results dataframe ────────────────────────────
def build_dataframe(abattoirs: list) -> pd.DataFrame:
    rows = []
    for a in sorted(abattoirs, key=lambda x: x.get('confidence_score', 0), reverse=True):
        cap_val  = a.get('capacity_day') or a.get('capacity_week') or a.get('capacity_year')
        cap_unit = ('cap/day'  if a.get('capacity_day')  else
                    'cap/week' if a.get('capacity_week') else
                    'cap/year' if a.get('capacity_year') else '')
        rows.append({
            'Est.#'       : a.get('establishment_number', ''),
            'Address'     : a.get('address', ''),
            'City'        : a.get('city', ''),
            'Country'     : a.get('country', ''),
            'Species'     : ', '.join(a.get('species', [])),
            'Capacity'    : cap_val or '',
            'Unit'        : cap_unit,
            'Cap.Year'    : a.get('capacity_reference_year', ''),
            'Status'      : a.get('operational_status', ''),
            'Export'      : '✅' if a.get('export_certified') else '?',
            'Source'      : a.get('source_url', '')[:60]
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════

# Header
st.markdown('<div class="header-tag">Research Tool · Oxford 2026</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">🥩 Abattoir Intelligence Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Automated slaughterhouse data extraction from public sources</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Input form ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="step-badge">COMPANY</div>', unsafe_allow_html=True)
    company = st.text_input("Company name", placeholder="e.g. SuKarne", label_visibility="collapsed")

    st.markdown('<div class="step-badge" style="margin-top:16px">COUNTRY</div>', unsafe_allow_html=True)
    country = st.text_input("Country", placeholder="e.g. Mexico", label_visibility="collapsed")

with col2:
    st.markdown('<div class="step-badge">SEARCH SETTINGS</div>', unsafe_allow_html=True)

    col2a, col2b = st.columns(2)
    with col2a:
        gl = st.selectbox("Google country (gl)", ['mx','us','au','fr','jp','br','de','gb'], label_visibility="visible")
    with col2b:
        hl = st.selectbox("Language (hl)", ['es','en','fr','ja','de','pt'], label_visibility="visible")

    location = st.text_input("SerpAPI location", placeholder="e.g. Mexico", label_visibility="visible")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── PDF Upload ─────────────────────────────────────────
st.markdown('<div class="step-badge">OPTIONAL — MANUAL PDF UPLOAD</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size:12px;color:#666;margin-bottom:8px">If a government PDF returns 403, download and upload it here</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Upload PDFs", type=['pdf'],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Run button ─────────────────────────────────────────
run = st.button("🔍  RUN PIPELINE", use_container_width=True)

# ── Results ────────────────────────────────────────────
if run:
    if not company or not country:
        st.error("Please fill in Company name and Country.")
    else:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Progress
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        metric_queries   = col_m1.empty()
        metric_links     = col_m2.empty()
        metric_sources   = col_m3.empty()
        metric_abattoirs = col_m4.empty()

        def render_metric(container, number, label):
            container.markdown(f"""
            <div class="metric-box">
              <div class="metric-number">{number}</div>
              <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        render_metric(metric_queries,   "—", "Queries")
        render_metric(metric_links,     "—", "Links found")
        render_metric(metric_sources,   "—", "Sources extracted")
        render_metric(metric_abattoirs, "—", "Slaughterhouses")

        st.markdown("**Pipeline log**")
        log_container = st.empty()
        progress_bar  = st.progress(0)

        # Run
        with st.spinner(""):
            abattoirs, exclusions = run_pipeline(
                company, country, gl, hl, location,
                uploaded_files or [],
                log_container, progress_bar
            )

        render_metric(metric_abattoirs, len(abattoirs), "Slaughterhouses")

        # ── Results table ──────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(f"### 📊 Results — {company}")

        if abattoirs:
            df = build_dataframe(abattoirs)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"abattoirs_{company.replace(' ','_')}.csv",
                mime='text/csv'
            )
        else:
            st.markdown('<div class="warning-box">⚠️  No slaughterhouses found — try adding a PDF manually or adjusting search settings.</div>', unsafe_allow_html=True)

        # ── Exclusions ─────────────────────────────────
        if exclusions:
            with st.expander(f"⛔ Excluded facilities ({len(exclusions)})"):
                df_excl = pd.DataFrame(exclusions)
                st.dataframe(df_excl, use_container_width=True, hide_index=True)
