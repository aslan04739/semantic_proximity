import streamlit as st
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import requests
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
PAGE_TITLE = "SEO Semantic Auditor (URL vs URL)"
MODEL_NAME = "all-mpnet-base-v2"

st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# --- 1. SESSION STATE ---
# Store fetched text so it doesn't vanish when you click other buttons
if 'my_text' not in st.session_state:
    st.session_state['my_text'] = ""
if 'competitor_text' not in st.session_state:
    st.session_state['competitor_text'] = ""

# --- 2. LOAD MODELS (Cached) ---
@st.cache_resource
def load_models():
    sentence_model = SentenceTransformer(MODEL_NAME)
    kw_model = KeyBERT(model=sentence_model)
    return sentence_model, kw_model

with st.spinner(f"Loading AI Brains..."):
    model, kw_model = load_models()

# --- 3. HELPER FUNCTIONS ---
def fetch_url_content(url):
    """Fetches clean text from a URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Grab only significant text (paragraphs and headers)
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
        # Filter out short snippets (like menu items)
        extracted_text = " ".join([elem.get_text().strip() for elem in text_elements if len(elem.get_text().strip()) > 25])
        return extracted_text
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return None

def extract_keywords(text, top_n=20):
    if not text: return []
    return kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)

# --- 4. THE INTERFACE ---
st.title("⚔️ " + PAGE_TITLE)
st.markdown("Compare your page directly against a competitor to find semantic gaps.")

# --- TOP: KEYWORD INPUT ---
target_keyword = st.text_input("🎯 Target Keyword / Topic", placeholder="e.g. Enterprise SEO Services")

st.divider()

# --- MIDDLE: URL INPUTS (2 Columns) ---
col_comp, col_mine = st.columns(2)

# LEFT: COMPETITOR
with col_comp:
    st.subheader("🔴 The Competitor")
    comp_url = st.text_input("Competitor URL", placeholder="https://them.com/blog")
    if st.button("Fetch Competitor", type="secondary", key="fetch_competitor"):
        if comp_url:
            with st.spinner("Fetching competitor..."):
                text = fetch_url_content(comp_url)
                if text:
                    st.session_state['competitor_text'] = text
                    st.success(f"Fetched {len(text)} characters.")
                    st.rerun()
    
    # Show preview if fetched
    if st.session_state['competitor_text']:
        with st.expander("View Competitor Text"):
            st.write(st.session_state['competitor_text'][:1000] + "...")

# RIGHT: MY PAGE
with col_mine:
    st.subheader("🟢 My Page")
    my_url = st.text_input("My URL", placeholder="https://me.com/blog")
    if st.button("Fetch My Page", type="secondary", key="fetch_my_page"):
        if my_url:
            with st.spinner("Fetching my page..."):
                text = fetch_url_content(my_url)
                if text:
                    st.session_state['my_text'] = text
                    st.success(f"Fetched {len(text)} characters.")
                    st.rerun()

    # Editable Text Area (User can tweak fetched text)
    if st.session_state['my_text']:
        with st.expander("View My Page Text"):
            st.write(st.session_state['my_text'][:1000] + "...")
    
    my_content_final = st.text_area(
        "Content for Analysis (Editable)", 
        value=st.session_state['my_text'],
        height=200,
        placeholder="Paste your content here or fetch from URL above..."
    )

# --- BOTTOM: ACTION & RESULTS ---
st.divider()
analyze_btn = st.button("🚀 Analyze Semantic Gap", type="primary", use_container_width=True)

if analyze_btn:
    if not target_keyword:
        st.warning("⚠️ Please provide a Target Keyword.")
    elif not my_content_final or len(my_content_final.strip()) < 50:
        st.warning("⚠️ Please fetch or paste 'My Page' content (at least 50 characters).")
    else:
        # 1. SCORING
        with st.spinner("🔍 Analyzing semantic similarity..."):
            emb_kw = model.encode(target_keyword)
            emb_my = model.encode(my_content_final)
            score = float(model.similarity(emb_kw, emb_my)[0][0])
        
        # 2. DISPLAY SCORE
        st.markdown("### 📊 Semantic Proximity Score")
        col_score, col_msg = st.columns([1, 3])
        
        with col_score:
            st.metric("Score", f"{score:.4f}")
        
        with col_msg:
            st.progress(score)
            if score > 0.60:
                st.success("Strong relevance to the keyword.")
            elif score > 0.40:
                st.warning("Moderate relevance. Needs more depth.")
            else:
                st.error("Low relevance. Content may be off-topic.")

        # 3. GAP ANALYSIS (Only if competitor text exists)
        if st.session_state['competitor_text']:
            st.divider()
            st.markdown("### 🕵️‍♀️ Keyword Gap Analysis")
            
            with st.spinner("Extracting entities from both pages..."):
                # Extract Top Keywords
                comp_kws = [kw[0] for kw in extract_keywords(st.session_state['competitor_text'], top_n=25)]
                my_kws = [kw[0] for kw in extract_keywords(my_content_final, top_n=25)]
                
                # Find Missing
                # Logic: Words in Competitor list that are NOT in my text at all
                missing_kws = [kw for kw in comp_kws if kw.lower() not in my_content_final.lower()]
                
                col_gap1, col_gap2 = st.columns(2)
                
                with col_gap1:
                    st.error(f"MISSING TOPICS ({len(missing_kws)})")
                    st.caption("Competitor uses these, but you don't:")
                    if missing_kws:
                        # Display as bullet points for readability
                        for kw in missing_kws:
                            st.markdown(f"- **{kw}**")
                    else:
                        st.success("No major semantic gaps found!")

                with col_gap2:
                    st.info(f"SHARED TOPICS")
                    st.caption("You both cover these concepts:")
                    # Intersection
                    shared = [kw for kw in comp_kws if kw.lower() in my_content_final.lower()]
                    st.write(", ".join(shared))
        
        else:
            st.info("💡 Tip: Fetch a Competitor URL to see Gap Analysis.")