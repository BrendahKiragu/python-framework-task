"""
CORD-19 Data Explorer - script.py

Streamlit app that:
- Loads metadata.csv (from working folder or via file upload)
- Cleans basic fields (publish_time, abstracts, titles)
- Generates simple analyses and visualizations:
  - Publications over time
  - Top publishing journals
  - Word frequency from titles and a word cloud
- Allows interactive filtering by year and top-N controls
- Lets the user download the filtered CSV

Run:
    pip install -r requirements.txt
    streamlit run script.py

Author: Brenda (starter template)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from io import StringIO

# WordCloud requires pillow; included in requirements.txt
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

st.set_page_config(page_title="CORD-19 Data Explorer", layout="wide")

# Basic stopwords for title word frequency
STOPWORDS = {
    "the", "and", "of", "in", "to", "for", "with", "on", "a", "an", "by",
    "from", "is", "are", "using", "use", "as", "be", "we", "this", "that",
    "our", "at", "towards", "based", "analysis", "study", "studies", "using",
    "case", "cases", "can", "new", "research", "effects", "effect", "covid",
    "sarscov2", "sars", "covid19", "19"
}

@st.cache_data
def load_data(uploaded_file, nrows=None):
    """
    Load the metadata CSV. If uploaded_file is not None, read from it.
    Otherwise try to read 'metadata.csv' in working directory.
    Returns a pandas DataFrame or None on failure.
    """
    try:
        if uploaded_file is not None:
            # uploaded_file is a BytesIO / UploadedFile from streamlit
            df = pd.read_csv(uploaded_file, nrows=nrows, low_memory=False)
        else:
            df = pd.read_csv("metadata.csv", nrows=nrows, low_memory=False)
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

def clean_dataframe(df):
    """
    Make a copy and perform light cleaning:
    - Ensure title and abstract exist and are strings
    - Parse publish_time to datetime and extract year
    - Compute title and abstract word counts
    """
    df = df.copy()
    # Ensure we have the commonly used columns
    if "title" not in df.columns:
        df["title"] = ""
    else:
        df["title"] = df["title"].fillna("").astype(str)

    if "abstract" not in df.columns:
        df["abstract"] = ""
    else:
        df["abstract"] = df["abstract"].fillna("").astype(str)

    # Publish time parsing
    if "publish_time" in df.columns:
        df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
        df["year"] = df["publish_time"].dt.year
    else:
        df["publish_time"] = pd.NaT
        df["year"] = pd.NA

    # Journal cleanup
    if "journal" not in df.columns:
        df["journal"] = "Unknown"
    else:
        df["journal"] = df["journal"].fillna("Unknown").astype(str)

    # Authors
    if "authors" not in df.columns:
        df["authors"] = ""
    else:
        df["authors"] = df["authors"].fillna("").astype(str)

    # Word counts
    df["title_word_count"] = df["title"].apply(lambda x: len(re.findall(r"\w+", str(x))))
    df["abstract_word_count"] = df["abstract"].apply(lambda x: len(re.findall(r"\w+", str(x))))
    return df

def publications_by_year(df):
    """
    Return a pandas Series indexed by year containing counts.
    """
    years = df["year"].dropna().astype(int)
    if years.empty:
        return pd.Series(dtype=int)
    counts = years.value_counts().sort_index()
    return counts

def top_journals(df, top_n=10):
    """
    Return a Series of top journals and their counts.
    """
    return df["journal"].fillna("Unknown").value_counts().head(top_n)

def top_title_words(df, top_n=20):
    """
    Tokenize titles, remove stopwords and very short words, then return top_n words.
    """
    titles = df["title"].fillna("").str.lower().astype(str)
    # extract words, letters only, at least 2 chars
    words = re.findall(r"\b[a-z]{2,}\b", " ".join(titles))
    words = [w for w in words if w not in STOPWORDS]
    freq = Counter(words)
    return freq.most_common(top_n)

def plot_publications_over_time(df):
    counts = publications_by_year(df)
    fig, ax = plt.subplots(figsize=(8, 4))
    if counts.empty:
        ax.text(0.5, 0.5, "No year data available", ha="center", va="center")
    else:
        ax.plot(counts.index, counts.values, marker="o")
        ax.set_xticks(counts.index)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of papers")
        ax.set_title("Publications by Year")
    plt.tight_layout()
    return fig

def plot_top_journals(df, top_n=10):
    counts = top_journals(df, top_n=top_n)
    fig, ax = plt.subplots(figsize=(8, 4))
    if counts.empty:
        ax.text(0.5, 0.5, "No journal data available", ha="center", va="center")
    else:
        counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Journal")
        ax.set_ylabel("Paper count")
        ax.set_title(f"Top {len(counts)} Journals")
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def plot_top_title_words_bar(df, top_n=20):
    top = top_title_words(df, top_n=top_n)
    if not top:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No title words available", ha="center", va="center")
        return fig
    words, counts = zip(*top)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(words, counts)
    ax.set_title("Top words in titles")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def plot_wordcloud_from_titles(df, max_words=150):
    if not WORDCLOUD_AVAILABLE:
        return None
    text = " ".join(df["title"].fillna("").astype(str).tolist())
    if not text.strip():
        return None
    wc = WordCloud(width=900, height=400, background_color="white",
                   stopwords=STOPWORDS, max_words=max_words)
    wc = wc.generate(text)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    return fig

# --- Streamlit UI ---

st.title("CORD-19 Data Explorer")
st.write("Simple interactive exploration of the CORD-19 metadata file. Upload metadata.csv or place it in the working folder.")

# Sidebar controls
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload metadata.csv (optional)", type=["csv"])
nrows = st.sidebar.number_input("Read only first n rows (0 = all)", min_value=0, value=0, step=1000)
nrows = None if nrows == 0 else int(nrows)

df_raw = load_data(uploaded_file, nrows=nrows)
if df_raw is None:
    st.warning("No dataset found. Upload the metadata.csv using the sidebar or place metadata.csv in this folder.")
    st.stop()

with st.spinner("Cleaning data..."):
    df = clean_dataframe(df_raw)

# Year range slider
years = df["year"].dropna().astype(int)
if years.empty:
    min_year = 2018
    max_year = pd.Timestamp.now().year
else:
    min_year = int(years.min())
    max_year = int(years.max())

year_range = st.sidebar.slider("Select year range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
top_n_journals = st.sidebar.slider("Top N journals", min_value=5, max_value=30, value=10)
top_n_words = st.sidebar.slider("Top N title words", min_value=5, max_value=50, value=20)

# Filtered DataFrame based on year selection
df_filtered = df.copy()
if "year" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["year"].between(year_range[0], year_range[1], inclusive="both")]

st.subheader("Dataset summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total papers (filtered)", f"{len(df_filtered):,}")
col2.metric("Unique journals", f"{df_filtered['journal'].nunique():,}")
col3.metric("Avg abstract words", f"{int(df_filtered['abstract_word_count'].mean()):,}" if len(df_filtered) else "N/A")

# Main visualizations
st.subheader("Publications over time")
fig_time = plot_publications_over_time(df_filtered)
st.pyplot(fig_time)

st.subheader(f"Top {top_n_journals} Journals")
fig_journals = plot_top_journals(df_filtered, top_n=top_n_journals)
st.pyplot(fig_journals)

st.subheader(f"Top {top_n_words} Title Words")
fig_words = plot_top_title_words_bar(df_filtered, top_n=top_n_words)
st.pyplot(fig_words)

if WORDCLOUD_AVAILABLE:
    st.subheader("Word Cloud from Titles")
    fig_wc = plot_wordcloud_from_titles(df_filtered, max_words=120)
    if fig_wc is not None:
        st.pyplot(fig_wc)
    else:
        st.info("Word cloud could not be generated. Try with a larger dataset or ensure titles exist.")
else:
    st.info("wordcloud package not available. Install it to see a word cloud. The bar chart above shows top words instead.")

# Show a sample of the data
st.subheader("Sample records")
sample_cols = ["title", "publish_time", "journal", "authors", "abstract"]
sample_cols = [c for c in sample_cols if c in df_filtered.columns]
st.dataframe(df_filtered[sample_cols].head(50))

# Download filtered dataset
csv = df_filtered.to_csv(index=False)
st.download_button("Download filtered CSV", data=csv, file_name="cord19_filtered.csv", mime="text/csv")

# Helpful notes and reflections
with st.expander("Notes, limitations and next steps"):
    st.markdown(
        """
        - This app is a beginner-friendly starting point for exploring the CORD-19 metadata file.
        - The dataset may contain missing or inconsistent dates and journal names. We parsed dates using pandas with errors coerced to NaT.
        - Next steps:
          - Add more robust text cleaning for title/abstract (stemming, more stopwords).
          - Use topic modelling to cluster papers.
          - Add author-level or affiliation-level analysis.
        """
    )

st.caption("Tip: If the dataset is large, set a reasonable nrows value in the sidebar while exploring.")
