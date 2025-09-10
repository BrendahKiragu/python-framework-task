# CORD-19 Data Explorer

A beginner-friendly data exploration and Streamlit app for the CORD-19 `metadata.csv` file. This project demonstrates loading, cleaning, basic analysis, and interactive visualizations.

## What the app does

Loads metadata.csv (or a user-uploaded CSV)

Cleans and prepares basic columns: publish_time, year, title, abstract, journal

Visualizes publications over time, top journals, top words in titles

Generates a word cloud if the wordcloud package is installed

Lets you filter by year and download the filtered CSV

## Prerequisites

- Python 3.7+
- Git (optional)
- `metadata.csv` from the CORD-19 dataset. You can download the metadata file from Kaggle:
  https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge

Note: the full dataset is large. For quick exploration you can use a subset by setting "Read only first n rows" in the sidebar.

## Setup

1. Clone the repo or copy files into a folder:

```bash
git clone <your-repo-url>
cd Frameworks_Assignment
```

# USAGE INSTRUCIONS

## Step 1: Create a virtual environment

Inside your project folder (where script.py is located), run:
```bash
    python3 -m venv venv
```

## Step 2: Activate the virtual environment

```bash
    source venv/bin/activate
```

You should see (venv) at the start of your terminal prompt, meaning the environment is active.

## Step 3: Install dependencies from requirements.txt

```bash
    pip install -r requirements.txt
```
This will install the following libraries:

**Pandas → for data manipulation**
**Matplotlib → for visualizations**
**Seaborn → for advanced statistical charts**
**Streamlit → for building the interactive web app**

## Step 4: Verify installation

 ```bash
    pip list
```

You should see:
```bash
    pandas         2.2.2
    matplotlib     3.9.2
    seaborn        0.13.2
    streamlit     1.38.0
```

## Step 5: Run the script

```bash
    python script.py
```

## Step 6:Run the app

```bash
    streamlit run script.py
```

This opens a browser window with the interactive CORD-19 Data Explorer.

## 👩‍💻 Author

**Brendah Mwihaki Kiragu**  
Junior Software Developer | Data Analysis Enthusiast

---

## 📝 License

This project is for educational purposes only.
