# Gaurav's Restaurant Recommendation System

A Python + Streamlit capstone that recommends restaurants from the supplied historical Zomato Bengaluru dataset.

## Features
- Filter by area, cuisine, budget, rating, online ordering and table booking.
- Text-based matching using TF-IDF and cosine similarity.
- Ranking that blends text similarity, restaurant rating and vote popularity.
- Dataset overview charts and simple unit tests.
- Listing links where the source dataset provides a URL.

## Run in IntelliJ IDEA
1. Extract the project ZIP to a folder.
2. Open IntelliJ IDEA and choose **Open**; select the project folder.
3. Configure a Python interpreter/virtual environment (Python 3.10–3.12 recommended).
4. Open the terminal in the project root and run:

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # If PowerShell blocks activation, use: .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```
5. Open the local URL printed by Streamlit, usually `http://localhost:8501`.

The supplied archive is in `data/raw/zomato.csv.zip`. The app extracts `zomato.csv` on first load. The extracted CSV is roughly 574 MB, so ensure enough disk space and RAM. If you already extracted it, place it at `data/raw/zomato.csv`.

## Test
```bash
python -m pytest -q
```

## Dataset notes
The source file has 71,730 data rows and 17 columns (confirmed from the uploaded archive). It contains restaurant metadata plus a large `reviews_list` field. The app reads only the metadata columns it needs to avoid loading the huge review text into memory. Duplicate rows are removed using restaurant name, location and address; the resulting count may therefore be lower than the source row count.

This is historical data, not a live Zomato API. Do not interpret old ratings, costs, or listing URLs as current facts. The ranking score is a heuristic, not a probability or accuracy metric.

See `docs/PROJECT_DOCUMENTATION.md` for the full project report.
