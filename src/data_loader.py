from pathlib import Path
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CSV_PATH = RAW_DIR / "zomato.csv"
ZIP_PATH = RAW_DIR / "zomato.csv.zip"
USECOLS = ["name", "online_order", "book_table", "rate", "votes", "location", "rest_type", "dish_liked", "cuisines", "approx_cost(for two people)", "listed_in(type)", "listed_in(city)", "address", "url"]

def ensure_csv() -> Path:
    """Extract the supplied dataset archive when the raw CSV is not present."""
    if CSV_PATH.exists():
        return CSV_PATH
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Dataset not found. Put zomato.csv or zomato.csv.zip in {RAW_DIR}")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        candidates = [n for n in archive.namelist() if n.lower().endswith('.csv')]
        if not candidates:
            raise ValueError("The dataset ZIP does not contain a CSV file.")
        # Extract only the CSV into the expected raw data directory.
        source = candidates[0]
        with archive.open(source) as src, CSV_PATH.open('wb') as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk: break
                dst.write(chunk)
    return CSV_PATH

def load_restaurants() -> pd.DataFrame:
    path = ensure_csv()
    # The original review and menu columns are very large; exclude them for app responsiveness.
    df = pd.read_csv(path, usecols=lambda c: c in USECOLS, low_memory=False, encoding_errors='replace')
    df = df.rename(columns={
        "approx_cost(for two people)": "cost_for_two",
        "listed_in(type)": "listed_type",
        "listed_in(city)": "listed_city",
    })
    # Normalize text columns explicitly: pandas may infer rate/votes/cost as numeric
    # when a CSV column contains mostly numbers. Calling .str on those columns fails.
    text_cols = [
        'name', 'online_order', 'book_table', 'rate', 'location', 'rest_type',
        'dish_liked', 'cuisines', 'cost_for_two', 'listed_type', 'listed_city',
        'address', 'url'
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype('string').fillna('').str.strip()

    # Convert mixed-format numeric fields safely after normalizing to strings.
    df['rating'] = pd.to_numeric(
        df['rate'].str.extract(r'(\d+(?:\.\d+)?)', expand=False),
        errors='coerce'
    )
    df['votes'] = pd.to_numeric(
        df['votes'].astype('string').str.replace(',', '', regex=False).str.strip(),
        errors='coerce'
    ).fillna(0)
    df['cost_for_two'] = pd.to_numeric(
        df['cost_for_two'].astype('string').str.replace(',', '', regex=False).str.strip(),
        errors='coerce'
    )
    df['name'] = df['name'].astype('string').fillna('').str.strip()
    df = df[df['name'].ne('')].copy()
    # Duplicate listing rows can occur across Zomato listing categories. Keep one row per name/location.
    df = df.drop_duplicates(subset=['name', 'location', 'address'], keep='first').reset_index(drop=True)
    df['search_text'] = df[[c for c in ['name','location','rest_type','dish_liked','cuisines','listed_type','listed_city'] if c in df]].fillna('').agg(' '.join, axis=1)
    return df
