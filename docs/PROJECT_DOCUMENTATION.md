# Gaurav's Restaurant Recommendation System
## Project Documentation and Capstone Report

**Project type:** Content-based restaurant recommendation system  
**Implementation:** Python, Pandas, scikit-learn, Streamlit  
**Dataset:** Uploaded `zomato.csv.zip`, containing `zomato.csv`  
**Report version:** 1.0

---

# Page 1 — Executive Summary

Gaurav's Restaurant Recommendation System is a local restaurant discovery application built around the supplied historical Zomato restaurant dataset. It lets a user narrow restaurants by area, cuisine, maximum cost for two, minimum rating, online-order support and table-booking support. A free-text request such as “North Indian family dinner” can influence ranking through text similarity. The app presents each recommendation with its area, restaurant type, cuisines, estimated cost for two, rating, vote count and source listing URL when available.

The system is deliberately designed as an explainable, compact capstone rather than a claim of a production-scale personalization platform. The dataset is a snapshot and does not provide a live feed of restaurant opening hours, price changes, reservations or delivery status. The project therefore recommends candidates from recorded attributes and clearly communicates that limitation.

The uploaded archive was inspected before implementation. It contains one CSV named `zomato.csv`, with 71,730 data rows and 17 columns. Important fields include `name`, `online_order`, `book_table`, `rate`, `votes`, `location`, `rest_type`, `dish_liked`, `cuisines`, `approx_cost(for two people)`, `reviews_list`, `menu_item`, `listed_in(type)` and `listed_in(city)`. The reviews field makes the raw CSV large, approximately 574 MB when extracted. To keep the app practical, it loads only the metadata fields needed for recommendations and exploration.

The main deliverable is a runnable IntelliJ IDEA project. It includes a Streamlit user interface, data loading and cleaning, a TF-IDF content representation, a ranking method, exploratory charts, unit tests, requirements and this report. No external LLM API is required.

---

# Page 2 — Problem Statement and Motivation

Restaurant discovery involves balancing multiple constraints. A person may want a particular cuisine near a chosen neighbourhood, within a budget, with a minimum acceptable rating and specific service options. Browsing an unfiltered list forces the user to compare these attributes manually. A recommendation interface can consolidate the constraints and rank the remaining candidates.

The project addresses four questions:

1. Which restaurant listings satisfy the user's explicit filters?
2. Which listings have metadata that resembles the user's free-text request?
3. How can rating and vote count contribute to a useful ordering without being mistaken for ground truth?
4. How can the results and dataset characteristics be exposed in a simple application?

The objective is not to assert that the system knows a user's personal taste. The provided dataset contains restaurant attributes and aggregate ratings/votes, but it does not contain a user-by-restaurant interaction matrix, user profiles or reliable historical clicks. As a result, a content-based method is a sensible baseline: it compares the request with descriptive restaurant text and supplements that similarity with recorded rating and popularity signals.

The scope is limited to restaurant recommendations from the supplied data. It does not perform live availability checks, route planning, payments, table booking, delivery ordering, or current web scraping. It does not require credentials or paid APIs. The implementation should be treated as a learning and demonstration project, with a clear path for future evaluation and improvement.

---

# Page 3 — Dataset Inspection and Data Dictionary

The archive contains a single CSV file. The raw dataset has 17 columns and 71,730 data records. The fields are not all equally useful for a lightweight recommender.

| Field | Meaning / use in this project |
|---|---|
| `name` | Restaurant display name; required to identify a result |
| `location` | Neighbourhood/area used for filtering |
| `address` | Address text; used as part of duplicate detection |
| `url` | Historical source listing URL, if present |
| `online_order` | Recorded online ordering flag |
| `book_table` | Recorded table booking flag |
| `rate` | Text rating such as `4.1/5`; parsed to numeric rating |
| `votes` | Vote count; parsed to numeric value |
| `rest_type` | Restaurant type, such as cafe or casual dining |
| `dish_liked` | Comma-separated liked dishes from the source |
| `cuisines` | Cuisine descriptions used for filters and text matching |
| `approx_cost(for two people)` | Approximate cost for two; converted to numeric |
| `listed_in(type)` | Source listing category |
| `listed_in(city)` | Source listing city/category label |
| `reviews_list` | Large review text field; excluded from app loading |
| `menu_item` | Menu-related field; excluded from app loading |
| `phone` | Contact data; excluded from app display |

Some values can be blank or malformed. Rating values may contain text such as `NEW`, `-` or missing values. Cost values may contain commas or blanks. The loader converts ratings and costs to numeric fields and keeps missing values as missing rather than fabricating values. Votes are parsed after removing commas; absent vote counts become zero for ranking purposes.

The data is historical and associated with Bengaluru locations. It should not be generalized to every city or treated as current business information.

---

# Page 4 — Architecture and Project Structure

The implementation uses a small layered architecture. The Streamlit interface collects user preferences and displays results. `data_loader.py` locates and extracts the dataset archive, selects useful columns, normalizes data types and constructs combined descriptive text. `recommender.py` builds a TF-IDF matrix and exposes filtering and ranking functions. Unit tests exercise a small in-memory example without depending on the full dataset.

```text
Gaurav's Restaurant Recommendation System/
├── app.py                         # Streamlit UI and charts
├── requirements.txt               # Python dependencies
├── README.md                      # Setup and run instructions
├── data/
│   ├── raw/zomato.csv.zip          # User-provided dataset archive
│   └── processed/                 # Reserved for generated artifacts
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Load, clean and transform source data
│   └── recommender.py              # TF-IDF and ranking logic
├── tests/test_recommender.py      # Unit tests
├── notebooks/                     # Reserved for optional EDA notebook
└── docs/PROJECT_DOCUMENTATION.md  # This report
```

The archive is included so the project is portable. On first load, the data loader extracts the CSV to `data/raw/zomato.csv`. This requires several hundred megabytes of free disk space. The app reads selected columns rather than loading every review and menu string into the working DataFrame. The original archive remains intact.

The project is compatible with a standard Python environment in IntelliJ IDEA with Python support. Streamlit runs as a local web application while the Python project remains editable in the IDE.

---

# Page 5 — Data Cleaning and Feature Engineering

Data cleaning is applied when the app loads the dataset. The loader first ensures the CSV exists, extracting it from the ZIP if needed. It reads only the selected metadata columns, which reduces memory use compared with reading every field. Object columns are converted to strings, missing object values become empty strings, and surrounding whitespace is removed.

The restaurant name is required; rows with an empty name are removed. Duplicate listings are reduced using the combination of restaurant name, location and address. This choice is a practical compromise: a restaurant can appear in multiple listing categories, but similarly named restaurants at different addresses should not automatically be merged. This rule may still merge or retain edge cases, so a future production implementation would ideally use a stable restaurant identifier.

The `rate` field is textual in the original source. A numeric value is extracted when present, so `4.1/5` becomes `4.1`. Values without a usable number become missing. The vote field has commas removed before numeric conversion. The cost field is converted after removing commas. These parsed numeric fields support comparisons and ranking.

A `search_text` field combines name, area, restaurant type, liked dishes, cuisines, listing type and listing city. This creates a compact representation of the restaurant's descriptive attributes. The project intentionally does not parse the enormous `reviews_list` column in the interactive app. Consequently, this version does not claim review sentiment analysis. That is a potential extension, not an implemented feature.

---

# Page 6 — Recommendation Method: TF-IDF

Term Frequency–Inverse Document Frequency (TF-IDF) represents text as weighted terms. Terms that occur in a restaurant description but are not universal across all descriptions can contribute more strongly to its vector. The implementation uses scikit-learn's `TfidfVectorizer`, English stop-word removal, unigrams and bigrams, and a maximum vocabulary size to constrain memory use.

Each restaurant's combined metadata becomes a document. When a user enters a request, the same vectorizer transforms the query into the same feature space. Cosine similarity measures the angle-based similarity between the query vector and each restaurant vector. Scores closer to one indicate a stronger text match; zero indicates no shared weighted terms in the representation.

TF-IDF is suitable for a first content-based recommender because it is interpretable and does not require training labels. A query containing “Italian pasta” should tend to favour records whose descriptions contain related terms. It is not semantic language understanding: synonyms, spelling mistakes, multilingual phrases and context can be missed. It also cannot infer a user's preferences from prior behaviour because no such history is available.

If the query is blank, the system does not fabricate a text match. It ranks filtered restaurants using rating and popularity signals. The sidebar filters are applied before the final list is returned, ensuring that recommendations obey the explicit area, cuisine, budget, rating and service constraints.

---

# Page 7 — Ranking and Filtering Logic

The system has two distinct steps: eligibility filtering and ranking. Filters decide which records are eligible. Ranking orders the eligible candidates. This separation makes the result easier to reason about and test.

Available filters are area, cuisine, maximum cost for two, minimum rating, online ordering, table booking and result count. “Any” choices do not impose a restriction. If the budget is set to zero, the UI treats it as no cost cap. Unknown costs are retained by the budget filter because excluding every unknown-cost restaurant could unfairly remove valid candidates; users should interpret these entries carefully.

For a non-empty text query, the ranking score is a weighted combination:

`score = 0.60 × text similarity + 0.30 × normalized rating + 0.10 × normalized popularity`

Rating is scaled by five. Popularity is derived from the natural logarithm of one plus votes and normalized within the candidate set. Logarithmic scaling prevents a restaurant with extremely high vote counts from overwhelming every other signal. For an empty query, the ranking score is `0.75 × normalized rating + 0.25 × normalized popularity`.

These weights are engineering choices for a demonstrable baseline; they have not been optimized using user feedback or a labelled relevance dataset. The score is not a probability and should not be called “accuracy.” The weights can be tuned later with an evaluation protocol and representative relevance judgements.

---

# Page 8 — User Interface and User Journey

The Streamlit application is designed for a user who wants to discover restaurants quickly. The sidebar contains the structured preferences. A free-text field at the top accepts a natural-language-style description, such as “quiet cafe,” “biryani,” or “North Indian family dinner.” The results are shown as individual cards with the restaurant name, area, type, cuisines, cost estimate, rating, vote count and listing link.

A user journey can be described in five steps:

1. Start the app and wait for the dataset and TF-IDF index to load.
2. Select an area or keep the area set to All.
3. Choose a cuisine and budget, then adjust rating and service preferences.
4. Enter descriptive terms if there is a particular style of restaurant in mind.
5. Compare the ranked results and follow a source URL if useful, remembering that the URL and listing details may be historical.

The application includes three tabs. **Recommendations** displays ranked candidates. **Explore dataset** shows headline counts and simple charts for area counts, rating distribution and cost distribution. **About & limitations** explains the algorithm and cautions about historical data. Streamlit caching prevents repeated data loads and repeated model construction during ordinary UI interactions.

The UI is intentionally functional rather than elaborate. A later release could add a map, restaurant images, responsive cards, saved favourites, downloadable results or an interactive explanation of why each candidate ranked highly.

---

# Page 9 — Technology Stack and Setup

Python is the implementation language. Pandas handles tabular data cleaning; NumPy provides numeric transformations; scikit-learn supplies TF-IDF vectorization and cosine similarity; Streamlit provides the interface; and pytest runs unit tests. The project does not depend on OpenAI or another paid LLM API.

Recommended setup is Python 3.10–3.12. Open the extracted project folder in IntelliJ IDEA and configure a project interpreter. From the integrated terminal, create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell prevents virtual-environment activation, invoke the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a local URL, commonly `http://localhost:8501`. Keep the terminal process running while using the application. To run tests, execute `python -m pytest -q` from the project root. If the IDE does not recognize the `src` package, confirm that the project root is opened as the project and that the chosen interpreter has the requirements installed.

The first run extracts a CSV of roughly 574 MB. The complete CSV plus the in-memory DataFrame and TF-IDF matrix may require substantial memory. The app intentionally omits the large review and menu columns to reduce this burden.

---

# Page 10 — Testing and Quality Assurance

The included unit tests use a small, synthetic DataFrame so they run quickly and do not require loading the 574 MB CSV. One test checks that area and cost filters return the expected record. A second checks that a text query containing “Italian pasta” ranks the matching restaurant first. These tests validate important behaviour of the filtering and text-matching logic.

To run them:

```bash
python -m pytest -q
```

The project should also be manually checked with several scenarios: no filters and no query; a restrictive area and budget; a cuisine with no matching records; a text query with relevant terms; a minimum rating higher than all available restaurants; and missing ratings or costs. Empty results should produce a helpful message rather than an exception.

Data-quality checks worth adding include the percentage of missing ratings, costs and locations; duplicate counts before and after deduplication; unexpected rating ranges; and cost outliers. Recommendation quality needs a different kind of testing from software correctness. A ranking algorithm can pass unit tests while recommending unhelpful restaurants. Human-labelled relevance judgements or click/selection feedback would be needed to measure relevance properly.

No benchmark accuracy or precision score is claimed in this report because the uploaded data does not include a ground-truth set of user queries and relevant restaurants. If a labelled test set becomes available, Precision@K, Recall@K and NDCG@K can be calculated and compared against simple baselines.

---

# Page 11 — Limitations, Privacy and Responsible Use

The dataset is historical and reflects the source's recorded information, not necessarily today's business conditions. Prices, ratings, restaurant names, addresses, menu details, order options and URLs may have changed. The application should not be used as a guarantee that a restaurant is open, bookable or delivering. Users should verify important details directly before travelling or making plans.

The system's ranking is based on metadata, rating and vote counts. Rating distributions can be biased by who chose to submit reviews. Vote count is a rough popularity signal, not an objective measure of food quality. Popularity normalization is calculated among current candidates, so scores can change when filters change. Unknown costs are allowed through a budget filter, which means a result with no listed cost may need manual verification.

The source file includes contact and review fields, but the application does not display phone numbers and does not load review text into its main DataFrame. This reduces unnecessary processing of personal or free-text content. The project does not send dataset contents to an external service. If the project is published publicly, review the dataset's licence and attribution requirements and avoid committing private credentials or additional personal data.

This is a decision-support tool. It presents candidates to the user rather than making purchases, reservations or other consequential decisions on the user's behalf. Clear limitations help users understand what the ranking can and cannot establish.

---

# Page 12 — Future Enhancements and Conclusion

There are several logical next steps. First, build an optional exploratory notebook that profiles missing values, duplicate patterns, rating/cost distributions and area-level cuisine patterns. Second, create a carefully labelled query–relevance dataset and evaluate ranking quality using Precision@K and NDCG@K. Third, improve text handling with cuisine aliases, typo normalization, multilingual tokenization or embeddings. Such upgrades should be compared against the current TF-IDF baseline rather than assumed to be better.

A review sentiment module could be developed separately by sampling and parsing the `reviews_list` field. It should be added only with a documented data-extraction process, memory strategy and validation set. Sentiment labels should not be invented from ratings alone, and automated sentiment predictions should be described as estimates. Collaborative filtering would require user-item interactions, which are not included in the fields used by this project. A local LLM assistant could optionally translate conversational requests into structured filters or explain recommendation factors, but the deterministic ranking engine should remain testable independently.

Other enhancements include model persistence, downloadable CSV results, saved preferences, maps, refreshed datasets, improved duplicate resolution and deployment packaging. Before deploying publicly, verify dataset usage rights and replace stale source links or data with a permitted, current data source.

In conclusion, Gaurav's Restaurant Recommendation System demonstrates a complete end-to-end data application: inspecting a real dataset, cleaning metadata, building a content-based ranking engine, providing interactive filters, visualizing dataset characteristics, documenting limitations and testing core behaviour. It is a solid foundation for a capstone and can be extended as evaluation data and requirements become available.

---

# Appendix A — Formula Reference

- Numeric rating: extract the first decimal/integer-like number from the original rating string; invalid strings become missing.
- Popularity raw value: `log(1 + votes)`.
- Normalized rating: `rating / 5`, with missing rating treated as zero for ranking.
- Normalized popularity: candidate-set popularity divided by the maximum candidate popularity, or zero if the maximum is zero.
- Query score: `0.60 × cosine similarity + 0.30 × normalized rating + 0.10 × normalized popularity`.
- No-query score: `0.75 × normalized rating + 0.25 × normalized popularity`.

These formulas define a transparent baseline, not a learned model. Any change to weights should be documented and tested against a consistent evaluation set.

# Appendix B — Troubleshooting

**Dataset not found:** check that `data/raw/zomato.csv.zip` is present, or put the extracted file at `data/raw/zomato.csv`.  
**Out of memory:** close other applications, use a machine with more RAM, or create a smaller metadata-only CSV once and configure the loader to use it.  
**Missing package:** activate the project virtual environment and run `python -m pip install -r requirements.txt`.  
**Streamlit command not found:** use `python -m streamlit run app.py`.  
**No recommendations:** loosen filters, select Any for cuisine/area, or lower the minimum rating.  
**Old listing details:** this project uses a historical snapshot; verify current details independently.

# Appendix C — Academic Presentation Summary

**Aim:** recommend restaurants from descriptive attributes and user-selected constraints.  
**Input:** historical restaurant listing metadata.  
**Preprocessing:** column selection, whitespace cleanup, numeric conversion, removal of blank names, deduplication and combined text construction.  
**Algorithm:** TF-IDF and cosine similarity, blended with rating and log-vote popularity.  
**Output:** a filtered and ranked restaurant list with source attributes.  
**Evaluation:** unit tests are provided; ranking relevance metrics require labelled query–restaurant judgements.  
**Key limitation:** historical dataset does not provide live status or user-item interactions.
