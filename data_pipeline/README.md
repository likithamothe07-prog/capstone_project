Module 1: Data Pipeline (`/data_pipeline`)

1. Setup & Execution
Run the following commands from inside the `data_pipeline/` directory:

```powershell
pip install -r requirements.txt
python pipeline.py
Executing pipeline.py scrapes the site, cleans and normalizes the data, creates and populates books_catalog.db, runs 5 SQL benchmark queries, and verifies DataFrame merge equivalence.

2. Scraping Architecture
Source: http://books.toscrape.com/

Coverage: Scrapes across 4 distinct categories (Travel, Mystery, Historical Fiction, Sequential Art), following pagination (li.next a) to collect 89 books (exceeding the requirement of at least 60 books across at least 3 categories).

Extracted Fields: Book title, raw price (£), star rating string, inventory availability text, and category name.

3. Data Cleaning & Type Parsing Decisions
Price Parsing (price_gbp): Extracted numeric values via regular expressions (r"[\d.]+") and cast to float.

Missing Value Imputation: If any numeric price field is missing or corrupt, it is imputed using the median price of valid books. This preserves sample size without skewing the distribution.

Rating Mapping (rating): Text rating classes (One, Two, Three, Four, Five) are converted to integers (1 through 5).

Stock Parsing (in_stock): Converted text into a binary integer flag (1 for "In stock", 0 for out of stock).

Currency Conversion (price_inr): Calculated strictly using the fixed baseline conversion rate:

1 GBP = 105.50 INR

Formula: round(price_gbp * 105.50, 2)

4. Relational Database Schema
A normalized SQLite database (books_catalog.db) with Foreign Key constraints enabled (PRAGMA foreign_keys = ON;) implements a two-table schema:

SQL
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL,
    in_stock INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories (category_id)
);
5. SQL Verification & Equivalence
SQL Queries Executed: 5 queries covering SELECT, WHERE, ORDER BY, LIMIT, DISTINCT, IN, BETWEEN, and an inner JOIN between books and categories.

Join Equivalence: Queried data using pd.read_sql() was compared against an in-memory pandas pd.merge() operation on identical attributes. The comparison returned True via df_sql.equals(df_merge), confirming relational parity.

4. Save the file (`Ctrl + S`).