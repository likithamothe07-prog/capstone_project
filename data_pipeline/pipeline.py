import os
import re
import sqlite3
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "http://books.toscrape.com/"
GBP_TO_INR_RATE = 105.50
DB_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
DB_PATH = os.path.join(DB_DIR, "books_catalog.db")

WORD_TO_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

def scrape_categories():
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(BASE_URL, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    category_tags = soup.select(".side_categories ul li ul li a")[:4]

    scraped_data = []
    for cat in category_tags:
        cat_name = cat.text.strip()
        current_page_url = BASE_URL + cat['href']

        while current_page_url:
            page_resp = requests.get(current_page_url, headers=headers)
            page_resp.raise_for_status()
            page_soup = BeautifulSoup(page_resp.content, "html.parser")

            product_pods = page_soup.select("article.product_pod")
            for pod in product_pods:
                title = pod.h3.a["title"]
                price_raw = pod.select_one(".price_color").text.strip()
                rating_classes = pod.select_one("p.star-rating")["class"]
                star_class = [c for c in rating_classes if c != "star-rating"][0]
                availability_raw = pod.select_one(".availability").text.strip()

                scraped_data.append({
                    "title": title,
                    "price_raw": price_raw,
                    "star_rating_raw": star_class,
                    "availability_raw": availability_raw,
                    "category": cat_name
                })

            next_tag = page_soup.select_one("li.next a")
            if next_tag and len(scraped_data) < 75:
                parent_dir = "/".join(current_page_url.split("/")[:-1])
                current_page_url = parent_dir + "/" + next_tag["href"]
            else:
                current_page_url = None

    print(f"[OK] Scraped {len(scraped_data)} records across categories.")
    return pd.DataFrame(scraped_data)

def clean_and_enrich_data(df):
    cleaned = df.copy()
    cleaned["price_gbp"] = cleaned["price_raw"].apply(
        lambda x: float(re.search(r"[\d.]+", x).group()) if re.search(r"[\d.]+", x) else None
    )
    if cleaned["price_gbp"].isnull().any():
        cleaned["price_gbp"] = cleaned["price_gbp"].fillna(cleaned["price_gbp"].median())

    cleaned["rating"] = cleaned["star_rating_raw"].str.lower().map(WORD_TO_NUM).fillna(1).astype(int)
    cleaned["in_stock"] = cleaned["availability_raw"].apply(lambda x: 1 if "in stock" in x.lower() else 0)
    cleaned["price_inr"] = (cleaned["price_gbp"] * GBP_TO_INR_RATE).round(2)
    return cleaned

def setup_sqlite_database(df, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")
    cursor.execute("DROP TABLE IF EXISTS books;")
    cursor.execute("DROP TABLE IF EXISTS categories;")

    cursor.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        );
    """)

    cursor.execute("""
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
    """)

    category_id_map = {}
    for cat in df["category"].unique():
        cursor.execute("INSERT INTO categories (category_name) VALUES (?)", (cat,))
        category_id_map[cat] = cursor.lastrowid

    for _, row in df.iterrows():
        cat_id = category_id_map[row["category"]]
        cursor.execute("""
            INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (row["title"], row["price_gbp"], row["price_inr"], row["rating"], row["in_stock"], cat_id))

    conn.commit()
    print(f"[OK] Database initialized and loaded at: {db_path}")
    return conn

def execute_sql_queries(conn):
    cursor = conn.cursor()
    queries = {
        "Query 1: SELECT, WHERE, LIMIT": "SELECT title, price_gbp, price_inr FROM books WHERE in_stock = 1 LIMIT 5;",
        "Query 2: ORDER BY, LIMIT": "SELECT title, price_inr, rating FROM books ORDER BY price_inr DESC LIMIT 5;",
        "Query 3: DISTINCT": "SELECT DISTINCT rating FROM books ORDER BY rating ASC;",
        "Query 4: IN, BETWEEN, ORDER BY": "SELECT title, price_inr, rating FROM books WHERE rating IN (4, 5) AND price_inr BETWEEN 2000.00 AND 4000.00 ORDER BY price_inr ASC;",
        "Query 5: JOIN categories & books": "SELECT b.title, b.price_inr, b.rating, c.category_name FROM books b JOIN categories c ON b.category_id = c.category_id WHERE b.rating >= 4 ORDER BY b.title ASC;"
    }
    for label, query in queries.items():
        print(f"\n--- {label} ---\n{query}")
        for row in cursor.execute(query).fetchall()[:3]:
            print(" ", row)

def verify_equivalence(conn):
    join_sql = """
        SELECT b.title, b.price_inr, b.rating, c.category_name 
        FROM books b 
        JOIN categories c ON b.category_id = c.category_id 
        WHERE b.rating >= 4 
        ORDER BY b.title ASC;
    """
    df_sql = pd.read_sql(join_sql, conn)
    df_books = pd.read_sql("SELECT title, price_inr, rating, category_id FROM books;", conn)
    df_cats = pd.read_sql("SELECT category_id, category_name FROM categories;", conn)

    df_merge = (
        pd.merge(df_books, df_cats, on="category_id")
        .query("rating >= 4")[["title", "price_inr", "rating", "category_name"]]
        .sort_values(by="title")
        .reset_index(drop=True)
    )
    print(f"\nAre both DataFrames exactly equal? -> {df_sql.equals(df_merge)}")

if __name__ == "__main__":
    raw = scrape_categories()
    cleaned = clean_and_enrich_data(raw)
    conn = setup_sqlite_database(cleaned)
    execute_sql_queries(conn)
    verify_equivalence(conn)
    conn.close()