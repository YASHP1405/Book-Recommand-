from flask import Flask, request, render_template, redirect
import mysql.connector
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

app = Flask(__name__)

# --- DB Config ---
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '5Ganpati@',
    'database': 'book_recomander'
}

# --- Load Books from DB ---
def load_books():
    conn = mysql.connector.connect(**db_config)
    query = "SELECT * FROM books"
    df = pd.read_sql(query, conn)
    conn.close()

    # Handle missing values in fields used by the recommender
    df['title'] = df['title'].fillna('')
    df['author'] = df['author'].fillna('')
    df['genre'] = df['genre'].fillna('')

    return df   # works with mysql.connector


# --- Build Token Map ---
def build_token_index(df):
    token_map = {}
    for idx, row in df.iterrows():
        title_tokens = row['title'].lower().split()
        author_tokens = row['author'].lower().split()
        genre_tokens = [g.strip().lower() for g in row['genre'].split(',')]
        tokens = set(title_tokens + author_tokens + genre_tokens)
        for token in tokens:
            if token:
                token_map.setdefault(token, []).append(idx)
    return token_map

# --- Recommender Logic ---
def get_recommendations(user_query, top_n=5, threshold=60):
    df = load_books()
    if df.empty:
        return None, "No books found in the database."

    # Create combined content for TF-IDF
    df['content'] = (df['title'] + ' ' + df['author'] + ' ' + df['genre']).str.lower()
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['content'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Build token map for RapidFuzz matching
    token_index = build_token_index(df)
    token_list = list(token_index.keys())

    # Find closest token match
    match = process.extractOne(user_query.lower(), token_list, scorer=fuzz.token_sort_ratio)
    if not match or match[1] < threshold:
        return None, f"No match found for '{user_query}'."

    matched_token = match[0]
    seed_idx = token_index[matched_token][0]

    # Get top similar books (excluding the matched one)
    sim_scores = list(enumerate(cosine_sim[seed_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    recommendations = df.iloc[[i[0] for i in sim_scores]]
    matched_book = df.iloc[seed_idx]

    # Return structured result
    return {
        "matched_token": matched_token,
        "matched_book": matched_book,
        "recommendations": recommendations if not recommendations.empty else None
    }, None

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/add", methods=["POST"])
def add_book():
    title = request.form["title"]
    author = request.form["author"]
    genre = request.form["genre"]
    summary = request.form.get("summary", "")
    price = float(request.form.get("price", 0))
    platform = request.form.get("platform", "")

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO books (title, author, genre, summary, price, platform) VALUES (%s,%s,%s,%s,%s,%s)",
        (title, author, genre, summary, price, platform)
    )
    conn.commit()
    cursor.close()
    conn.close()

    return redirect("/")

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    result = None
    error = None

    if request.method == 'POST':
        query_text = request.form.get('query')

        try:
            # Use TF-IDF + RapidFuzz recommender
            result, error = get_recommendations(query_text, top_n=5, threshold=60)

        except Exception as e:
            error = str(e)

    return render_template('recommend.html', result=result, error=error)

@app.route("/book/<int:book_id>")
def book_details(book_id):
    conn = mysql.connector.connect(**db_config)
    query = "SELECT * FROM books WHERE id = %s"
    df = pd.read_sql(query, conn, params=(book_id,))
    conn.close()

    if df.empty:
        return render_template("book_details.html", book=None)

    book = df.iloc[0]
    return render_template("book_details.html", book=book)

# --- Run ---
if __name__ == "__main__":
    app.run(debug=True)
