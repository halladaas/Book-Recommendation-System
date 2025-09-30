from flask import Flask, request, render_template
from model import recommend_books

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        book_title = request.form["book_title"]
        recommendations = recommend_books(book_title, top_n=3)
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
