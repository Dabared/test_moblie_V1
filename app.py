from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

app = Flask(__name__)
CORS(app)  # HTML වලින් backend call කරන්න අවසර දෙයි

# 📄 PDF එක load කරලා එකෙ තියන text එක extract කරමු
pdf_path = "Fund-Transfer-Success-04-11-2025_14-23-10.pdf"
with pdfplumber.open(pdf_path) as pdf:
    text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])

# 🔍 Text එක කුඩා කොටස් වලට බෙදමු
def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

chunks = split_text(text)

# 🔢 Sentence embedding create කරන්න
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# 🤖 QA Model එක load කරමු
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# 🧠 ප්‍රශ්නයක් ඇසූවොත් පිළිතුර ලබාදෙන function එක
def get_answer(question):
    question_embedding = embed_model.encode([question])
    _, I = index.search(np.array(question_embedding), k=3)
    context = "\n".join([chunks[i] for i in I[0]])
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# 🌐 HTML UI එකෙන් ප්‍රශ්නයක් ආවොත් මෙතනින් පිළිතුර යවයි
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"answer": "කරුණාකර ප්‍රශ්නයක් ඇසන්න."})
    answer = get_answer(question)
    return jsonify({"answer": answer})

# 🚀 Server එක ක්‍රියාත්මක කරන්න
if __name__ == "__main__":
    app.run(debug=True)
