# 🤖 FAQ Chatbot — NLP Powered

A terminal-based **FAQ Chatbot** built with Python that understands what you're asking — even if you don't use the exact words. Powered by a full **NLP pipeline** using TF-IDF Vectorization and Cosine Similarity matching.

> 🎓 Built as part of my Python internship at **CodeAlpha**

---

## 🖼️ Demo

```
+======================================================================+
|                                                                      |
|                🤖 FAQ CHATBOT - Python Programming Q&A                |
|                                                                      |
+======================================================================+

> Ask me anything about Python programming.
> Type help for tips, or exit / quit / bye to leave.

----------------------------------------------------------------------

You > what is python?

+-  Bot  .  Turn 1  .  Category: Basics
[36m|[0m  [37mPython is a high-level, interpreted, general-purpose programming language prized for its clean syntax, readability, and huge ecosystem of libraries.[0m
|
|  Confidence  ##############################  100.0%
+--------------------------------------------------------------------+
......................................................................

You > 
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 NLP Pipeline | Tokenization → Stopword removal → Lemmatization |
| 📊 TF-IDF Matching | Understands meaning, not just exact keywords |
| 📐 Cosine Similarity | Finds the most relevant answer mathematically |
| 📈 Confidence Bar | Visual score showing how confident the match is |
| ⌨️ Typing Animation | Bot "types" the answer character by character |
| 🎨 Colored Terminal UI | Clean colored output using Colorama |
| 📂 8 Categories | Basics, Data Types, OOP, Advanced, File I/O and more |
| 📊 Session Summary | Tracks total turns, matched and unanswered queries |
| 💡 Help Command | Tips for getting the best results |

---

## 🧠 How It Works

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         NLP Preprocessing           │
│  lowercase → remove punctuation     │
│  → tokenize → drop stopwords        │
│  → lemmatize (WordNetLemmatizer)    │
└─────────────────────────────────────┘
    │
    ▼ cleaned query text
┌─────────────────────────────────────┐
│       TF-IDF Vectorization          │
│  Query → numeric vector             │
│  (fitted once at startup on FAQ     │
│   corpus — fast on every query)     │
└─────────────────────────────────────┘
    │
    ▼ query vector
┌─────────────────────────────────────┐
│       Cosine Similarity             │
│  Compare query vector against       │
│  all pre-computed FAQ vectors       │
│  → pick highest scoring match       │
└─────────────────────────────────────┘
    │
    ▼ best match (if score ≥ 0.15)
┌─────────────────────────────────────┐
│       Display Answer                │
│  Category label + confidence bar    │
│  + typing animation                 │
└─────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **NLTK** — tokenization, stopword removal, WordNet lemmatization
- **scikit-learn** — TF-IDF Vectorizer, Cosine Similarity
- **Colorama** — colored terminal output (cross-platform)

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/faq-chatbot.git
cd faq-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the chatbot
```bash
python src/faq_chatbot.py
```

> NLTK data (`punkt`, `stopwords`, `wordnet`) downloads automatically on first run.

---

## 📁 Project Structure

```
faq-chatbot/
│
├── src/
│   └── faq_chatbot.py      # Full chatbot — NLP pipeline + UI + chat loop
│
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

---

## 💬 Supported Commands

| Command | Action |
|---------|--------|
| Any question | Finds the best matching FAQ answer |
| `help` or `?` | Shows tips for better queries |
| `exit` / `quit` / `bye` / `q` | Exits with session summary |

---

## 📚 FAQ Categories Covered

| Category | Example Questions |
|----------|------------------|
| Basics | What is Python? |
| Installation | How to install Python? |
| Environment | What is a virtual environment? |
| Data Types | List vs tuple, dictionaries |
| Functions | Lambda functions |
| OOP | Classes, objects, inheritance |
| Advanced | Decorators, generators |
| Syntax | List comprehension, f-strings |
| File I/O | Reading and writing files |
| Error Handling | Try-except blocks |

---

## 🔬 NLP Pipeline — Under the Hood

| Step | Tool | Purpose |
|------|------|---------|
| Lowercasing | Python built-in | Normalize case |
| Punctuation removal | `re.sub()` | Clean noise |
| Tokenization | `nltk.word_tokenize` | Split into words |
| Stopword removal | `nltk.corpus.stopwords` | Drop "the", "is", "a" etc. |
| Lemmatization | `WordNetLemmatizer` | "running" → "run" |
| Vectorization | `TfidfVectorizer` | Words → numeric vectors |
| Similarity | `cosine_similarity` | Find closest FAQ match |
