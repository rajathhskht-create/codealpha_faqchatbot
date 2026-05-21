"""
FAQ Chatbot using NLP (NLTK + Cosine Similarity)
Install dependencies: pip install nltk scikit-learn colorama
"""

import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from colorama import Fore, Style, init

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

init(autoreset=True)  # colorama auto-reset

# ─────────────────────────────────────────────
# 1. FAQ DATA  (topic: Python Programming)
# ─────────────────────────────────────────────
FAQ_DATA = [
    {
        "question": "What is Python?",
        "answer": "Python is a high-level, interpreted, general-purpose programming language known for its simplicity and readability.",
    },
    {
        "question": "How do I install Python?",
        "answer": "Download the installer from https://python.org and follow the setup wizard. Make sure to check 'Add Python to PATH'.",
    },
    {
        "question": "What is a Python virtual environment?",
        "answer": "A virtual environment is an isolated Python environment that allows you to manage dependencies per project using 'python -m venv env'.",
    },
    {
        "question": "How do I install packages in Python?",
        "answer": "Use pip: 'pip install package_name'. For a specific version: 'pip install package_name==1.2.3'.",
    },
    {
        "question": "What is a list in Python?",
        "answer": "A list is an ordered, mutable collection of items defined with square brackets, e.g., my_list = [1, 2, 3].",
    },
    {
        "question": "What is the difference between a list and a tuple?",
        "answer": "Lists are mutable (can be changed), while tuples are immutable (cannot be changed after creation).",
    },
    {
        "question": "What is a dictionary in Python?",
        "answer": "A dictionary is an unordered collection of key-value pairs, defined with curly braces, e.g., {'name': 'Alice', 'age': 25}.",
    },
    {
        "question": "How do I read a file in Python?",
        "answer": "Use the open() function: 'with open(\"file.txt\", \"r\") as f: content = f.read()'.",
    },
    {
        "question": "What is a lambda function?",
        "answer": "A lambda is an anonymous, single-expression function: e.g., square = lambda x: x ** 2.",
    },
    {
        "question": "What is object-oriented programming in Python?",
        "answer": "OOP in Python uses classes and objects to structure code. Use 'class MyClass:' to define a class.",
    },
    {
        "question": "What are Python decorators?",
        "answer": "Decorators are functions that modify the behaviour of another function, applied using the '@decorator' syntax.",
    },
    {
        "question": "How do I handle exceptions in Python?",
        "answer": "Use try-except blocks: 'try: ... except Exception as e: print(e)'.",
    },
    {
        "question": "What is list comprehension?",
        "answer": "A concise way to create lists: e.g., squares = [x**2 for x in range(10)].",
    },
    {
        "question": "How do I exit the chatbot?",
        "answer": "Type 'exit', 'quit', or 'bye' to end the conversation.",
    },
]


# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    """Lowercase → remove punctuation → tokenize → remove stopwords → lemmatize."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)          # remove punctuation
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


# Pre-process all FAQ questions once
processed_questions = [preprocess(faq["question"]) for faq in FAQ_DATA]


# ─────────────────────────────────────────────
# 3. MATCHING ENGINE (TF-IDF + Cosine Similarity)
# ─────────────────────────────────────────────
def find_best_match(user_query: str, threshold: float = 0.2):
    """Return (best_answer, confidence_score) or (None, 0) if below threshold."""
    processed_query = preprocess(user_query)
    # Build corpus: all FAQ questions + user query at the end
    corpus = processed_questions + [processed_query]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Cosine similarity between user query (last row) and all FAQ questions
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    best_idx = similarities.argmax()
    best_score = similarities[best_idx]

    if best_score >= threshold:
        return FAQ_DATA[best_idx]["answer"], round(float(best_score), 2)
    return None, 0.0


# ─────────────────────────────────────────────
# 4. CHAT UI  (terminal-based)
# ─────────────────────────────────────────────
BANNER = f"""
{Fore.CYAN}{'='*55}
        🤖  FAQ CHATBOT  (Python Programming)
{'='*55}{Style.RESET_ALL}
{Fore.YELLOW}  Ask me anything about Python!
  Type 'exit', 'quit', or 'bye' to leave.
{Fore.CYAN}{'='*55}{Style.RESET_ALL}
"""

EXIT_COMMANDS = {"exit", "quit", "bye", "q"}


def chat():
    print(BANNER)
    while True:
        try:
            user_input = input(f"{Fore.GREEN}You >{Style.RESET_ALL} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Fore.YELLOW}Goodbye! 👋{Style.RESET_ALL}")
            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print(f"{Fore.YELLOW}Bot > Goodbye! Have a great day! 👋{Style.RESET_ALL}")
            break

        answer, score = find_best_match(user_input)

        if answer:
            print(f"{Fore.CYAN}Bot > {answer}")
            print(f"      {Fore.WHITE}(confidence: {score}){Style.RESET_ALL}\n")
        else:
            print(
                f"{Fore.RED}Bot > Sorry, I couldn't find a relevant answer. "
                f"Try rephrasing your question.{Style.RESET_ALL}\n"
            )


# ─────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    chat()