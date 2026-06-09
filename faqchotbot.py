"""
FAQ Chatbot  ─  NLP-powered (NLTK + TF-IDF + Cosine Similarity)
Install dependencies:
    pip install nltk scikit-learn colorama

Usage:
    python faqchatbot.py
"""

import re
import sys
import time
import os
import nltk
from colorama import Fore, Back, Style, init
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK downloads (silent)
for _pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Colorama init
init(autoreset=True)

C  = Fore.CYAN
Y  = Fore.YELLOW
G  = Fore.GREEN
M  = Fore.MAGENTA
W  = Fore.WHITE
R  = Fore.RED
DW = Fore.LIGHTBLACK_EX
RS = Style.RESET_ALL
BD = Style.BRIGHT


def _spinner_task(label: str, fn):
    """Run fn() while showing a spinner, then print done."""
    frames = ["|  ", "/  ", "-  ", "\\  "]
    result = [None]
    done   = [False]

    import threading
    def _run():
        result[0] = fn()
        done[0] = True

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    i = 0
    while not done[0]:
        sys.stdout.write(f"\r  {C}{frames[i % 4]}{RS} {DW}{label}{RS}")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    sys.stdout.write(f"\r  {G}OK {RS} {W}{label}{RS}\n")
    sys.stdout.flush()
    return result[0]


print(f"\n  {BD}{M}Initialising FAQ Chatbot ...{RS}\n")

# Step 1 - NLTK downloads
def _nltk_setup():
    for _pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        nltk.download(_pkg, quiet=True)

_spinner_task("Downloading NLTK data ...", _nltk_setup)

FAQ_DATA = [
    {
        "question": "What is Python?",
        "answer":   "Python is a high-level, interpreted, general-purpose programming language "
                    "prized for its clean syntax, readability, and huge ecosystem of libraries.",
        "category": "Basics",
    },
    {
        "question": "How to install Python?",
        "answer":   "Download the installer from https://python.org and follow the setup wizard. "
                    "On Windows, tick ✔ 'Add Python to PATH' before clicking Install.",
        "category": "Installation",
    },
    {
        "question": "What is a Python virtual environment?",
        "answer":   "A virtual environment is an isolated Python environment that lets you manage "
                    "project-specific dependencies without polluting the global install.\n"
                    "  Create:  python -m venv .venv\n"
                    "  Activate (Win): .venv\\Scripts\\activate\n"
                    "  Activate (Mac/Linux): source .venv/bin/activate",
        "category": "Environment",
    },
    {
        "question": "How to install packages in Python?",
        "answer":   "Use pip:  pip install package_name\n"
                    "  Specific version:  pip install package_name==1.2.3\n"
                    "  From requirements file:  pip install -r requirements.txt",
        "category": "Packages",
    },
    {
        "question": "What is a list in Python?",
        "answer":   "A list is an ordered, mutable collection of items.\n"
                    "  Example:  my_list = [1, 'hello', 3.14]\n"
                    "  Access:   my_list[0]   →  1\n"
                    "  Append:   my_list.append(True)",
        "category": "Data Types",
    },
    {
        "question": "What is the difference between a list and a tuple?",
        "answer":   "Lists [ ] are mutable — you can add, remove, or change items.\n"
                    "Tuples ( ) are immutable — once created they cannot be modified.\n"
                    "Tuples are slightly faster and are used for fixed data (e.g. coordinates).",
        "category": "Data Types",
    },
    {
        "question": "What is a dictionary in Python?",
        "answer":   "A dictionary is an ordered (Python 3.7+) collection of key-value pairs.\n"
                    "  Example:  person = {'name': 'Alice', 'age': 25}\n"
                    "  Access:   person['name']  →  'Alice'\n"
                    "  Keys must be hashable (e.g. strings, numbers, tuples).",
        "category": "Data Types",
    },
    {
        "question": "How to read a file in Python?",
        "answer":   "Use the open() context manager:\n"
                    "  with open('file.txt', 'r', encoding='utf-8') as f:\n"
                    "      content = f.read()     # entire file as string\n"
                    "      lines   = f.readlines() # list of lines",
        "category": "File I/O",
    },
    {
        "question": "How to write to a file in Python?",
        "answer":   "Open the file in write or append mode:\n"
                    "  with open('out.txt', 'w', encoding='utf-8') as f:\n"
                    "      f.write('Hello, world!\\n')\n"
                    "  Use mode='a' to append without overwriting.",
        "category": "File I/O",
    },
    {
        "question": "What is a lambda function?",
        "answer":   "A lambda is an anonymous, single-expression function.\n"
                    "  Syntax:  lambda args: expression\n"
                    "  Example: square = lambda x: x ** 2  →  square(5) = 25\n"
                    "  Useful in map(), filter(), and sorted() as inline callbacks.",
        "category": "Functions",
    },
    {
        "question": "What is object-oriented programming in Python?",
        "answer":   "OOP organises code around classes (blueprints) and objects (instances).\n"
                    "  class Dog:\n"
                    "      def __init__(self, name): self.name = name\n"
                    "      def bark(self): print(f'{self.name} says Woof!')\n"
                    "  d = Dog('Rex');  d.bark()",
        "category": "OOP",
    },
    {
        "question": "What are Python decorators?",
        "answer":   "Decorators are callables that wrap a function to extend its behaviour "
                    "without modifying its source code.  Apply with the @ syntax:\n"
                    "  @my_decorator\n"
                    "  def greet(): print('Hello')\n"
                    "  Common built-ins: @staticmethod, @classmethod, @property.",
        "category": "Advanced",
    },
    {
        "question": "How do I handle exceptions in Python?",
        "answer":   "Use try-except-else-finally blocks:\n"
                    "  try:\n"
                    "      result = 10 / 0\n"
                    "  except ZeroDivisionError as e:\n"
                    "      print(f'Error: {e}')\n"
                    "  finally:\n"
                    "      print('Always runs')",
        "category": "Error Handling",
    },
    {
        "question": "What is list comprehension?",
        "answer":   "A concise syntax to build a list from an iterable.\n"
                    "  squares  = [x**2 for x in range(10)]\n"
                    "  evens    = [x for x in range(20) if x % 2 == 0]\n"
                    "  flat     = [n for row in matrix for n in row]",
        "category": "Syntax",
    },
    {
        "question": "What are Python generators?",
        "answer":   "Generators are functions that yield values one at a time, saving memory.\n"
                    "  def count_up(n):\n"
                    "      for i in range(n): yield i\n"
                    "  Generator expressions:  (x**2 for x in range(10))",
        "category": "Advanced",
    },
    {
        "question": "How do I use f-strings in Python?",
        "answer":   "f-strings (Python 3.6+) embed expressions directly in string literals.\n"
                    "  name = 'World'\n"
                    "  print(f'Hello, {name}!')      → Hello, World!\n"
                    "  print(f'{2 ** 10 = }')         → 2 ** 10 = 1024",
        "category": "Syntax",
    },
    {
        "question": "How do I exit the chatbot?",
        "answer":   "Type  exit,  quit,  bye,  or  q  — and I'll sign off with a smile.👋",
        "category": "Meta",
    },
]


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# 2. TEXT PREPROCESSING

_lemmatizer = WordNetLemmatizer()
_stop_words  = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    """Lowercase -> strip punctuation -> tokenize -> drop stopwords -> lemmatize."""
    text   = text.lower()
    text   = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t not in _stop_words and t.isalpha()
    ]
    return " ".join(tokens)


# Step 2 - pre-process FAQ questions
_processed_questions = _spinner_task(
    "Pre-processing FAQ questions ...",
    lambda: [preprocess(faq["question"]) for faq in FAQ_DATA]
)

# Step 3 - fit TF-IDF vectorizer
def _build_tfidf():
    v = TfidfVectorizer()
    m = v.fit_transform(_processed_questions)
    return v, m

_vectorizer, _faq_tfidf = _spinner_task("Building TF-IDF model ...", _build_tfidf)

print(f"\n  {G}{BD}Ready!{RS} {DW}Ask away.{RS}\n")


# 3. MATCHING ENGINE

def find_best_match(user_query: str, threshold: float = 0.15):
    """
    Returns (answer, category, confidence) or (None, None, 0.0) if below threshold.

    Uses a pre-fitted TF-IDF vectorizer (fast) rather than re-fitting every call.
    """
    processed = preprocess(user_query)
    if not processed:
        return None, None, 0.0

    query_vec   = _vectorizer.transform([processed])
    sims        = cosine_similarity(query_vec, _faq_tfidf).flatten()
    best_idx    = int(sims.argmax())
    best_score  = float(sims[best_idx])

    if best_score >= threshold:
        faq = FAQ_DATA[best_idx]
        return faq["answer"], faq["category"], round(best_score, 3)
    return None, None, 0.0



# 4. TERMINAL UI HELPERS


#  Palette 
C  = Fore.CYAN
Y  = Fore.YELLOW
G  = Fore.GREEN
M  = Fore.MAGENTA
W  = Fore.WHITE
R  = Fore.RED
DW = Fore.LIGHTBLACK_EX   # dim white / grey
RS = Style.RESET_ALL
BD = Style.BRIGHT

WIDTH = 70


def _hr(char="-", color=C):
    """Horizontal rule."""
    return f"{color}{char * WIDTH}{RS}"


def _box_line(text, color=W, pad=2):
    """A line inside a box with side borders."""
    inner = WIDTH - 2 - pad * 2
    return f"{C}|{RS}{' ' * pad}{color}{text:<{inner}}{RS}{' ' * pad}{C}|{RS}"


def _typing(text: str, delay: float = 0.012):
    """Print text with a typing-animation effect."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def _confidence_bar(score: float, width: int = 30) -> str:
    """Render a coloured progress bar for the confidence score."""
    filled = int(round(score * width))
    empty  = width - filled
    if score >= 0.6:
        bar_color = G
    elif score >= 0.35:
        bar_color = Y
    else:
        bar_color = R
    bar = f"{bar_color}{'#' * filled}{DW}{'.' * empty}{RS}"
    pct = f"{score * 100:5.1f}%"
    return f"{bar}  {bar_color}{BD}{pct}{RS}"


def print_banner():
    os.system("cls" if os.name == "nt" else "clear")
    print()
    print(f"{C}+{'=' * WIDTH}+{RS}")
    print(f"{C}|{RS}{' ' * WIDTH}{C}|{RS}")
    print(f"{C}|{RS}{BD}{M}{'🤖 FAQ CHATBOT - Python Programming Q&A':^{WIDTH}}{RS}{C}|{RS}")
    print(f"{C}|{RS}{' ' * WIDTH}{C}|{RS}")
    print(f"{C}+{'=' * WIDTH}+{RS}")
    print()
    print(f"{DW}> Ask me anything about Python programming.")
    print(f"{DW}> Type {W}{BD}help{RS}{DW} for tips, or {W}{BD}exit{RS}{DW} / {W}{BD}quit{RS}{DW} / {W}{BD}bye{RS}{DW} to leave.")
    print()
    print(_hr())
    print()


def print_help():
    tips = [
        ("Be specific",   "e.g. 'How do I create a virtual environment?'"),
        ("Use keywords",  "e.g. 'list vs tuple', 'file reading', 'decorators'"),
        ("Short queries", "e.g. 'lambda', 'generators', 'f-strings'"),
        ("Exit",          "type  exit / quit / bye / q"),
    ]
    print()
    print(f"{BD}{Y}+-  Tips for best results  {'-' * 30}+{RS}")
    for title, detail in tips:
        print(f"{Y}|{RS}  {BD}{W}{title:<14}{RS}  {DW}{detail}{RS}")
    print(f"{BD}{Y}+{'-' * 50}+{RS}")
    print()


def format_answer(answer: str, category: str, score: float, turn: int) -> str:
    """Build the styled bot-response block."""
    lines = []
    lines.append("")
    lines.append(f"{C}+-  {BD}Bot{RS}{C}  .  {DW}Turn {turn}  .  Category: {M}{category}{RS}")

    # Answer lines (indent multi-line answers)
    for i, ln in enumerate(answer.splitlines()):
        prefix = f"{C}|{RS}  " if i == 0 else f"{C}|{RS}     "
        lines.append(f"{prefix}{W}{ln}{RS}")

    # Confidence bar
    bar_str = _confidence_bar(score)
    lines.append(f"{C}|{RS}")
    lines.append(f"{C}|{RS}  {DW}Confidence  {bar_str}")
    lines.append(f"{C}+{'-' * (WIDTH - 2)}+{RS}")
    lines.append("")
    return "\n".join(lines)


EXIT_COMMANDS = {"exit", "quit", "bye", "q", ":q", "quit()", "exit()"}
HELP_COMMANDS = {"help", "?", "commands", "h"}



# 5. MAIN CHAT LOOP

def chat():
    print_banner()

    turn       = 0
    matched    = 0
    not_found  = 0

    while True:
        #   Prompt     
        try:
            user_input = input(f"{G}{BD}You >{RS} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {Y}Session interrupted.  Goodbye! 👋{RS}\n")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        #    Special commands   
        if cmd in EXIT_COMMANDS:
            print()
            print(f"{Y}Thanks for chatting!  Goodbye!{RS}")
            print(f"{DW}Turns      : {W}{turn}{RS}")
            print(f"{DW}Matched    : {G}{matched}{RS}")
            print(f"{DW}Not found  : {R}{not_found}{RS}")
            print()
            break

        if cmd in HELP_COMMANDS:
            print_help()
            continue

        turn += 1

        #    Match  
        answer, category, score = find_best_match(user_input)

        if answer:
            matched += 1
            # Print formatted block with typing animation for the first line
            block_lines = format_answer(answer, category, score, turn).splitlines()
            # Print all lines except answer content lines instantly;
            # animate just the answer text for a "bot typing" feel
            answer_lines = answer.splitlines()
            in_answer = False
            answer_idx = 0
            for ln in block_lines:
                # Detect which lines are the answer payload
                stripped = re.sub(r"\x1b\[[0-9;]*m", "", ln)  # strip ANSI
                if answer_idx < len(answer_lines) and answer_lines[answer_idx].strip() in stripped:
                    _typing(ln, delay=0.008)
                    answer_idx += 1
                else:
                    print(ln)
        else:
            not_found += 1
            print()
            print(f"{R}+-  Bot  .  Turn {turn}  {'-' * 40}{RS}")
            _typing(
                f"{R}|  Sorry, I couldn't find a relevant answer.\n"
                f"{R}|  Try rephrasing, or type {W}{BD}help{RS}{R} for query tips.",
                delay=0.01,
            )
            print(f"{R}+{'-' * (WIDTH - 2)}{RS}")
            print()

        print(_hr(char=".", color=DW))
        print()


#  ENTRY POINT

if __name__ == "__main__":
    chat()
