# codealpha_faqchatbot
A terminal-based FAQ chatbot focused on Python programming topics. It uses NLTK for text preprocessing (lemmatization, stopword removal) and TF-IDF with cosine similarity via scikit-learn to match user queries to the most relevant FAQ answer, displaying results with color-coded output using Colorama.

Purpose:

A rule-based NLP chatbot that answers Python programming FAQs using text similarity — no LLM or API required.

Tech Stack:

LibraryRolenltkTokenization, lemmatization, stopword removalscikit-learnTF-IDF vectorization + cosine similaritycoloramaColored terminal outputrePunctuation cleaning.

How It Works:

FAQ Data — 14 hardcoded Python Q&A pairs stored as a list of dicts.

Preprocessing — Lowercases → strips punctuation → tokenizes → removes stopwords → lemmatizes each question.

Matching Engine — Combines all preprocessed FAQ questions + the user query into a TF-IDF matrix, then computes cosine similarity. Returns the best 
match if score ≥ 0.2 threshold.

Chat Loop — Continuously reads user input, finds a match, prints the answer with a confidence score, and exits on exit / quit / bye / q.


Key Functions:

FunctionDescriptionpreprocess(text)Cleans and normalizes any input stringfind_best_match(query, threshold)Returns (answer, score) or (None, 0.0)chat()Main terminal UI loop.

Limitations:

FAQ is static — answers must be manually added to FAQ_DATA.

TF-IDF is keyword-dependent — paraphrased questions may score low.

No context/memory — each query is treated independently.
