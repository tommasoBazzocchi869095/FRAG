import pandas as pd
import sys
import re
from textblob import TextBlob
import nltk
import spacy
import os


INPUT_FILES = ["dataset_train.csv", "dataset_val.csv", "dataset_test.csv"]
OUTPUT_FILES = ["features_train.csv", "features_val.csv", "features_test.csv"]

try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
except OSError:
    print("Scaricamento modello spaCy...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

required_nltk_packages = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']
for package in required_nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{package}')
    except LookupError:
        try:
            nltk.data.find(f'taggers/{package}')
        except LookupError:
            nltk.download(package, quiet=True)

HEDGE_WORDS = {
    "may", "might", "could", "suggests", "appears", 
    "potentially", "seems", "likely", "probably"
}

def clean_html(text):
    if not isinstance(text, str): return ""
    return re.sub(r'<[^>]+>', ' ', text)

def calculate_numeric_density(text):
    clean_text = clean_html(text)
    numbers = re.findall(r'\b\d+[\.,]?\d*\b', clean_text)
    word_count = len(clean_text.split())
    if word_count == 0: return 0
    return len(numbers) / word_count

def calculate_subjectivity(text):
    clean_text = clean_html(text)
    if not clean_text: return 0
    return TextBlob(clean_text).sentiment.subjectivity

def calculate_hedge_density(text):
    clean_text = clean_html(text)
    words = clean_text.lower().split()
    word_count = len(words)
    if word_count == 0: return 0
    return sum(1 for w in words if w in HEDGE_WORDS) / word_count

def calculate_adjective_density(text):
    clean_text = clean_html(text)
    if not clean_text: return 0
    blob = TextBlob(clean_text)
    adj_count = sum(1 for word, tag in blob.tags if tag.startswith('JJ'))
    word_count = len(blob.words)
    return adj_count / word_count if word_count > 0 else 0

def calculate_avg_word_length(text):
    clean_text = clean_html(text)
    words = clean_text.split()
    if len(words) == 0: return 0
    return sum(len(word) for word in words) / len(words)

def calculate_exclamation_ratio(text):
    if not isinstance(text, str) or len(text) == 0: return 0
    return text.count('!') / len(text)

def calculate_first_person_pronouns(text):
    clean_text = clean_html(text).lower()
    words = clean_text.split()
    if len(words) == 0: return 0
    pronouns = {"i", "me", "my", "mine", "myself"}
    return sum(1 for word in words if word in pronouns) / len(words)

def calculate_lexical_diversity(text):
    clean_text = clean_html(text).lower()
    words = clean_text.split()
    if len(words) == 0: return 0
    return len(set(words)) / len(words)


def calculate_fact_density(text):
    clean_text = clean_html(text)

    if not clean_text.strip(): return 0
    doc = nlp(clean_text[:100000])
    fact_count = 0
    sentence_count = 0
    for sent in doc.sents:
        sentence_count += 1
        has_subj = False
        has_obj = False
        for token in sent:
            if "subj" in token.dep_:
                has_subj = True
            if "obj" in token.dep_:
                has_obj = True
        if has_subj and has_obj:
            fact_count += 1
    if sentence_count == 0: return 0
    return fact_count / sentence_count


print("Inizio calcolo feature sui dataset sincronizzati...")
for file_in, file_out in zip(INPUT_FILES, OUTPUT_FILES):
    
    print(f"\n--- Elaborazione: {file_in} -> {file_out} ---")
    
    if not os.path.exists(file_in):
        print(f"ERRORE: Non trovo il file {file_in}. Hai eseguito lo script di split?")
        continue
    df = pd.read_csv(file_in)
    df['body'] = df['body'].astype(str)
    print(f"Articoli da analizzare: {len(df)}")
    df_features = pd.DataFrame()
    df_features['label'] = df['label']
    print("  > Calcolo Numeric Density...")
    df_features['numeric_density'] = df['body'].apply(calculate_numeric_density)
    print("  > Calcolo Subjectivity...")
    df_features['subjectivity'] = df['body'].apply(calculate_subjectivity)
    print("  > Calcolo Hedge Density...")
    df_features['hedge_density'] = df['body'].apply(calculate_hedge_density)
    print("  > Calcolo Adjective Density (Lento)...")
    df_features['adjective_density'] = df['body'].apply(calculate_adjective_density)
    print("  > Calcolo Avg Word Length...")
    df_features['avg_word_len'] = df['body'].apply(calculate_avg_word_length)
    print("  > Calcolo Exclamation Ratio...")
    df_features['exclamation_ratio'] = df['body'].apply(calculate_exclamation_ratio)
    print("  > Calcolo First Person Pronouns...")
    df_features['first_person_pronouns'] = df['body'].apply(calculate_first_person_pronouns)
    print("  > Calcolo Lexical Diversity...")
    df_features['lexical_diversity'] = df['body'].apply(calculate_lexical_diversity)
    print("  > Calcolo Fact Density (Molto Lento - usa spaCy)...")
    df_features['fact_density'] = df['body'].apply(calculate_fact_density)
    df_features.to_csv(file_out, index=False)
    print(f"Salvato: {file_out}")

print("\n--- TUTTO COMPLETATO! ---")
