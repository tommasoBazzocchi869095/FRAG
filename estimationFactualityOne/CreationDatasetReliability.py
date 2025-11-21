import pandas as pd
import sys
import json
import re
from textblob import TextBlob
import nltk

required_nltk_packages = [
    'punkt', 
    'punkt_tab', 
    'averaged_perceptron_tagger', 
    'averaged_perceptron_tagger_eng',
    'brown'
]

for package in required_nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{package}')
    except LookupError:
        try:
            nltk.data.find(f'taggers/{package}')
        except LookupError:
            print(f"Download automatico del pacchetto NLTK mancante: {package}...")
            nltk.download(package, quiet=True)

COLONNA_TESTO = 'body' 
COLONNA_SOURCE_ID = 'source_id'

FILE_ETICHETTE = "entity_annotations.csv"
FILE_ARTICOLI_GRANDE = "articles.csv"
FILE_FEATURE_FINALE = "features_dataset_per_analisi.csv" 
HEDGE_WORDS = {
    "may", "might", "could", "suggests", "appears", 
    "potentially", "seems", "likely", "probably"
}

def clean_html(text):
    """Pulisce il testo dall'HTML."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'<[^>]+>', ' ', text)

def calculate_numeric_density(text):
    """Calcola la densità di numeri."""
    clean_text = clean_html(text)
    numbers = re.findall(r'\b\d+[\.,]?\d*\b', clean_text)
    word_count = len(clean_text.split())
    if word_count == 0: return 0
    return len(numbers) / word_count

def calculate_subjectivity(text):
    """Calcola la soggettività usando TextBlob."""
    clean_text = clean_html(text)
    if not clean_text: return 0
    return TextBlob(clean_text).sentiment.subjectivity

def calculate_hedge_density(text):
    """Calcola la densità di parole caute (hedge words)."""
    clean_text = clean_html(text)
    words = clean_text.lower().split()
    word_count = len(words)
    if word_count == 0: return 0
    
    hedge_count = 0
    for word in words:
        if word in HEDGE_WORDS:
            hedge_count += 1
            
    return hedge_count / word_count

def calculate_adjective_density(text):
    """Calcola la densità di aggettivi usando TextBlob."""
    clean_text = clean_html(text)
    if not clean_text: return 0
    
    blob = TextBlob(clean_text)
    adj_count = 0
    for word, tag in blob.tags:
        if tag.startswith('JJ'):
            adj_count += 1
            
    word_count = len(blob.words)
    if word_count == 0: return 0
    return adj_count / word_count

def calculate_avg_word_length(text):
    """Media lunghezza parole (Proxy per complessità tecnica)."""
    clean_text = clean_html(text)
    words = clean_text.split()
    if len(words) == 0: return 0
    total_chars = sum(len(word) for word in words)
    return total_chars / len(words)

def calculate_exclamation_ratio(text):
    """Sensazionalismo (Punti esclamativi)."""
    if not isinstance(text, str) or len(text) == 0: return 0
    return text.count('!') / len(text)

def calculate_first_person_pronouns(text):
    """Aneddotica (Io, me, mio)."""
    clean_text = clean_html(text).lower()
    words = clean_text.split()
    if len(words) == 0: return 0
    pronouns = {"i", "me", "my", "mine", "myself"}
    count = sum(1 for word in words if word in pronouns)
    return count / len(words)

def calculate_lexical_diversity(text):
    """Ricchezza del vocabolario."""
    clean_text = clean_html(text).lower()
    words = clean_text.split()
    if len(words) == 0: return 0
    unique_words = set(words)
    return len(unique_words) / len(words)



print(f"Carico le etichette da {FILE_ETICHETTE}...")
try:
    df_labels = pd.read_csv(FILE_ETICHETTE)
except FileNotFoundError:
    print(f"ERRORE: File '{FILE_ETICHETTE}' non trovato.")
    sys.exit()

df_labels_sources = df_labels[
    (df_labels['annotation_type_id'] == 1) & 
    (df_labels['entity_type'] == 'sources')
].copy()

def parse_label_json(value_str):
    try:
        data = json.loads(value_str)
        label = data.get('value', '').lower()
        if label == 'reliable': return 1
        elif label == 'unreliable': return 0
        else: return None
    except:
        return None

df_labels_sources['label'] = df_labels_sources['value'].apply(parse_label_json)
df_labels_sources = df_labels_sources.dropna(subset=['label'])
mappa_etichette = df_labels_sources.set_index('entity_id')['label'].to_dict()

print(f"Mappa etichette creata. Trovate {len(mappa_etichette)} fonti etichettate.")
if not mappa_etichette:
    print("ERRORE: La mappa delle etichette è vuota.")
    sys.exit()

chunk_size = 10000
results_list = [] 
TARGET_PER_CLASSE = 10000 
conteggio_reliable = 0
conteggio_unreliable = 0

print(f"Avvio analisi da {FILE_ARTICOLI_GRANDE}...")

try:
    for chunk in pd.read_csv(FILE_ARTICOLI_GRANDE, chunksize=chunk_size):
        
        chunk['label'] = chunk[COLONNA_SOURCE_ID].map(mappa_etichette)
        labeled_chunk = chunk.dropna(subset=['label'])
        
        if not labeled_chunk.empty:
            features = pd.DataFrame()
            features['label'] = labeled_chunk['label']
            features['numeric_density'] = labeled_chunk[COLONNA_TESTO].apply(calculate_numeric_density)
            features['subjectivity'] = labeled_chunk[COLONNA_TESTO].apply(calculate_subjectivity)
            features['hedge_density'] = labeled_chunk[COLONNA_TESTO].apply(calculate_hedge_density)
            features['adjective_density'] = labeled_chunk[COLONNA_TESTO].apply(calculate_adjective_density)
            features['avg_word_len'] = labeled_chunk[COLONNA_TESTO].apply(calculate_avg_word_length)
            features['exclamation_ratio'] = labeled_chunk[COLONNA_TESTO].apply(calculate_exclamation_ratio)
            features['first_person_pronouns'] = labeled_chunk[COLONNA_TESTO].apply(calculate_first_person_pronouns)
            features['lexical_diversity'] = labeled_chunk[COLONNA_TESTO].apply(calculate_lexical_diversity)
            results_list.append(features)
            counts = features['label'].value_counts()
            conteggio_reliable += counts.get(1, 0)
            conteggio_unreliable += counts.get(0, 0)
            
            print(f"Trovati finora: {conteggio_reliable} Affidabili, {conteggio_unreliable} Non Affidabili")
            if conteggio_reliable >= TARGET_PER_CLASSE and conteggio_unreliable >= TARGET_PER_CLASSE:
                print("Raggiunto il numero target di articoli! Interrompo l'analisi.")
                break

except FileNotFoundError:
    print(f"ERRORE: File '{FILE_ARTICOLI_GRANDE}' non trovato.")
    sys.exit()
except KeyError as e:
    print(f"ERRORE: Colonna non trovata: {e}. Controlla i nomi delle colonne.")
    sys.exit()

print("Elaborazione completata. Creazione del file di feature finale...")
if not results_list:
    print("ERRORE: Non sono stati trovati articoli etichettati.")
else:
    df_feature_finale = pd.concat(results_list, ignore_index=True)
    df_feature_finale.to_csv(FILE_FEATURE_FINALE, index=False)

    print(f"\nFile '{FILE_FEATURE_FINALE}' creato con successo!")
    print(f"Contiene {len(df_feature_finale)} articoli analizzati.")
    print("Questo file è piccolo e puoi aprirlo con il tuo editor.")

    print("\n--- ANALISI COMPARATIVA DELLE CARATTERISTICHE (Metodo 1) ---")
    print(df_feature_finale.groupby('label').mean())