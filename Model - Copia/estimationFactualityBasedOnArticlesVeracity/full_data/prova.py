import pandas as pd
import re
from transformers import AutoTokenizer

# Configurazione
FILE_TRAIN = "dataset_train.csv"
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# Carica solo un campione per fare prima
try:
    df = pd.read_csv(FILE_TRAIN)
    # Prendiamo 5 righe casuali e 5 righe che contengono sicuramente '<'
    sample_random = df.sample(n=min(5, len(df)))
    sample_html = df[df['body'].str.contains('<', na=False, regex=False)].head(5)
    
    samples = pd.concat([sample_random, sample_html]).drop_duplicates()
except Exception as e:
    print(f"Errore caricamento: {e}")
    exit()

print(f"--- DIAGNOSTICA HTML SU {FILE_TRAIN} ---")

# Controlla quanti record hanno tag HTML potenziali
html_pattern = re.compile(r'<[^>]+>')
total_html = df['body'].astype(str).apply(lambda x: bool(html_pattern.search(x))).sum()
print(f"Record totali: {len(df)}")
print(f"Record con potenziali tag HTML: {total_html} ({total_html/len(df):.2%})")

if total_html == 0:
    print(" IL DATASET È GIÀ PULITO! Puoi rimuovere la funzione clean_html.")
else:
    print(f" ATTENZIONE: Trovati {total_html} articoli con HTML. La pulizia è NECESSARIA.")
    
    # Esempio pratico di spreco token
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Prendi il primo testo sporco che trova
    dirty_text = samples[samples['body'].str.contains('<', na=False, regex=False)].iloc[0]['body']
    dirty_text = str(dirty_text)[:500] # Prendiamo solo l'inizio
    
    # Simula pulizia
    clean_text = re.sub(r'<[^>]+>', ' ', dirty_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # Conta i token
    tokens_dirty = len(tokenizer.encode(dirty_text, add_special_tokens=False))
    tokens_clean = len(tokenizer.encode(clean_text, add_special_tokens=False))
    
    print("\n--- ESEMPIO DI SPRECO TOKEN ---")
    print(f"Testo Originale (primi 100 char): {dirty_text[:100]}...")
    print(f"Token SPORCHI: {tokens_dirty}")
    print(f"Token PULITI:  {tokens_clean}")
    print(f"Token risparmiati: {tokens_dirty - tokens_clean}")
    
    if tokens_dirty > tokens_clean:
        print("\nCONCLUSIONE: Mantieni la funzione clean_html. Senza di essa, BERT spreca spazio analizzando tag inutili.")