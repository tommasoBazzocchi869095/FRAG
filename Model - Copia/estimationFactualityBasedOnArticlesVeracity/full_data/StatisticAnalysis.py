import pandas as pd
import json
import sys
import numpy as np
import re

FILE_ARTICOLI = "articles.csv"
FILE_ANNOTAZIONI = "entity_annotations.csv"
ANNOTATION_TYPE_ID_VERACITY = 6 

def clean_html_robust(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("--- CALCOLO STATISTICHE COMPLETE PER LA TESI (TESTO PULITO) ---\n")

print(f"1. Analisi del file '{FILE_ARTICOLI}' (Statistiche Testuali)...")
try:
    df_art = pd.read_csv(FILE_ARTICOLI, usecols=['id', 'body'])
    num_tot_articoli = len(df_art)
    df_art_clean = df_art.dropna(subset=['body']).copy()
    df_art_clean['body'] = df_art_clean['body'].astype(str)
    print("   -> Pulizia testo in corso (rimozione HTML/JS)...")
    clean_bodies = df_art_clean['body'].apply(clean_html_robust)
    
    word_counts = clean_bodies.apply(lambda x: len(x.split()))
    avg_words = word_counts.mean()
    median_words = word_counts.median()
    idx_max = word_counts.idxmax()
    max_words = word_counts.max()
    
    print(f"   -> Totale Articoli nel DB: {num_tot_articoli}")
    print(f"   -> Lunghezza Media (parole pulite): {avg_words:.2f}")
    print(f"   -> Lunghezza Mediana (parole pulite): {median_words:.2f}")
    print(f"   -> Lunghezza Max (parole pulite): {max_words}")
    print(f"   -> ID Articolo più lungo: {df_art_clean.loc[idx_max, 'id']}")

    del df_art, df_art_clean, word_counts, clean_bodies

except FileNotFoundError:
    print("   [ERRORE] File articoli non trovato.")
    sys.exit()

print(f"\n2. Analisi del file '{FILE_ANNOTAZIONI}'...")
try:
    df_ann = pd.read_csv(FILE_ANNOTAZIONI)
    tot_ann = len(df_ann)
    print(f"   -> Totale righe annotazioni: {tot_ann}")
    df_type1 = df_ann[
        (df_ann['annotation_type_id'] == 1) & 
        (df_ann['entity_type'] == 'sources')
    ].copy()
    print(f"   -> Annotazioni Tipo 1 (Affidabilità Fonte): {len(df_type1)}")
    df_type6 = df_ann[
        (df_ann['annotation_type_id'] == ANNOTATION_TYPE_ID_VERACITY) & 
        (df_ann['entity_type'] == 'articles')
    ].copy()
    print(f"   -> Annotazioni Tipo 6 (Veridicità Articolo): {len(df_type6)}")

except FileNotFoundError:
    print("   [ERRORE] File annotazioni non trovato.")
    sys.exit()

print("\n3. Dettaglio qualità Annotazioni Tipo 6 (Claims)...")

cnt_vuoti = 0
cnt_solo_unknown = 0
cnt_validi = 0
cnt_errore_json = 0

def analizza_dettaglio(value_str):
    global cnt_vuoti, cnt_solo_unknown, cnt_validi, cnt_errore_json
    try:
        data = json.loads(value_str)
        claims = data.get('claims', [])
        if not claims:
            cnt_vuoti += 1
            return
        
        has_valid_claim = False
        for c in claims:
            r = c.get('rating', '').lower()
            s = c.get('stance', '').lower()
            
            is_useful = r in ['true', 'mostly-true', 'false', 'mostly-false']
            is_stance_ok = s in ['supporting', 'contradicting']

            if is_useful and is_stance_ok:
                has_valid_claim = True
                break
        if not has_valid_claim:
            cnt_solo_unknown += 1
        else:
            cnt_validi += 1
            
    except:
        cnt_errore_json += 1

df_type6['value'].apply(analizza_dettaglio)
print(f"   Analisi su {len(df_type6)} record di Tipo 6:")
print(f"   ------------------------------------------------")
print(f"   A. Record con lista Claims VUOTA ([]):           {cnt_vuoti}")
print(f"   B. Record con solo Claims UNKNOWN/Inutili:       {cnt_solo_unknown}")
print(f"   C. Record VALIDI (Almeno 1 claim giudicabile):   {cnt_validi}")
print(f"   ------------------------------------------------")
print(f"   (Totale controllato: {cnt_vuoti + cnt_solo_unknown + cnt_validi + cnt_errore_json})")

if cnt_validi > 0:
    print(f"   [INFO] Il dataset Metodo 2 è stato estratto da questi {cnt_validi} record.")

print("\n4. Dettaglio Copertura Fonti (Tipo 1)...")
ids_fonti_annotate = set(df_type1['entity_id'].dropna().astype(int))
print(f"   -> Numero di fonti uniche annotate: {len(ids_fonti_annotate)}")

if len(ids_fonti_annotate) > 0:
    print("   -> Calcolo articoli coperti (scansione articles.csv in corso)...")
    
    articoli_coperti = 0
    articoli_senza_fonte = 0
    chunk_size = 50000 
    try:
        for chunk in pd.read_csv(FILE_ARTICOLI, usecols=['source_id'], chunksize=chunk_size):
            chunk_validi = chunk.dropna(subset=['source_id'])
            articoli_senza_fonte += (len(chunk) - len(chunk_validi))
            source_ids = pd.to_numeric(chunk_validi['source_id'], errors='coerce').dropna().astype(int)
            match = source_ids.isin(ids_fonti_annotate).sum()
            articoli_coperti += match
        print(f"   -> Articoli appartenenti alle {len(ids_fonti_annotate)} fonti annotate: {articoli_coperti}")
        print(f"   -> Percentuale sul totale ({num_tot_articoli}): {(articoli_coperti/num_tot_articoli)*100:.2f}%")
        print(f"   [INFO] Il dataset Metodo 1 è stato estratto da questi {articoli_coperti} articoli.")      
    except Exception as e:
        print(f"   [ERRORE nel calcolo fonti] {e}")
else:
    print(" Nessuna fonte annotata trovata, salto il calcolo copertura.")