import pandas as pd
import sys
import json 

CAMPIONI_PER_CLASSE = 10000 
COLONNE_DA_SALVARE = ['id', 'body', 'label']

FILE_ETICHETTE = "entity_annotations.csv"
FILE_ARTICOLI_GRANDE = "articles.csv"
FILE_CAMPIONE_FINALE = "prova.csv"
ANNOTATION_TYPE_ID_VERACITY = 6 

print(f"Carico le annotazioni da {FILE_ETICHETTE}...")
try:
    df_labels = pd.read_csv(FILE_ETICHETTE)
except FileNotFoundError:
    print(f"ERRORE: File '{FILE_ETICHETTE}' non trovato.")
    sys.exit()

print(f"Filtraggio per annotation_type_id == {ANNOTATION_TYPE_ID_VERACITY}...")

df_veracity = df_labels[
    (df_labels['annotation_type_id'] == ANNOTATION_TYPE_ID_VERACITY) & 
    (df_labels['entity_type'] == 'articles')
].copy()

print(f"Trovate {len(df_veracity)} annotazioni di tipo 'veracity'.")

def parse_claims_logic(value_str):
    """
    Analizza il JSON. 
    Restituisce 1 SOLO se TUTTI i claim validi sono corretti.
    Restituisce 0 se c'è ALMENO un claim scorretto (disinformazione).
    """
    try:
        data = json.loads(value_str)
        claims_list = data.get('claims', [])
        
        if not claims_list:
            return None 
            
        claims_validi_totali = 0
        claims_errati = 0
        
        for claim in claims_list:
            rating = claim.get('rating', '').lower()
            stance = claim.get('stance', '').lower()
            is_true = rating in ['true', 'mostly-true']
            is_false = rating in ['false', 'mostly-false']
            is_supporting = stance == 'supporting'
            is_contradicting = stance == 'contradicting'
            if (is_true and is_supporting) or (is_false and is_contradicting):
                claims_validi_totali += 1
            elif (is_false and is_supporting) or (is_true and is_contradicting):
                claims_validi_totali += 1
                claims_errati += 1

        if claims_validi_totali == 0:
            return None
        if claims_errati > 0:
            return 0
        else:
            return 1
    except Exception as e:
        return None

print("Parsing dei Claims con logica RIGOROSA (1 solo se 100% corretto)...")
df_veracity['label'] = df_veracity['value'].apply(parse_claims_logic)
df_veracity = df_veracity.dropna(subset=['label'])
df_veracity['label'] = df_veracity['label'].astype(int)
mappa_etichette = df_veracity.set_index('entity_id')['label'].to_dict()

print(f"Mappa etichette pronta. {len(mappa_etichette)} articoli validi trovati.")

if not mappa_etichette:
    print("ERRORE: Nessuna etichetta valida trovata.")
    sys.exit()

chunk_size = 10000
lista_affidabili = []
lista_non_affidabili = []
conteggio_affidabili = 0
conteggio_non_affidabili = 0

print(f"Avvio lettura e join con {FILE_ARTICOLI_GRANDE}...")

try:
    for chunk in pd.read_csv(FILE_ARTICOLI_GRANDE, chunksize=chunk_size):
        
        chunk['label'] = chunk['id'].map(mappa_etichette)
        labeled_chunk = chunk.dropna(subset=['label'])
        
        if not labeled_chunk.empty:
            affidabili = labeled_chunk[labeled_chunk['label'] == 1][COLONNE_DA_SALVARE]
            non_affidabili = labeled_chunk[labeled_chunk['label'] == 0][COLONNE_DA_SALVARE]
            
            if conteggio_affidabili < CAMPIONI_PER_CLASSE:
                ne_servono = CAMPIONI_PER_CLASSE - conteggio_affidabili
                da_aggiungere = affidabili.head(ne_servono)
                lista_affidabili.append(da_aggiungere)
                conteggio_affidabili += len(da_aggiungere)
                
            if conteggio_non_affidabili < CAMPIONI_PER_CLASSE:
                ne_servono = CAMPIONI_PER_CLASSE - conteggio_non_affidabili
                da_aggiungere = non_affidabili.head(ne_servono)
                lista_non_affidabili.append(da_aggiungere)
                conteggio_non_affidabili += len(da_aggiungere)

        if (conteggio_affidabili >= CAMPIONI_PER_CLASSE and 
            conteggio_non_affidabili >= CAMPIONI_PER_CLASSE):
            print("Target raggiunto! Interrompo la lettura.")
            break 

except FileNotFoundError:
    print(f"ERRORE: File '{FILE_ARTICOLI_GRANDE}' non trovato.")
    sys.exit()

print("Assemblaggio dataset finale...")

if not lista_affidabili and not lista_non_affidabili:
    print("ERRORE: Nessun articolo trovato.")
else:
    df_campioni_aff = pd.concat(lista_affidabili) if lista_affidabili else pd.DataFrame()
    df_campioni_non_aff = pd.concat(lista_non_affidabili) if lista_non_affidabili else pd.DataFrame()
    df_campione_finale = pd.concat([df_campioni_aff, df_campioni_non_aff])
    df_campione_finale = df_campione_finale.sample(frac=1, random_state=42).reset_index(drop=True)
    df_campione_finale.to_csv(FILE_CAMPIONE_FINALE, index=False)
    print(f"\nFile '{FILE_CAMPIONE_FINALE}' creato con successo!")
    print(f"Totale articoli: {len(df_campione_finale)}")
    print(df_campione_finale['label'].value_counts())