import pandas as pd
import sys
import json 

CAMPIONI_PER_CLASSE = 10000 
COLONNE_DA_SALVARE = ['id', 'source_id', 'body', 'label'] 

FILE_ETICHETTE = "entity_annotations.csv"
FILE_ARTICOLI_GRANDE = "articles.csv"
FILE_CAMPIONE_FINALE = "campione_per_analisi_metodo1.csv"

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
        
        if label == 'reliable':
            return 1
        elif label == 'unreliable':
            return 0
        else:
            return None
    except:
        return None

df_labels_sources['label'] = df_labels_sources['value'].apply(parse_label_json)
df_labels_sources = df_labels_sources.dropna(subset=['label'])

mappa_etichette = df_labels_sources.set_index('entity_id')['label'].to_dict()

print(f"Mappa etichette creata. Trovate {len(mappa_etichette)} fonti etichettate.")
if not mappa_etichette:
    print("ERRORE: La mappa delle etichette è vuota. Controlla i filtri.")
    sys.exit()

chunk_size = 10000
lista_affidabili = []
lista_non_affidabili = []
conteggio_affidabili = 0
conteggio_non_affidabili = 0

print(f"Avvio campionamento da {FILE_ARTICOLI_GRANDE} (obiettivo: {CAMPIONI_PER_CLASSE} per classe)...")

try:
    for chunk in pd.read_csv(FILE_ARTICOLI_GRANDE, chunksize=chunk_size):
        
        chunk['label'] = chunk['source_id'].map(mappa_etichette)
        labeled_chunk = chunk.dropna(subset=['label'])
        
        if not labeled_chunk.empty:
            affidabili = labeled_chunk[labeled_chunk['label'] == 1][COLONNE_DA_SALVARE]
            non_affidabili = labeled_chunk[labeled_chunk['label'] == 0][COLONNE_DA_SALVARE]
            
            if conteggio_affidabili < CAMPIONI_PER_CLASSE:
                lista_affidabili.append(affidabili)
                conteggio_affidabili += len(affidabili)
                
            if conteggio_non_affidabili < CAMPIONI_PER_CLASSE:
                lista_non_affidabili.append(non_affidabili)
                conteggio_non_affidabili += len(non_affidabili)

        if (conteggio_affidabili >= CAMPIONI_PER_CLASSE and 
            conteggio_non_affidabili >= CAMPIONI_PER_CLASSE):
            print("Campionamento completato! Raggiunto il numero di campioni desiderato.")
            break 
            
        print(f"Processato blocco. Campioni: {conteggio_affidabili} Affidabili, {conteggio_non_affidabili} Non Affidabili")

except FileNotFoundError:
    print(f"ERRORE: File '{FILE_ARTICOLI_GRANDE}' non trovato.")
    sys.exit()
except KeyError as e:
    print(f"ERRORE: Colonna non trovata: {e}. Controlla che i nomi in COLONNE_DA_SALVARE siano corretti.")
    sys.exit()

print("Creazione del dataset finale...")

if not lista_affidabili or not lista_non_affidabili:
    print("ERRORE: Non sono stati trovati campioni sufficienti.")
else:
    df_campioni_aff = pd.concat(lista_affidabili).head(CAMPIONI_PER_CLASSE)
    df_campioni_non_aff = pd.concat(lista_non_affidabili).head(CAMPIONI_PER_CLASSE)
    
    df_campione_finale = pd.concat([df_campioni_aff, df_campioni_non_aff])
    
    df_campione_finale.to_csv(FILE_CAMPIONE_FINALE, index=False)

    print(f"\nFile '{FILE_CAMPIONE_FINALE}' creato con successo!")
    print(f"Contiene {len(df_campione_finale)} articoli totali.")
    print(f"Colonne salvate: {', '.join(COLONNE_DA_SALVARE)}")