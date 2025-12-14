import pandas as pd
from sklearn.model_selection import train_test_split
import os

FILE_MASTER = "campione_per_analisi_metodo1_BILANCIATO.csv"
SEED = 42

print("Caricamento Master Dataset...")
if not os.path.exists(FILE_MASTER):
    print(f"ERRORE: Devi prima creare '{FILE_MASTER}'")
    exit()

df = pd.read_csv(FILE_MASTER)
print(f"Righe totali PRIMA della pulizia: {len(df)}")
df = df.dropna(subset=['body', 'label'])
print(f"Articoli totali nel Master: {len(df)}")
df_train_full, df_test = train_test_split(
    df, test_size=0.20, stratify=df['label'], random_state=SEED
)

df_train, df_val = train_test_split(
    df_train_full, test_size=0.20, stratify=df_train_full['label'], random_state=SEED
)

def stampa_statistiche(nome_dataset, dataframe):
    totale = len(dataframe)
    conteggio = dataframe['label'].value_counts().sort_index()
    n_0 = conteggio.get(0.0, 0)
    n_1 = conteggio.get(1.0, 0)
    
    print(f"\n--- {nome_dataset.upper()} ---")
    print(f"Numero Righe: {totale}")
    print(f"  > Classe 0 (Non Affidabile): {n_0} ({n_0/totale:.2%})")
    print(f"  > Classe 1 (Affidabile):     {n_1} ({n_1/totale:.2%})")

stampa_statistiche("Training Set", df_train)
stampa_statistiche("Validation Set", df_val)
stampa_statistiche("Test Set", df_test)
print("Salvataggio file CSV...")
df_train.to_csv("dataset_train.csv", index=False)
df_val.to_csv("dataset_val.csv", index=False)
df_test.to_csv("dataset_test.csv", index=False)

print(f"\nFatto! File salvati: Train ({len(df_train)}) | Val ({len(df_val)}) | Test ({len(df_test)})")