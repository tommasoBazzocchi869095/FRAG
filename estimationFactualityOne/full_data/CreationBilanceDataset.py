import pandas as pd
import sys

FILE_INPUT = "campione_per_analisi_metodo1.csv"
FILE_OUTPUT = "campione_per_analisi_metodo1_BILANCIATO.csv"

print(f"Caricamento {FILE_INPUT}...")
try:
    df = pd.read_csv(FILE_INPUT)
except FileNotFoundError:
    print("Errore: File non trovato.")
    sys.exit()

print(f"Righe totali iniziali: {len(df)}")
print("Pulizia righe vuote...")
df = df.dropna(subset=['body', 'label'])
df = df[df['body'].astype(str).str.strip().astype(bool)]

print(f"Righe valide dopo la pulizia: {len(df)}")
counts = df['label'].value_counts()
min_count = counts.min()

print("\n--- SITUAZIONE PULITA ---")
print(counts)
print(f"Target di pareggio (minimo reale): {min_count}")
df_balanced = df.groupby('label').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n--- SITUAZIONE FINALE ---")
print(df_balanced['label'].value_counts())
print(f"Totale articoli finali: {len(df_balanced)}")

df_balanced.to_csv(FILE_OUTPUT, index=False)
print(f"File salvato: {FILE_OUTPUT}")