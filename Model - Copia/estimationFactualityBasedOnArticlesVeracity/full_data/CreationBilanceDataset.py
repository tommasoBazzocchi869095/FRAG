import pandas as pd

FILE_INPUT = "campione_per_analisi_metodo2_claims.csv"
FILE_OUTPUT = "campione_per_analisi_metodo2_claims_BILANCIATO.csv"

print(f"Caricamento {FILE_INPUT}...")
try:
    df = pd.read_csv(FILE_INPUT)
except FileNotFoundError:
    print("Errore: File non trovato.")
    exit()


counts = df['label'].value_counts()
min_count = counts.min()

print("\n--- SITUAZIONE ATTUALE ---")
print(counts)
print(f"Target di pareggio: {min_count} per classe")
df_balanced = df.groupby('label').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n--- SITUAZIONE FINALE ---")
print(df_balanced['label'].value_counts())
print(f"Totale articoli finali: {len(df_balanced)}")

df_balanced.to_csv(FILE_OUTPUT, index=False)
print(f" File salvato: {FILE_OUTPUT}")