import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

FILE_TRAIN = "features_train.csv"
FILE_VAL   = "features_val.csv"
FILE_TEST  = "features_test.csv"

all_dfs = []
file_list = [FILE_TRAIN, FILE_VAL, FILE_TEST]

print("Caricamento e unione dei dataset (Train, Val, Test)...")
for file in file_list:
    try:
        df_temp = pd.read_csv(file)
        all_dfs.append(df_temp)
        print(f"Caricato: {file} ({len(df_temp)} righe)")
    except FileNotFoundError:
        print(f"ERRORE: File {file} non trovato. Assicurati che tutti e tre i file esistano.")
        exit()

df_total = pd.concat(all_dfs, ignore_index=True)
print(f"\nDataFrame totale unito: {len(df_total)} righe.")
correlazione = df_total.corr(method='pearson')
corr_con_label = correlazione['label'].drop('label') 
corr_ordinata = corr_con_label.sort_values(ascending=False)

print("\n--- CLASSIFICA CORRELAZIONE CON L'AFFIDABILITÀ (Label 1) ---")
print(corr_ordinata)

try:
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlazione, 
                annot=True,       
                fmt=".2f",         
                cmap='coolwarm',   
                vmin=-1, vmax=1,  
                linewidths=.5,
                cbar_kws={'label': 'Correlation Coefficient (R)'})
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()

except ImportError:
    print("Seaborn/Matplotlib non installati. Salto il grafico.")