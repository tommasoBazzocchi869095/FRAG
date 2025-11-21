import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE_FEATURE = "features_dataset_per_analisi.csv"

print(f"Carico i dati da {FILE_FEATURE}...")
try:
    df = pd.read_csv(FILE_FEATURE)
except FileNotFoundError:
    print("ERRORE: File non trovato. Esegui prima lo script di analisi!")
    exit()

correlazione = df.corr(method='pearson')
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
                linewidths=.5)
    
    plt.title("Matrice di Correlazione Completa (Feature vs. Feature)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()

except ImportError:
    print("Seaborn/Matplotlib non installati. Salto il grafico.")