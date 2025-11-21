import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier


FILE_FEATURE = "features_dataset_per_analisi.csv"

print("Caricamento del dataset...")
try:
    df = pd.read_csv(FILE_FEATURE)
except FileNotFoundError:
    print("ERRORE: File non trovato. Assicurati che il file CSV sia nella cartella.")
    exit()

print(f"Dati caricati: {df.shape[0]} righe, {df.shape[1]} colonne.")
X = df[['numeric_density', 'subjectivity', 'hedge_density', 'adjective_density' , 
    'avg_word_len', 'exclamation_ratio', 'first_person_pronouns', 'lexical_diversity']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


print("\n--- BILANCIAMENTO DEL TRAINING SET ---")
conteggio = y_train.value_counts()
percentuale = y_train.value_counts(normalize=True) * 100

print(f"Classe 0 (Non Affidabile): {conteggio.get(0, 0)} ({percentuale.get(0, 0):.2f}%)")
print(f"Classe 1 (Affidabile):     {conteggio.get(1, 0)} ({percentuale.get(1, 0):.2f}%)")

if abs(conteggio.get(0, 0) - conteggio.get(1, 0)) < 500:
    print("Il dataset è ben bilanciato.")
else:
    print("Attenzione: c'è un certo sbilanciamento.")

print(f"Training Set: {X_train.shape[0]} articoli")
print(f"Test Set:     {X_test.shape[0]} articoli")
print("\nAddestramento del modello XGBoost in corso...")
model = XGBClassifier(
    n_estimators=400,   
    learning_rate=0.1,  
    max_depth=5,        
    random_state=42
)

model.fit(X_train, y_train)
print("Modello addestrato!")


print("\nValutazione sul Test Set (dati mai visti prima)...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- RISULTATO FINALE ---")
print(f"ACCURATEZZA: {accuracy:.2%}")
print("-" * 30)
print("\nReport di Classificazione:")
print(classification_report(y_test, y_pred, target_names=['Non Affidabile (0)', 'Affidabile (1)']))
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importanza': model.feature_importances_
}).sort_values(by='Importanza', ascending=False)

print("\n--- CLASSIFICA DELLE FEATURE PIÙ IMPORTANTI ---")
print(feature_importances)
try:
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred: Fake', 'Pred: Vero'],
                yticklabels=['Reale: Fake', 'Reale: Vero'])
    plt.title('Matrice di Confusione')
    plt.ylabel('Etichetta Reale')
    plt.xlabel('Etichetta Predetta')
    plt.show()
except Exception:
    pass