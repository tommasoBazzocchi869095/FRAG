import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import os
import numpy as np
import re

FILE_TRAIN = "dataset_train.csv"
FILE_VAL   = "dataset_val.csv"
FILE_TEST  = "dataset_test.csv"

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
OUTPUT_DIR = "./modello_finale_bert"

MAX_LEN = 512
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-5

def clean_html(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Caricamento dei 3 dataset (Train, Val, Test)...")

if not (os.path.exists(FILE_TRAIN) and os.path.exists(FILE_VAL) and os.path.exists(FILE_TEST)):
    print("ERRORE: Uno dei file dataset_*.csv non esiste. Esegui prima lo script di split.")
    exit()

df_train = pd.read_csv(FILE_TRAIN)
df_val   = pd.read_csv(FILE_VAL)
df_test  = pd.read_csv(FILE_TEST)

print("Pulizia del testo (Rimozione HTML)...")
for df in [df_train, df_val, df_test]:
    df.dropna(subset=['body', 'label'], inplace=True)
    df['body'] = df['body'].astype(str)
    df['body'] = df['body'].apply(clean_html)

print(f"TRAIN Set: {len(df_train)} articoli")
print(f"VAL   Set: {len(df_val)} articoli")
print(f"TEST  Set: {len(df_test)} articoli")

print("\nCaricamento Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def analizza_lunghezze(df, nome_set):
    print(f"\n--- Analisi Lunghezza Token: {nome_set} ---")
    texts = df['body'].astype(str).tolist()
    encodings = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
    lengths = [len(ids) for ids in encodings['input_ids']]
    
    num_totale = len(lengths)
    num_tagliati = sum(1 for l in lengths if l > MAX_LEN)
    percentuale = (num_tagliati / num_totale) * 100
    avg_len = np.mean(lengths)
    max_len = np.max(lengths)
    
    print(f"Totale documenti: {num_totale}")
    print(f"Documenti che superano {MAX_LEN} token: {num_tagliati} ({percentuale:.2f}%)")
    print(f"Lunghezza media: {avg_len:.1f} token")
    print(f"Lunghezza massima trovata: {max_len} token")

    plt.figure(figsize=(8, 4))
    sns.histplot(lengths, bins=50, kde=True)
    plt.axvline(x=MAX_LEN, color='r', linestyle='--', label=f'Taglio a {MAX_LEN}')
    plt.title(f'Distribuzione lunghezza token - {nome_set} (TESTO PULITO)')
    plt.xlabel('Numero di Token')
    plt.legend()
    plt.show()


analizza_lunghezze(df_train, "Train Set")
analizza_lunghezze(df_val, "Validation Set")
analizza_lunghezze(df_test, "Test Set")

def encode_dataset(texts, labels):
    encodings = tokenizer(
        texts.tolist(), 
        truncation=True, 
        padding=True, 
        max_length=MAX_LEN
    )
    return tf.data.Dataset.from_tensor_slices((dict(encodings), labels.tolist()))

print("Creazione Dataset TensorFlow...")
train_ds = encode_dataset(df_train['body'], df_train['label']).shuffle(1000).batch(BATCH_SIZE)
val_ds   = encode_dataset(df_val['body'],   df_val['label']).batch(BATCH_SIZE)
test_ds  = encode_dataset(df_test['body'],  df_test['label']).batch(BATCH_SIZE)

print("\nScaricamento PubMedBERT...")
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, from_pt=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

print("\n--- INIZIO TRAINING ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

print("\n--- INIZIO TEST FINALE (Su dati mai visti) ---")
predictions = model.predict(test_ds)
y_pred_logits = predictions.logits
y_pred = tf.argmax(y_pred_logits, axis=1).numpy()
y_true = df_test['label'].tolist()

accuracy = accuracy_score(y_true, y_pred)
print(f"\nACCURATEZZA FINALE SU TEST SET: {accuracy:.2%}")
print("-" * 40)
print(classification_report(y_true, y_pred, target_names=['Non Affidabile', 'Affidabile']))

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Pred: Fake', 'Pred: Vero'],
            yticklabels=['Reale: Fake', 'Reale: Vero'])
plt.title('Matrice di Confusione - PubMedBERT')
plt.show()

print(f"\nSalvataggio modello in {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Fatto! Modello salvato e pronto.")