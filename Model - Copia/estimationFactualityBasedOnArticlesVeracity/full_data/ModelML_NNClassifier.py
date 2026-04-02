import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import os


FILE_TRAIN = "features_train.csv"
FILE_VAL   = "features_val.csv"
FILE_TEST  = "features_test.csv"

print("Caricamento dataset fissi...")
if not os.path.exists(FILE_TRAIN):
    print(f"ERRORE: Non trovo {FILE_TRAIN}. Hai lanciato lo script 'calcola_features_sincronizzate.py'?")
    exit()


df_train = pd.read_csv(FILE_TRAIN)
df_val   = pd.read_csv(FILE_VAL)
df_test  = pd.read_csv(FILE_TEST)

print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
FEATURE_COLS = [
    'numeric_density', 
    'subjectivity', 
    'hedge_density', 
    'adjective_density', 
    'avg_word_len', 
    'exclamation_ratio', 
    'first_person_pronouns', 
    'lexical_diversity', 
    'fact_density'
]

X_train = df_train[FEATURE_COLS]
y_train = df_train['label']
X_val   = df_val[FEATURE_COLS]
y_val   = df_val['label']
X_test  = df_test[FEATURE_COLS]
y_test  = df_test['label']

print("Scaling dei dati...")
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

dims = X_train.shape[1] 
model = Sequential()
model.add(Input(shape=(dims,))) 
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())
print("\nInizio training...")
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    epochs=500,           
    batch_size=32,
    validation_data=(X_val_scaled, y_val), 
    callbacks=[early_stop],
    verbose=1
)

print("\nValutazione sul Test Set (Fisso)...")
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- RISULTATO FINALE (Keras NN) ---")
print(f"ACCURATEZZA: {accuracy:.2%}")
print("-" * 30)
print("\nReport di Classificazione:")
print(classification_report(y_test, y_pred, target_names=['Non Affidabile (0)', 'Affidabile (1)']))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss Training')
plt.plot(history.history['val_loss'], label='Loss Validazione')
plt.title('Errore durante il Training')
plt.xlabel('Epoche')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy Training')
plt.plot(history.history['val_accuracy'], label='Accuracy Validazione')
plt.title('Accuratezza durante il Training')
plt.xlabel('Epoche')
plt.legend()

plt.show()