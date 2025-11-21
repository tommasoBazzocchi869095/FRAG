import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping


FILE_FEATURE = "features_dataset_per_analisi.csv"

print("Caricamento dati...")
try:
    df = pd.read_csv(FILE_FEATURE)
except FileNotFoundError:
    print("ERRORE: File non trovato.")
    exit()

X = df[['numeric_density', 'subjectivity', 'hedge_density', 'adjective_density' , # Le 4 vecchie
    'avg_word_len', 'exclamation_ratio', 'first_person_pronouns', 'lexical_diversity']]
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


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
    epochs=100,           
    batch_size=32,        
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

print("\nValutazione sul Test Set...")
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