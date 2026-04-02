import os

# === MODIFICA CRUCIALE: IMPEDIAMO A TENSORFLOW DI TOCCARE LA GPU ===
# TensorFlow serve solo per caricare i pesi iniziali (from_tf=True).
# Se non facciamo questo, TF occupa tutta la VRAM e PyTorch crashta.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
try:
    # Nascondiamo la GPU a TensorFlow
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass
# ===================================================================

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import numpy as np


class FactualityScorer:
    def __init__(self, model_path="../Model/estimationFactualityOne/modello_finale_bertSourceBased", max_len=512):
        print(f"[FactualityScorer] Inizializzazione modello da: {model_path}")

        # Ora possiamo usare CUDA tranquillamente perché TF è confinato alla CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[FactualityScorer] Device selezionato per inferenza: {self.device}")

        self.max_len = max_len

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Caricamento Modello
            # TensorFlow caricherà i pesi nella RAM di sistema (CPU)
            # Poi verranno convertiti in PyTorch
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            except OSError:
                print("[FactualityScorer] Pesi PyTorch non trovati. Conversione automatica da TensorFlow (from_tf=True)...")
                # Qui TF lavora su CPU, lasciando la GPU libera
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path, from_tf=True)

            # Sposta il modello (ora PyTorch puro) sulla GPU
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            print(f"ERRORE CRITICO nel caricamento del modello: {e}")
            raise e

    def clean_html_robust(self, text):
        """Pulisce il testo da HTML, Script e CSS."""
        if not isinstance(text, str): return ""
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'', ' ', text, flags=re.DOTALL)
        text = re.sub(r'<!\[CDATA\[.*?\]\]>', ' ', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict(self, texts):
        """
        Input: Lista di stringhe.
        Output: Numpy array con probabilità classe 1 (Reliable).
        """
        if not texts:
            return np.array([])

        cleaned_texts = [self.clean_html_robust(t) for t in texts]

        # Tokenizzazione
        encodings = self.tokenizer(
            cleaned_texts,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Sposta gli input sulla GPU
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        # Inferenza
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)

        # Ritorna numpy array (spostando prima su CPU)
        return probs[:, 1].cpu().numpy()