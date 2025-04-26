import os
import pandas as pd
import re
import tensorflow as tf
from keras.api import layers, Model, Sequential, models, losses
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class EmailClassifier:
    def __init__(self, model_path="../trained_models/transformer_model.h5", data_dir="../data/", test_size=0.2, max_features=20000,
                 max_len=200):
        tf.random.set_seed(42)

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU")
            except RuntimeError as e:
                pass
        else:
            print("CPU")

        self.model_path = model_path
        self.max_features = max_features
        self.max_len = max_len
        self.test_df = None

        if not os.path.exists(model_path):
            print("Modelul nu există. Se încarcă datele și se construiește unul nou...")
            if os.path.exists(data_dir):
                df = self._load_data(data_dir)
                if df is not None:
                    self._build_and_train_model(df, test_size)
        else:
            print(f"Loading model from {model_path}...")
            self.model = models.load_model(
                model_path,
                custom_objects={'TransformerBlock': TransformerBlock}
            )

            if os.path.exists(data_dir):
                self._load_and_split_data(data_dir, test_size)

        self.vectorizer = layers.TextVectorization(
            max_tokens=max_features,
            output_mode='int',
            output_sequence_length=max_len
        )

        self.vectorizer_adapted = False

        if self.test_df is not None and len(self.test_df) > 0:
            self.adapt_vectorizer(self.test_df['text'].values)

    def _load_data(self, data_dir):
        csv_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".csv")]

        df_list = []
        for file in csv_files:
            try:
                df_temp = pd.read_csv(file, on_bad_lines='skip', encoding='utf-8')
                if {'subject', 'body', 'label'}.issubset(df_temp.columns):
                    df_list.append(df_temp)
                else:
                    print(f"Fișierul {file} a fost sărit - lipsesc coloane.")
            except Exception as e:
                print(f"Eroare la citirea fișierului {file}: {e}")

        if not df_list:
            print("No valid CSV files found in data directory.")
            return None

        df = pd.concat(df_list, ignore_index=True)

        df['text'] = df['subject'].astype(str) + " " + df['body'].astype(str)
        df['text'] = df['text'].apply(self.preprocess_text)
        df['label'] = df['label'].astype(int)

        return df

    def _load_and_split_data(self, data_dir, test_size):
        if not self.test_df:
            df = self._load_data(data_dir)
            if df is not None:
                _, self.test_df = train_test_split(df, test_size=test_size, random_state=42)
                print(f"Loaded {len(df)} total samples, {len(self.test_df)} test samples.")

    def _build_and_train_model(self, df, test_size):
        train_df, self.test_df = train_test_split(df, test_size=test_size, random_state=42)

        self.vectorizer.adapt(train_df['text'].values)
        self.vectorizer_adapted = True

        X_train = self.vectorizer(train_df['text'].values)
        X_test = self.vectorizer(self.test_df['text'].values)
        y_train = train_df['label'].values
        y_test = self.test_df['label'].values

        embedding_dim = 128
        num_heads = 4
        ff_dim = 128
        num_transformer_blocks = 2

        inputs = layers.Input(shape=(self.max_len,))
        x = layers.Embedding(input_dim=self.max_features, output_dim=embedding_dim, input_length=self.max_len)(inputs)
        for _ in range(num_transformer_blocks):
            x = TransformerBlock(embedding_dim, num_heads, ff_dim)(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(20, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        batch_size = 32
        epochs = 5

        history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        self.model.save(self.model_path)
        print("Model salvat.")

        self.model.summary()
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print("Loss:", loss)
        print("Acuratețe:", accuracy)

        y_pred_probs = self.model.predict(X_test)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()

        print("\nRaport clasificare:\n")
        print(classification_report(y_test, y_pred, digits=4))

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text

    def adapt_vectorizer(self, texts):
        self.vectorizer.adapt(texts)
        self.vectorizer_adapted = True

    def predict_email(self, subject: str, body: str) -> tuple[bool, float]:
        text = f"{subject} {body}"

        preprocessed_text = self.preprocess_text(text)

        if not self.vectorizer_adapted:
            print("Warning: Vectorizer not adapted. Adapting to the current input only.")
            self.adapt_vectorizer([preprocessed_text])

        vectorized_text = self.vectorizer([preprocessed_text])

        prediction_prob = self.model.predict(vectorized_text)[0][0]

        is_spam = bool(prediction_prob > 0.5)

        return is_spam, float(prediction_prob)

    def evaluate(self):
        if self.test_df is None or len(self.test_df) == 0:
            print("No test data available. Please provide a data directory when initializing the classifier.")
            return None, None

        if not self.vectorizer_adapted:
            print("Adapting vectorizer to test data...")
            self.adapt_vectorizer(self.test_df['text'].values)

        X_test = self.vectorizer(self.test_df['text'].values)
        y_test = self.test_df['label'].values

        y_pred_probs = self.model.predict(X_test)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()

        loss = losses.BinaryCrossentropy()(y_test, y_pred_probs).numpy()
        accuracy = (y_pred == y_test).mean()

        report = classification_report(y_test, y_pred, digits=4)

        print(f"Loss: {loss:.2f}\nAccuracy: {accuracy:.2f}\n\nClassification Report:\n{report}")

        return accuracy, report


if __name__ == "__main__":
    classifier = EmailClassifier("../trained_models/transformer_model.h5", data_dir="../data/")

    subject = "URGENT: You've won $1,000,000"
    body = "Claim your prize now by sending your bank details"

    is_spam, confidence = classifier.predict_email(subject, body)
    print(f"Is spam: {is_spam}")
    print(f"Confidence: {confidence:.4f}")

    accuracy, report = classifier.evaluate()