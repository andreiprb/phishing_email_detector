import os
import pandas as pd
import re
import tensorflow as tf
from keras.api import layers, Model, Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

data_dir = "../data/"
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

df = pd.concat(df_list, ignore_index=True)

df['text'] = df['subject'].astype(str) + " " + df['body'].astype(str)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

df['text'] = df['text'].apply(preprocess_text)
df['label'] = df['label'].astype(int)

print(len(df))

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

max_features = 20000
max_len = 200

vectorizer = layers.TextVectorization(max_tokens=max_features, output_mode='int', output_sequence_length=max_len)
vectorizer.adapt(train_df['text'].values)

X_train = vectorizer(train_df['text'].values)
X_test = vectorizer(test_df['text'].values)
y_train = train_df['label'].values
y_test = test_df['label'].values

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

embedding_dim = 128
num_heads = 4
ff_dim = 128
num_transformer_blocks = 2

model_path = "transformer_model.h5"

if os.path.exists(model_path):
    print("Se încarcă modelul salvat...")
    model = tf.keras.models.load_model(model_path, custom_objects={'TransformerBlock': TransformerBlock})
else:
    print("Modelul nu există. Se construiește unul nou...")

    inputs = layers.Input(shape=(max_len,))
    x = layers.Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=max_len)(inputs)
    for _ in range(num_transformer_blocks):
        x = TransformerBlock(embedding_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    batch_size = 32
    epochs = 5

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    model.save(model_path)
    print("Model salvat.")

model.summary()
loss, accuracy = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Acuratețe:", accuracy)

y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print("\nRaport clasificare:\n")
print(classification_report(y_test, y_pred, digits=4))
