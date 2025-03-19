import tensorflow as tf
from keras.api import layers, Model, Sequential
import pandas as pd
from sklearn.model_selection import train_test_split
import re

tf.random.set_seed(42)

df = pd.read_csv("8339691/CEAS_08.csv")
df['text'] = df['subject'].astype(str) + " " + df['body'].astype(str)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

df['text'] = df['text'].apply(preprocess_text)
df['label'] = df['label'].astype(int)

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
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
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
model.summary()

batch_size = 32
epochs = 5

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
loss, accuracy = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)
