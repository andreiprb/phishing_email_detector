from data_holder import EmailDataset
import tensorflow as tf
from keras.api import layers, Model, Sequential
from sklearn.model_selection import train_test_split

tf.random.set_seed(42)

# Load the dataset
emails = [
    "data/CEAS_08.csv",
    "data/Nazario_5.csv",
    "data/Nazario.csv",
    "data/Nigerian_5.csv",
    "data/Nigerian_Fraud.csv",
    "data/SpamAssasin.csv"
]

label_positions = [
    "second_last",
    "second_last",
    "last",
    "second_last",
    "last",
    "second_last"
]

dataset = EmailDataset(emails, label_positions)

# Convert dataset to DataFrame for compatibility with TensorFlow
data = []
labels = []
for i in range(len(dataset)):
    email, label = dataset[i]
    data.append(email)
    labels.append(label)

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Preprocess the text data
max_features = 20000
max_len = 200

vectorizer = layers.TextVectorization(max_tokens=max_features, output_mode='int', output_sequence_length=max_len)
vectorizer.adapt(train_data)

X_train = vectorizer(train_data)
X_test = vectorizer(test_data)
y_train = tf.convert_to_tensor(train_labels, dtype=tf.float32)
y_test = tf.convert_to_tensor(test_labels, dtype=tf.float32)

# Define the transformer model
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