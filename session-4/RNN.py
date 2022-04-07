import tensorflow as tf
import numpy as np
from global_vars import *
from data_reader import load_datasets, get_vocab_size

class RNN:
    def __init__(self,
                 output_size,
                 embedding_size,
                 lstm_size,
                ):
        self._vocab_size = get_vocab_size()
        self._output_size = output_size
        self._embedding_size = embedding_size
        self._lstm_size = lstm_size

    def embedding_layer(self, indices):

        return tf.nn.embedding_lookup(self._embedding_matrix, indices)

    def lstm_layer(self, embeddings, sentence_lengths):
        lstm_outputs = self.rnn(inputs=embeddings) 
        # -> [batch_size, MAX_SENTENCE_LENGTH, lstm_size]

        lstm_outputs = tf.reshape(lstm_outputs, [self._batch_size * MAX_SENTENCE_LENGTH, -1])
        # -> [batch_size * MAX_SENTENCE_LENGTH, lstm_size]

        mask = tf.sequence_mask(
            lengths=sentence_lengths,
            maxlen=MAX_SENTENCE_LENGTH,
            dtype=tf.float32,
        )
        # -> [batch_size, MAX_SENTENCE_LENGTH]
        
        mask = tf.reshape(mask, [self._batch_size * MAX_SENTENCE_LENGTH, -1])
        # -> [batch_size * MAX_SENTENCE_LENGTH, 1]

        lstm_outputs = mask * lstm_outputs

        lstm_outputs_split = tf.split(lstm_outputs, num_or_size_splits=self._batch_size)
        lstm_outputs_sum = tf.reduce_sum(lstm_outputs_split, axis=1)
        lstm_outputs_average = lstm_outputs_sum / tf.expand_dims(
            tf.cast(sentence_lengths, tf.float32), -1
        )

        return lstm_outputs_average

    def build_graph(self):
        self.trainables = []
        
        np.random.seed(42)
        pretrained_vectors = np.zeros((self._vocab_size+2, self._embedding_size))
        pretrained_vectors[1:, :] = np.random.normal(
            loc=0., 
            scale=1., 
            size=(self._vocab_size+1, self._embedding_size)
        )

        self._embedding_matrix = tf.Variable(
            pretrained_vectors,
            name='embedding_layer',
        )
        self.trainables.append(self._embedding_matrix)

        self.lstm_cell = tf.keras.layers.LSTMCell(self._lstm_size)
        self.trainables += self.lstm_cell.weights

        self.rnn = tf.keras.layers.RNN(
            cell=self.lstm_cell,
            return_sequences=True,
        )
        self.trainables += self.rnn.weights

        self.W = tf.Variable(
            tf.random.normal(
                shape=[self._lstm_size, self._output_size],
                seed=42,
            ),
            name='final_layer_weights'
        )

        self.b = tf.Variable(
            tf.random.normal(
                shape=[self._output_size],
                seed=42,
            ),
            name='final_layer_biases'
        )

        self.trainables += [self.W, self.b]

    def forward(self, X, sentence_lengths):
        self._batch_size = X.shape[0]

        embeddings = self.embedding_layer(X)
        lstm_outputs_average = self.lstm_layer(embeddings, sentence_lengths)
        logits = tf.matmul(lstm_outputs_average, self.W) + self.b

        return logits

    def predict(self, x):
        logits = self.forward(x)
        probs = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(probs, axis=1)
        predicted_labels = tf.squeeze(predicted_labels)

        return predicted_labels

    def loss(self, y_true, logits):
        labels_one_hot = tf.one_hot(
            indices=y_true,
            depth=self._output_size,
            dtype=tf.float32,
            )

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_one_hot,
            logits=logits,
        )

        loss = tf.reduce_mean(loss)
        return loss

    def fit(self, X, y, sentence_lengths, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        with tf.GradientTape() as tape:
            logits = self.forward(X, sentence_lengths)
            current_loss = self.loss(y, logits)
        grads = tape.gradient(current_loss, self.trainables)
        optimizer.apply_gradients(zip(grads, self.trainables))

        return current_loss

if __name__ == '__main__':
    train_data_reader, _ = load_datasets()
    step, MAX_STEP = 0, 100000

    rnn = RNN(
        output_size=20,
        embedding_size=300,
        lstm_size=50,
    )
    rnn.build_graph()

    for step in range(MAX_STEP):
        train_data, train_labels, train_sentence_lengths = train_data_reader.next_batch()

        loss = rnn.fit(
            X=train_data,
            y=train_labels,
            sentence_lengths=train_sentence_lengths,
            learning_rate=1e-2,
        )

        if step % 1 == 0:
            print(f'Step: {step}, Loss: {loss.numpy()}')