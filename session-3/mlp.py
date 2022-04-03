import numpy as np
import tensorflow as tf
from data_reader import load_datasets
from save_and_load import save_parameters

PATH = '../data/20news-bydate'

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size

    def build_graph(self):
        self.W1 = tf.Variable(
            tf.random.normal(
                shape=[self._input_size, self._hidden_size],
                stddev=0.1,
                seed=42,
            ),
            name='weights_input_hidden',
        )

        self.b1 = tf.Variable(
            tf.zeros([1, self._hidden_size]),
            name='biases_input_hidden',
        )

        self.W2 = tf.Variable(
            tf.random.normal(
                shape=[self._hidden_size, self._output_size],
                stddev=0.1,
                seed=42,
            ),
            name='weights_hidden_output',
        )

        self.b2 = tf.Variable(
            tf.zeros([1, self._output_size]),
            name='biases_hidden_output',
        )

        self.trainable_variables = [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X):
        X_tf = tf.cast(X, dtype=tf.float32)
        Z1 = tf.matmul(X_tf, self.W1) + self.b1
        Z1 = tf.sigmoid(Z1)
        logits = tf.matmul(Z1, self.W2) + self.b2
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

    def trainer(self, X, y, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        with tf.GradientTape() as tape:
            logits = self.forward(X)
            current_loss = self.loss(y, logits)
        grads = tape.gradient(current_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return current_loss

if __name__ == '__main__':
    train_data_reader, _ = load_datasets()
    step, MAX_STEP = 0, 100000

    model = MLP(
        input_size=train_data_reader.data.shape[1],
        hidden_size=100,
        output_size=20,
    )
    model.build_graph()

    for step in range(MAX_STEP):
        train_data, train_labels = train_data_reader.next_batch()

        loss = model.trainer(
            X=train_data,
            y=train_labels,
            learning_rate=0.01,
        )

        if step % 100 == 0:
            print(f'Step: {step}, Loss: {loss.numpy()}')

    trainable_variables = model.trainable_variables
    for variable in trainable_variables:
        save_parameters(
            name=variable.name, 
            value=variable.numpy(),
            epoch=train_data_reader.num_epoch,
        )