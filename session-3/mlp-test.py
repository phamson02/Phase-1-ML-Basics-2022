import tensorflow as tf
import numpy as np
from data_reader import load_datasets
from save_and_load import restore_parameters
from mlp import MLP

if __name__ == '__main__':
    _, test_data_reader = load_datasets()
    epoch = 442

    model = MLP(
        input_size=test_data_reader.data.shape[1],
        hidden_size=100,
        output_size=20,
    )
    model.build_graph()

    trainable_variables = model.trainable_variables
    for variable in trainable_variables:
        saved_value = restore_parameters(variable.name, epoch)
        variable.assign(saved_value)

    num_true_preds = 0
    while True:
        test_data, test_labels = test_data_reader.next_batch()
        predicted_labels = model.predict(test_data)
        matches = np.equal(predicted_labels, test_labels)
        num_true_preds += np.sum(matches.astype(float))

        if test_data_reader.batch_id == 0:
            break
        
    print('Epoch:', epoch)
    print('Accuracy on test data:', num_true_preds / len(test_data_reader.data))