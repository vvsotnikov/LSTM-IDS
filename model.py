import tensorflow as tf

# Parameters
training_epochs = 100
batch_size = 20
start_rate = 0.00025

# Network Parameters
# n_hidden_1 = 100
n_hidden_2 = 50
n_features = 196  # There are 194 different features for each packet.
n_classes = 10  # There are 9 different types of malicious packets + Normal
lstm_size = 100

X = tf.placeholder(tf.float32, [None, n_features])
Y = tf.placeholder(tf.int32, [None, ])

# decay step for learning rate decay
decay_step = tf.placeholder(tf.int32)

# Create model
# Hidden fully connected layer with 100 neurons
# layer_1 = tf.layers.dense(X, n_hidden_1, activation=tf.nn.relu)
# LSTM layer
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
outputs, states = tf.contrib.rnn.static_rnn(lstm, [X], dtype=tf.float32)
# Hidden fully connected layer with 50 neurons
layer_2 = tf.layers.dense(outputs[0], n_hidden_2, activation=tf.nn.relu)
# Output fully connected layer with a neuron for each class
logits = tf.layers.dense(layer_2, n_classes)

# Define loss and optimizer
# Converting categories into one hot labels
labels = tf.one_hot(indices=tf.cast(Y, tf.int32), depth=n_classes)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=labels))
global_step = tf.Variable(0, trainable=False)

# Using a learning rate which has polynomial decay
starter_learning_rate = start_rate
end_learning_rate = 0.00005  # we will use a polynomial decay to reach this learning rate
decay_steps = decay_step
learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                          decay_steps, end_learning_rate,
                                          power=0.5)
# Using adam optimizer to reduce loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initializing the variables
init = tf.global_variables_initializer()
# Model for testing
pred = tf.nn.softmax(logits)  # Apply softmax to logits
# Model for prediction: Used to just return predicted values
prediction = tf.argmax(pred, 1)
