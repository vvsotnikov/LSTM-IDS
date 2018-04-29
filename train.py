import numpy as np
import time

from data_processing import *
from model import *
import tensorflow as tf

training, testing, le = parse_dataset()
export_dir = 'saved_model'


def get_accuracy(df):
    # Calculate accuracy for label classification
    categories = prediction.eval(feed_dict={X: df.iloc[:, 0:-2]})  # Getting back the predictions

    # Function to convert categories back into binary labels
    f = lambda x: 0 if le.inverse_transform(x) == "Normal" else 1

    # Preparing the necessary predictions and labels for comparision; converting categories to normal/malicious
    binary_prediction = np.fromiter((f(xi) for xi in categories), categories.dtype, count=len(categories))
    binary_labels = df.iloc[:, -1].values

    # Comparing predictions and labels to calculate accuracy
    correct_labels = tf.equal(binary_prediction, binary_labels)
    label_accuracy = tf.reduce_mean(tf.cast(correct_labels, tf.float32))
    result = label_accuracy.eval()
    print("Label accuracy: {:.2f}%".format(result * 100))

    # Calculate accuracy for category classification
    correct_prediction = tf.equal(prediction, tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = accuracy.eval({X: df.iloc[:, 0:-2], Y: df.iloc[:, -2]})
    print("Category accuracy: {:.2f}%".format(result * 100))


def train_and_test_model(do_loading, do_saving):
    with tf.Session() as sess:
        sess.run(init)
        if do_loading:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
            print("Training results: ")
            get_accuracy(training)
            print("Testing results: ")
            get_accuracy(testing)
        # Training cycle
        # TODO: Fix input pipeline bottleneck (move mo tf.Data API?)
        for epoch in range(training_epochs):
            # Shuffling dataset before training
            df = training.sample(frac=1)
            avg_cost = 0.
            total_data = df.index.shape[0]
            num_batches = total_data // batch_size + 1
            i = 0
            # Loop over all batches
            while i < total_data:
                batch_x = df.iloc[i:i + batch_size, 0:-2].values
                batch_y = df.iloc[i:i + batch_size, -2].values  # Last two columns are categories and labels
                i += batch_size
                # Run optimization op and cost op (to get loss value)
                _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                                Y: batch_y,
                                                                decay_step: num_batches * training_epochs})
                # Compute average loss
                avg_cost += c / num_batches
                # print("Epoch: {:04} | Cost={:.9f}".format(epoch + 1, avg_cost))
            # Display logs per epoch step
            print("Epoch: {:04} | Cost={:.9f}".format(epoch + 1, avg_cost))
            # get_accuracy(testing)
            # TODO: implement timer between epochs
            print(time.time())
            print()
        print("Training complete")
        print("Training results: ")
        get_accuracy(training)
        print("Testing results: ")
        get_accuracy(testing)
        if do_saving:
            tf.saved_model.simple_save(sess, export_dir+'1', inputs={"x": X}, outputs={"y": Y})
