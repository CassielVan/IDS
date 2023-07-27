from models.rnn import RNNModel
from models.lstm import LSTMModel
from models.gru import GRUModel
from models.dnn import DNNModel
from models.conv1d import Conv1DModel
from keras.optimizers import adam_v2
import time
import numpy as np


def RNNRunner(x_train, x_test, y_train, y_test, classification_type, dataset, epochs):
    model = RNNModel.model(classification_type, dataset, (x_train.shape[1], x_train.shape[2]))
    model.summary()
    adam = adam_v2.Adam(learning_rate=0.0005)
    start = time.time()
    if classification_type == 0:
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=32)

    print("--- %s seconds ---" % (time.time() - start))

    loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

    predict_x = model.predict(x_test)
    y_pred = np.argmax(predict_x, axis=1)
    print("\nAnomaly in Test: ", np.count_nonzero(y_test, axis=0))
    print("\nAnomaly in Prediction: ", np.count_nonzero(y_pred, axis=0))


def LSTMRunner(x_train, x_test, y_train, y_test, classification_type, dataset, epochs):
    model = LSTMModel.model(classification_type, dataset, (x_train.shape[1], x_train.shape[2]))
    model.summary()
    adam = adam_v2.Adam(learning_rate=0.0005)
    start = time.time()
    if classification_type == 0:
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=32)

    print("--- %s seconds ---" % (time.time() - start))

    loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

    predict_x = model.predict(x_test)
    y_pred = np.argmax(predict_x, axis=1)
    print("\nAnomaly in Test: ", np.count_nonzero(y_test, axis=0))
    print("\nAnomaly in Prediction: ", np.count_nonzero(y_pred, axis=0))


def GRURunner(x_train, x_test, y_train, y_test, classification_type, dataset, epochs):
    model = GRUModel.model(classification_type, dataset, (x_train.shape[1], x_train.shape[2]))
    model.summary()
    adam = adam_v2.Adam(learning_rate=0.0005)
    start = time.time()
    if classification_type == 0:
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=32)

    print("--- %s seconds ---" % (time.time() - start))

    loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

    predict_x = model.predict(x_test)
    y_pred = np.argmax(predict_x, axis=1)
    print("\nAnomaly in Test: ", np.count_nonzero(y_test, axis=0))
    print("\nAnomaly in Prediction: ", np.count_nonzero(y_pred, axis=0))


def DNNRunner(x_train, x_test, y_train, y_test, classification_type, dataset, epochs):
    model = DNNModel.model(classification_type, dataset, (x_train.shape[1], x_train.shape[2]))
    model.summary()
    adam = adam_v2.Adam(learning_rate=0.0005)
    start = time.time()
    if classification_type == 0:
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=32)

    print("--- %s seconds ---" % (time.time() - start))

    loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

    predict_x = model.predict(x_test)
    y_pred = np.argmax(predict_x, axis=1)
    print("\nAnomaly in Test: ", np.count_nonzero(y_test, axis=0))
    print("\nAnomaly in Prediction: ", np.count_nonzero(y_pred, axis=0))


def Conv1dRunner(x_train, x_test, y_train, y_test, classification_type, dataset, epochs):
    model = Conv1DModel.model(classification_type, dataset, (x_train.shape[1], x_train.shape[2]))
    model.summary()
    adam = adam_v2.Adam(learning_rate=0.0005)
    start = time.time()
    if classification_type == 0:
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=32)

    print("--- %s seconds ---" % (time.time() - start))

    loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

    predict_x = model.predict(x_test)
    y_pred = np.argmax(predict_x, axis=1)
    print("\nAnomaly in Test: ", np.count_nonzero(y_test, axis=0))
    print("\nAnomaly in Prediction: ", np.count_nonzero(y_pred, axis=0))
