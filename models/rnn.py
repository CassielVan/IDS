from keras.layers import Dense, Activation, Dropout, SimpleRNN
from keras.models import Sequential


class RNNModel:

    @classmethod
    def model(cls, run_type, dataset, shapes):
        model = Sequential()
        model.add(SimpleRNN(120, input_shape=shapes, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(120, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(120, return_sequences=False))
        model.add(Dropout(0.2))

        if run_type == 0:
            model.add(Dense(2))
            model.add(Activation('sigmoid'))
        else:
            if dataset == 0:
                model.add(Dense(5))
            else:
                model.add(Dense(4))
            model.add(Activation('softmax'))

        return model
