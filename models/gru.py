from keras.layers import Dense, Activation, Dropout, GRU
from keras.models import Sequential


class GRUModel:

    @classmethod
    def model(cls, run_type, dataset, shapes):
        model = Sequential()
        model.add(GRU(120, input_shape=shapes, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(120, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(120, return_sequences=False))
        model.add(Dropout(0.2))

        if run_type == 0:
            model.add(Dense(2))
            model.add(Activation('hard_sigmoid'))
        else:
            if dataset == 0:
                model.add(Dense(5))
            else:
                model.add(Dense(4))
            model.add(Activation('softmax'))

        return model
