from keras.layers import Dense, Activation, Dropout, Conv1D, Flatten
from keras.models import Sequential


class Conv1DModel:

    @classmethod
    def model(cls, run_type, dataset, shapes):
        model = Sequential()
        model.add(
            Conv1D(60, kernel_size=1, activation='relu', input_shape=shapes)
        )
        model.add(Dropout(0.1))
        model.add(Conv1D(60, kernel_size=1, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Conv1D(60, kernel_size=1, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Flatten())

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
