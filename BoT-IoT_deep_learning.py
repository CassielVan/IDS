from services import BoT_IoT_process as BoT_IoT_process
from models import deep_learning_runner as dl_runner

classification_type = 0
if classification_type == 0:
    output_path = "output/BoT_IoT_binary.csv"
else:
    output_path = "output/BoT_IoT_multiclass.csv"
data_path = "data/BoT-IoT"
balance_num = 3000
x_train, x_test, y_train, y_test = BoT_IoT_process.get_BoT_IoT_deep_learning_data(classification_type, data_path, output_path, balance_num)

dataset = 1
epochs = 10

dl_runner.RNNRunner(x_train, x_test, y_train, y_test, classification_type, dataset, epochs)
dl_runner.LSTMRunner(x_train, x_test, y_train, y_test, classification_type, dataset, epochs)
dl_runner.GRURunner(x_train, x_test, y_train, y_test, classification_type, dataset, epochs)
dl_runner.DNNRunner(x_train, x_test, y_train, y_test, classification_type, dataset, epochs)
dl_runner.Conv1dRunner(x_train, x_test, y_train, y_test, classification_type, dataset, epochs)
