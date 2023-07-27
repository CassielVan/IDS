import pandas as pd
from models import ml_classification as ml_classification

from services import BoT_IoT_process as BoT_IoT_process

classification_type = 1
if classification_type == 0:
    output_path = "output/BoT_IoT_binary.csv"
else:
    output_path = "output/BoT_IoT_multiclass.csv"
data_path = "data/BoT-IoT"
balance_num = 3000
x_train, x_test, y_train, y_test = BoT_IoT_process.get_BoT_IoT_machine_learning_data(classification_type, data_path, output_path, balance_num)
print(y_test)
print(y_train)
kneighbors_pred = ml_classification.KNeighbors(x_train, x_test, y_train, y_test)
BAG_pred = ml_classification.Bagging(x_train, x_test, y_train, y_test)
DT_pred = ml_classification.DecisionTree(x_train, x_test, y_train, y_test)
RFC_pred = ml_classification.RandomForest(x_train, x_test, y_train, y_test)
svm_pred = ml_classification.svmClassification(x_train, x_test, y_train, y_test)
gnb_pred = ml_classification.Gaussian(x_train, x_test, y_train, y_test)
lr_pred = ml_classification.LogisticRegress(x_train, x_test, y_train, y_test)
GradBost_pred = ml_classification.GradientBoosting(x_train, x_test, y_train, y_test)
ADA_pred = ml_classification.AdaBoost(x_train, x_test, y_train, y_test)

# result = [kneighbors_pred, BAG_pred, DT_pred, RFC_pred, svm_pred,
#           gnb_pred, lr_pred, GradBost_pred, ADA_pred, y_test]
result = [kneighbors_pred, BAG_pred, DT_pred, RFC_pred, svm_pred,
          gnb_pred, lr_pred, GradBost_pred, ADA_pred, y_test]
result_pd = pd.DataFrame(result).transpose()
rs_path = "output/ml_classification_result.csv"
result_pd.to_csv(rs_path)
print("all over")
