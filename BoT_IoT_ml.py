import pandas as pd

from dataset import BoT_IoT_datasets as BoT_IoT_datasets
from dataset import datasets as datasets
from sklearn.model_selection import train_test_split
from models import ml_classification as ml_classification

from sklearn import preprocessing

# 读取数据集目录下所有的数据文件名称
directory = "data/BoT-IoT"
remove_file = ['header.csv', 'num_header.csv', '.DS_Store']
header_path = directory + '/' + remove_file[0]
files_name = BoT_IoT_datasets.get_all_files_name(directory, remove_file)

# 读取所有数据文件的数据，并对地址字段做初步处理
handle_colunms = ['saddr', 'sport', 'daddr', 'dport']
data = BoT_IoT_datasets.read_files(files_name, directory, handle_colunms, header_path)
print("read files over")
# 针对label做数据平衡处理
label = "category"
data = datasets.balace_data(data, label)

# 提取需要的列，即需要的特征以及label，label包括二分类标签和多分类标签
col_list = ['flgs', 'proto', 'pkts', 'bytes', 'state', 'ltime', 'seq', 'dur', 'mean', 'stddev', 'sum', 'min',
                'max', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate', 'attack', 'category']
new_data = datasets.get_need_col_data(data, col_list)

feature_list = BoT_IoT_datasets.get_emb_feature_list()
num_data = datasets.get_str_feature_to_num(new_data, feature_list)

col_list = ['flgs', 'proto', 'state']
onehot_data = datasets.data_to_onehot(num_data, col_list)
print("to one hot over")

feature = onehot_data.iloc[:, :-2]
label = onehot_data.iloc[:, -1]

# min_max_scaler = preprocessing.MinMaxScaler()
# new_feature = min_max_scaler.fit_transform(feature)

scaler = preprocessing.StandardScaler()
new_feature = scaler.fit_transform(feature)

x_train_all, x_test, y_train_all, y_test = train_test_split(new_feature, label, test_size=0.25, random_state=37)

kneighbors_pred = ml_classification.KNeighbors(x_train_all, x_test, y_train_all, y_test)
BAG_pred = ml_classification.Bagging(x_train_all, x_test, y_train_all, y_test)
DT_pred = ml_classification.DecisionTree(x_train_all, x_test, y_train_all, y_test)
RFC_pred = ml_classification.RandomForest(x_train_all, x_test, y_train_all, y_test)
svm_pred = ml_classification.svmClassification(x_train_all, x_test, y_train_all, y_test)
gnb_pred = ml_classification.Gaussian(x_train_all, x_test, y_train_all, y_test)
lr_pred = ml_classification.LogisticRegress(x_train_all, x_test, y_train_all, y_test)
GradBost_pred = ml_classification.GradientBoosting(x_train_all, x_test, y_train_all, y_test)
ADA_pred = ml_classification.AdaBoost(x_train_all, x_test, y_train_all, y_test)

# result = [kneighbors_pred, BAG_pred, DT_pred, RFC_pred, svm_pred,
#           gnb_pred, lr_pred, GradBost_pred, ADA_pred, y_test]
result = [kneighbors_pred, BAG_pred, DT_pred, RFC_pred, svm_pred,
          gnb_pred, lr_pred, GradBost_pred, ADA_pred, y_test]
result_pd = pd.DataFrame(result).transpose()
rs_path = "output/ml_classification_result.csv"
result_pd.to_csv(rs_path)
print("all over")
