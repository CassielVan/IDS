import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from dataset import BoT_IoT_datasets as BoT_IoT_datasets
from dataset import datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# 读取数据集目录下所有的数据文件名称
def get_BoT_IoT_deep_learning_data(classification_type, data_path, output_path, balance_num):
    directory = data_path
    remove_file = ['header.csv', 'num_header.csv', '.DS_Store']
    header_path = directory + '/' + remove_file[0]
    files_name = BoT_IoT_datasets.get_all_files_name(directory, remove_file)
    handle_colunms = ['saddr', 'sport', 'daddr', 'dport']
    col_list = ['flgs', 'proto', 'pkts', 'bytes', 'state', 'ltime', 'seq', 'dur', 'mean', 'stddev', 'sum', 'min',
                'max', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate', 'attack', 'category']
    header = pd.read_csv(header_path)
    headerList = header.columns.to_list()

    if not os.path.isfile(output_path):
        for file_name in tqdm(files_name):
            file_path = directory + '/' + file_name
            file_data = pd.read_csv(file_path, header=None)
            file_data.columns = headerList
            for col in handle_colunms:
                file_data[col] = file_data[col].astype('string')
                file_data[col] = file_data[col].str.replace('.0', '')
                file_data[col] = file_data[col].str.replace(r'[\W]', '')
            file_data.reset_index().drop(['index'], axis=1)
            file_data = datasets.get_need_col_data(file_data, col_list)
            file_data.to_csv(output_path, index=False, header=False, mode='a')

    data = pd.read_csv(output_path, header=None)
    data.columns = col_list
    data = data.reset_index().drop(['index'], axis=1)
    print("read files over")
    # 针对label做数据平衡处理
    if classification_type == 0:
        label = "attack"
    else:
        label = "category"
    data = datasets.balace_data(data, label, balance_num)
    print("data balanced")
    print(np.unique(np.array(data[label]), return_counts=True))

    # 提取需要的列，即需要的特征以及label，label包括二分类标签和多分类标签

    feature_list = BoT_IoT_datasets.get_emb_feature_list()
    num_data = datasets.get_str_feature_to_num(data, feature_list)
    print("num data over!!!")

    feature_col = ['flgs', 'proto', 'state']
    label_col = ['attack', 'category']
    feature = num_data.iloc[:, :-2]
    label = num_data.iloc[:, -2:]
    onehot_feature = datasets.data_to_onehot(feature, feature_col)
    onehot_label = datasets.data_to_onehot(label, label_col)
    print("convert to one hot over")

    # min_max_scaler = preprocessing.MinMaxScaler()
    # new_feature = min_max_scaler.fit_transform(feature)

    scaler = preprocessing.StandardScaler()
    final_feature = pd.DataFrame(scaler.fit_transform(onehot_feature))
    if classification_type == 0:
        final_label = onehot_label.iloc[:, :2]
    else:
        final_label = onehot_label.iloc[:, 2:]
    print("data process over")

    x_train, x_test, y_train, y_test = train_test_split(final_feature, final_label, test_size=0.25, random_state=37)
    print("data split over!!!")

    x_train = np.array(x_train).reshape(x_train.shape[0], 1, x_train.shape[1])
    x_test = np.array(x_test).reshape(x_test.shape[0], 1, x_test.shape[1])
    print("reshape over")
    return x_train, x_test, y_train, y_test


def get_BoT_IoT_machine_learning_data(classification_type, data_path, output_path, balance_num):
    directory = data_path
    remove_file = ['header.csv', 'num_header.csv', '.DS_Store']
    header_path = directory + '/' + remove_file[0]
    files_name = BoT_IoT_datasets.get_all_files_name(directory, remove_file)
    handle_colunms = ['saddr', 'sport', 'daddr', 'dport']
    col_list = ['flgs', 'proto', 'pkts', 'bytes', 'state', 'ltime', 'seq', 'dur', 'mean', 'stddev', 'sum', 'min',
                'max', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate', 'attack', 'category']
    header = pd.read_csv(header_path)
    headerList = header.columns.to_list()

    if not os.path.isfile(output_path):
        for file_name in tqdm(files_name):
            file_path = directory + '/' + file_name
            file_data = pd.read_csv(file_path, header=None)
            file_data.columns = headerList
            for col in handle_colunms:
                file_data[col] = file_data[col].astype('string')
                file_data[col] = file_data[col].str.replace('.0', '')
                file_data[col] = file_data[col].str.replace(r'[\W]', '')
            file_data.reset_index().drop(['index'], axis=1)
            file_data = datasets.get_need_col_data(file_data, col_list)
            file_data.to_csv(output_path, index=False, header=False, mode='a')

    data = pd.read_csv(output_path, header=None)
    data.columns = col_list
    data = data.reset_index().drop(['index'], axis=1)
    print("read files over")
    # 针对label做数据平衡处理
    if classification_type == 0:
        label = "attack"
    else:
        label = "category"
    data = datasets.balace_data(data, label, balance_num)
    print("data balanced")
    print(np.unique(np.array(data[label]), return_counts=True))

    # 提取需要的列，即需要的特征以及label，label包括二分类标签和多分类标签

    feature_list = BoT_IoT_datasets.get_emb_feature_list()
    num_data = datasets.get_str_feature_to_num(data, feature_list)
    print("num data over!!!")

    feature_col = ['flgs', 'proto', 'state']
    feature = num_data.iloc[:, :-2]
    onehot_feature = datasets.data_to_onehot(feature, feature_col)
    print("convert to one hot over")

    # min_max_scaler = preprocessing.MinMaxScaler()
    # new_feature = min_max_scaler.fit_transform(feature)

    scaler = preprocessing.StandardScaler()
    final_feature = pd.DataFrame(scaler.fit_transform(onehot_feature))
    final_label = num_data[label]
    print("data process over")

    x_train, x_test, y_train, y_test = train_test_split(final_feature, final_label, test_size=0.25, random_state=37)
    print("data split over!!!")

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    classification_type = 1
    if classification_type == 0:
        output_path = "../output/BoT_IoT_binary.csv"
    else:
        output_path = "../output/BoT_IoT_multiclass.csv"
    data_path = "../data/BoT-IoT"
    balance_num = 3000
    x_train, x_test, y_train, y_test = get_BoT_IoT_deep_learning_data(classification_type, data_path, output_path, balance_num)
    print(x_train)
    print(x_test)
    print(y_train)
    print(y_test)
