import os
from dataset import datasets as datasets
import pandas as pd
import numpy as np


# 获取目录下所有文件名称，返回文件名称的数组
def get_all_files_name(directory, remove_file):
    files_name = os.listdir(directory)
    for f in remove_file:
        if f in files_name:
            files_name.remove(f)
    return files_name


def read_files(files_name, directory, handle_colunms, header_path):
    data = pd.DataFrame()
    header = pd.read_csv(header_path)
    headerList = header.columns.to_list()
    for file in files_name:
        file_path = directory + '/' + file
        file_data = pd.read_csv(file_path, header=None)
        data = pd.concat([data, file_data])

    data.columns = headerList

    for col in handle_colunms:
        data[col] = data[col].astype('string')
        data[col] = data[col].str.replace('.0', '')
        data[col] = data[col].str.replace(r'[\W]', '')

    return data.reset_index().drop(['index'], axis=1)


def get_train_test_data(data, duplicates_col, label):
    train_data, rest_data_1 = datasets.get_data(data, duplicates_col)
    test_data, rest_data_2 = datasets.get_data(rest_data_1, duplicates_col)
    train_data = train_data.reset_index()
    test_data = test_data.reset_index()

    train_category, train_num = np.unique(np.array(train_data[label]), return_counts=True)
    train_balace_num = min(train_num)
    train_data = datasets.balace_data(train_data, train_category, train_balace_num, label)

    test_category, test_num = np.unique(np.array(test_data[label]), return_counts=True)
    test_balace_num = min(test_num)
    test_data = datasets.balace_data(test_data, test_category, test_balace_num, label)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)
    return train_data, test_data


def merge_addr_col(data, addr_list):
    src_col_list = ["saddr", "sport"]
    datasets.merge_col(data, src_col_list, addr_list[0])
    dst_col_list = ["daddr", "dport"]
    datasets.merge_col(data, dst_col_list, addr_list[1])


def get_dic(train_data, test_data, addr_list):
    train_addr_dic = datasets.get_addr_dict(train_data, addr_list)
    test_addr_dic = datasets.get_addr_dict(test_data, addr_list)
    return train_addr_dic, test_addr_dic


def get_feature_nodes_label(data, feature_list, label_list, nodes_list):
    feature_df = pd.DataFrame()
    for feature in feature_list:
        feature_df = pd.concat([feature_df, data[feature]], axis=1)

    label_df = pd.DataFrame()
    for label in label_list:
        label_df = pd.concat([label_df, data[label]], axis=1)

    u = data[nodes_list[0]]
    v = data[nodes_list[1]]
    return feature_df, label_df, u, v


def get_emb_feature_list():
    feature_list = ['flgs', 'proto', 'state', 'attack', 'category']
    return feature_list


def get_str_feature_to_num(data, feature_list):
    # all_dict = {}
    for feature in feature_list:
        value = np.unique(np.array(data[feature]))
        length = len(value)
        value_dict = {}
        for i in range(length):
            value_dict[value[i]] = i

        data[feature] = data[feature].map(value_dict, na_action=None)
        # all_dict[feature] = value_dict
    return data


def get_bot_iot_data(directory):
    # directory = "../data/BoT-IoT"
    remove_file = ['header.csv', 'num_header.csv']
    header_path = directory + '/' + remove_file[0]
    handle_colunms = ['saddr', 'sport', 'daddr', 'dport']
    files_name = get_all_files_name(directory, remove_file)
    label = "category"
    data = read_files(files_name, directory, handle_colunms, header_path)
    feature_list = get_emb_feature_list()
    # attack, subcategory = "attack", "subcategory"
    data = get_str_feature_to_num(data, feature_list)
    print(data)
    train_data, test_data = get_train_test_data(data, handle_colunms, label)

    addr_list = ['src_addr', 'dst_addr']
    merge_addr_col(train_data, addr_list)
    merge_addr_col(test_data, addr_list)

    train_addr_dic, test_addr_dic = get_dic(train_data, test_data, addr_list)
    train_data = datasets.addr_change_index(train_data, addr_list, train_addr_dic)
    test_data = datasets.addr_change_index(test_data, addr_list, test_addr_dic)
    print(train_data, test_data)

    feature_list = ['flgs', 'proto', 'pkts', 'bytes', 'state', 'ltime', 'seq', 'dur', 'mean', 'stddev', 'sum', 'min',
                    'max', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate']
    label_list = ['attack', 'category']

    train_feature, train_label, train_u, train_v = get_feature_nodes_label(train_data, feature_list, label_list,
                                                                           addr_list)
    test_feature, test_label, test_u, test_v = get_feature_nodes_label(test_data, feature_list, label_list,
                                                                       addr_list)
    return train_feature, train_label, train_u, train_v, test_feature, test_label, test_u, test_v


def counts_label(data_path, directory, header_path, label):
    all_cate = []
    all_num = []
    header = pd.read_csv(header_path)
    headerList = header.columns.to_list()
    for path in data_path:
        new_path = directory + '/' + path
        print(new_path)
        data = pd.read_csv(new_path, header=None)
        data.columns = headerList
        cate, num = np.unique(data[label], return_counts=True)
        for index in range(len(cate)):
            element = cate[index]
            if element not in all_cate:
                all_cate.append(element)
                all_num.append(0)
            new_index = all_cate.index(element)
            all_num[new_index] = all_num[new_index] + num[index]
        print(cate, num)
    return all_cate, all_num


# if __name__ == '__main__':
#     # 读取数据集目录下所有的数据文件名称
#     directory = "../data/BoT-IoT"
#     remove_file = ['header.csv', 'num_header.csv', '.DS_Store']
#     header_path = directory + '/' + remove_file[0]
#     files_name = get_all_files_name(directory, remove_file)
#     print(files_name)
#     label = 'category'
#     all_cate, all_num = counts_label(files_name, directory, header_path, label)
#     print("over")
#     print(all_cate, all_num)
