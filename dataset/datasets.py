import pandas as pd
import numpy as np
import torch as th
import torch.nn.functional as F


def get_no_duplicates_data(data, duplicates_col):
    '''
    获得dataframe中按duplicates_col列无重复的行，
    :param data:
    :param duplicates_col: 需要检查重复的列名的list
    :return:
    '''
    # data = pd.read_csv(data_path)
    duplicates_list = pd.DataFrame(data.duplicated(subset=duplicates_col))
    duplicates_list.columns = ['isDuplicates']
    new_data = pd.concat([data, duplicates_list], axis=1)
    no_duplicates_data = new_data[new_data['isDuplicates'] == False].iloc[:, :-1]
    rest_data = new_data[new_data['isDuplicates'] == True].iloc[:, :-1]
    return no_duplicates_data, rest_data


def merge_col(data, col_list, new_col_name):
    '''
    dataframe中多列合并成成一列
    :param data:
    :param col_list:
    :param new_col_name:
    :return:
    '''
    data[new_col_name] = data[col_list[0]]
    for col in range(1, len(col_list)):
        data[new_col_name] = data[new_col_name].map(str) + "," + data[col_list[col]].map(str)


def get_addr_dict(data, col_list):
    '''
    对打他frame中的多列数据建立索引
    :param data: 操作的dataframe
    :param col_list: 需要建立索引的列名的list
    :return:
    '''
    new_col = pd.DataFrame()
    for col in range(len(col_list)):
        new_col = pd.concat([new_col, data[col_list[col]]])
    new_col = new_col.drop_duplicates().reset_index().drop(['index'], axis=1)
    new_col_dict = new_col.to_dict()
    new_col_dict = new_col_dict[0]
    new_dict = dict(zip(new_col_dict.values(), new_col_dict.keys()))
    return new_dict


def addr_change_index(data, col_list, need_dict):
    for col in col_list:
        data[col] = data[col].map(need_dict, na_action=None)
    return data


def balace_data(data, label):
    '''
    数据平衡处理
    :param data:
    :param label: 需要按某一列做平衡处理的列名，string
    :return:
    '''
    category, num = np.unique(np.array(data[label]), return_counts=True)
    balace_num = min(num)
    result_data = pd.DataFrame()
    for cate in category:
        sub = data[data[label] == cate]
        sub_sample = sub.sample(n=balace_num)
        result_data = pd.concat([result_data, sub_sample])
    return result_data.reset_index().drop(['index'], axis=1)


def get_str_feature_to_num(data, feature_list):
    '''
    将dataframe中类型为字符形式的离散型数据列改为用0，1，2等数字表示，以方便后一步做one-hot处理
    :param data:
    :param feature_list: 需要处理的列的列名的list
    :return:
    '''
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


def data_to_onehot(data, feature_list):
    '''
    将数据中离散型数据的列转换为one-hot向量
    :param data:
    :param feature_list:
    :return:
    '''
    new_data = pd.DataFrame()
    col_num = len(data.columns)
    header = data.columns
    for i in range(col_num):
        if header[i] not in feature_list:
            new_data = pd.concat([new_data, data.iloc[:, i]], axis=1)
        else:
            sub_list = []
            sub_col = th.tensor(np.array(data[header[i]]))
            sub_data = pd.DataFrame(F.one_hot(sub_col))
            for j in range(sub_data.shape[1]):
                sub_list.append(header[i] + '_' + str(j))
            sub_data.columns = sub_list
            new_data = pd.concat([new_data, sub_data], axis=1)
    return new_data


def get_feature_label(data, feature_list, label_list):
    feature_pd = pd.DataFrame()
    for feature in feature_list:
        feature_pd = pd.concat([feature_pd, data[feature]], axis=1)
    label_pd = pd.DataFrame()
    for label in label_list:
        label_pd = pd.concat([label_pd, data[label]], axis=1)
    return pd.concat([feature_pd, label_pd], axis=1)

def get_need_col_data(data, col_list):
    '''
    将数据中需要的列提取出来
    :param data:
    :param col_list: 需要的列名的list
    :return:
    '''
    new_data = pd.DataFrame()
    for col in col_list:
        new_data = pd.concat([new_data, data[col]], axis=1)
    return new_data