import datasets as datasets
import pandas as pd


def get_train_data_data():
    data_path = "../data/NF-BoT-IoT/NF-BoT-IoT.csv"
    duplicates_col = ['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT']
    data = pd.read_csv(data_path)
    train_data, rest_data_1 = datasets.get_data(data, duplicates_col)
    test_data, rest_data_2 = datasets.get_data(rest_data_1, duplicates_col)
    train_data = train_data.reset_index()
    test_data = test_data.reset_index()
    return train_data, test_data


def merge_addr_col(data, addr_list):
    src_col_list = ["IPV4_SRC_ADDR", "L4_SRC_PORT"]
    datasets.merge_col(data, src_col_list, addr_list[0])
    dst_col_list = ["IPV4_DST_ADDR", "L4_DST_PORT"]
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


def get_nf_bot_iot_data():
    train_data, test_data = get_train_data_data()
    addr_list = ['src_addr', 'dst_addr']
    merge_addr_col(train_data, addr_list)
    merge_addr_col(test_data, addr_list)

    train_addr_dic, test_addr_dic = get_dic(train_data, test_data, addr_list)
    train_data = datasets.addr_change_index(train_data, addr_list, train_addr_dic)
    test_data = datasets.addr_change_index(test_data, addr_list, test_addr_dic)
    feature_list = ['PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'TCP_FLAGS',
                    'FLOW_DURATION_MILLISECONDS']
    label_list = ['Label', 'Attack']

    train_feature, train_label, train_u, train_v = get_feature_nodes_label(train_data, feature_list, label_list, addr_list)
    test_feature, test_label, test_u, test_v = get_feature_nodes_label(test_data, feature_list, label_list,
                                                                           addr_list)
    return train_feature, train_label, train_u, train_v, test_feature, test_label, test_u, test_v


if __name__ == '__main__':
    train_feature, train_label, train_u, train_v, test_feature, test_label, test_u, test_v = get_nf_bot_iot_data()
    print(test_u, test_v)

