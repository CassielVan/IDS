from dataset import BoT_IoT_datasets
from models import E_GraphSAGE

import torch as th
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import dgl.function as fn
import dgl


def trainer(train_feature, train_label, train_u, train_v):
    u = th.tensor(np.array(train_u))
    v = th.tensor(np.array(train_v))
    length = train_feature.shape[1]
    g = dgl.graph((u, v))
    g.ndata['h'] = th.ones(g.num_nodes(), length, dtype=th.float32)
    # g.ndata['h_0'] = g.ndata['h']
    g.edata['w'] = th.tensor(np.array(train_feature), dtype=th.float32)
    g.edata['attack'] = th.tensor(np.array(train_label['attack']))
    g.edata['category'] = th.tensor(np.array(train_label['category']))
    print(g)


    loss_list = []
    node_features = g.ndata['h']
    category_y = g.edata['category']
    print("data detail")
    print(np.unique(np.array(category_y), return_counts=True))
    output_num = len(np.unique(np.array(category_y)))
    category_y = th.tensor(F.one_hot(category_y), dtype=th.float)
    model = E_GraphSAGE.Model(length, length, length, output_num)
    opt = th.optim.Adam(model.parameters())

    for epoch in range(2000):
        pred = model(g, node_features)
        level = th.argmax(pred, 1)
        print(np.unique(level, return_counts=True))
        loss = F.mse_loss(pred, category_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        print("epoch = ", epoch, loss.item())
        one_loss = [epoch, loss.item()]
        loss_list.append(one_loss)

    loss_list = pd.DataFrame(loss_list)
    x = np.array(loss_list.iloc[:, 0])
    y = np.array(loss_list.iloc[:, 1])

    plt.figure(figsize=(8, 6))  # 定义图的大小
    plt.xlabel("time(s)")  # X轴标签
    plt.ylabel("Volt")  # Y轴坐标标签
    plt.title("Example")  # 曲线图的标题

    plt.plot(x, y)  # 绘制曲线图
    # 在ipython的交互环境中需要这句话才能显示出来
    plt.show()



if __name__ == '__main__':
    # directory = "../data/BoT-IoT"
    directory = "data/BoT-IoT"
    train_feature, train_label, train_u, train_v, test_feature, test_label, test_u, test_v = BoT_IoT_datasets.get_bot_iot_data(directory)
    trainer(train_feature, train_label, train_u, train_v)