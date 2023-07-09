import torch as th
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F


class SAGE(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(SAGE, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)  # W

    def forward(self, g, h):
        with g.local_scope():  # 在这个区域内对g的修改不会同步到原始的图上
            g.ndata['h'] = h
            g.update_all(  # 对所有的节点和边采用下面的message函数和reduce函数
                message_func=fn.copy_u("h", "m"),  # message函数：将节点特征'h'作为消息传递给邻居，命名为'm'
                reduce_func=fn.mean("m", "h_N"),  # reduce函数：将接收到的'm'信息取平均，保存至节点特征'h_N'
            )
            h_N = g.ndata["h_N"]
            h_total = th.cat([h, h_N], dim=1)
            return self.linear(h_total)


class Model(nn.Module):
    def __init__(self, in_features, out_features, mlp_in, mlp_out):
        super().__init__()
        self.sage = SAGE(in_features, out_features)
        self.dense2 = th.nn.Linear(mlp_in * 2, 200)
        self.dense3 = th.nn.Linear(200, 200)
        self.dense4 = th.nn.Linear(200, mlp_out)

    def forward(self, g, x):
        h = self.sage(g, x)
        h_1 = self.sage(g, h)
        h_2 = self.sage(g, h_1)
        g.ndata['new_h'] = h_2
        g.apply_edges(lambda edges: {'x': th.cat([edges.src['new_h'], edges.dst['new_h']], dim=1)})
        rs = F.relu(self.dense2(g.edata['x']))
        rs = F.relu(self.dense3(rs))
        rs = self.dense4(rs)

        return rs
