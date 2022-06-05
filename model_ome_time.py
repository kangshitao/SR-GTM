import copy
import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import ome

MemoryState = torch.zeros(0).detach()  # 记忆矩阵，存放过去的M个session表示
STARTING = True  # 每一次epoch开始时，记忆矩阵为空
ModelState = 'train'  # 模型状态，测试状态不更新记忆矩阵


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))  # [3d,2d]
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))  # [3d,d]
        self.b_ih = Parameter(torch.Tensor(self.gate_size))  # [3d]
        self.b_hh = Parameter(torch.Tensor(self.gate_size))  # [3d]
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))  # [d]
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))  # [d]

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)  # (1) [b,len,2d]
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)  # [b,len,d]
        h_r, h_i, h_n = gh.chunk(3, 2)  # [b,len,d]
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)  # [b,len,d]
        return hy  # [b,len,d]

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.memory_size = opt.memory_size
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.pos_embedding = nn.Embedding(21, self.hidden_size)  # 位置编码
        self.time_embedding = nn.Embedding(601, self.hidden_size)  # 驻留时间嵌入
        self.pos = opt.pos
        self.time = opt.time
        self.fus = opt.fus
        self.sigma = opt.sigma
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.time_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_four = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform2 = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=True)
        self.linear_transform3 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        ##-----------------------------------------------
        # OME模块
        self.ome_cell = ome.OME(mem_size=(self.memory_size, self.hidden_size))  # 前一个数表示topk个邻居，后一个是嵌入维度
        ##_______________________________________________

        self.linear_transform_fg1 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_transform_fg2 = nn.Linear(self.hidden_size * 2, 1, bias=True)

        self.dropout = nn.Dropout(p=0.1)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get_position_emb(self, hidden, mask):  # 顺序位置编码
        # mask: 100*16
        mask_pos = torch.arange(0, hidden.size(1)).reshape(1, hidden.size(1)).expand(hidden.size(0), -1)  # 从1开始，100*16
        mask_pos = trans_to_cuda(mask_pos)
        pos_emb = self.pos_embedding(mask_pos)  # 100*16*d 获取位置编码表示
        pos_emb = pos_emb * mask.unsqueeze(-1)  # 100*16*d   将padding的地方置零
        return pos_emb

    def get_reverse_position_emb(self, hidden, mask):  # 逆序位置编码
        # mask: 100*16
        mask_sum = torch.sum(mask, dim=1, keepdim=True)  # 100*1 将mask的每个序列上求和，结果为每个序列的有效item个数
        # 对于[3,4,5,2,2,3,0,0,0]，其位置编码为[-6,-5,-4,-3,-2,-1,0,0,0],然后求相反数。
        # [0,1,2,3,4,5,6,7,8]
        mask_pos = torch.arange(0, hidden.size(1)).reshape(1, hidden.size(1)).expand(hidden.size(0), -1)  # 从1开始，100*16
        mask_pos = trans_to_cuda(mask_pos)
        mask_pos = -1 * (mask_pos - mask_sum)  # 100*16    [6,5,4,3,2,1,0,-1,-2]
        reverse_position = mask_pos * mask  # 100*16 [6,5,4,3,2,1,0,0,0]
        pos_emb = self.pos_embedding(reverse_position)  # 100*16*d 获取位置编码表示
        pos_emb = pos_emb * mask.unsqueeze(-1)  # 100*16*d   将padding的地方置零
        return pos_emb

    def compute_position_weight(self, hidden, mask):
        # 使用位置衰减权重，越靠近last item 的item位置权重越大。位置权重w = 1/(len-p)
        mask = mask.unsqueeze(-1).long()  # 100 * 16 * 1
        mask_sum = torch.sum(mask, dim=1, keepdim=True)  # 100*1*1 将mask的每个序列上求和，结果为每个序列的有效item个数
        mask_pos = torch.arange(hidden.size(1)).reshape(1, hidden.size(1), 1).expand(hidden.size(0), -1, -1)  # 100*16*1
        mask_pos = trans_to_cuda(mask_pos)
        position_weight = torch.clamp(1 / (mask_sum - mask_pos - 1e-8), min=0)  # 100*16*1    [1/3, 1/2, 1, 0, ...0]
        return position_weight

    def compute_scores(self, hidden, mask, time_emb, hidden_avg):
        # session_avg = torch.sum(hidden, 1) / torch.sum(mask, 1, keepdim=True)  # [b,d]

        # 为每个item表示加入位置编码嵌入
        if self.pos == 'fp':  # 顺序位置编码
            hidden = hidden + self.get_position_emb(hidden, mask)
        elif self.pos == 'rp':  # 逆序位置编码
            hidden = hidden + self.get_reverse_position_emb(hidden, mask)
        elif self.pos == 'wp':  # 使用位置衰减权重
            hidden = hidden * self.compute_position_weight(hidden, mask)

        # 加入时间信息
        if self.time != 'none':
            if self.time == "cat":
                hidden_time = self.time_transform(torch.cat([hidden, time_emb], -1))
            elif self.time == "add":
                hidden_time = hidden + time_emb

            ht_t = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
            q1_t = self.linear_one(ht_t).view(ht_t.shape[0], 1, ht_t.shape[1])  # batch_size x 1 x latent_size
            q2_t = self.linear_two(hidden_time)  # batch_size x seq _length x
            q3_t = self.linear_four(hidden_avg)
            alpha_t = self.linear_three(torch.sigmoid(q1_t + q2_t + q3_t))  # (b,s,1)
            a_t = torch.sum(alpha_t * hidden_time * mask.view(mask.shape[0], -1, 1).float(), 1)  # (b,d)

        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        q3 = self.linear_four(hidden_avg)
        alpha = self.linear_three(torch.sigmoid(q1 + q2 + q3))  # (b,s,1)
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)  # (b,d)
        # 有时间信息就根据带时间信息的会话求协同信息；没有则只根据会话表示a计算协同
        # tag : 7.19
        global MemoryState
        if self.time == 'none':
            memory_network_reads, memory_new_state = self.ome_cell(MemoryState, a, STARTING,
                                                                   ModelState)  # [b,d],[10000,d]
        else:
            # OME，计算协同信息
            memory_network_reads, memory_new_state = self.ome_cell(MemoryState, a_t, STARTING,
                                                                   ModelState)  # [b,d],[10000,d]

        MemoryState = memory_new_state.detach()  # 更新M

        if self.fus == 'cat':
            # 方式二：使用cat
            a = self.linear_transform2(torch.cat([a, ht, memory_network_reads], 1))
        # 方式三：使用sum
        elif self.fus == 'sum':
            hidden_seq = self.linear_transform_fg1(torch.cat([a, ht], 1))
            a = hidden_seq + memory_network_reads
        elif self.fus == 'max':
            hidden_seq = self.linear_transform_fg1(torch.cat([a, ht], 1))
            a = torch.max(hidden_seq,memory_network_reads)
        else:
            # 默认方式：使用门控机制将协同信息加入进来
            hidden_seq = self.linear_transform_fg1(torch.cat([a, ht], 1))
            w = torch.sigmoid(self.linear_transform_fg2(torch.cat([hidden_seq, memory_network_reads], -1)))
            a = w * hidden_seq + (1 - w) * memory_network_reads
        ###
        # 会话表示和预测层
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        a = a / torch.norm(a, dim=1, keepdim=True)  # 对session表示进行norm
        score = torch.matmul(a, b.transpose(1, 0))

        """
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        hidden_gen = hidden_gen / torch.norm(hidden_gen, dim=1, keepdim=True)  # 对session表示进行norm
        score = torch.matmul(hidden_gen, b.transpose(1, 0))
        """
        # return score
        return score * self.sigma

    def forward(self, inputs, A, times):  # input是唯一item序列，[[0,274,0,0,0,0],[0,295,296,0,0,0],...]
        # layernorm item
        self.embedding.weight.data = self.embedding.weight / torch.norm(self.embedding.weight, dim=1, keepdim=True)

        hidden = self.embedding(inputs)
        # hidden = self.dropout(hidden)
        time_emb = self.time_embedding(times)
        hidden_avg = torch.sum(hidden, 1, keepdim=True) / hidden.shape[1]  # 初始item的均值

        hidden = self.gnn(A, hidden)  # 使用GNN求序列兴趣
        return hidden, time_emb, hidden_avg


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets, times = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())  # [b,16,100],每个session中item的顺序，用于还原session
    items = trans_to_cuda(torch.Tensor(items).long())  # [b,6,100]，每个session中唯一item列表，包括
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    times = trans_to_cuda(torch.Tensor(times).long())
    hidden, time_emb, hidden_avg = model(items, A, times)  # time_emb: [b,16,100]

    get = lambda i: hidden[i][alias_inputs[i]]  # 还原真实seq，有重复，[274,274,0,0,...0]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])  # [b,16,100]
    return targets, model.compute_scores(seq_hidden, mask, time_emb, hidden_avg)


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    model.scheduler.step()
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [[], [], []], [[], [], []]
    hit_mrr_list = [5, 10, 20]  # k=5,10,20
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        for k in range(len(hit_mrr_list)):
            sub_scores = scores.topk(hit_mrr_list[k])[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            h, m = get_hit_mrr(sub_scores, targets, test_data.mask)
            hit[k] += h
            mrr[k] += m
    hit = list(np.mean(hit, 1) * 100)
    mrr = list(np.mean(mrr, 1) * 100)
    return hit, mrr


def get_hit_mrr(sub_scores, targets, test_data_mask):
    hit_temp, mrr_temp = [], []
    for score, target, mask in zip(sub_scores, targets, test_data_mask):
        hit_temp.append(np.isin(target - 1, score))
        if len(np.where(score == target - 1)[0]) == 0:
            mrr_temp.append(0)
        else:
            mrr_temp.append(1 / (np.where(score == target - 1)[0][0] + 1))
    return hit_temp, mrr_temp
