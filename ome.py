import torch
import torch.nn.functional as F

# OME模块，存放最近的M个session的信息，包括每个session的item序列和GNN后的session表示
# 设置两个矩阵，一个存放每个session的item序列，一个存放每个session经过GNN后的表示


class OME():
    def __init__(self, mem_size):
        self.memory_size, self.memory_dim = mem_size
    @property
    def state_size(self):
        return(self.memory_size, self.memory_size)
    @property
    def output_size(self):
        return self.memory_dim

    def __call__(self, memory_state, session_represention, starting, state):
        def direct_assign():
            # 每轮最开始时
            read_memory = session_represention      # [b,d]
            new_memory_state = session_represention  # [b,d]
            return read_memory, new_memory_state

        def update_memory():
            # 求最近邻-余弦相似度，memory_state是外部记忆矩阵，[n_session,d]
            cos_similarity = self.smooth_cosine_similarity(session_represention, memory_state)  # [batch, n_session]
            if memory_state.shape[0] >= self.memory_size:
                neigh_sim , neigh_num = torch.topk(cos_similarity, k=self.memory_size)  # [batch_size, memory_size]
            else:
                neigh_sim, neigh_num = torch.topk(cos_similarity, k=memory_state.shape[0])
            # 从M矩阵中找出索引为neigh_num的session
            session_neighborhood = memory_state[neigh_num]  # [batch_size, memory_size, memory_dim]
            neigh_sim = torch.unsqueeze(F.softmax(neigh_sim, dim=1), dim=2)   # [b, m_s, 1]
            read_memory = (neigh_sim * session_neighborhood).sum(dim=1).squeeze()  # [batch_size, memory_dim]
            new_memory_state = torch.cat((memory_state, session_represention), dim=0)[-10000:]  # 更新M，只存放最新10000个
            return read_memory, new_memory_state

        if state == 'train':
            if starting:        # 如果是每次迭代的开始，则M矩阵为空，用当前session表示填充M
                read_memory, new_memory_state = direct_assign()
            else:
                read_memory, new_memory_state = update_memory()
        elif state == 'test':
            read_memory, new_memory_state = update_memory()
            new_memory_state = memory_state
        return read_memory, new_memory_state

    def smooth_cosine_similarity(self, session_emb, sess_all_representations):
        """
        :param session_emb: a [batch_size*hidden_units] tensor, session representation
        :param sess_all_representations: a [n_session*hidden_units] tensor,M
        :return: a [batch_size*n_session] weighting vector
        """
        # Cosine Similarity
        # 添加维度，然后扩展为b，[b, n_s, h_u]
        sess_all_representations = torch.unsqueeze(sess_all_representations, dim=0).expand(session_emb.shape[0], -1, -1)
        session_emb = torch.unsqueeze(session_emb, dim=2)  # [b,h_u,1]
        inner_product = torch.matmul(sess_all_representations, session_emb)  # [batch_size,n_session,1]
        k_norm = torch.sqrt(torch.square(session_emb).sum(dim=1,keepdim=True))  # [b,1,1]
        M_norm = torch.sqrt(torch.square(sess_all_representations).sum(dim=2, keepdim=True))  # [b,n_s,1]
        norm_product = M_norm * k_norm  # [b,n_s,1]
        similarity = torch.squeeze(inner_product / (norm_product + 1e-8), dim=2)  # [b,n_s]
        return similarity
