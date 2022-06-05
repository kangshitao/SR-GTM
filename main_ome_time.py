import argparse
import pickle
import time
import utils_exp
from model_ome_time import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample',
                    help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/retailrocket/lastfm/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=256, help='hidden state size')
parser.add_argument('--memory_size', type=int, default=512, help='number of neighborhood session')
parser.add_argument('--epoch', type=int, default=11, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=1, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=3, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--pos', default='rp', help='postion embedding:rp/fp/wp/none')
parser.add_argument('--time', default='add', help='method of time embedding:add/cat/none')
parser.add_argument('--fus', default='gate', help='method of fusion:gate/cat/sum/max')
parser.add_argument('--sigma', type=int, default=12, help='sigma:12/16')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1,
                    help='split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)
with open('result.txt', 'a') as f:  # 将结果写入文档
    f.write(str(opt) + '\n')

def main():

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = utils_exp.split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)
    train_data = utils_exp.Data(train_data, shuffle=True)
    test_data = utils_exp.Data(test_data, shuffle=False)
    # del all_train_seq, g
    if opt.dataset == 'diginetica' or opt.dataset == 'diginetica_short' or opt.dataset == 'diginetica_long':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4' \
            or opt.dataset == 'yoochoose1_64_short' or opt.dataset == 'yoochoose1_64_long' \
            or opt.dataset == 'yoochoose1_4_short' or opt.dataset == 'yoochoose1_4_long':
        n_node = 37484
    elif opt.dataset == 'lastfm':
        n_node = 39186
        # n_node = 526
    elif opt.dataset == 'retailrocket':
        n_node = 48965
        # n_node = 288
    else:
        n_node = 310

    model = trans_to_cuda(SessionGraph(opt, n_node))
    torch.set_num_threads(2) # 限制线程数为2

    start = time.time()
    best_result = [[0, 0], [0, 0], [0, 0]]
    best_epoch = [[0, 0], [0, 0], [0, 0]]
    bad_counter = [0, 0, 0]
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        flag = [0, 0, 0]
        k = [5, 10, 20]
        curtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        curepoch = str(epoch)
        with open('result.txt', 'a') as f:  # 将结果写入文档
            f.write(curtime + '\t\t' + 'Current epoch:' + curepoch + '\n' + 'Best Result:' + '\n')
            for i in range(len(k)):
                if hit[i] >= best_result[i][0]:
                    best_result[i][0] = hit[i]
                    best_epoch[i][0] = epoch
                    flag[i] = 1
                if mrr[i] >= best_result[i][1]:
                    best_result[i][1] = mrr[i]
                    best_epoch[i][1] = epoch
                    flag[i] = 1
                    # 将内容写入文件
                content = 'Recall@' + str(k[i]) + ':\t' + str(best_result[i][0])[:7] + '\t' + \
                          'MRR@' + str(k[i]) + ':\t' + str(best_result[i][1])[:7] + '\t' +\
                          'Epoch:\t' + str(best_epoch[i][0]) + ',\t' + str(best_epoch[i][1]) + '\n'
                f.write(content)
                print('Best Result:')
                print('\tRecall@%d:\t%.4f\tMRR@%d:\t%.4f\tEpoch:\t%d,\t%d'
                      % (k[i], best_result[i][0], k[i], best_result[i][1], best_epoch[i][0], best_epoch[i][1]))
                bad_counter[i] += 1 - flag[i]
            f.write('--------------------------------------------------------' + '\n')

        bad_count = list(map(lambda x: x >= opt.patience, bad_counter))
        if sum(bad_count) == 3: break  # 当5,10,20的指标都超过了容忍度才退出
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
