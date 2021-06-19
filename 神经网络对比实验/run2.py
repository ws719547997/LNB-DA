# coding: UTF-8
import time
import xlwt
import torch
import numpy as np
from train_eval2 import train,test,evaluate, init_network
from importlib import import_module
import argparse
import os

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', default='DPCNN', type=str,  help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=True, type=bool, help='True for word, False for char')
parser.add_argument('--domMode',default='ST',type=str,help="Is include past domains or currenet domain when training")
parser.add_argument('--domNum',default=21,type=int,help='how many domain you want to train with?')
parser.add_argument('--dataSet',default='JD21',type=str,help='THUCNews or JD21')
# parser.add_argument('--isLL',default=False,type=bool,help='turn on lifelong mode or not')
args = parser.parse_args()

domain_list = ['褪黑素','维生素', '无线耳机', '蛋白粉', '游戏机', '电视', 'MacBook', '洗面奶', '智能手表', '吹风机', '小米手机', '红米手机', '护肤品', '电动牙刷', 'iPhone', '海鲜', '酒', '平板电脑', '修复霜', '运动鞋', '智能手环']

if __name__ == '__main__':
    dataset = args.dataSet  # 数据集
    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset_LL, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset_LL, build_iterator, get_time_dif

    start_time = time.time()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    acc,f1 = [],[]

    for domain in range(args.domNum):
        domain_name = domain_list[domain]
        print(domain_name)
        x = import_module('models.' + model_name)
        config = x.Config(dataset, embedding)

        print("Loading data...")
        vocab, train_data, dev_data, test_data = build_dataset_LL(config, args.word,args.domMode,domain_name,args.domNum,domain_list)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

        # train
        config.n_vocab = len(vocab)
        model = x.Model(config).to(config.device)
        if model_name != 'Transformer':
            init_network(model)
        print(model.parameters)
        model = train(config, model,train_iter, dev_iter, test_iter)
        _acc,_f1 = test(config, model, test_iter)

        acc.append(_acc)
        f1.append(_f1)
    
    xls = xlwt.Workbook()
    sheet1 = xls.add_sheet('sheet1')
    sheet1.write(0,0,args.model)
    sheet1.write(0,1,args.embedding)
    sheet1.write(0,2,args.domMode)
    sheet1.write(0,3,args.domNum)
    sheet1.write(1,0,'domain')
    sheet1.write(1,1,'acc')
    sheet1.write(1,2,'f1')
    sheet1.write(2,0,'avg:')
    sheet1.write(2,1,sum(acc)/len(acc))
    sheet1.write(2,2,sum(f1)/len(f1))
    for i in range(args.domNum):
        sheet1.write(i+3,0,domain_list[i])
        sheet1.write(i+3,1,acc[i])
        sheet1.write(i+3,2,f1[i])
    xls.save('JD21/result/'+args.model+args.domMode+str(args.domNum)+'.xls')
