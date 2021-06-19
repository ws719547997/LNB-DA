#!/usr/bin/env python3

"""
@author: xi
@since: 2018-02-02
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pickle as pkl

import common
import photinia as ph
import os
from transformers import BertTokenizer
from bert_serving.client import BertClient

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def act(x):
    return tf.minimum(tf.maximum(1e-2 * x, x), 0.9 + 1e-2 * x)


class VectorStat(ph.Widget):

    def __init__(self, name):
        super(VectorStat, self).__init__(name)
        self._tracks = {}

    def _build(self):
        pass

    def _setup(self, tensor):
        name = tensor.name
        basename = ph.utils.get_basename(name)
        init_value = tf.zeros((tensor.shape[1],))
        stat = ph.variable(basename, init_value)
        stat_track = ph.variable(basename + '_track', init_value, trainable=False)
        tensor = tf.reduce_sum(tf.abs(tensor), 0)
        update = tf.assign(stat_track, stat_track + tensor)
        self._tracks[name] = (stat, stat_track, update)

    @property
    def updates(self):
        return [update for _, _, update in self._tracks.values()]

    def read_stat(self, tensor):
        name = tensor.name
        if name not in self._tracks:
            return None
        stat = self._tracks[name][0]
        return ph.utils.read_variables(stat)

    def update_stats(self):
        for stat, stat_track, _ in self._tracks.values():
            value1 = ph.utils.read_variables(stat)
            value2 = ph.utils.read_variables(stat_track)
            ph.utils.write_variables(stat, value1 + value2)


class MaskGrad(ph.OptimizerWrapper):

    def __init__(self, optimizer):
        super(MaskGrad, self).__init__(optimizer)
        self._masks = {}
        self._threshold = 0.5

    def add_mask(self, w):
        name = w.name
        if name in self._masks:
            raise ValueError('Mask exists for %s.' % name)
        mask_var = ph.variable(
            ph.utils.get_basename(name) + '_mask',
            np.zeros((w.shape[0],), dtype=np.float32),
            trainable=False
        )
        self._masks[name] = mask_var

    def update_mask(self, w, stat, n=0):
        t = 0.2
        lam = 10
        self._threshold = 1.0 - ((1.0 - t) * np.exp(-lam * n) + t)
        # print(self._threshold)
        name = w.name
        if name not in self._masks:
            raise ValueError('No mask added for %s.' % name)
        mask_var = self._masks[name]
        mask = self._compute_mask(stat)
        # for e in mask:
            # print(e, end='\t')
        # print()
        ph.utils.write_variables(mask_var, mask)

    def _compute_mask(self, stat):
        min_value = np.min(stat)
        max_value = np.max(stat)
        if max_value == 0:
            return 1 - stat
        normalized_stat = 1 - (stat - min_value) / (max_value - min_value + 1e-5)
        #
        ordered_stat = [(index, value) for index, value in enumerate(normalized_stat)]
        ordered_stat.sort(key=lambda a: a[1])
        end = int(len(ordered_stat) * self._threshold)
        ordered_stat = ordered_stat[: end]
        #
        for i, _ in ordered_stat:
            normalized_stat[i] = 0.0
        return normalized_stat

    def _process_gradients(self, pair_list):
        # pair_list, _, _ = ph.clip_gradient(pair_list, 1)
        new_list = []
        for g, w in pair_list:
            # g = tf.clip_by_value(g, -1e-4, 1e-4)
            name = w.name
            if name in self._masks:
                mask = self._masks[name]
                g *= tf.reshape(mask, (-1, 1))
            new_list.append((g, w))
        return new_list


class Embedding(ph.Widget):

    def __init__(self,
                 name,
                 wemb_size,
                 state_size,
                 activation=tf.nn.tanh):
        self._wemb_size = wemb_size
        self._state_size = state_size
        self._activation = activation
        super(Embedding, self).__init__(name)

    def _build(self):
        ph.GRUCell(
            'cell', self._wemb_size, self._state_size,
            with_bias=False,
            activation=self._activation,
            w_init=ph.TruncatedNormal(0, 1e-3),
            u_init=ph.TruncatedNormal(0, 1e-3)
        )

    def _setup(self, seq):
        seq_len = ph.sequence_length(seq)
        states = self.cell.setup_sequence(seq)
        h = ph.last_elements(states, seq_len)
        return h, states


class Main(ph.Trainer):

    def __init__(self,
                 name,
                 wemb_size=300):
        self._wemb_size = wemb_size
        super(Main, self).__init__(name)

    def _build(self):
        shared = Embedding('shared', self._wemb_size, 500, act)
        specific = Embedding('specific', self._wemb_size, 500)
        gate = ph.Gate('gate', (500, 500), 500)
        lin = ph.Linear('lin', 500, 1000)
        out = ph.Linear('out', 1000, 2)
        stat = VectorStat('stat')
        drop = ph.Dropout('drop')
        #
        seq = ph.placeholder('seq', (None, None, self._wemb_size))
        h1, states1 = shared.setup(seq)
        stat.setup(tf.reshape(seq, (-1, self._wemb_size), name='flat_seq'))
        stat.setup(tf.reshape(states1, (-1, 500), name='flat_states'))
        h2, _ = specific.setup(seq)
        g = gate.setup(h1, h2)
        h = g * h1 + (1.0 - g) * h2
        y_pred = ph.setup(
            h, [
                drop, lin, ph.lrelu,
                drop, out, tf.nn.sigmoid
            ]
        )
        y_pred_ = ph.setup(
            h1, [
                drop, lin, ph.lrelu,
                drop, out, tf.nn.sigmoid
            ]
        )
        y_pred__ = ph.setup(
            h1, [
                drop, lin, ph.lrelu,
                drop, out, tf.nn.sigmoid
            ]
        )
        label_pred = tf.argmax(y_pred, 1)
        label = ph.placeholder('label', (None, 2))
        loss = tf.reduce_mean((y_pred - label) ** 2, axis=1)
        loss += tf.reduce_mean((y_pred_ - label) ** 2, axis=1)
        loss += tf.reduce_mean((y_pred__ - label) ** 2, axis=1)
        loss_sum = tf.reduce_sum(loss)
        loss_mean = tf.reduce_mean(loss)
        #
        correct = tf.cast(tf.equal(label_pred, tf.argmax(label, 1)), ph.D_TYPE)
        correct_pos = correct * label[:, 1]
        correct_neg = correct * label[:, 0]
        hit_pos = tf.reduce_sum(correct_pos)
        hit_neg = tf.reduce_sum(correct_neg)
        pred_pos = tf.reduce_sum(label_pred)
        pred_neg = tf.reduce_sum(1 - label_pred)
        error = tf.reduce_sum(1 - correct)
        #
        reg = ph.Regularizer()
        reg.add_l1(self.get_trainable_variables())
        #
        optimizer = MaskGrad(tf.train.RMSPropOptimizer(1e-4, 0.8, 0.9))
        self._optimizer = optimizer
        optimizer.add_mask(shared.cell.wz)
        optimizer.add_mask(shared.cell.wr)
        optimizer.add_mask(shared.cell.wh)
        optimizer.add_mask(shared.cell.uz)
        optimizer.add_mask(shared.cell.ur)
        optimizer.add_mask(shared.cell.uh)
        #
        self._add_train_slot(
            inputs=(seq, label),
            outputs={
                'Loss': loss_mean,
                'Norm': tf.norm(self.specific.cell.uz, 1)
            },
            updates=(
                optimizer.minimize(loss_mean + reg.get_loss(2e-7)),
                stat.updates
            ),
            givens={drop.keep_prob: 0.5}
        )
        self._add_validate_slot(
            inputs=(seq, label),
            outputs={
                'Loss': loss_sum,
                'hit_pos': hit_pos * 100,
                'hit_neg': hit_neg * 100,
                'pred_pos': pred_pos * 100,
                'pred_neg': pred_neg * 100,
                'Error': error * 100,
            },
            givens={drop.keep_prob: 1.0}
        )


def padding_list(list, args=None):
    """
    把每个list都补长到指定维数
    """
    while len(list) < args.max_seq_length:
        list.append(0.0)
    return list

def build_dataset_LL(index,args,bert_client):
    # tokenizer = BertTokenizer.from_pretrained(args.bert_model)  # huggingface的transformers

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in f:
                lin = line.strip()
                if not lin:
                    continue
                if len(lin.split('\t')) < 5:
                    continue
                lin_sp = lin.split('\t')
                content = lin_sp[4]
                label = 0 if lin_sp[2] == 'NEG' else 1
                # token = tokenizer(content,truncation='longest_first',max_length=args.max_seq_length)
                token = bert_client.encode([content])[0]
                for key in token:
                    key = padding_list(key,args)
                contents.append((token, label)) # 返回BERTtoken和label token中有 input_ids segmengt_musk attention_musk
        return contents
    _train = []
    _dev = []
    _train = load_dataset('JD21/data/train/' + dom_list[index] + '.txt', args.max_seq_length)
    _dev = load_dataset('JD21/data/dev/' + dom_list[index] + '.txt', args.max_seq_length)

    # only test on target domain
    _test = load_dataset('JD21/data/test/' + dom_list[index] + '.txt', args.max_seq_length)

    return _train, _dev, _test



# the order can be changed manually by random
domain_num = 21
MAX_VOCAB_SIZE = 50000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
dom_list = ['褪黑素', '维生素', '无线耳机', '蛋白粉', '游戏机', '电视', 'MacBook', '洗面奶', '智能手表', '吹风机', '小米手机', '红米手机', '护肤品',
                   '电动牙刷', 'iPhone', '海鲜', '酒', '平板电脑', '修复霜', '运动鞋', '智能手环']


def main1(args):
    log_file = 'logs_' + args.name + str(args.start)  # 加start是为了互相区分开
    model_dir = 'models_' + args.name+ str(args.start)
    exp = common.Experiment(log_file,model_dir)  # 在这里内次运行前会把已经存在model_dir的同名目录删除
    # we = ph.utils.WordEmbedding()  # 修改了原函数 初始化之后可以直接通过embeding查询
    bert_client = BertClient(ip='202.201.242.38')
    trainer = Main(args.name+ str(args.start), 768)  # 词向量 300维  输入了模型名称和300维
    ph.initialize_global_variables()
    #
    test_list = []

    for i in range(domain_num):
        print('***********************')
        print('domain:'+str(i)+dom_list[i])
        train_data, dev_data, test_data = build_dataset_LL(i,args,bert_client)  # 重写的数据读取 这里是读取一个领域的数据
        train_ds = common.TrainSource(train_data, i)  # 实现了源代码中的DataSource作为Dataloader
        dev_ds = common.TrainSource(dev_data, i)
        test_ds = common.TrainSource(test_data, i)
        test_list.append(test_ds)
        #
        exp.load_model(trainer)  # 先把之前的模型拿出来接着训练 第一次就从头训
        seq_stat = trainer.stat.read_stat(trainer.flat_seq)
        states_stat = trainer.stat.read_stat(trainer.flat_states)
        trainer._optimizer.update_mask(trainer.shared.cell.wz, seq_stat, i)
        trainer._optimizer.update_mask(trainer.shared.cell.wr, seq_stat, i)
        trainer._optimizer.update_mask(trainer.shared.cell.wh, seq_stat, i)
        trainer._optimizer.update_mask(trainer.shared.cell.uz, states_stat, i)
        trainer._optimizer.update_mask(trainer.shared.cell.ur, states_stat, i)
        trainer._optimizer.update_mask(trainer.shared.cell.uh, states_stat, i)
        trainer.add_data_trainer(train_ds, 64)  # 32 batch_size
        # trainer.add_screen_logger('train', ('Loss', 'Norm'), interval=1)  # 这里是输出训练过程的
        trainer.add_data_validator(test_ds, 64, interval=20)  # 相当于model.eval()
        # trainer.add_screen_logger(  # eval的结果
        #     "validate",
        #     ('hit_pos', 'hit_neg', 'pred_pos', 'pred_neg', 'Error'),
        #     message='[%d]' % i,
        #     interval=20
        # )
        trainer.add_fitter(common.DevFitter(dev_ds, 64, 20))
        trainer.fit(args.num_loops)
        trainer.clear_fitters()  # 训练过程到此结束
        #
        exp.dump_model(trainer)  # 模型存起来

        # test turn
        for test_data in test_list:
            trainer.add_data_validator(test_data, 64, interval=1)  # model.eval()
            trainer.add_screen_logger(  # 输出测试结果
                "validate",
                ('hit_pos', 'hit_neg', 'pred_pos', 'pred_neg', 'Error'),
                message='[%d]' % i,
                interval=1
            )
            trainer.fit(1)
            trainer.clear_fitters()
        trainer.stat.update_stats()
    bert_client.close()
    return 0

if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--gpu', default='0', help='Choose which GPU to use.')
    _parser.add_argument('--batch-'
                         '', default=64, help='Batch size.', type=int)
    _parser.add_argument('--num-loops', default=20, help='Max fit loops.', type=int)  #  实验中500
    _parser.add_argument('--name', default='ws')  # 起个名字吧~
    _parser.add_argument('--start',default=0,type=int )  # 这里是设定domain_list的起始点，用于修改序列
    _parser.add_argument('--max_seq_length', default=128, type=int)
    _parser.add_argument('--bert_model', default='/home/wangsong/resource/bert-base-chinese', type=str)
    _args = _parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpu

    for i in range(_args.start):
        dom_list.append(dom_list.pop(0))



    exit(main1(_args))
