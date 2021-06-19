from bert_serving.client import BertClient
import numpy as np
bc = BertClient(ip='202.201.242.38')


def cos_similar(sen_a_vec, sen_b_vec):
    '''
    计算两个句子的余弦相似度
    :param sen_a_vec:
    :param sen_b_vec:
    :return:
    '''
    vector_a = np.mat(sen_a_vec)
    vector_b = np.mat(sen_b_vec)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    return cos

if __name__=='__main__':
    # 从候选集condinates 中选出与sentence_a 最相近的句子
    sentence_a=['妈妈']
    sentence_b=['他妈']
    sentence_a_vec = bc.encode(sentence_a)[0]
    sentence_b_vec = bc.encode(sentence_b)[0]
    cos_similar = cos_similar(sentence_a_vec[2],sentence_b_vec[2])

    bc.close_bert()