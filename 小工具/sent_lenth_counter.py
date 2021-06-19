# coding:utf-8
import os
import random
import time
import xlwt
import openpyxl as oxl
import numpy as np
import pandas as pd

path2 = '/home/wangsong/resource/snap/all/'


def file_name(file_dir):
    for root, dirs, filess in os.walk(file_dir):
        print(root)  # 当前目录路径
        print(dirs)  # 当前路径下所有子目录
        print(filess)  # 当前路径下所有非目录子文件
        if filess == '.DS_Store':
            continue

        return filess


files = file_name(path2)
# files.remove('.DS_Store')
amount = 1
domain_id = 0

posneg = np.zeros((21, 2), dtype=int)
frenq = np.zeros((21, 202), dtype=int)

for file in files:
    if file != '.DS_Store':
        pos, neg = 0 ,0
        f = open(path2 + file, 'r', encoding='utf-8')
        raw_list = f.readlines()
        for i in range(int(len(raw_list))):
            if len(raw_list[i].split('\t')) < 5:
                continue
            sent = raw_list[i].split('\t')[4]
            tag = raw_list[i].split('\t')[2]
            sent = sent.split()

            if tag == 'NEG':
                neg += 1
            else:
                pos += 1

            sent_len = 0
            for st in sent:
                sent_len += len(st)
            if sent_len > 200:
                frenq[domain_id, 201] += 1
            else:
                frenq[domain_id, sent_len] += 1

        f.close()
        posneg[domain_id, 0] = pos
        posneg[domain_id, 1] = neg
        domain_id += 1


    sheet1 = pd.DataFrame(posneg)
    sheet1.index = files
    sheet2 = pd.DataFrame(frenq)
    sheet2.index = files

    write = pd.ExcelWriter('dataSetsINFO.xlsx')
    sheet1.to_excel(write,'sheet1')
    sheet2.to_excel(write,'sheet2')
write.save()
print('saved!')
write.close()
