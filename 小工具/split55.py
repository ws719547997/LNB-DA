# coding:utf-8
import os
import random
import time

path1 = '/home/wangsong/resource/snap/all/'
path2 = '/home/wangsong/resource/snap/snap10k/dev/'
path3 = '/home/wangsong/resource/snap/snap10k/test/'
path4 = '/home/wangsong/resource/snap/snap10k/train/'


def file_name(file_dir):
    for root, dirs, filess in os.walk(file_dir):
        print(root)  # 当前目录路径
        print(dirs)  # 当前路径下所有子目录
        print(filess)  # 当前路径下所有非目录子文件
        return filess


files = file_name(path1)


for file in files:
    if file != '.DS_Store':
        f = open(path1 + file, 'r',encoding='utf-8')
        ftest = open(path3 + file.split('.')[0]+'.txt','w',encoding='utf-8')
        fdev = open(path2 + file.split('.')[0]+'.txt','w',encoding='utf-8')
        ftrain = open(path4 + file.split('.')[0]+'.txt','w',encoding='utf-8')

        raw_list = f.readlines()
        random.seed(10)
        random.shuffle(raw_list)
        for i in range(0,int(len(raw_list)*0.2),1):  # 随机抽取数目 n
            ftest.writelines(raw_list[i])
        for i in range(int(len(raw_list)*0.2),int(len(raw_list)*0.3),1):
            fdev.writelines(raw_list[i])
        for i in range(int(len(raw_list)*0.3),len(raw_list),1):
            ftrain.writelines(raw_list[i])

        ftest.close()
        fdev.close()
        ftrain.close()
        f.close()

# for f in files:
#     print(f.split('.')[0])