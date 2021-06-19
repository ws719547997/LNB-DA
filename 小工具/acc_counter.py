import os
import xlrd
import xlwt


def file_name(file_dir):
    for root, dirs, filess in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        # print(filess)  # 当前路径下所有非目录子文件
        return filess


def subfolder_name(file_dir):
    for root, dirs, filess in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        # print(filess)  # 当前路径下所有非目录子文件
        return dirs


def readacc_LSC(path, files):
    number = 0
    flag = 0
    for file in files:
        f = open(path + '/' + file, encoding='UTF-8')
        for line in f.readlines():
            count = line.split()
            # print(count)
            number += float(count[-1])
            flag += 1
    print(number / flag ,'DOMAIN NUM:' + str(flag))
    f.close


def readacc_LNB_acc(path, files):
    number = 0
    flag = 0
    for file in files:
        f = open(path + '/' + file, encoding='UTF-8')
        t = f.readline()
        for line in f.readlines():
            count = line.split()
            number += float(count[0])
            flag += 1
    acces.append(number / flag)
    f.close

def readacc_LNB_f1(path, files):
    number = 0
    flag = 0
    for file in files:
        f = open(path + '/' + file, encoding='UTF-8')
        t = f.readline()
        for line in f.readlines():
            count = line.split()
            number += float(count[0])
            flag += 1
    f1es.append(number / flag)
    f.close

DOMAINNUM = '21'
# path = '/Users/kuroneko/科研/LNBb/Data/Output/SentimentClassificaton/Accuracy_Reuters10'+DOMAINNUM
path = '/Users/kuroneko/科研/LNBb/Data/Output/SentimentClassificaton/Accuracy_1KReviewNaturalClassDistributionDomains20'
acces = []
f1es = []
param = []

wb=xlwt.Workbook()
ws=wb.add_sheet('sheet1')
mathod_files = subfolder_name(path)  # acc level
print(path)
for meth in mathod_files:
    item = path + '/' + meth
    print(meth)
    if len(file_name(item)) > 1:
        if len(file_name(item)) == 3:
            print("skip LLSGD")
            continue
        readacc_LNB_acc(item, file_name(item))
        param.append(meth)
    else:
        readacc_LSC(item, file_name(item))


# path = '/Users/kuroneko/科研/LNBb/Data/Output/SentimentClassificaton/F1Score_Reuters10'+DOMAINNUM
path = '/Users/kuroneko/科研/LNBb/Data/Output/SentimentClassificaton/F1Score_1KReviewNaturalClassDistributionDomains20'
mathod_files = subfolder_name(path)  # acc level
print(path)

for meth in mathod_files:
    item = path + '/' + meth
    print(meth)
    if len(file_name(item)) > 1:
        if len(file_name(item)) == 3:
            print("skip LLSGD")
            continue
        readacc_LNB_f1(item, file_name(item))
    else:
        readacc_LSC(item, file_name(item))

for index in range(len(param)):
    ws.write(index, 0, param[index])
    ws.write(index, 1, acces[index])
    ws.write(index, 2, f1es[index])

wb.save('../data/turning/lnb_on_english.xls')

