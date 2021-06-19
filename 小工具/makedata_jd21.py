# encode = utf-8

import jieba
import pandas as pd
import codecs
import re
import os


def file_name(file_dir):
    for root, dirs, filess in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        print(filess)  # 当前路径下所有非目录子文件
        return filess

def dir_name(file_dir):
    for root, dirs, filess in os.walk(file_dir):
        # print(root)  # 当前目录路径
        print(dirs)  # 当前路径下所有子目录
        # print(filess)  # 当前路径下所有非目录子文件
        return dir


def stopword(review):
    characters = {',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '-',
                  '...', '^', '{', '}', '~', '，', '。', '：', '……', '\n'}
    words = [w for w in review if not w in characters]
    return words


def get_cut(index, Domain, Label, Rating, Review):
    try:
        review_cut = re.findall('[\u4e00-\u9fa5]', Review)
        content = ''
        for s in review_cut:
            content += s
        content = jieba.cut(content,cut_all=False)
        output = '{}\t{}\t{}\t{}\t{}\n'.format(index, Domain, Label, Rating, ' '.join(content))
        f = codecs.open(path + '/' + Domain + '.txt', 'a+', 'utf-8')
        f.write(output)
        f.close()
        # print(index, content)
    except Exception as e:
        print(e)


def get_cut_by_char(index, Domain, Label, Rating, Review):
    try:
        review_cut = re.findall('[\u4e00-\u9fa5]', Review)
        content = ''
        for s in review_cut:
            content += s
            content += ' '
        output = '{}\t{}\t{}\t{}\t{}\n'.format(index, Domain, Label, Rating, content)
        f = codecs.open(path + '/' + Domain + '.txt', 'a+', 'utf-8')
        f.write(output)
        f.close()
        print(index, review_cut)
    except Exception as e:
        print(e)


def get_content(path, DOMAIN,counter):
    data = pd.DataFrame(pd.read_csv(path))
    # print(data)
    for i in range(1, len(data)):
        index = counter + i
        Domain = DOMAIN
        if data.loc[i, '评论类型'] == '好评':
            Label = 'POS'
            Rating = 5
        else:
            Label = 'NEG'
            Rating = 1
        Review = data.loc[i, '评价内容']
        if Review == '':
            continue
        get_cut(index, Domain, Label, Rating, Review)
    return index
        # get_cut_by_char(index, Domain, Label, Rating, Review)


path = '/Users/kuroneko/科研/jd_comment_raw/cutchar'
input_path = '/Users/kuroneko/科研/jd_comment_raw/JD21/'
filess = os.listdir(input_path)
for i in range(len(filess)):
    if filess[i]== '.DS_Store':
        continue
    else:
        counter = 0
        print(input_path+filess[i])
        sub_file = os.listdir(input_path+filess[i])
        for item in sub_file:
            if item.split('.')[1] =='csv':
                counter = get_content(input_path+filess[i]+'/'+item, 'all',counter)
                counter += 1
# get_content('/Users/kuroneko/科研/语料库/吹风机 1800.csv','吹风机')

