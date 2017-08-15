#coding=utf-8
'''
构造普惠端用户指标
@Author:
@E-mail：dlutfeipeng@gmail.com
'''
import pandas as pd
import numpy as np
import csv
import re
import os
import jieba
import codecs
import pickle
from numpy import log
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

data_path = '../rawdata/'
#用户普惠指标
file_puhui_user='bak_m4data.tsv'

print('Part I:')
# --------------------------------------------load data---------------------------------------------------
print('@processing data...')
with codecs.open(data_path + file_puhui_user, 'r', encoding='utf-8') as fin,\
    codecs.open(data_path + 'processed_' + file_puhui_user, 'w', encoding='utf-8') as fout:
        for index, line in enumerate(fin):
            items = line.strip().split('\t')
            for i, item in enumerate(items):
                item = item.strip()
                if i < 45:
                    fout.write(item + '\t')
                else:
                    fout.write(item + '\n')
        print('{} lines in train_95598'.format(index))

puhui_user = pd.read_csv(data_path + 'processed_' + file_puhui_user, sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE,low_memory=False)
puhui_user = puhui_user.loc[~puhui_user.ecif_id.isnull()]
puhui_user['ecif_id'] = puhui_user.ecif_id
#缺失值补充为0
puhui_user.fillna(0, inplace=True)

#删除为空字段和标签字段
puhui_user.drop(['total_m4_num','push_back_amount','collection_num'], axis=1, inplace=1)

print puhui_user



# train_info = train_info.loc[~train_info.CUST_NO.isnull()]




