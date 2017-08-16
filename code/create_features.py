#coding=utf-8
'''
构造普惠端用户指标
@Author:李阳
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
puhui_user.drop(['total_m4_num','push_back_amount','collection_num','live_address','org_addr'], axis=1, inplace=1)
puhui_user.reset_index(drop=True, inplace=1)
# puhui_user = puhui_user.rename(columns={'is_m4': 'label'})

#分别取出m4和非m4的数据
# no_m4_user = puhui_user.loc[puhui_user.is_m4 == 0].copy()
# is_m4_user = puhui_user.loc[puhui_user.is_m4 == 1].copy()

# --------------------------------------------create features---------------------------------------------------
print('@creating features...')
#排序
# puhui_user['rank_ecif_id'] =puhui_user.ecif_id.rank(method='max')
# puhui_user['rank_ecif_id'] = MinMaxScaler().fit_transform(puhui_user.rank_ecif_id)

# one-hot
temp = pd.get_dummies(puhui_user.age_group, prefix='onehot_age_group', dummy_na=True)
puhui_user = pd.concat([puhui_user, temp], axis=1)
puhui_user.drop(['age_group'], axis=1, inplace=1)


puhui_user['sum_contract_amount'] = log(puhui_user.ecif_id.map(puhui_user.groupby('ecif_id').total_contract_amount.sum()) + 1)
puhui_user['mean_contract_amount'] = log(puhui_user.ecif_id.map(puhui_user.groupby('ecif_id').total_contract_amount.mean()) + 1)
puhui_user['max_contract_amount'] = log(puhui_user.ecif_id.map(puhui_user.groupby('ecif_id').total_contract_amount.max()) + 1)
puhui_user['min_contract_amounte'] = log(puhui_user.ecif_id.map(puhui_user.groupby('ecif_id').total_contract_amount.min()) + 1)

puhui_user['sum_approved_amount'] = log(puhui_user.ecif_id.map(puhui_user.groupby('ecif_id').total_approved_amount.sum()) + 1)
puhui_user['mean_approved_amount'] = log(puhui_user.ecif_id.map(puhui_user.groupby('ecif_id').total_approved_amount.mean()) + 1)
puhui_user['max_approved_amount'] = log(puhui_user.ecif_id.map(puhui_user.groupby('ecif_id').total_approved_amount.max()) + 1)
puhui_user['min_approved_amount'] = log(puhui_user.ecif_id.map(puhui_user.groupby('ecif_id').total_approved_amount.min()) + 1)

puhui_user['sum_apply_amount'] = log(puhui_user.ecif_id.map(puhui_user.groupby('ecif_id').total_apply_amount.sum()) + 1)
puhui_user['mean_apply_amount'] = log(puhui_user.ecif_id.map(puhui_user.groupby('ecif_id').total_apply_amount.mean()) + 1)
puhui_user['max_apply_amount'] = log(puhui_user.ecif_id.map(puhui_user.groupby('ecif_id').total_apply_amount.max()) + 1)
puhui_user['min_apply_amount'] = log(puhui_user.ecif_id.map(puhui_user.groupby('ecif_id').total_apply_amount.min()) + 1)

puhui_user['sum_urge_amount'] = log(puhui_user.ecif_id.map(puhui_user.groupby('ecif_id').should_urge_amount.sum()) + 1)
puhui_user['mean_urge_amount'] = log(puhui_user.ecif_id.map(puhui_user.groupby('ecif_id').should_urge_amount.mean()) + 1)
puhui_user['max_urge_amount'] = log(puhui_user.ecif_id.map(puhui_user.groupby('ecif_id').should_urge_amount.max()) + 1)
puhui_user['min_urge_amount'] = log(puhui_user.ecif_id.map(puhui_user.groupby('ecif_id').should_urge_amount.min()) + 1)


puhui_user.drop(['total_contract_amount','total_approved_amount','total_apply_amount','should_urge_amount'], axis=1, inplace=1)
pd.set_option('display.max_rows', None)

# puhui_user=puhui_user.unstack(level=0)
# print puhui_user.unstack()

no_m4 = puhui_user.loc[puhui_user.is_m4 ==0]
is_m4 = puhui_user.loc[puhui_user.is_m4 ==1]

train_user_nom4 = no_m4.sample(n=None, frac=0.8, replace=False, weights=None, random_state=None, axis=0)
train_user_ism4 = is_m4.sample(n=None, frac=0.8, replace=False, weights=None, random_state=None, axis=0)
test_user_nom4 = no_m4.sample(n=None, frac=0.2, replace=False, weights=None, random_state=None, axis=0)
test_user_ism4 = is_m4.sample(n=None, frac=0.2, replace=False, weights=None, random_state=None, axis=0)

train_user = pd.concat([train_user_nom4,train_user_ism4])
train_user['is_train']=1
test_user = pd.concat([test_user_nom4,test_user_ism4])
test_user['is_train']=0

flow = train_user.append(test_user).copy()

pickle.dump(flow, open('../myfeatures/features.pkl', 'wb'))
print('done!')








# train_info = train_info.loc[~train_info.CUST_NO.isnull()]




