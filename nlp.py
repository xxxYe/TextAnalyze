# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:14:11 2021

@author: lenovo
"""

import sys
import re
import os
import time
import random

import jieba
import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

# import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import pylab

pylab.rcParams["font.family"] = "simhei"
import cv2
from PIL import Image, ImageDraw, ImageFont
import wave

# 读取词表
def read_words_list(path):
    with open(path,encoding='utf-8') as f:
        lines = f.readlines()

    strlist = []
    for l in lines:
        if "#" != l[0] and "" != l.strip():
            l = l.strip()
            strlist.append(l)
    return strlist


# 去除常用词
def remove_stop_words(text, stop_words):
    # 保存过滤词数量的字典
    swords_cnt = {}

    while "  " in text:  # 去掉多余空格
        text = text.replace("  ", " ")
    for key, words in stop_words.items():
        swords_cnt[key] = np.zeros(len(words))  # 创建向量
        for i, stpwd in enumerate(words):
            if (stpwd) in text:
                text = text.replace(" " + stpwd + " ", " ")
                #                 swords_cnt[key][i] += text.count(stpwd)
                swords_cnt[key][i] += 1
    return text, swords_cnt


# 查找敏感词
def check_sens_words(text, sens_words):
    ttext = text.strip()
    sw_dict = {}  # 敏感词
    for sw in sens_words:
        n = ttext.count(sw)  # 敏感词出现次数
        if n > 0:
            if not sw_dict.__contains__(sw):
                sw_dict[sw] = 0
            sw_dict[sw] += n
    return sw_dict


# 切割句子
def sentence_cut(text):
    pattern = pattern = r"\.|/|;|\?|!|。|；|！|……"
    pre_sentence = re.split(pattern, text)
    sentence = []
    section_pos = []
    max_cnt = 2
    cnt = max_cnt + 1
    for pre in pre_sentence:
        if len(pre) > 6:
            if cnt > max_cnt:
                sentence.append(pre)
                cnt = 0
            else:
                sentence[-1] += pre
                cnt += 1
            if "\n" in pre:
                sentence[-1].replace("\n", "")
                section_pos.append(len(sentence) - 1)
                cnt = max_cnt + 1
    return sentence, section_pos


# 中文比例
def str_ratio_zh(instr):
    cnt = 0
    for s in instr:
        # 中文字符范围
        if "\u4e00" <= s <= "\u9fa5":
            cnt += 1
    return cnt / len(instr)


stop_words_path = "hit_stopwords.txt"
stop_words = read_words_list(stop_words_path)

# torch专用切割
def cut_sentence(sentence):
    return [token for token in jieba.lcut(sentence) if token not in stop_words]

# get_dataset构造并返回Dataset所需的examples和fields
def get_test_dataset(test_data, text_field, label_field, fields):
    examples = []
    """
    for text in test_data:
        examples.append(torchtext.legacy.data.Example.fromlist([text], fields))
    test = torchtext.legacy.data.Dataset(examples, fields)
    """
    examples.append(torchtext.legacy.data.Example.fromlist([test_data], fields))
    test = torchtext.legacy.data.Dataset(examples, fields)
    return test


class TextCNN(nn.Module):
    def __init__(
        self,
        class_num,  # 分类数
        filter_sizes,  # 卷积核的长也就是滑动窗口的长
        filter_num,  # 卷积核的数量
        vocabulary_size,  # 词表的大小
        embedding_dimension,  # 词向量的维度
        vectors,  # 词向量
        dropout,
    ):  # dropout率
        super(TextCNN, self).__init__()  # 继承nn.Module

        chanel_num = 1  # 通道数，也就是一篇文章一个样本只相当于一个feature map

        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)  # 嵌入层
        self.embedding = self.embedding.from_pretrained(vectors)  # 嵌入层加载预训练词向量

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(chanel_num, filter_num, (fsz, embedding_dimension))
                for fsz in filter_sizes
            ]
        )  # 卷积层
        self.dropout = nn.Dropout(dropout)  # dropout
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)  # 全连接层

    def forward(self, x):
        # x维度[句子长度,一个batch中所包含的样本数] 例:[3451,128]
        x = self.embedding(x) # #经过嵌入层之后x的维度，[句子长度,一个batch中所包含的样本数,词向量维度] 例：[3451,128,300]
        x = x.permute(1,0,2) # permute函数将样本数和句子长度换一下位置，[一个batch中所包含的样本数,句子长度,词向量维度] 例：[128,3451,300]
        x = x.unsqueeze(1) # # conv2d需要输入的是一个四维数据，所以新增一维feature map数 unsqueeze(1)表示在第一维处新增一维，[一个batch中所包含的样本数,一个样本中的feature map数，句子长度,词向量维度] 例：[128,1,3451,300]
        x = [conv(x) for conv in self.convs] # 与卷积核进行卷积，输出是[一个batch中所包含的样本数,卷积核数，句子长度-卷积核size+1,1]维数据,因为有[3,4,5]三张size类型的卷积核所以用列表表达式 例：[[128,16,3459,1],[128,16,3458,1],[128,16,3457,1]]
        x = [sub_x.squeeze(3) for sub_x in x]#squeeze(3)判断第三维是否是1，如果是则压缩，如不是则保持原样 例：[[128,16,3459],[128,16,3458],[128,16,3457]]
        x = [F.relu(sub_x) for sub_x in x] # ReLU激活函数激活，不改变x维度 
        x = [F.max_pool1d(sub_x,sub_x.size(2)) for sub_x in x] # 池化层，根据之前说的原理，max_pool1d要取出每一个滑动窗口生成的矩阵的最大值，因此在第二维上取最大值 例：[[128,16,1],[128,16,1],[128,16,1]]
        x = [sub_x.squeeze(2) for sub_x in x] # 判断第二维是否为1，若是则压缩 例：[[128,16],[128,16],[128,16]]
        x = torch.cat(x, 1) # 进行拼接，例：[128,48]
        x = self.dropout(x) # 去除掉一些神经元防止过拟合，注意dropout之后x的维度依旧是[128,48]，并不是说我dropout的概率是0.5，去除了一半的神经元维度就变成了[128,24]，而是把x中的一些神经元的数据根据概率全部变成了0，维度依旧是[128,48]
        logits = self.fc(x) # 全接连层 例：输入x是[128,48] 输出logits是[128,10]
        return logits
    
    
#分类
#评估            
def classify(model,vectors):
    model.eval()
    print("请输入")
    input_text = input("待检测文本：")
    input_text = cut_sentence(input_text)
    
    """
    #input_text = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:：。？、~@#￥%……&*（）]+", " ", input_text)
    #input_text2 = remove_stop_words(input_text, stop_words)
    
    #input_text2 = list(filter(None, input_text))
    #转成int
    #input_text = input_text.split(',')
    """
    
    #print(input_text)
    """
    for data in input_text:
        text1 = torchtext.legacy.data.Field(sequential=True,lower=True,tokenize=cut_sentence)
        text1.build_vocab(data)
        #data = data.split(',')
        #sentence = np.array(data)
        #sentence = torch.from_numpy(sentence)
        print(text1)
    """
    text1 = torchtext.legacy.data.Field(sequential=True,lower=True,tokenize=cut_sentence)
    #调用torchtext的build_vocab函数建立词表
    text1.build_vocab(input_text, vectors=vectors)
    #for data in text1:
    
    """
    if torch.cuda.is_available(): # 如果有GPU将特征更新放在GPU上
        feature = text1.cuda()
    """
    
    logits = model(text1)
    
    corrects = 0
    corrects = torch.argmax(logits,dim=1)
    
    return corrects


text_field_path = "./word_embedding/wen_text.field"
label_field_path = "./word_embedding/wen_label.field"
model_path = "./model/bestmodel_steps800.pt"



wen_text = torch.load(text_field_path)
wen_label = torch.load(label_field_path)

class_num = len(wen_label.vocab)  # 类别数目
filter_size = [3, 4, 5]  # 卷积核种类数
filter_num = 16  # 卷积核数量
vocab_size = len(wen_text.vocab)  # 词表大小
embedding_dim = wen_text.vocab.vectors.size()[-1]  # 词向量维度
vectors = wen_text.vocab.vectors  # 词向量
dropout = 0.5  # (not used)
"""
learning_rate = 0.001  # (not used) 学习率
epochs = 5  # (not used) 迭代次数
save_dir = "./model"  # 模型保存路径
steps_show = 10  # (not used) 每10步查看一次训练集loss和mini batch里的准确率
steps_eval = 200  # (not used) 每100步测试一下验证集的准确率
early_stopping = 1000  # (not used) 若发现当前验证集的准确率在1000步训练之后不再提高 一直小于best_acc,则提前停止训练
"""
textcnn_model = TextCNN(
    class_num=class_num,
    filter_sizes=filter_size,
    filter_num=filter_num,
    vocabulary_size=vocab_size,
    embedding_dimension=embedding_dim,
    vectors=vectors,
    dropout=dropout,
)

#定义预训练词向量
pretrained_name = 'sgns.sogou.word' # 预训练词向量文件名
pretrained_path = './word_embedding' #预训练词向量存放路径
vectors = torchtext.vocab.Vectors(name=pretrained_name, cache=pretrained_path)

textcnn_model.load_state_dict(torch.load(model_path))  # 加载训练好的参数

while True:
    textcnn_model.eval()

    print("请输入")
    input_text = input("待检测文本：")
    #input_text = cut_sentence(input_text)
    #input_text = input_text.split()

    text_field = torchtext.legacy.data.Field(sequential=True,lower=True,tokenize=cut_sentence)
    label_field = torchtext.legacy.data.LabelField(sequential=False, dtype=torch.int64)
    fields=[('content',text_field)] 
    """
    examples = []
    fields=[('content',text1)] 
    examples.append(torchtext.legacy.data.Example.fromlist([input_text], fields))
    textdata = torchtext.legacy.data.Dataset(examples, fields)
    """
    text1 = get_test_dataset(input_text, text_field, label_field, fields)
    #text2 = torchtext.legacy.data.TabularDataset.splits(input_text)
    #调用torchtext的build_vocab函数建立词表
    text_field.build_vocab(text1,vectors=vectors)
    #生成迭代器
    test_iter = torchtext.legacy.data.Iterator(text1, batch_size=1, device=-1, sort=False, sort_within_batch=False, repeat=False)

    #print(len(text1.vocab))
    #text3 = text_field.vocab.vectors
    for batch in test_iter:
        feature = batch.content
        logits = textcnn_model(feature)
    
    output = 0
    output = torch.max(logits, 1)[1].numpy()
    #使用model对输入文本进行分类
    #result = classify(textcnn_model,vectors)
    print('模型检测结果：')
    if output == 0:
        print('合格。')
    if output == 1:
        print('存在敏感片段！')

