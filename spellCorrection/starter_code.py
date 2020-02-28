# -*- coding: utf-8 -*-

# 词典库
vocab = set([line.rstrip() for line in open('vocab.txt')])

#生成所有编辑距离为1的集合
def generate_one_distance_words(mistake_word):
    # 假设使用26个字符
    letters = 'abcdefghijklmnopqrstuvwxyz'
    #吧单词分成左右两个部分
    splits = [ (mistake_word[:i],mistake_word[i:]) for i in range(len(mistake_word)+1)]
    
    #insert操作
    inserts = [L+c+R for L,R in splits for c in letters]
    #delete操作
    deletes = [L+R[1:] for L,R in splits if R]
    #relpace操作
    replaces = [L+c+R[1:] for L,R in splits if R for c in letters]
    
    return set(inserts + deletes + replaces)

#生成所有的候选集合
def generate_candidates(mistake_word):
    """
    word:给定的输入（错误的输入）
    返回所有可用的候选集合
    """
    words = generate_one_distance_words(mistake_word)
    # 如果生成的编辑距离为1的集合大小为0，则生成编辑距离为二的集合
    if len([word for word in words if word in vocab])==0:
        candidates = []
        for word in words:
           candidates +=  generate_one_distance_words(word)
        candidates = set(candidates)
        return [word for word in candidates if word in vocab]
            
    else :
        return [word for word in words if word in vocab]

#print(generate_candidates("Tkyos"))
  

from nltk.corpus import reuters
#读取语料库
categories = reuters.categories()
corpus = reuters.sents(categories=categories)

#构建语言模型：bigram
term_count = {}
bigram_count = {}
for doc in corpus:
    doc = ['<s>'] + doc 
    for i in range(0, len(doc)-1):
        # bigram : [i,i+1]
        term = doc[i]
        bigram = doc[i:i+2]
        
        if term in term_count:
            term_count[term] += 1
        else:
            term_count[term] = 1
        
        bigram = ' '.join(bigram)
        if bigram in bigram_count:
            bigram_count[bigram] += 1
        else:
            bigram_count[bigram] = 1
        
#用户打错的概率统计 --channel probability
#这里是模拟数据，假设每个打错的单词的概率是等比例的
channel_prob = {}

for line in open('spell-errors.txt'):
    items = line.split(':')
    correct = items[0].strip()
    mistakes = [item.strip() for item in items[1].split(',')]
    channel_prob[correct] = {}
    for mis in mistakes:
        channel_prob[correct][mis] = 1.0/len(mistakes)
#print(channel_prob)

import numpy as np
V = len(term_count.keys())

file = open("testdata.txt",'r')
for line in file:
    items = line.rstrip().split('\t')
    line = items[2].split()
    for word in line:
        word = word.rstrip('.').rstrip(',')
        if word not in vocab:
 
            # 生成所有的有效的候选集和
            candidates = generate_candidates(word)
            if len(candidates) < 1:
                continue
            
            probs = []
            #对于每个candidate，计算它的score
            #score = p(correct)* p(mistake/correct)
            #      = log p(correct) + log p(mistake/correct)
            #返回score最大的candidate
            for candi in candidates:
                prob = 0
                # a.计算channel probability
                if candi in channel_prob and word in channel_prob[candi]:
                    prob += np.log(channel_prob[candi][word])
                else :
                    prob += np.log(0.0001)
                
                # b.计算语言模型的概率
                idx = items[2].index(word)
                if items[2][idx] in bigram_count and candi in bigram_count[items[2][idx]]:
                    prob += np.log((bigram_count[items[2][idx]][candi]+1.0)/
                                   (term_count[bigram_count[items[2][idx]]]+V))
                
                else :
                    prob += np.log(1.0/V)
                
                probs.append(prob)
                
            max_idx = probs.index(max(probs))
            print(word, candidates[max_idx])
            
    
