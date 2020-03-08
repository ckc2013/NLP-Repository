# -*- coding: utf-8 -*-

# 创建词性和单词的映射字典
tag2id, id2tag = {}, {}
word2id, id2word = {}, {}

for line in open('traindata.txt'):
    items = line.split('/')
    word, tag = items[0], items[1].rstrip()
    
    if word not in word2id :
        word2id[word] = len(word2id)
        id2word[len(id2word)] = word
    if tag not in tag2id:
        tag2id[tag] = len(tag2id)
        id2tag[len(id2tag)] = tag

M = len(word2id)
N = len(tag2id)


# 构建pi, A, B
import numpy as np
pi = np.zeros(N)
A = np.zeros((N,M))
B = np.zeros((N,N))


# 计算模型的所有参数：pi A B 
prev_tag = ""
for line in open("traindata.txt"):
    items = line.split('/')
    wordId, tagId = word2id[items[0]], tag2id[items[1].rstrip()]
    if prev_tag == "":
        pi[tagId] += 1
        A[tagId][wordId] += 1
    else : #如果不是句子的开头
        A[tagId][wordId] += 1
        B[tag2id[prev_tag]][tagId] += 1
        
    if items[0] == ".":
        prev_tag = ""
    else :
        prev_tag = items[1].rstrip()
        
# normalize
pi = pi/sum(pi)
for i in range(N):
    A[i] /= sum(A[i])
    B[i] /= sum(B[i])
    

def log(v):
    if v==0:
        return np.log(v + 0.000001)
    return np.log(v)

def viterbi(x, pi, A, B):
    
    x = [word2id[word] for word in x.split(" ")]
    T = len(x)
    
    dp = np.zeros((T,N))
    ptr = np.zeros((T,N), dtype=int)
#    ptr = np.array([[0 for x in range(N)] for y in range(T)] )
    
    # basecase for DP算法
    for j in range(N):
        dp[0][j] = log(A[j][x[0]]) + log(pi[j])
    
    for i in range(1,T): #每个单词
        for j in range(N): #每个词性
            dp[i][j] = -999999
            for k in range(N):
                score = dp[i-1][k] + log(A[j][x[i]]) + log(B[k][j])
                if score > dp[i][j]:
                    dp[i][j] = score
                    ptr[i][j] = k
                    
    #decoding: 把最好的tag sequence 打印出来
    best_seq = [0]*T
    #step1:找出对应于最后一个的单词的词性
    best_seq[T-1] = np.argmax(dp[T-1])
    #step2:通过从后往前的循环来依次求出每个单词的词性
    for i in range(T-2,-1,-1):
        best_seq[i] = ptr[i+1][best_seq[i+1]]
    
    #打印出对应的词性
    for i in range(len(best_seq)):
        print(id2tag[best_seq[i]])
        
        
        
# 测试
x = "Social Security number , passport number and details about the services provided for the payment"
viterbi(x, pi, A, B)       
    
