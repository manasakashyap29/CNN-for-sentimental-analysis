# python3.5 run_sc.py <test_file_path> <model_file_path> <output_file_path>

import os
import math
import sys
import torch
import re

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
import time
warnings.filterwarnings("ignore")

start = time.time()

stopwords_list = ['a', 'about', 'across', 'all', 'already', 'also', 'an', 'and', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'be', 'been', 'before', 'being', 'beings', 'by', 'came', 'come', 'did', 'do', 'does', 'during', 'each', 'either', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'face', 'faces', 'for', 'four', 'from', 'further', 'furthered', 'furthering', 'furthers', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'got', 'group', 'grouped', 'grouping', 'groups', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'herself', 'him', 'himself', 'his', 'how', 'however', 'if', 'in', 'is', 'it', 'its', 'itself', 'just', 'keep', 'keeps', 'let', 'lets', 'made', 'make', 'making', 'man', 'many', 'me', 'member', 'members', 'men', 'mr', 'mrs', 'much', 'my', 'myself', 'number', 'numbers', 'of', 'one', 'or', 'other', 'others', 'our', 'per', 'place', 'places', 'put', 'puts', 'room', 'rooms', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems', 'sees', 'several', 'shall', 'she', 'should', 'side', 'sides', 'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states', 'still', 'such', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'thing', 'things', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 'through', 'thus', 'to', 'today', 'too', 'took', 'two', 'us', 'was', 'way', 'ways', 'we', 'wells', 'went', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whose', 'why', 'with', 'would', 'year', 'years', 'you', 'your', 'yours']

punct_list = ["#", "&", ".", ",", "(", ")", ":", ";", "/", "?", "'"]

def isValid(word):
    if "_" in word:
        return False
    if re.fullmatch("[A-Za-z][a-z]+[A-Z]+[a-z]*", word):
        return False
    if any(x in word for x in punct_list):
        return True
    if word.lower() in stopwords_list:
        return False
    if len(word) == 1:
        return False  
    return True


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout,embed_index):
        super().__init__()
        self.filter_sizes = filter_sizes
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        self.fc1 = nn.Linear(len(filter_sizes) * n_filters, n_filters)
        self.fc2 = nn.Linear(n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
        self.embedding_index_dict = embed_index
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        #print(text)
        text = text.permute(1, 0)

        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)

        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim=1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        res_fc1 = self.fc1(cat)

        return self.fc2(res_fc1)


def load_test_files(test_text_file):
    documents = []

    with open(test_text_file, 'r', encoding="utf8") as f:
        for doc in f:
            doc = [word for word in doc.strip().split() if isValid(word)]
            documents.append(doc)
        f.close()
    
    return documents



def test_model(test_text_file, model_file, out_file):
    documents = load_test_files(test_text_file)
    
    model = torch.load(model_file)
    model.eval()

    padding_size = max(model.filter_sizes)

    with torch.no_grad():

        with open(out_file, 'w', encoding="utf-8") as f:
            for doc in documents:
                indexed_doc = []
                for word in doc:
                    word_index = -1
                    try:
                        word_index = model.embedding_index_dict[word]
                    except:
                        word_index = model.embedding_index_dict['<pad>']
                    indexed_doc.append(word_index)
                
                padding = [model.embedding_index_dict['<pad>'] for i in range(padding_size)]
                indexed_doc = torch.LongTensor(padding + indexed_doc + padding).to(model.device).unsqueeze(1)
                prediction = int(torch.round(torch.sigmoid(model(indexed_doc)))[0]) + 1

                f.write(str(prediction) + "\n")
    
    print(time.time() - start)
    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_text_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    test_model(test_text_file, model_file, out_file)
