# python3.5 build_sc.py <pretrained_vectors_gzipped_file_absolute_path> <train_text_path> <train_label_path> <model_file_path>

import os
import math
import sys
import gzip
import io
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

start = time.time()

######---PARAMETERS---######

random_seed = 123
train_batch_size = 50
num_filters = 25
n_gram_sizes = [3,4,5,6]
output_dimension = 1    #1 or 2
dropout_rate = 0.7
epochs = 7

#To be fixed in function load_embeddings_file()
embedding_vector_size = -1     
vocabulary_size = -1

############################
random_seed = 123

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

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
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim=1))

        res_fc1 = self.fc1(cat)

        return self.fc2(res_fc1)



def load_embeddings_file(embeddings_file):
    global embedding_vector_size, vocabulary_size

    gz = gzip.open(embeddings_file, 'rb')
    f = io.BufferedReader(gz)
    #file_content = f.read().decode('utf8').strip().split("\n")
    print("time", 1.1, time.time() - start)

    embeddings_indices = dict()
    embedding_matrix = []
    count = 0
    flag = False

    for embedding in f:
        if not flag:
            flag = True
            continue
        
        embedding = embedding.decode('utf-8').strip().split()

        if not isValid(embedding[0]):
            continue

        for i in range(1, len(embedding)):
            embedding[i] = float(embedding[i])
        embeddings_indices[embedding[0]] = count
        count+=1
        if count%100000 == 0:
            print(count, "time", time.time() - start)
        
        embedding_vector = embedding[1:]
        if len(embedding_vector) < 300:
            print(count - 1)
            embedding_vector.extend([0 for i in range(300 - len(embedding_vector))])
        embedding_matrix.append(embedding_vector)
        
        if count > 250000:
            break
    print(count, "time", time.time() - start)
    vocabulary_size = len(embeddings_indices)
    embedding_vector_size = len(embedding_matrix[0])
    embeddings_indices['<pad>'] = count
    default_embedding = [0 for i in range(embedding_vector_size)]
    embedding_matrix.append(default_embedding)
    vocabulary_size += 1
    gz.close()
    return embeddings_indices, embedding_matrix

def load_train_files(train_text_file, train_label_file):
    documents = []
    labels = []

    with open(train_text_file, 'r', encoding="utf8") as f:
        for doc in f:
            doc = [word for word in doc.strip().split() if isValid(word)]
            documents.append(doc)
        f.close()

    with open(train_label_file, 'r') as f:
        for label in f:
            label = int(label.strip()) - 1
            labels.append(label)
        f.close()
    
    return documents, labels

def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    #rounded_preds = torch.round(preds)

    #print(max(torch.sigmoid(preds)), min(torch.sigmoid(preds)))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc
    

def modeltrain(model, x_batches, y_batches, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    num_batches = len(x_batches)
    for i in range(num_batches):
        
        optimizer.zero_grad()
        
        predictions = model(x_batches[i]).squeeze(1)
        
        loss = criterion(predictions, y_batches[i])
        
        acc = binary_accuracy(predictions, y_batches[i])
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / num_batches, epoch_acc / num_batches

    
def get_batches(documents, labels, model):
    no_documents = len(documents)
    no_batches = int(no_documents/train_batch_size)
    text_batches = []
    label_batches = []
    
    for i in range(no_batches):
        start_pos = i*train_batch_size
        end_pos = min((i+1)*train_batch_size, no_documents)
        
        text_batch = documents[start_pos:end_pos]
        label_batch = labels[start_pos:end_pos]
        
        label_batches.append(torch.cuda.FloatTensor(label_batch))
        
        indexed_text_batch = []
        padding_size = max(n_gram_sizes) - 1
        batch_doc_size = max([len(doc) for doc in text_batch])

        for doc in text_batch:
            indexed_doc = []
            for word in doc:
                word_index = -1
                try:
                    word_index = model.embedding_index_dict[word]
                except:
                    word_index = model.embedding_index_dict['<pad>']
                indexed_doc.append(word_index)
            
            padding = [model.embedding_index_dict['<pad>'] for i in range(padding_size)]
            batch_doc_padding = [model.embedding_index_dict['<pad>'] for i in range(batch_doc_size - len(indexed_doc))]
            indexed_doc = padding + indexed_doc + padding + batch_doc_padding

            indexed_text_batch.append(indexed_doc)
        
        text_batches.append(torch.cuda.LongTensor(indexed_text_batch))
    
    return text_batches, label_batches


def train_model(embeddings_file, train_text_file, train_label_file, model_file):
    documents, labels = load_train_files(train_text_file, train_label_file)
    print("time", 1, time.time() - start)


    embeddings_indices, embedding_matrix = load_embeddings_file(embeddings_file)
    print("time", 2, time.time() - start)


    model = CNN(vocab_size=vocabulary_size, 
                embedding_dim=embedding_vector_size, 
                n_filters=num_filters, 
                filter_sizes=n_gram_sizes, 
                output_dim=output_dimension, 
                dropout=dropout_rate,
                embed_index = embeddings_indices
                )
    
    embedding_matrix = torch.cuda.FloatTensor(embedding_matrix)

    model.embedding.weight = nn.Parameter(embedding_matrix)
    print("time", 3, time.time() - start)

    optimizer = optim.Adam(model.parameters())

    criterion = nn.BCEWithLogitsLoss()

    model = model.to(model.device)
    criterion = criterion.to(model.device)

    text_batches, label_batches = get_batches(documents, labels, model)
    print("time", 4, time.time() - start)

    for epoch in range(epochs):
        loss, accuracy = modeltrain(model, text_batches, label_batches, optimizer, criterion)
        print(epoch, loss, accuracy)

        print("time", 5 + epoch, time.time() - start)

    torch.save(model, model_file)
    
    print('Finished...')
        
if __name__ == "__main__":
    # make no changes here
    embeddings_file = sys.argv[1]
    train_text_file = sys.argv[2]
    train_label_file = sys.argv[3]
    model_file = sys.argv[4]
    train_model(embeddings_file, train_text_file, train_label_file, model_file)
    print(time.time() - start)
