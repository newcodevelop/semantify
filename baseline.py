# -*- coding: utf-8 -*-








import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import random
import torch
import numpy as np
import os
from tqdm import tqdm
torch.use_deterministic_algorithms(True)
def set_seed(seed):

    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

from PIL import Image



import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# Simulated batch sizes and vector dimensions
batch_size = 32
vector_dim = 1024

import torch
import torch.nn as nn
import torch.nn.functional as F
class fusion(nn.Module):
    def __init__(self,img_feat_size, txt_feat_size, is_first, K, O, DROPOUT_R):
        super(fusion, self).__init__()
        #self.__C = __C
        self.K = K
        self.O = O
        self.DROPOUT_R = DROPOUT_R

        self.is_first = is_first
        self.proj_i = nn.Linear(img_feat_size, K * O)
        self.proj_t = nn.Linear(txt_feat_size, K * O)

        self.dropout = nn.Dropout(DROPOUT_R)
        self.pool = nn.AvgPool1d(K, stride = K)

    def forward(self, img_feat, txt_feat, exp_in=1):

        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)
        txt_feat = self.proj_t(txt_feat)

        exp_out = img_feat * txt_feat
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)
        z = self.pool(exp_out) * self.K
        z = F.normalize(z.view(batch_size, -1))
        z = z.view(batch_size, -1, self.O)
        return z

import os



import jsonlines

id2text = {}
with jsonlines.open('./data/train.jsonl') as f:
    for line in tqdm(f):

        id2text[str(line['img']).split('/')[1]] = line['text']

import os
import json
import pickle
from tqdm import tqdm
import requests
import pandas as pd

#nlp = spacy.load("en_core_web_sm")

kb_fb = torch.load('./kb_fb.pt')

prefix = './tensors/'
im_tensor = torch.load(prefix+'im_tensor.pt')
im_tensor_ = torch.load(prefix+'im_tensor_.pt')
tx_tensor = torch.load(prefix+'tx_tensor.pt')
tx_tensor_ = torch.load(prefix+'tx_tensor_.pt')

gl = torch.load(prefix+'gl.pt')
gl_ = torch.load(prefix+'gl_.pt')

img_id = torch.load(prefix+'img_id.pt')
img_id_ = torch.load(prefix+'img_id_.pt')

import torch

class TinyModel(torch.nn.Module):

    def __init__(self, mdl, mdl_rand, rand=False):
        super(TinyModel, self).__init__()
        if rand:
            self.linear1 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 512))
        else:
            self.linear1 = mdl_rand
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(512, 128)
        self.linear3 = torch.nn.Linear(128, 2)
        self.rand = rand
        self.softmax = torch.nn.Softmax()
        self.proj = torch.nn.Linear(512,768)
        # self.proj = torch.nn.Linear(512,5120)

    def forward(self, x, y):
        joint_tensor = torch.cat((x, y), dim=1)
        if self.rand:
            m_ = self.linear1(joint_tensor)
        else:
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
            m_ = self.linear1(x, y).squeeze(dim=1)
        m = self.activation(m_)
        m = self.linear3(self.linear2(m))
        # m = self.softmax(m)
        return m, self.proj(m_)

import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, tensor1, tensor2, gold_label, ids):
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.gl = gold_label
        self.ids = ids

    def __len__(self):
        return len(self.tensor1)

    def __getitem__(self, idx):
        return self.tensor1[idx], self.tensor2[idx], self.gl[idx], self.ids[idx]

# def collate_fn(batch):
#     tensor1_batch, tensor2_batch = zip(*batch)
#     return torch.stack(tensor1_batch), torch.stack(tensor2_batch)

# Create your tensors
#N = 100  # Example number of samples
tensor1 = im_tensor
tensor2 = tx_tensor

# Create a custom dataset
custom_dataset = CustomDataset(tensor1, tensor2, gl, img_id)
train_size = int(0.8 * len(custom_dataset))
test_size = len(custom_dataset) - train_size
# torch.manual_seed(42)
set_seed(42)
train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

# Create a DataLoader with your collate_fn
batch_size = 4
import torch
torch.manual_seed(42)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

set_seed(42)
dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=g,
)



# dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)

# Iterate through the DataLoader
counter = 0
for batch_tensor1, batch_tensor2, b3, b4 in dataloader:
    print("Tensor 1 batch shape:", batch_tensor1.shape)
    print("Tensor 2 batch shape:", batch_tensor2.shape)
    print("Tensor 3 batch shape:", b3.shape)
    print("Tensor 4 batch shape:", b4)
    print("-" * 30)
    counter+=1
    if counter==10:
        break

from transformers import GPT2Tokenizer, GPT2LMHeadModel

import gc
gc.collect()
torch.cuda.empty_cache()

def get_tokens(prpmt,begin=False):
    set_seed(42)
    if begin:
        prepended_inp = [tokenizer1.encode(i) for i in prpmt]
    else:
        prepended_inp = [tokenizer1.encode(i) for i in prpmt]
    max_len = max([len(i) for i in prepended_inp])
    #print(max_len)
    attn_mask = []
    bs = len(prpmt)
    for i in range(bs):
        tmp_len = max_len - len(prepended_inp[i])
        tmp_mask = torch.tensor([1]* len(prepended_inp[i]) + [0]* tmp_len)
        attn_mask.append(tmp_mask)
        extra_tokens = tokenizer1.encode(tokenizer1.eos_token)*tmp_len
        prepended_inp[i] = prepended_inp[i]+extra_tokens

    attn_mask = torch.stack(attn_mask)
    #print(prepended_inp)
    #print(attn_mask, attn_mask.shape)
    fin = []
    for i in prepended_inp:
        inter = []
        for j in i:
            inter.append(E[j,:])
        fin.append(torch.stack(inter))






    return torch.stack(fin), torch.tensor(prepended_inp), attn_mask

from sklearn.metrics import *

def get_performance_test():
    torch.use_deterministic_algorithms(mode=True)
    set_seed(42)
    tinymodel.eval()
    model1.eval()
    #set_seed(42)
    all_labs = []
    all_labs_clf = []
    counter = 0
    for ii,ti, gol, ids in test_dataset:

        #     print(0/0)
        ti = ti.unsqueeze(0)
        ii = ii.unsqueeze(0)
        gol = [gol]
        ids = [ids]
        # print(batch_model1, batch_model1)

        ids = list(map(lambda x: x.split('/')[1], ids))
        #         ids_ = []
        #         for i in range(len(ids)):
        #             kbs = ids[i].split('[KB]')
        #             caps = ids[i].split('[CAPTION]')[-1]
        #             try:
        #                 kb = '[KB] '+kbs[1]+'[KB] '+kbs[2] + '[CAPTION] '+ caps
        #             except IndexError as e:
        #                 kb = '[CAPTION] '+ caps
        #             ids_.append(kb)
        #         ids = ids_

        ids_ = []
        for i in range(len(ids)):
            kbs = ids[i].split('[KB]')
            #print('kbs ', kbs)
            caps = ids[i].split('[CAPTION]')[-1]
            #print(caps)
            kb = ''
            if len(kbs)>3:
                kb = '[KB] '+kbs[1]+'[KB] '+kbs[2] + '[CAPTION] '+ caps
            elif len(kbs)==3:
                #print('in len3')
                kb = '[KB] '+kbs[1]
                caps = kbs[2].split('[CAPTION]')
                kb += '[KB] '+ caps[0]+ ' [CAPTION] '+caps[1]
            elif len(kbs)==2:
                #print('in len 1')
                caps = kbs[1].split('[CAPTION]')
                kb += '[KB] '+ caps[0]+ ' [CAPTION] '+caps[1]
            elif len(kbs)==1:
                kb = kbs[0].strip()
                #print('int ', kb)



            ids_.append(kb)

        ids = ids_



        #         print(ids)
        texts = [id2text[i] for i in ids]
        ids = [kb_fb[i] for i in ids]

        #     prpmt = []
        #     for i,j in zip(ids,texts):
        #         prpmt.append(i+'. The meme text reads :'+j + '. Classifier thinks the meme is')



        i = ii.float().to(device)
        t = ti.float().to(device)
        batch_size = i.shape[0]
        with torch.no_grad():
            logits, m = tinymodel(i,t)
        # print(m)
        m = m.unsqueeze(dim=1)
        all_labs_clf.append(logits.argmax(dim=-1)[0].item())
        #         prmpt0, lab0 = get_tokens([tokenizer1.bos_token]*batch_size, begin=True)

        #         print(prmpt0.shape)

        gumbel_logits = F.gumbel_softmax(logits, tau=1, hard=True)
        # print(gumbel_logits)
        agmx = gumbel_logits.argmax(dim=1)

        # print(agmx, logits.argmax(dim=1))
        #         prpmt = ['model thinks the meme is ']*batch_size

        #portion1,lab1, a1 = get_tokens(prpmt)

        portion2,lab3, _ = get_tokens(['output of the classifier multimodal embedding is ']*batch_size)
        lab4 = [tokenizer1.encode('-')[:] for i in range(batch_size)]
        lab4 = torch.tensor(lab4)

        # portion3 = get_tokens(['the meme is actually'])






        kk = []
        labs_verbalized = []
        # batch_prompt = ['the model thinks the meme is ']*32
        lab2 = []
        for i in gumbel_logits:
            agmx = i.argmax()
            # print(agmx)
            # print(dix[agmx.item()])
            tokenized_ = tokenizer1.encode(dix[agmx.item()])[:]
            labs_verbalized.append(dix[agmx.item()])
            # print(tokenized_)
            lab2.append(tokenized_)
            embedding = E[tokenized_[0]]
            one_hot = i.view(-1, 1)
            # print(embedding.shape, one_hot, one_hot.shape)

            e = torch.sum(one_hot*embedding.to(device), dim=0)
            kk.append(e)
        lab2 = torch.tensor(lab2)
        prpmt = []
        for i,j,z in zip(ids,texts,labs_verbalized):
            prpmt.append(i+'. The meme text reads :'+j + '. Classifier thinks the meme is {}'.format(z))
            #all_labs_clf.append(z)
        portion1,lab1, a1 = get_tokens(prpmt)
        string = ['the meme is actually']
        portion3,lab5,_ = get_tokens(string)
        inp_embed = torch.stack(kk).unsqueeze(dim=1)
        # print(inp_embed)
        # print(inp_embed.shape)
        # print(portion1.shape,inp_embed.shape,portion2.shape,m.shape,portion3.shape)

        final_embeds = torch.cat((portion1,inp_embed,portion2,m,portion3),dim=1)
        # print(final_embeds.shape)
        # print(lab1.shape, lab2.shape, lab3.shape, lab4.shape, lab5.shape)
        labs_shape = torch.cat((lab2,lab3,lab4,lab5),dim=1).shape
        remaining_mask = torch.ones(labs_shape[0], labs_shape[1]).long()
        #print(a1)
        #print(remaining_mask)

        full_mask = torch.cat((a1,remaining_mask),dim=1)
        #print('fm ', full_mask.shape)
        labs = torch.cat((lab1,lab2,lab3,lab4,lab5),dim=1).to('cuda')
        #print(full_mask)
        #print(0/0)
        #         print(final_embeds.shape, labs.shape)
        #         print(0/0)
        with torch.no_grad():
            output2 = model1(inputs_embeds = final_embeds.float(), labels = labs.long(), attention_mask=full_mask.to('cuda'))
        # print(output2.loss)
        # print(0/0)
        loss_lm = output2.loss
        #print(tokenizer1.batch_decode(output2.logits.argmax(dim=-1)))

        #print(loss_lm)
        llm_lab = tokenizer1.decode(output2.logits.argmax(dim=-1)[0][-1])
        all_labs.append(llm_lab)
        counter+=1
        #if counter==10:
        #    break

    return all_labs, all_labs_clf

def get_performance_test_nm():
    torch.use_deterministic_algorithms(mode=True)
    set_seed(42)
    tinymodel.eval()
    model1.eval()
    #set_seed(42)
    all_labs = []
    all_labs_clf = []
    counter = 0
    for ii,ti, gol, ids in test_dataset:

        #     print(0/0)
        ti = ti.unsqueeze(0)
        ii = ii.unsqueeze(0)
        gol = [gol]
        ids = [ids]
        # print(batch_model1, batch_model1)

        ids = list(map(lambda x: x.split('/')[1], ids))
        #         print(ids)
        texts = [id2text[i] for i in ids]
        ids = [kb_fb[i] for i in ids]
        #         ids_ = []
        #         for i in range(len(ids)):
        #             kbs = ids[i].split('[KB]')
        #             caps = ids[i].split('[CAPTION]')[-1]
        #             try:
        #                 kb = '[KB] '+kbs[1]+'[KB] '+kbs[2] + '[CAPTION] '+ caps
        #             except IndexError as e:
        #                 kb = '[CAPTION] '+ caps
        #             ids_.append(kb)
        #         ids = ids_


        ids_ = []
        for i in range(len(ids)):
            kbs = ids[i].split('[KB]')
            #print('kbs ', kbs)
            caps = ids[i].split('[CAPTION]')[-1]
            #print(caps)
            kb = ''
            if len(kbs)>3:
                kb = '[KB] '+kbs[1]+'[KB] '+kbs[2] + '[CAPTION] '+ caps
            elif len(kbs)==3:
                #print('in len3')
                kb = '[KB] '+kbs[1]
                caps = kbs[2].split('[CAPTION]')
                kb += '[KB] '+ caps[0]+ ' [CAPTION] '+caps[1]
            elif len(kbs)==2:
                #print('in len 1')
                caps = kbs[1].split('[CAPTION]')
                kb += '[KB] '+ caps[0]+ ' [CAPTION] '+caps[1]
            elif len(kbs)==1:
                kb = kbs[0].strip()
                #print('int ', kb)



            ids_.append(kb)

        ids = ids_

        #     prpmt = []
        #     for i,j in zip(ids,texts):
        #         prpmt.append(i+'. The meme text reads :'+j + '. Classifier thinks the meme is')



        i = ii.float().to(device)
        t = ti.float().to(device)
        batch_size = i.shape[0]
        with torch.no_grad():
            logits, m = tinymodel(i,t)
        # print(m)
        m = m.unsqueeze(dim=1)
        #         prmpt0, lab0 = get_tokens([tokenizer1.bos_token]*batch_size, begin=True)

        #         print(prmpt0.shape)

        gumbel_logits = F.gumbel_softmax(logits, tau=1, hard=True)
        # print(gumbel_logits)
        agmx = gumbel_logits.argmax(dim=1)

        # print(agmx, logits.argmax(dim=1))
        #         prpmt = ['model thinks the meme is ']*batch_size

        #portion1,lab1, a1 = get_tokens(prpmt)


        # portion3 = get_tokens(['the meme is actually'])






        kk = []
        labs_verbalized = []
        # batch_prompt = ['the model thinks the meme is ']*32
        lab2 = []
        for i in gumbel_logits:
            agmx = i.argmax()
            # print(agmx)
            # print(dix[agmx.item()])
            tokenized_ = tokenizer1.encode(dix[agmx.item()])[:]
            labs_verbalized.append(dix[agmx.item()])
            # print(tokenized_)
            lab2.append(tokenized_)
            embedding = E[tokenized_[0]]
            one_hot = i.view(-1, 1)
            # print(embedding.shape, one_hot, one_hot.shape)

            e = torch.sum(one_hot*embedding.to(device), dim=0)
            kk.append(e)
        lab2 = torch.tensor(lab2)
        prpmt = []
        for i,j,z in zip(ids,texts,labs_verbalized):
            prpmt.append(i+'. The meme text reads :'+j)
            all_labs_clf.append(z)
        portion1,lab1, a1 = get_tokens(prpmt)
        string = ['the meme is actually']
        portion3,lab5,_ = get_tokens(string)
        inp_embed = torch.stack(kk).unsqueeze(dim=1)
        # print(inp_embed)
        # print(inp_embed.shape)
        # print(portion1.shape,inp_embed.shape,portion2.shape,m.shape,portion3.shape)

        final_embeds = torch.cat((portion1,portion3),dim=1)
        # print(final_embeds.shape)
        # print(lab1.shape, lab2.shape, lab3.shape, lab4.shape, lab5.shape)
        labs_shape = lab5.shape
        remaining_mask = torch.ones(labs_shape[0], labs_shape[1]).long()
        #print(a1)
        #print(remaining_mask)

        full_mask = torch.cat((a1,remaining_mask),dim=1)
        #print('fm ', full_mask.shape)
        labs = torch.cat((lab1,lab5),dim=1)
        #print(full_mask)
        #print(0/0)
        #         print(final_embeds.shape, labs.shape)
        #         print(0/0)
        with torch.no_grad():
            output2 = model1(inputs_embeds = final_embeds.float(), labels = labs.long(), attention_mask=full_mask.to('cuda'))
        # print(output2.loss)
        # print(0/0)
        loss_lm = output2.loss
        #print(tokenizer1.batch_decode(output2.logits.argmax(dim=-1)))

        #print(loss_lm)
        llm_lab = tokenizer1.decode(output2.logits.argmax(dim=-1)[0][-1])
        all_labs.append(llm_lab)
        counter+=1

    return all_labs, all_labs_clf




from captum.attr import  LayerIntegratedGradients, LayerGradientShap, LayerGradCam, LayerGradientXActivation, InternalInfluence, LayerFeatureAblation, GradientShap, KernelShap, InputXGradient, Saliency, FeatureAblation






gl_test  = []
for _,_,g,_ in test_dataset:
    gl_test.append(g)

def fwd_model(inputs, labels, attention_mask):
    
    #pred = model1(inputs_embeds = inputs, labels = labels, attention_mask=attention_mask)
    pred = model1(input_ids = inputs, labels = labels, attention_mask=attention_mask)
    #print(tokenizer1.decode(pred.logits[:,-1,:].argmax(1)))
    pred = pred.logits[:,-1,:].max(1)
    
    #print(pred.values)
    return pred.values


def fwd_model_others(inputs, labels, attention_mask):
    
    #pred = model1(inputs_embeds = inputs, labels = labels, attention_mask=attention_mask)
    pred = model1(inputs_embeds = inputs, labels = labels, attention_mask=attention_mask)
    #print(tokenizer1.decode(pred.logits[:,-1,:].argmax(1)))
    pred = pred.logits[:,-1,:].max(1)
    
    #print(pred.values)
    return pred.values




def get_performance_test_baseline(mode):
    torch.use_deterministic_algorithms(mode=True)
    set_seed(42)
    tinymodel.eval()
    model1.eval()
    #set_seed(42)
    all_labs = []
    all_labs_clf = []
    d_ids = {}
    counter = 0
    cnnt  = 0
    for ii,ti, gol, ids in tqdm(test_dataset):

        #     print(0/0)
        t_id = ids
        ti = ti.unsqueeze(0)
        ii = ii.unsqueeze(0)
        gol = [gol]
        ids = [ids]
        
        # print(batch_model1, batch_model1)

        ids = list(map(lambda x: x.split('/')[1], ids))
        #         ids_ = []
        #         for i in range(len(ids)):
        #             kbs = ids[i].split('[KB]')
        #             caps = ids[i].split('[CAPTION]')[-1]
        #             try:
        #                 kb = '[KB] '+kbs[1]+'[KB] '+kbs[2] + '[CAPTION] '+ caps
        #             except IndexError as e:
        #                 kb = '[CAPTION] '+ caps
        #             ids_.append(kb)
        #         ids = ids_
        
        ids_ = []
        texts = [id2text[i] for i in ids]
        #ids = [kb_fb[i] for i in ids]
        for i in range(len(ids)):
            kbs = ids[i].split('[KB]')
            # print('kbs ', kbs)
            caps = ids[i].split('[CAPTION]')[-1]
            #print(caps)
            kb = ''
            if len(kbs)>3:
                print('in len >3')
                kb = '[KB] '+kbs[1]+'[KB] '+kbs[2] + '[CAPTION] '+ caps
            elif len(kbs)==3:
                print('in len3')
                kb = '[KB] '+kbs[1]
                caps = kbs[2].split('[CAPTION]')
                kb += '[KB] '+ caps[0]+ ' [CAPTION] '+caps[1]
            elif len(kbs)==2:
                print('in len 1')
                caps = kbs[1].split('[CAPTION]')
                kb += '[KB] '+ caps[0]+ ' [CAPTION] '+caps[1]
            elif len(kbs)==1:
                print('in len1')
                kb = kbs[0].strip()
                #print('int ', kb)



            ids_.append(kb)
        
        ids = ids_
        
        
        
        #print('ids',ids)
        
        
        
        #     prpmt = []
        #     for i,j in zip(ids,texts):
        #         prpmt.append(i+'. The meme text reads :'+j + '. Classifier thinks the meme is')



        i = ii.float().to(device)
        t = ti.float().to(device)
        batch_size = i.shape[0]
        #with torch.no_grad():
        logits, m = tinymodel(i,t)
        # print(m)
        m = m.unsqueeze(dim=1)
        all_labs_clf.append(logits.argmax(dim=-1)[0].item())
        #         prmpt0, lab0 = get_tokens([tokenizer1.bos_token]*batch_size, begin=True)

        #         print(prmpt0.shape)

        gumbel_logits = F.gumbel_softmax(logits, tau=1, hard=True)
        # print(gumbel_logits)
        agmx = gumbel_logits.argmax(dim=1)

        # print(agmx, logits.argmax(dim=1))
        #         prpmt = ['model thinks the meme is ']*batch_size

        #portion1,lab1, a1 = get_tokens(prpmt)

        portion2,lab3, _ = get_tokens(['output of the classifier multimodal embedding is ']*batch_size)
        lab4 = [tokenizer1.encode('-')[:] for i in range(batch_size)]
        lab4 = torch.tensor(lab4)

        # portion3 = get_tokens(['the meme is actually'])






        kk = []
        labs_verbalized = []
        # batch_prompt = ['the model thinks the meme is ']*32
        lab2 = []
        for i in gumbel_logits:
            agmx = i.argmax()
            # print(agmx)
            # print(dix[agmx.item()])
            tokenized_ = tokenizer1.encode(dix[agmx.item()])[:]
            labs_verbalized.append(dix[agmx.item()])
            # print(tokenized_)
            lab2.append(tokenized_)
            embedding = E[tokenized_[0]]
            one_hot = i.view(-1, 1)
            # print(embedding.shape, one_hot, one_hot.shape)

            e = torch.sum(one_hot*embedding.to(device), dim=0)
            kk.append(e)
        lab2 = torch.tensor(lab2)
        prpmt = []
        for i,j,z in zip(ids,texts,labs_verbalized):
            prpmt.append(i+'. The meme text reads :'+j + '. Classifier thinks the meme is {}'.format(z))
            #all_labs_clf.append(z)
        portion1,lab1, a1 = get_tokens(prpmt)
        string = ['the meme is actually']
        portion3,lab5,_ = get_tokens(string)
        inp_embed = torch.stack(kk).unsqueeze(dim=1)
        # print(inp_embed)
        # print(inp_embed.shape)
        # print(portion1.shape,inp_embed.shape,portion2.shape,m.shape,portion3.shape)

        final_embeds = torch.cat((portion1,inp_embed,portion2,m,portion3),dim=1)
        # print(final_embeds.shape)
        # print(lab1.shape, lab2.shape, lab3.shape, lab4.shape, lab5.shape)
        labs_shape = torch.cat((lab2,lab3,lab4,lab5),dim=1).shape
        remaining_mask = torch.ones(labs_shape[0], labs_shape[1]).long()
        #print(a1)
        #print(remaining_mask)

        full_mask = torch.cat((a1,remaining_mask),dim=1)
        #print('fm ', full_mask.shape)
        labs = torch.cat((lab1,lab2,lab3,lab4,lab5),dim=1)
        #print(full_mask)
        #print(0/0)
        #         print(final_embeds.shape, labs.shape)
        #         print(0/0)
        #ig = LayerIntegratedGradients(fwd_model, model1.transformer.wte)   # 1
        #ig = LayerGradientShap(fwd_model, model1.transformer.wte)
        #ig = LayerGradCam(fwd_model, model1.transformer.wte)   
        # ig = LayerGradientXActivation(fwd_model, model1.transformer.wte)
        # ig = InternalInfluence(fwd_model, model1.transformer.wte)
        # ig = LayerFeatureAblation(fwd_model, model1.transformer.wte) #2
        # print('LIG')
        # print(tokenizer1.batch_decode(labs.long()))
        # attributions, delta_start = ig.attribute(inputs=labs.long().to('cuda'),
        #                           additional_forward_args=(labs.long().to('cuda'), full_mask.to('cuda')),
        #                           return_convergence_delta=True)

        if mode=='Integrated Gradient':
            ig = LayerIntegratedGradients(fwd_model, model1.transformer.wte)
            attributions, delta_start = ig.attribute(inputs=labs.long().to('cuda'),
                                   additional_forward_args=(labs.long().to('cuda'), full_mask.to('cuda')),
                                   return_convergence_delta=True)
            attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        elif mode=='KernelShap':
            ig = KernelShap(fwd_model_others)
            attributions = ig.attribute(inputs=final_embeds.to(torch.float32).to('cuda'),
                                    additional_forward_args=(labs.long().to('cuda'), full_mask.to('cuda')), return_input_shape=True)
            print(attributions.shape)
            attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

        elif mode=='InputXGradient':
            ig = InputXGradient(fwd_model_others)
            attributions = ig.attribute(inputs=final_embeds.to(torch.float32).to('cuda'),
                                    additional_forward_args=(labs.long().to('cuda'), full_mask.to('cuda')))
            print(attributions.shape)
            attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

        elif mode=='Saliency':
            ig = Saliency(fwd_model_others)
            attributions = ig.attribute(inputs=final_embeds.to(torch.float32).to('cuda'),
                                    additional_forward_args=(labs.long().to('cuda'), full_mask.to('cuda')))
            print(attributions.shape)
            attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        
        elif mode=='FeatureAblation':
            ig = FeatureAblation(fwd_model_others)
            attributions = ig.attribute(inputs=final_embeds.to(torch.float32).to('cuda'),
                                    additional_forward_args=(labs.long().to('cuda'), full_mask.to('cuda')))
            print(attributions.shape)
            attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

           

        # attributions = ig.attribute(inputs=labs.long().to('cuda'),
        #                           additional_forward_args=(labs.long().to('cuda'), full_mask.to('cuda')))
        
        #print(attributions)
        
      
        # print(attributions.max())
        # print(attributions.min())
        # Normalize the attributions
        attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())
        print(attributions.shape)
        # print(0/0)
        # print(attributions)
        # Map the attributions back to tokens
        tokens = tokenizer1.convert_ids_to_tokens(labs[0].tolist())
        token_importance_dict = {}
        with torch.no_grad():
            pred = model1(input_ids = labs.long().to('cuda'), labels = labs.long().to('cuda'), attention_mask=full_mask.to('cuda'))
        fin_token = tokenizer1.decode(pred.logits[:,-1,:].argmax(1))
        # Display the feature attributions for each token
        # for token, attribution in zip(tokens[1:]+[fin_token], attributions):
        #     token_importance_dict[token] = attribution
        #     # print(f"{token}: {attribution:.3f}")
        for token, attribution in zip(tokens, attributions):
            token_importance_dict[token] = attribution
        
        begin, end = 0,0
        first_time_over = False
        counter = 0
        arr = []
        # for i in token_importance_dict:
        #     if i[0]=='Ġ':
        #         if not first_time_over:
        #             first_time_over=True
        #             begin=counter
                
        #         if first_time_over:
        #             begin=counter
        #             end=counter-1
        #     counter+=1
        #     arr.append((begin,end))

        for i in token_importance_dict:
            if i[0]=='Ġ':
                arr.append(counter)
            counter+=1
        


       
        fin_arr = []
        for i,j in zip(arr[:-1], arr[1:]):
           
            fin_arr.append((i,j))
        
        tok_imp = []

        for i in token_importance_dict:
            tok_imp.append((i, token_importance_dict[i]))
        
        word_attributions = []
        for i in fin_arr:
            start, end = i[0], i[1]
            word_attributions.append((tok_imp[start:end]))
        
        word_importance_dict = {}
        prohibited_words = ['meme', '-', 'The', 'the', 'reads', 'Classifier', 'thinks', 'is', 'reads', 'multimodal', 'embedding', 'normal', ':-', 'class', '-the', 'and', 'you', 'text', 'for', 'shape']
        for i in word_attributions:
            k = ''.join(list(zip(*i))[0])
            l = list(zip(*i))[1]
            if k[1:] not in prohibited_words and len(k[1:])>2:
                if k[1:]=='offensiveoffensiveoutput':
                    word_importance_dict['offensive'] = np.mean(l)
                elif k[1:]=='normalnormaloutput':
                    word_importance_dict['normal'] = np.mean(l)
                else:
                    word_importance_dict[k[1:]] = np.mean(l)
        
       

        sorted_tokens = sorted(word_importance_dict.items(), key=lambda x: x[1], reverse=True)

        # Select the top 4 tokens
        top_4_tokens = sorted_tokens[:4]

       

        toks = []
        for i in top_4_tokens:
            toks.append(i[0])
        
  
        string = ''
        for i in toks:
            string+=i+'\t'

        #print(string)
        #print(string.split('\t'))

        d_ids[t_id] = string
        all_labs.append(fin_token.strip())

        cnnt+=1
        print(cnnt)
       
        
            
    return d_ids, all_labs
        
                
       








import torch
#os.environ['CUBLAS_WORKSPACE_CONFIG']=:4096:8
torch.use_deterministic_algorithms(True)
set_seed(42)







import numpy as np
model1_name = 'gpt2'
tokenizer1 = GPT2Tokenizer.from_pretrained(model1_name)
model1 = GPT2LMHeadModel.from_pretrained(model1_name,output_hidden_states=True).to(device)
mdl = fusion(512,512,True,256,512,0.1).to(device)
mdl_rand = fusion(512,512,True,256,512,0.1).to(device)
tinymodel = TinyModel(mdl,mdl_rand,rand=False).to(device)

model1.load_state_dict(torch.load('./gpt2_backbone.pt'), strict=False)
tinymodel.load_state_dict(torch.load('./classifier (1).pt'), strict=False)



# class ToyModel(nn.Module):
#     def __init__(self, model1):
#         super().__init__()
#         self.model1 = model1
       

#     def forward(self, input):
#         output2 = model1(inputs_embeds = input[0].float(), labels = input[1].long(), attention_mask=input[2].to('cuda'))
#         llm_lab = tokenizer1.decode(output2.logits.argmax(dim=-1)[0][-1])
#         tmp = 0 if llm_lab=='normal' else 1
#         return tmp


# model1 = ToyModel(model1)
# model1.eval()




E = model1.transformer.wte.weight.detach()
model1.eval()
tinymodel.eval()
dix = {0:'normal', 1:'offensive'}
set_seed(42)
d_ids, al = get_performance_test_baseline(mode='FeatureAblation')





all_labs = []
for i in al:
    if i.strip()=='normal':
        all_labs.append(0)
    else:
        all_labs.append(1)

import pickle

list_of_dicts = [d_ids,all_labs]

print(list_of_dicts)


with open('iclr_baseline_fa.pkl', 'wb') as f:
    pickle.dump(list_of_dicts, f)

