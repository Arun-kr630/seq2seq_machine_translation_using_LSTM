from dataset import get_dataset,validation
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

class Config:
    def __init__(self,src_vacab_size,tgt_vacab_size):
        self.embed_size=768
        self.src_vocab_size=src_vacab_size
        self.tgt_vocab_size=tgt_vacab_size
        self.hidden_size=1024



class Encoder(nn.Module):
    def __init__(self,config:Config):
        super().__init__()
        self.embeds=nn.Embedding(config.src_vocab_size,config.embed_size)
        self.lstm=nn.LSTM(input_size=config.embed_size,hidden_size=config.hidden_size,batch_first=True)

    def forward(self,x):
        x=self.embeds(x)
        output,(h_n,c_n)=self.lstm(x)

        return h_n,c_n
    
class Decoder(nn.Module):
    def __init__(self,config:Config):
        super().__init__()
        self.embeds=nn.Embedding(config.tgt_vocab_size,config.embed_size)
        self.lstm=nn.LSTM(input_size=config.embed_size,hidden_size=config.hidden_size,batch_first=True)
        self.linear=nn.Linear(config.hidden_size,config.tgt_vocab_size)

    def forward(self,x,h_0,c_0):
        x=self.embeds(x)
        output,(h_n,c_n)=self.lstm(x,(h_0,c_0))
        return self.linear(output)
    

class seq2seq(nn.Module):
    def __init__(self,config:Config):
        super().__init__()
        self.encoder=Encoder(config)
        self.decoder=Decoder(config)
    def forward(self,encoder_input,decoder_input):
        h_0,c_0=self.encoder(encoder_input)
        output=self.decoder(decoder_input,h_0,c_0)
        return output
    

num_epoch=50
lr=3e-4
device='cuda:2' if torch.cuda.is_available() else 'cpu'
print(f"using device {device}")
train_loader,val_loader,src_vacab_size,tgt_vocab_size,src_tokenizer,tgt_tokenizer=get_dataset("opus_books","en","it")

config=Config(src_vacab_size=src_vacab_size,tgt_vacab_size=tgt_vocab_size)

model=seq2seq(config).to(device)
criterian=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr)

for epoch in range(num_epoch):
    model.train()   
    loop=tqdm(train_loader,leave=False,total=len(train_loader),desc=f"Processing Epoch {epoch:02d}")

    for data in loop:
        encoder_input=data['encoder_input'].to(device)
        decoder_input=data['decoder_input'].to(device)
        label=data['label'].to(device)
        label=label.unsqueeze(-1)
        y_pred=model(encoder_input,decoder_input)
        y_pred=y_pred.view(-1,y_pred.size(2))
        label=label.view(-1)
        loss=criterian(y_pred,label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    validation(model,val_loader,device,src_tokenizer,tgt_tokenizer,epoch)
