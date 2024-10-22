import torch

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from pathlib import Path

from torch.utils.data import Dataset,DataLoader,random_split
def get_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]
def get_tokenizer(ds,lang):
    tokenizer_path=Path(f'tokenizer_{lang}.json')
    if Path.exists(tokenizer_path):
        tokenizer=Tokenizer.from_file(str(tokenizer_path))
    else:
        tokenizer=Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer=Whitespace()
        trainer=BpeTrainer(special_tokens=["[UNK]","[EOS]","[SOS]","[PAD]"],min_frequency=2)
        tokenizer.train_from_iterator(get_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    return tokenizer
class BilingualDataset(Dataset):
    def __init__(self,ds,src_lang,tgt_lang,src_tokenizer,tgt_tokenier):
        self.ds=ds
        self.src_lang=src_lang
        self.tgt_lang=tgt_lang
        self.src_tokenizer=src_tokenizer
        self.tgt_tokenizer=tgt_tokenier

        self.sos_token=torch.tensor([src_tokenizer.token_to_id("[SOS]")],dtype=torch.int64)
        self.eos_token=torch.tensor([src_tokenizer.token_to_id("[EOS]")],dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self,index):
        data_pair=self.ds[index]
        src_sentence=data_pair['translation'][self.src_lang]
        tgt_sentence=data_pair['translation'][self.tgt_lang]

        src_tokens=self.src_tokenizer.encode(src_sentence).ids
        tgt_tokens=self.tgt_tokenizer.encode(tgt_sentence).ids

        encoder_input=torch.cat(
            [self.sos_token,torch.tensor(src_tokens,dtype=torch.int64),self.eos_token],dim=-1
        )
        decoder_input=torch.cat([
            self.sos_token,torch.tensor(tgt_tokens,dtype=torch.int64)
        ],dim=-1)
        label=torch.cat([
            torch.tensor(tgt_tokens,dtype=torch.int64),self.eos_token
        ],dim=-1)

        return {"encoder_input":encoder_input,'decoder_input':decoder_input,'label':label}


def get_dataset(source,src_lang,tgt_lang):
    ds=load_dataset(source,f"{src_lang}-{tgt_lang}",split="train")
    src_tokenizer=get_tokenizer(ds,src_lang)
    tgt_tokenizer=get_tokenizer(ds,tgt_lang)


    train_data_size=int(0.9 * len(ds))
    train_raw_data,val_raw_data=random_split(ds,[train_data_size,len(ds)-train_data_size])

    train_dataset=BilingualDataset(train_raw_data,src_lang,tgt_lang,src_tokenizer,tgt_tokenizer)
    val_dataset=BilingualDataset(val_raw_data,src_lang,tgt_lang,src_tokenizer,tgt_tokenizer)


    train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
    val_laoder=DataLoader(val_dataset,batch_size=1,shuffle=True)

    src_vocab_size=src_tokenizer.get_vocab_size()
    tgt_vacab_size=tgt_tokenizer.get_vocab_size()

    return train_loader,val_laoder,src_vocab_size,tgt_vacab_size,src_tokenizer,tgt_tokenizer





def validation(model,test_loader,device,src_tokenizer,tgt_tokenizer,epoch):
    model.eval()
    with torch.no_grad():
        first_batch = next(iter(test_loader))
        data = first_batch
        encoder_input=data['encoder_input'].to(device)
        decoder_input=data['decoder_input'].to(device)
        y_pred=model(encoder_input,decoder_input)
        _,predicted=torch.max(y_pred,dim=-1)
        predicted=predicted.view(-1).tolist()
        encoder_input=encoder_input.view(-1).tolist()
        decoder_input=decoder_input.view(-1).tolist()
        src_sentence= src_tokenizer.decode(encoder_input, skip_special_tokens=True)
        tgt_sentence= tgt_tokenizer.decode(decoder_input, skip_special_tokens=True)
        predicted_sentence= tgt_tokenizer.decode(predicted, skip_special_tokens=True)
        print("--------------------------------------------------------------------------------------------------------------------------")
        print(f'epoch {epoch}')
        print("src -->",src_sentence)
        print("tgt -->",tgt_sentence)
        print("predicted -->",predicted_sentence)
