from mydatasets import get_dataloader,ChEBIdataset
import torch
import transformers
from mytokenizers import SimpleSmilesTokenizer,regexTokenizer
from transformers import AutoModel
from transformers import AutoTokenizer
import argparse
import pdb

import os
os.environ['CUDA_DEVICES_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='5'

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--dataset", default="mmp", required=True)
args = parser.parse_args()
split = args.input
dataset = args.dataset
smtokenizer = regexTokenizer(max_len=128)


train_dataset = ChEBIdataset(
        dir=f'../../datasets/{dataset}/',
        smi_tokenizer=smtokenizer,
        split=split,
        replace_desc=False,
        load_state=False
        # pre = pre
    )
model = AutoModel.from_pretrained('../../scibert')
tokz = AutoTokenizer.from_pretrained('../../scibert')

volume = {}


model = model.cuda()
    # alllen = []
model.eval()
with torch.no_grad():
    for i in range(len(train_dataset)):
        if i%190 == 0:
            print(i)
        id = train_dataset[i]['cid']
        desc = train_dataset[i]['desc']
        # pdb.set_trace()
        tok_op = tokz(
            desc,max_length=512, truncation=True,padding='max_length'
            )
        toked_desc = torch.tensor(tok_op['input_ids']).unsqueeze(0)
        toked_desc_attentionmask = torch.tensor(tok_op['attention_mask']).unsqueeze(0)
        assert(toked_desc.shape[1]==512)
        # pdb.set_trace()
        lh = model(toked_desc.cuda()).last_hidden_state
        volume[id] = {'states':lh.to('cpu'),'mask':toked_desc_attentionmask}



torch.save(volume,f'../../datasets/{dataset}/{split}_desc_states.pt')