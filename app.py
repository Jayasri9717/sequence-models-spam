import json,time,re
from pathlib import Path
import torch
from torch import nn
torch.set_num_threads(1)
from fastapi import FastAPI
from pydantic import BaseModel,Field
ROOT=Path(__file__).parent; ART=ROOT/'artifacts/best_transformer'
class Net(nn.Module):
 def __init__(self,n):
  super().__init__();self.emb=nn.Embedding(n,64,padding_idx=0);self.pos=nn.Embedding(40,64);self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(64,4,128,.1,batch_first=True),2);self.head=nn.Linear(64,1)
 def forward(self,x):
  mask=x.ne(0);h=self.enc(self.emb(x)+self.pos(torch.arange(40)[None,:]),src_key_padding_mask=~mask);h=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1).clamp(min=1).unsqueeze(-1);return self.head(h).squeeze(-1)
meta=json.load(open(ART/'meta.json'));vocab=json.load(open(ART/'vocab.json'));model=Net(len(vocab));model.load_state_dict(torch.load(ART/'model.pt',map_location='cpu',weights_only=True));model.eval()
app=FastAPI(title='SMS spam classifier');stats={'requests':0,'spam':0,'latency_sum':0.0}
class Request(BaseModel): text:str=Field(min_length=1,max_length=1000)
def predict(text):
 ids=[vocab.get(w,1) for w in re.findall(r'[a-z0-9]+',text.lower())][:40];ids += [0]*(40-len(ids))
 with torch.no_grad():return float(torch.sigmoid(model(torch.tensor([ids])))[0])
@app.get('/health')
def health():return {'status':'ok','model':meta['model']}
@app.post('/predict')
def classify(req:Request):
 start=time.perf_counter();p=predict(req.text);lat=(time.perf_counter()-start)*1000;label='spam' if p>=.5 else 'ham';stats['requests']+=1;stats['spam']+=int(label=='spam');stats['latency_sum']+=lat;print(json.dumps({'spam_probability':p,'latency_ms':lat}));return {'spam_probability':p,'label':label,'latency_ms':lat}
@app.get('/metrics')
def metrics():
 n=stats['requests'];return {'request_count':n,'spam_rate':stats['spam']/n if n else 0.0,'average_latency_ms':stats['latency_sum']/n if n else 0.0}

