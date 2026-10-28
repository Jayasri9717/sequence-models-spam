"""Small reproduction of scaled dot-product multi-head self-attention."""
import json, math, random, re, sys, time, urllib.request, zipfile
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'work'/'pydeps'))
import numpy as np, pandas as pd, torch, mlflow
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

torch.set_num_threads(1)
URL='https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
def tokenize(s): return re.findall(r'[a-z0-9]+',str(s).lower())
def metrics(y,p):
 pred=(p>=.5).astype(int)
 return {'f1':float(f1_score(y,pred,zero_division=0)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p))}

class ManualSelfAttention(nn.Module):
 """Q/K/V projections and scaled dot-product attention written with matmul."""
 def __init__(self,d_model=64,heads=4,dropout=.1):
  super().__init__(); self.heads=heads; self.head_dim=d_model//heads
  self.qkv=nn.Linear(d_model,3*d_model); self.output=nn.Linear(d_model,d_model); self.attn_dropout=nn.Dropout(dropout)
 def forward(self,x,pad_mask):
  b,length,width=x.shape
  q,k,v=self.qkv(x).reshape(b,length,3,self.heads,self.head_dim).permute(2,0,3,1,4).unbind(0)
  scores=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim)
  safe_pad=pad_mask.clone(); empty=safe_pad.all(dim=1); safe_pad[empty,0]=False
  scores=scores.masked_fill(safe_pad[:,None,None,:],float('-inf'))
  weights=self.attn_dropout(torch.softmax(scores,dim=-1))
  context=(weights@v).transpose(1,2).contiguous().reshape(b,length,width)
  return self.output(context)
class ManualEncoderLayer(nn.Module):
 def __init__(self):
  super().__init__();self.attn=ManualSelfAttention();self.drop1=nn.Dropout(.1);self.norm1=nn.LayerNorm(64)
  self.ff1=nn.Linear(64,128);self.ffdrop=nn.Dropout(.1);self.ff2=nn.Linear(128,64);self.drop2=nn.Dropout(.1);self.norm2=nn.LayerNorm(64)
 def forward(self,x,pad_mask):
  x=self.norm1(x+self.drop1(self.attn(x,pad_mask)))
  ff=self.ff2(self.ffdrop(torch.relu(self.ff1(x))))
  return self.norm2(x+self.drop2(ff))
class ManualTransformer(nn.Module):
 def __init__(self,vocab_size):
  super().__init__();self.embedding=nn.Embedding(vocab_size,64,padding_idx=0);self.position=nn.Embedding(40,64)
  self.layers=nn.ModuleList([ManualEncoderLayer(),ManualEncoderLayer()]);self.head=nn.Linear(64,1)
 def forward(self,ids):
  mask=ids.eq(0);x=self.embedding(ids)+self.position(torch.arange(40,device=ids.device)[None,:])
  for layer in self.layers:x=layer(x,mask)
  pooled=(x*~mask.unsqueeze(-1)).sum(1)/(~mask).sum(1).clamp(min=1).unsqueeze(-1)
  return self.head(pooled).squeeze(-1)
def main():
 data=ROOT/'data/SMSSpamCollection';data.parent.mkdir(exist_ok=True)
 if not data.exists():
  z=ROOT/'data/sms.zip';urllib.request.urlretrieve(URL,z)
  with zipfile.ZipFile(z) as f:f.extract('SMSSpamCollection',data.parent)
 frame=pd.read_csv(data,sep='\t',header=None,names=['label','text'],encoding='latin-1');texts=frame.text.astype(str).tolist();labels=(frame.label=='spam').astype(int).to_numpy()
 train_x,remaining_x,train_y,remaining_y=train_test_split(texts,labels,test_size=.30,stratify=labels,random_state=42)
 val_x,test_x,val_y,test_y=train_test_split(remaining_x,remaining_y,test_size=.5,stratify=remaining_y,random_state=42)
 counts={}
 for text in train_x:
  for word in tokenize(text):counts[word]=counts.get(word,0)+1
 vocab={'<PAD>':0,'<UNK>':1}
 for word,count in sorted(counts.items(),key=lambda x:(-x[1],x[0])):
  if len(vocab)>=8000:break
  vocab[word]=len(vocab)
 def encode(items):
  arr=np.zeros((len(items),40),dtype=np.int64)
  for i,text in enumerate(items):
   ids=[vocab.get(w,1) for w in tokenize(text)][:40];arr[i,:len(ids)]=ids
  return torch.tensor(arr)
 X,Xv,Xt=encode(train_x),encode(val_x),encode(test_x)
 mlflow.set_tracking_uri('sqlite:///'+str(ROOT/'mlflow.db'));mlflow.set_experiment('sequence-models-spam')
 runs=[]
 for seed in [0,1,2]:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);started=time.time();model=ManualTransformer(len(vocab));opt=torch.optim.Adam(model.parameters())
  pos_weight=torch.tensor([(len(train_y)-sum(train_y))/sum(train_y)]);lossfn=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  loader=DataLoader(TensorDataset(X,torch.tensor(train_y,dtype=torch.float32)),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed));best=-1.;best_state=None;best_epoch=0
  for epoch in range(1,9):
   model.train()
   for bx,by in loader:
    opt.zero_grad();loss=lossfn(model(bx),by);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
   model.eval()
   with torch.no_grad():vp=torch.sigmoid(model(Xv)).numpy()
   score=average_precision_score(val_y,vp)
   if score>best:best=float(score);best_epoch=epoch;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
  model.load_state_dict(best_state);model.eval()
  with torch.no_grad():p=torch.sigmoid(model(Xt)).numpy()
  result=metrics(test_y,p);elapsed=time.time()-started
  with mlflow.start_run(run_name=f'ManualSelfAttention-seed{seed}') as run:
   mlflow.log_params({'model':'ManualSelfAttention','seed':seed,'embedding':64,'hidden':64,'heads':4,'layers':2,'feed_forward':128,'dropout':.1,'maxlen':40,'vocab':len(vocab),'epochs':8,'batch':64,'best_epoch':best_epoch})
   mlflow.log_metrics({'test_'+k:v for k,v in result.items()}|{'best_val_pr_auc':best,'training_time_seconds':elapsed})
   mlflow.set_tag('paper_reproduction','Vaswani et al. 2017, scaled dot-product multi-head attention')
   run_id=run.info.run_id
  runs.append({'seed':seed,'best_epoch':best_epoch,'best_val_pr_auc':best,'training_time_seconds':elapsed,'metrics':result,'run_id':run_id})
  print(f'Finished manual attention seed {seed}',flush=True)
 native=json.load(open(ROOT/'results/results.json',encoding='utf-8'));native_runs=[r for r in native['runs'] if r['model']=='Transformer']
 summary={}
 for key in ['f1','precision','recall','roc_auc','pr_auc']:
  vals=[r['metrics'][key] for r in runs];summary[key+'_mean']=float(np.mean(vals));summary[key+'_std']=float(np.std(vals))
  nvals=[r['metrics'][key] for r in native_runs];summary['nn_'+key+'_mean']=float(np.mean(nvals));summary['nn_'+key+'_std']=float(np.std(nvals))
 summary['manual_minus_nn_pr_auc_mean']=summary['pr_auc_mean']-summary['nn_pr_auc_mean']
 output={'paper':'Vaswani et al. (2017), Attention Is All You Need, arXiv:1706.03762','runs':runs,'summary':summary,'matched':'64-dimensional token embeddings, learned positions, two encoder layers, four heads, 128-dimensional feed-forward network, dropout 0.1, key-padding mask, masked mean pooling, same split, seeds, optimizer, loss, epoch selection, and test metrics.','differences':'Q/K/V projections, head splitting, score scaling, key mask, softmax, and weighted value aggregation are explicit matmul code. The native PyTorch encoder uses its built-in attention implementation and may use fused kernels. Parameter initialization and random-number consumption differ, so numerical outputs are not expected to match exactly.'}
 (ROOT/'results').mkdir(exist_ok=True);json.dump(output,open(ROOT/'results/reproduction.json','w'),indent=2)
 mean=lambda key:f"{summary[key+'_mean']} ± {summary[key+'_std']}"
 section='## Reproduction: hand-written scaled dot-product attention\n\nThe attention block follows the scaled dot-product multi-head attention idea in [Vaswani et al. (2017), *Attention Is All You Need*](https://arxiv.org/abs/1706.03762). This implementation computes Q, K, and V with linear projections, reshapes into four heads, scales QKᵀ by √16, masks padded keys, applies softmax, and combines the values. Two encoder blocks keep the remaining architecture aligned with `nn.TransformerEncoder`.\n\n| Model | Test F1 (mean ± population std) | Test PR-AUC (mean ± population std) |\n|---|---:|---:|\n| Hand-written attention | '+mean('f1')+' | '+mean('pr_auc')+' |\n| `nn.TransformerEncoder` | '+mean('nn_f1')+' | '+mean('nn_pr_auc')+' |\n\n**Matched:** embedding width 64, two layers, four heads, feed-forward width 128, dropout 0.1, learned positional embeddings, padding mask, masked mean pooling, same train/validation/test split, seeds 0–2, Adam/BCE training, validation PR-AUC checkpoint selection, and test metrics at 0.5.\n\n**Different:** Q/K/V projection, head splitting, score scaling, key masking, softmax, and value aggregation are written with explicit tensor operations. PyTorch may use fused attention kernels, and parameter initialization/random-number consumption differs, so weights and outputs are not numerically identical. The measured test metrics above compare task performance, not layer-by-layer numerical equivalence.\n\nManual run IDs and all full-precision metrics are in `results/reproduction.json`; each seed is also tracked in MLflow.\n'
 readme=(ROOT/'README.md').read_text(encoding='utf-8');marker='## Limitations'
 if '## Reproduction: hand-written scaled dot-product attention' in readme:readme=readme[:readme.index('## Reproduction: hand-written scaled dot-product attention')]+readme[readme.index(marker):]
 readme=readme.replace(marker,section+'\n'+marker);(ROOT/'README.md').write_text(readme,encoding='utf-8')
 print(json.dumps(output,indent=2));print('REPRODUCTION_COMPARISON');print(section.encode('ascii','backslashreplace').decode('ascii'))
if __name__=='__main__':main()

