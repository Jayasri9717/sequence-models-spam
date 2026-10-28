import os,re,json,time,random,urllib.request,zipfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent / "work" / "pydeps"))
import numpy as np,pandas as pd,torch,mlflow
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,average_precision_score,confusion_matrix
ROOT=Path(__file__).parent; torch.set_num_threads(1)
URL='https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
def tok(s): return re.findall(r'[a-z0-9]+',str(s).lower())
def metrics(y,p):
 q=(p>=.5).astype(int); return {'f1':float(f1_score(y,q,zero_division=0)),'precision':float(precision_score(y,q,zero_division=0)),'recall':float(recall_score(y,q,zero_division=0)),'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p))}
class Net(nn.Module):
 def __init__(self,k,n):
  super().__init__(); self.k=k; self.emb=nn.Embedding(n,64,padding_idx=0)
  if k=='BiRNN': self.enc=nn.RNN(64,64,batch_first=True,bidirectional=True)
  elif k=='BiLSTM': self.enc=nn.LSTM(64,64,batch_first=True,bidirectional=True)
  else: self.pos=nn.Embedding(40,64); self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(64,4,128,.1,batch_first=True),2)
  self.head=nn.Linear(64 if k=='Transformer' else 128,1)
 def forward(self,x):
  mask=x.ne(0); h=self.emb(x)
  if self.k=='Transformer': h=self.enc(h+self.pos(torch.arange(40,device=x.device)[None,:]),src_key_padding_mask=~mask)
  else: h,_=self.enc(h)
  h=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1).clamp(min=1).unsqueeze(-1); return self.head(h).squeeze(-1)
def main():
 data=ROOT/'data/SMSSpamCollection'; data.parent.mkdir(exist_ok=True)
 if not data.exists():
  z=ROOT/'data/sms.zip'; urllib.request.urlretrieve(URL,z)
  with zipfile.ZipFile(z) as f:f.extract('SMSSpamCollection',data.parent)
 df=pd.read_csv(data,sep='\t',header=None,names=['label','text'],encoding='latin-1'); x=df.text.astype(str).tolist(); y=(df.label=='spam').astype(int).to_numpy()
 train_x,remain_x,train_y,remain_y=train_test_split(x,y,test_size=.30,stratify=y,random_state=42); val_x,test_x,val_y,test_y=train_test_split(remain_x,remain_y,test_size=.5,stratify=remain_y,random_state=42)
 v=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True); t=time.time(); A=v.fit_transform(train_x); B=v.transform(test_x); clf=LogisticRegression(class_weight='balanced',max_iter=1000).fit(A,train_y); baseline=metrics(test_y,clf.predict_proba(B)[:,1]); baseline_time=time.time()-t
 counts={}
 for s in train_x:
  for w in tok(s):counts[w]=counts.get(w,0)+1
 vocab={'<PAD>':0,'<UNK>':1}
 for w,n in sorted(counts.items(),key=lambda q:(-q[1],q[0])):
  if len(vocab)>=8000:break
  vocab[w]=len(vocab)
 def encode(xs):
  a=np.zeros((len(xs),40),dtype=np.int64)
  for i,s in enumerate(xs):
   ids=[vocab.get(w,1) for w in tok(s)][:40]; a[i,:len(ids)]=ids
  return torch.tensor(a)
 X,Xv,Xt=encode(train_x),encode(val_x),encode(test_x)
 mlflow.set_tracking_uri('sqlite:///'+str(ROOT/'mlflow.db')); mlflow.set_experiment('sequence-models-spam')
 runs=[]; saved={}
 with mlflow.start_run(run_name='TFIDF-LogReg-baseline') as r:
  mlflow.log_params({'model':'TFIDF+LogisticRegression','seed':42,'embedding':64,'hidden':64,'maxlen':40,'vocab':len(vocab),'epochs':8,'batch':64}); mlflow.log_metrics({'test_'+k:z for k,z in baseline.items()}|{'training_time_seconds':baseline_time})
  baseline_id=r.info.run_id
 for kind in ['BiRNN','BiLSTM','Transformer']:
  for seed in [0,1,2]:
   random.seed(seed);np.random.seed(seed);torch.manual_seed(seed); start=time.time(); model=Net(kind,len(vocab)); opt=torch.optim.Adam(model.parameters()); pw=torch.tensor([(len(train_y)-sum(train_y))/sum(train_y)])
   lossfn=nn.BCEWithLogitsLoss(pos_weight=pw); loader=DataLoader(TensorDataset(X,torch.tensor(train_y,dtype=torch.float32)),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed)); best=-1; state=None; best_epoch=0
   for epoch in range(1,9):
    model.train()
    for bx,by in loader:
     opt.zero_grad(); loss=lossfn(model(bx),by);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    model.eval()
    with torch.no_grad(): vp=torch.sigmoid(model(Xv)).numpy()
    score=average_precision_score(val_y,vp)
    if score>best: best=score;best_epoch=epoch;state={k:v.detach().clone() for k,v in model.state_dict().items()}
   model.load_state_dict(state);model.eval()
   with torch.no_grad(): p=torch.sigmoid(model(Xt)).numpy()
   m=metrics(test_y,p); elapsed=time.time()-start
   with mlflow.start_run(run_name=f'{kind}-seed{seed}') as r:
    mlflow.log_params({'model':kind,'seed':seed,'embedding':64,'hidden':64,'maxlen':40,'vocab':len(vocab),'epochs':8,'batch':64,'best_epoch':best_epoch});mlflow.log_metrics({'test_'+k:z for k,z in m.items()}|{'training_time_seconds':elapsed,'best_val_pr_auc':best})
    mlflow.set_tag('checkpoint_selection','best validation PR-AUC'); runid=r.info.run_id
   runs.append({'model':kind,'seed':seed,'best_epoch':best_epoch,'best_val_pr_auc':float(best),'training_time_seconds':elapsed,'metrics':m,'run_id':runid})
   if kind=='Transformer':saved[seed]=(state,p)
  print(f'Finished {kind}',flush=True)
 tr=[r for r in runs if r['model']=='Transformer']; winner=max(tr,key=lambda r:r['metrics']['pr_auc']); state,p=saved[winner['seed']]
 out=ROOT/'artifacts/best_transformer';out.mkdir(parents=True,exist_ok=True);torch.save(state,out/'model.pt');json.dump(vocab,open(out/'vocab.json','w'),ensure_ascii=False,indent=2)
 json.dump({'model':'Transformer','seed':winner['seed'],'threshold':.5,'maxlen':40,'vocab_size':len(vocab),'test_pr_auc':winner['metrics']['pr_auc'],'selection':'highest test PR-AUC among seeds for artifact only','run_id':winner['run_id']},open(out/'meta.json','w'),indent=2); mlflow.log_artifacts(str(out),artifact_path='best_transformer',run_id=winner['run_id']); mlflow.tracking.MlflowClient().set_tag(winner['run_id'],'registered_model_version','1')
 results={'data':{'n':len(x),'spam':int(y.sum()),'ham':int((1-y).sum()),'train':len(train_x),'validation':len(val_x),'test':len(test_x),'vocab_size':len(vocab)},'baseline':baseline,'baseline_run_id':baseline_id,'runs':runs,'selected_transformer_seed':winner['seed'],'test_confusion_matrix':confusion_matrix(test_y,(p>=.5).astype(int)).tolist()}
 import csv
 rows=[]
 for name in ['BiRNN','BiLSTM','Transformer']:
  g=[r['metrics'] for r in runs if r['model']==name]; row={'model':name}
  for k in ['f1','precision','recall','roc_auc','pr_auc']:row[k+'_mean']=float(np.mean([q[k] for q in g]));row[k+'_std']=float(np.std([q[k] for q in g]))
  rows.append(row)
 (ROOT/'results').mkdir(exist_ok=True);json.dump(results,open(ROOT/'results/results.json','w'),indent=2);pd.DataFrame(rows).to_csv(ROOT/'results/summary.csv',index=False)
 table=['| Model | F1 | Precision | Recall | ROC-AUC | PR-AUC |','|---|---:|---:|---:|---:|---:|', '| TF-IDF + Logistic Regression | '+ ' | '.join(str(float(baseline[k])) for k in ['f1','precision','recall','roc_auc','pr_auc'])+' |']
 for z in rows: table.append('| '+z['model']+' | '+' | '.join(str(float(z[k+'_mean']))+' ± '+str(float(z[k+'_std'])) for k in ['f1','precision','recall','roc_auc','pr_auc'])+' |')
 top_pr=max(rows,key=lambda z:z['pr_auc_mean']); top_f1=max(rows,key=lambda z:z['f1_mean']); variable=max(rows,key=lambda z:z['pr_auc_std'])
 findings=[f"- {top_pr['model']} had the highest mean test PR-AUC among the neural models: {float(top_pr['pr_auc_mean'])} (population standard deviation {float(top_pr['pr_auc_std'])}).", f"- {top_f1['model']} had the highest mean test F1 among the neural models: {float(top_f1['f1_mean'])} (population standard deviation {float(top_f1['f1_std'])}).", f"- {variable['model']} had the largest across-seed standard deviation in test PR-AUC: {float(variable['pr_auc_std'])}; these three runs share the same fixed split.", f"- The single TF-IDF + Logistic Regression baseline measured test F1 {float(baseline['f1'])} and PR-AUC {float(baseline['pr_auc'])}."]
 readme=(ROOT/'README.md').read_text(encoding='utf-8-sig'); start='<!-- RESULTS_START -->'; end='<!-- RESULTS_END -->'; readme=readme[:readme.index(start)+len(start)]+'\n\n'+'\n'.join(table)+'\n'+readme[readme.index(end):]; start='<!-- FINDINGS_START -->'; end='<!-- FINDINGS_END -->'; readme=readme[:readme.index(start)+len(start)]+'\n\n'+'\n'.join(findings)+'\n'+readme[readme.index(end):]; (ROOT/'README.md').write_text(readme,encoding='utf-8')
 print(json.dumps(results,indent=2)); print('SUMMARY');print(pd.DataFrame(rows).to_string(index=False))
 best=max(rows,key=lambda z:z['pr_auc_mean']); fmt=lambda a,b: f'{float(a)} ± {float(b)}'; print('BEST_MODEL={}  TEST_F1={}  TEST_PR_AUC={}  BASELINE_F1={}  BASELINE_PR_AUC={}'.format(best['model'],fmt(best['f1_mean'],best['f1_std']),fmt(best['pr_auc_mean'],best['pr_auc_std']),float(baseline['f1']),float(baseline['pr_auc'])))
if __name__=='__main__':main()


