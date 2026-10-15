"""Recompute summary statistics from raw outputs; no model calls."""
import json,pathlib,collections,re,csv,hashlib
import numpy as np
import sys
V2="--v2" in sys.argv
SUFFIX="_v2" if V2 else ""
P=pathlib.Path(__file__).resolve().parents[1];R=P/'results';Q=json.load(open(P/'data/processed/testset.json'));IDS=[q['id'] for q in Q];QT={q['id']:q['type'] for q in Q}
def loadall(sub):return [json.load(open(p)) for p in sorted((R/sub).glob('*.json'))]
def mean(xs):return float(np.mean(xs))
def charf1(a,b):
 a=collections.Counter(re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]','',a.lower()));b=collections.Counter(re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]','',b.lower()));same=sum((a&b).values());return 2*same/max(1,sum(a.values())+sum(b.values()))
def bootstrap(diff):
 x=np.array(diff);rng=np.random.default_rng(20261010);v=x[rng.integers(0,len(x),(2000,len(x)))].mean(1);return {'delta':float(x.mean()),'ci95':[float(v) for v in np.quantile(v,[.025,.975])]}
def main():
 ret=json.load(open(R/'retrieval.json'));G=loadall('generations');J=loadall('judgments_v2' if V2 else 'judgments');gmap={(g['arm'],g['id'],g['repeat']):g for g in G};jmap={(g['arm'],g['id'],g['repeat']):g for g in J}
 assert len(ret)==702 and len(G)==426 and len(J)==426
 assert gmap.keys()==jmap.keys();assert all(g['finish_reason']=='stop' for g in G+J)
 corpus=json.load(open(P/'data/processed/corpus.json'));D={d['id']:d for d in corpus}
 for q in Q:
  for group in q['evidence_groups']:
   for g in group:assert D[g['doc_id']]['text'][g['start']:g['end']]==g['quote']
 for r in ret:
  for c in r['chunks']:assert D[c['doc_id']]['text'][c['start']:c['end']]==c['text']
  if r['mode']=='budget768':assert r['source_chars']==768
 arms=sorted({g['arm'] for g in G});rows=[];flat=[]
 for arm in arms:
  gs=[gmap[arm,id,0] for id in IDS];js=[jmap[arm,id,0]['score'] for id in IDS];rr=[r for r in ret if r['config']+'__'+r['mode']==arm]
  d={'arm':arm,'n':len(gs),'correct_n':sum(s['correct'] for s in js),'correct_rate':mean([s['correct'] for s in js]),'completeness':mean([s['completeness'] for s in js]),'unsupported_rate':mean([s['unsupported'] for s in js]),'mean_prompt_tokens':mean([g['usage']['prompt_tokens'] for g in gs]),'mean_completion_tokens':mean([g['usage']['completion_tokens'] for g in gs]),'mean_seconds':mean([g['latency_seconds'] for g in gs]),'char_f1':mean([charf1(g['answer'],q['reference_answer']) for g,q in zip(gs,Q)])}
  for metric in ['coverage','evidence_recall','all_evidence','duplicate_ratio','source_chars','search_ms']:d[metric]=mean([r[metric] for r in rr]) if rr else None
  rows.append(d)
  for g,j,q in zip(gs,js,Q):flat.append({'arm':arm,'id':q['id'],'type':q['type'],**j,'prompt_tokens':g['usage']['prompt_tokens'],'answer':g['answer']})
 rsummary=[]
 for config in sorted({r['config'] for r in ret}):
  for mode in ['top3','budget768']:
   rr=[r for r in ret if r['config']==config and r['mode']==mode];rsummary.append({'config':config,'mode':mode,**{m:mean([r[m] for r in rr]) for m in ['coverage','evidence_recall','all_evidence','duplicate_ratio','source_chars']}})
 comparisons=[]
 for mode in ['top3','budget768']:
  base='fixed_256__'+mode
  for kind in ['overlap','structure']:
   arm=kind+'_256__'+mode
   for metric in ['correct','completeness']:
    d=bootstrap([jmap[arm,id,0]['score'][metric]-jmap[base,id,0]['score'][metric] for id in IDS]);comparisons.append({'arm':arm,'baseline':base,'metric':metric,**d})
   for metric in ['coverage','evidence_recall']:
    vals={(r['config'],r['id']):r[metric] for r in ret if r['mode']==mode};d=bootstrap([vals[kind+'_256',id]-vals['fixed_256',id] for id in IDS]);comparisons.append({'arm':arm,'baseline':base,'metric':metric,**d})
 repeat=[]
 for arm in ['fixed_256__top3','overlap_256__top3','structure_256__top3']:
  ids=[id for id in IDS if (arm,id,1) in gmap];repeat.append({'arm':arm,'n':len(ids),'text_changed':sum(gmap[arm,id,0]['answer']!=gmap[arm,id,1]['answer'] for id in ids),'judgment_changed':sum(jmap[arm,id,0]['score']['correct']!=jmap[arm,id,1]['score']['correct'] for id in ids),'first_correct':sum(jmap[arm,id,0]['score']['correct'] for id in ids),'repeat_correct':sum(jmap[arm,id,1]['score']['correct'] for id in ids)})
 bytype=[]
 for arm in ['fixed_256__top3','overlap_256__top3','structure_256__top3']:
  for t in sorted(set(QT.values())):
   ids=[id for id in IDS if QT[id]==t];bytype.append({'arm':arm,'type':t,'n':len(ids),'correct_n':sum(jmap[arm,id,0]['score']['correct'] for id in ids),'completeness':mean([jmap[arm,id,0]['score']['completeness'] for id in ids])})
 summary={'main':rows,'retrieval_sweep':rsummary,'paired_bootstrap':comparisons,'repeat':repeat,'by_type':bytype,'models':dict(collections.Counter(g['model'] for g in G+J)),'usage':{'prompt_tokens':sum(g['usage']['prompt_tokens'] for g in G+J),'completion_tokens':sum(g['usage']['completion_tokens'] for g in G+J),'total_tokens':sum(g['usage']['total_tokens'] for g in G+J)},'generation_count':len(G),'judgment_count':len(J),'doc_chars':sum(len(d['text']) for d in corpus),'articles':sum(len(d['articles']) for d in corpus)}
 (R/('summary'+SUFFIX+'.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 for name,rs in [('metrics.csv',rows),('per_question_scores.csv',flat),('retrieval_metrics.csv',rsummary)]:
  with open(R/(pathlib.Path(name).stem+SUFFIX+pathlib.Path(name).suffix),'w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rs[0].keys());w.writeheader();w.writerows(rs)
 (R/('analysis_audit'+SUFFIX+'.json')).write_text(json.dumps({'n_questions':len(Q),'retrieval_rows':len(ret),'unique_generations':len(gmap),'unique_judgments':len(jmap),'all_finish_stop':True,'all_evidence_offsets_verified':True,'all_context_offsets_verified':True,'all_budget768_exact':True},indent=2))
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
