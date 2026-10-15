"""Checks the recorded chunk partitions and evidence coordinates without API calls."""
import json,pathlib,hashlib
P=pathlib.Path(__file__).resolve().parents[1]; R=P/'results';D={d['id']:d for d in json.load(open(P/'data/processed/corpus.json'))};tests=[]
for path in sorted(R.glob('*_chunks.json')):
 kind,size=path.stem.removesuffix('_chunks').split('_');size=int(size);cs=json.load(open(path))
 for did,d in D.items():
  sub=[c for c in cs if c['doc_id']==did];seen=set()
  for c in sub:
   assert 0<c['end']-c['start']<=size
   assert c['text']==d['text'][c['start']:c['end']]
   positions=set(range(c['start'],c['end']))
   if kind!='overlap':assert not seen&positions
   seen|=positions
  assert len(seen)==len(d['text'])
 tests.append({'configuration':path.stem,'valid':True,'chunks':len(cs)})
for s in json.load(open(P/'data/source_manifest.json')):
 assert hashlib.sha256((P/'data/raw'/s['file']).read_bytes()).hexdigest()==s['sha256']
result={'chunk_checks':tests,'source_hashes_verified':True}
(R/'offline_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
