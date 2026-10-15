"""Download official sources into a new directory, preserving the experiment snapshot."""
import argparse,pathlib,json,subprocess,hashlib
P=pathlib.Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--out',default=str(P/'data/downloaded_raw'));args=a.parse_args();dest=pathlib.Path(args.out);dest.mkdir(parents=True,exist_ok=True)
for s in json.load(open(P/'data/source_manifest.json')):
 f=dest/s['file']
 if f.exists():raise SystemExit('Refusing to overwrite '+str(f))
 subprocess.run(['curl','--compressed','--fail','--location','--retry','3','--max-time','180','--output',str(f),s['url']],check=True)
 sha=hashlib.sha256(f.read_bytes()).hexdigest()
 print(s['id'],'snapshot_match='+str(sha==s['sha256']),'sha256='+sha)
