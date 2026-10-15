"""Extract official documents and downloaded official questions; no generated questions."""
import re,json,hashlib,pathlib,zipfile,xml.etree.ElementTree as ET
import pymupdf
from bs4 import BeautifulSoup
ROOT=pathlib.Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw';OUT=ROOT/'data/processed';OUT.mkdir(exist_ok=True)
def clean(s):return re.sub(r'\s+','',s).replace('\u200b','')
def pdftext(p):
 pages=[]
 for page in pymupdf.open(p):
  lines=page.get_text().splitlines();lines=[l for l in lines if not re.fullmatch(r'\s*[—－-]?\s*\d+\s*[—－-]?\s*',l)]
  pages.append('\n'.join(lines))
 return '\n'.join(pages)
def extract(name):
 p=next(RAW.glob(name+'.*'))
 if p.suffix=='.pdf':return pdftext(p)
 if p.suffix=='.docx':
  r=ET.fromstring(zipfile.ZipFile(p).read('word/document.xml'));n={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
  return '\n'.join(''.join(x.text or '' for x in q.findall('.//w:t',n)) for q in r.findall('.//w:p',n))
 s=BeautifulSoup(p.read_bytes(),'html.parser');t=s.get_text('\n',strip=True);t=t[t.index('第一章'):];return t.split('常用链接')[0]
def make_doc(name,title):
 t=extract(name)
 # Strip chapter headings BEFORE joining line wraps, keeping chapter metadata separately.
 lines=t.splitlines();out=[];chapters=[];pos=0
 for l in lines:
  l=clean(l)
  if not l:continue
  if re.match(r'^第[一二三四五六七八九十百]+章',l):chapters.append((pos,l));continue
  out.append(l);pos+=len(l)
 t=''.join(out)
 matches=list(re.finditer(r'第([一二三四五六七八九十百零〇]+)条',t))
 # Only structural article starts: in-line references are not starts. Parse from paragraph starts instead.
 starts=[];offset=0
 for l in out:
  if re.match(r'^第[一二三四五六七八九十百零〇]+条',l):starts.append(offset)
  offset+=len(l)
 if not starts:raise ValueError(name)
 base=starts[0];t=t[base:];starts=[x-base for x in starts];arts=[]
 for i,a in enumerate(starts):
  b=starts[i+1] if i+1<len(starts) else len(t)
  tx=t[a:b];label=re.match(r'第[一二三四五六七八九十百零〇]+条',tx)[0]
  ch=next((c for o,c in reversed(chapters) if o<=a+base),'')
  arts.append({'id':f'{name}:a{i+1:03}','label':label,'start':a,'end':b,'chapter':ch,'text':tx})
 return {'id':name,'title':title,'text':t,'articles':arts}
def main():
 specs=[('rules2021','中山大学本科生学籍管理规定（2021）'),('transfer','中山大学全日制本科生转专业实施办法（2019）'),('exchange','中山大学本科交流生学籍管理办法（2015）'),('thesis','中山大学本科生毕业论文（设计）工作管理规定（2020）')]
 docs=[make_doc(*s) for s in specs]
 json.dump(docs,open(OUT/'corpus.json','w'),ensure_ascii=False,indent=2)
 for d in docs:print(d['id'],len(d['text']),len(d['articles']))
 # Keep source pages for every question; remove page number artifacts and following section headers.
 pages=[]
 for i,p in enumerate(pymupdf.open(RAW/'sysu_qa.pdf')):
  t=re.sub(r'—\s*\d+\s*—','',p.get_text());pages.append(t)
 t='\n'.join(pages);qa=[]
 for i,m in enumerate(re.finditer(r'Q[：:]\s*(.*?)\s*A[：:]\s*(.*?)(?=Q[：:]|\Z)',t,re.S)):
  q=clean(m[1]);a=clean(m[2]);a=a.split('')[0]
  a=re.split(r'[一二三四五六七八九十]+、(?:课程及考试业务|证明/证件办理|交换交流管理|国际学生管理|辅修专业和辅修学士学位)',a)[0]
  page=t[:m.start()].count('\n') # below locate by cumulative lengths
  off=0;pn=1
  for j,p in enumerate(pages):
   if off+len(p)+1>m.start():pn=j+1;break
   off+=len(p)+1
  qa.append({'id':f'S{i+1:03}','question':q,'reference_answer':a,'source_pdf_page':pn})
 json.dump(qa,open(OUT/'all_downloaded_qa.json','w'),ensure_ascii=False,indent=2)
 print('qa',len(qa))
if __name__=='__main__':main()
