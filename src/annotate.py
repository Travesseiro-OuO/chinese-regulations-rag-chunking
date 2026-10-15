"""Frozen source-grounded evidence annotations, created before any model outputs."""
import json,re,pathlib,hashlib
P=pathlib.Path(__file__).resolve().parents[1]/'data/processed'
D=json.load(open(P/'corpus.json'));A={a['id']:(d,a) for d in D for a in d['articles']}
Q={q['id']:q for q in json.load(open(P/'all_downloaded_qa.json'))}
# Each selector is (article ID, start substring, end substring); empty = full article.
# Alternative evidence spans can be represented within one group; default conjunctive groups.
R='rules2021:';T='transfer:';E='exchange:';B='thesis:'
S={
3:[(R+'a009','未经请假','给予退学处理。')],
5:[(R+'a005','新生因创业','退役后两年。'),(R+'a005','新生保留入学资格期间','不具有学籍。')],
13:[(R+'a055','','')],
15:[(R+'a057','','')],
16:[(R+'a060','','')],
18:[(R+'a058','因病申请休学','签署意见'),(R+'a061','（二）','办理复学手续；')],
19:[(R+'a032','一学年所修学分','作退学处理。')],
20:[(R+'a032','学生降级试读期结束后','报送教务部审批。')],
21:[(R+'a051','第五十一条','（五）')],
22:[(R+'a059','在校生应征','退役后两年。')],
27:[(T+'a005','','')],
28:[(T+'a006','','')],
31:[(T+'a007','每位学生','一个专业。')],
35:[(T+'a010','','')],
38:[(T+'a014','','')],
50:[(R+'a041','辅修专业是','不低于25学分。')],
51:[(R+'a041','辅修学士学位是','不低于75学分。')],
52:[(R+'a067','修读辅修专业的学生','同时修完辅修专业；'),(R+'a044','未修满辅修专业学分者','公共选修课学分。')],
53:[(R+'a067','修读辅修学士学位的学生','不得超过最长在籍年限')],
54:[(R+'a044','修读辅修专业的学生','不单独发放学位证书。')],
55:[(R+'a044','不符合辅修学士学位授予条件','申请辅修专业证书。')],
61:[(R+'a067','学生修完主修专业','办理结业；'),(R+'a067','申请延读的学生','送分管教学校领导审批。')],
63:[(R+'a068','','')],
66:[(R+'a069','学生根据本规定第六十八条第（一）项','不得再申请重修或者补考。'),(R+'a069','换发的毕业证书','日期为准。'),(R+'a065','符合《中华人民共和国学位条例》','发给学士学位证书。')],
69:[(B+'a018','除外国语言文学类','简化汉语文字撰写。')],
70:[(B+'a010','','')],
71:[(B+'a018','论文正文部分','教务部备案。')],
72:[(B+'a020','','')],
73:[(B+'a020','','')],
74:[(B+'a027','',''),(B+'a028','','')],
93:[(R+'a064','获得批准退学','由学校发给肄业证书')],
149:[(E+'a009','5．课程学分','逾期不予办理。')],
151:[(E+'a008','2）交流生在合作院校','成绩可按较高的一个登录）。')],
152:[(E+'a008','1）如果学生在合作院校所修课程成绩以百分制','相应的百分制成绩。'),(E+'a008','3）如果学生在合作院校','为依据进行转换。')],
153:[(E+'a008','对公共选修课程','备案处理。')],
154:[(E+'a009','6．学生应在','原则上不予以转换。')],
196:[(B+'a007','','')],
198:[(B+'a009','（六）选题','完全相同。')],
203:[(B+'a022','答辩委员会成员','职称的教师担任。')]
}
# Preserve downloaded question wording. Remove non-answer section headings only.
items=[]
for k,sels in S.items():
 q=dict(Q[f'S{k:03}']);groups=[]
 for aid,left,right in sels:
  d,a=A[aid];t=a['text'];l=t.index(left) if left else 0;r=t.index(right,l)+len(right) if right else len(t)
  groups.append([{'doc_id':d['id'],'article_id':aid,'start':a['start']+l,'end':a['start']+r,'quote':t[l:r]}])
 q['evidence_groups']=groups
 q['reference_answer']=re.sub('五、学生竞赛$','',q['reference_answer'])
 q['type']='cross_clause' if len({s[0] for s in sels})>1 else 'conditional' if k in [3,5,13,19,20,21,22,27,28,52,53,54,55,63,70,74,93,153,154] else 'factual'
 q['annotation_note']='AI-assisted source alignment, not human expert annotation; original downloaded reference retained.'
 items.append(q)
# Non-conjunctive alternative clauses for the same prohibition.
aid=T+'a005';d,a=A[aid];t=a['text'];l=t.index('第五条');r=t.index('（五）')
next(q for q in items if q['id']=='S021')['evidence_groups'][0].append({'doc_id':d['id'],'article_id':aid,'start':a['start']+l,'end':a['start']+r,'quote':t[l:r]})
# Semantic reference additions may exist outside corpus. Judge only source-supported claims,
# with gold evidence as authoritative scope; report this limitation explicitly.
json.dump(items,open(P/'testset.json','w'),ensure_ascii=False,indent=2)
print('frozen_test_questions',len(items));print('types',{t:sum(q['type']==t for q in items) for t in ['factual','conditional','cross_clause']})
print('sha256',hashlib.sha256((P/'testset.json').read_bytes()).hexdigest())
