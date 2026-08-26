from __future__ import annotations

import base64,math,os,re
from datetime import datetime,timezone
from typing import Any
import cv2,imagehash,numpy as np,requests
from bs4 import BeautifulSoup
from fastapi import FastAPI,HTTPException,Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from PIL import Image

APP_VERSION='3.0.0'
SUPABASE_URL=os.getenv('SUPABASE_URL','').rstrip('/')
SUPABASE_KEY=os.getenv('SUPABASE_PUBLISHABLE_KEY','') or os.getenv('SUPABASE_ANON_KEY','')
REQUEST_TIMEOUT=float(os.getenv('HTTP_TIMEOUT','10'))
SESSION=requests.Session();SESSION.headers.update({'user-agent':'Mozilla/5.0 CardDistrictVisual/3.0','accept-language':'de-DE,de;q=0.9,en;q=0.8'})
_rng=np.random.default_rng(20260827);PROJ=_rng.standard_normal((256,512)).astype(np.float32);PROJ/=np.maximum(np.linalg.norm(PROJ,axis=1,keepdims=True),1e-8)
app=FastAPI(title='CardDistrict Visual Search AI',version=APP_VERSION);app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['GET','POST','OPTIONS'],allow_headers=['*'])

class ImagePayload(BaseModel): data:str; mimeType:str='image/jpeg'
class RecognizeRequest(BaseModel): front:ImagePayload; category_key:str|None=None; max_candidates:int=Field(default=10,ge=1,le=30); include_market:bool=True
class DescribeRequest(BaseModel): image:ImagePayload

def now_iso():return datetime.now(timezone.utc).isoformat()
def norm(s:str):return re.sub(r'[^a-z0-9]+',' ',s.lower().replace('ü','u').replace('ö','o').replace('ä','a').replace('ß','ss')).strip()
def toks(s:str):return [x for x in norm(s).split() if len(x)>1 and x not in {'card','cards','trading','rookie','rc','topps','chrome','panini','upper','deck','the','and','fc'}]
def decode(payload:ImagePayload):
    try:raw=base64.b64decode(payload.data.split(',',1)[-1],validate=False)
    except Exception as e:raise HTTPException(400,'Ungültige Bilddaten') from e
    img=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
    if img is None or img.size==0:raise HTTPException(400,'Bild konnte nicht gelesen werden')
    return img
def download(url:str):
    if not re.match(r'^https?://',url,re.I):raise HTTPException(400,'Ungültige Bild-URL')
    try:r=SESSION.get(url,timeout=REQUEST_TIMEOUT);r.raise_for_status();raw=r.content
    except Exception as e:raise HTTPException(502,'Referenzbild konnte nicht geladen werden') from e
    img=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
    if img is None or img.size==0:raise HTTPException(502,'Referenzbild konnte nicht dekodiert werden')
    return img
def order_quad(pts):
    pts=pts.astype(np.float32);s=pts.sum(axis=1);d=np.diff(pts,axis=1).reshape(-1);return np.array([pts[np.argmin(s)],pts[np.argmin(d)],pts[np.argmax(s)],pts[np.argmax(d)]],dtype=np.float32)
def rectify(img):
    h,w=img.shape[:2];scale=min(1.0,1400.0/max(h,w));work=cv2.resize(img,(max(1,int(w*scale)),max(1,int(h*scale))),interpolation=cv2.INTER_AREA) if scale<1 else img.copy();gray=cv2.GaussianBlur(cv2.cvtColor(work,cv2.COLOR_BGR2GRAY),(5,5),0);edges=cv2.dilate(cv2.Canny(gray,45,135),np.ones((3,3),np.uint8),iterations=1);contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);total=work.shape[0]*work.shape[1];quad=None;best=0.0
    for c in sorted(contours,key=cv2.contourArea,reverse=True)[:35]:
        area=cv2.contourArea(c)
        if area<total*.16:continue
        ap=cv2.approxPolyDP(c,.024*cv2.arcLength(c,True),True)
        if len(ap)==4 and area>best:best=area;quad=ap.reshape(4,2)
    if quad is not None:
        q=order_quad(quad);tl,tr,br,bl=q;mw=int(max(np.linalg.norm(br-bl),np.linalg.norm(tr-tl)));mh=int(max(np.linalg.norm(tr-br),np.linalg.norm(tl-bl)))
        if mw>180 and mh>250:work=cv2.warpPerspective(work,cv2.getPerspectiveTransform(q,np.array([[0,0],[mw-1,0],[mw-1,mh-1],[0,mh-1]],dtype=np.float32)),(mw,mh))
    if work.shape[1]>work.shape[0]:work=cv2.rotate(work,cv2.ROTATE_90_CLOCKWISE)
    hh,ww=work.shape[:2];expected=640/896;ratio=ww/max(1,hh)
    if ratio>expected*1.14:nw=max(1,int(hh*expected));x=max(0,(ww-nw)//2);work=work[:,x:x+nw]
    elif ratio<expected*.86:nh=max(1,int(ww/expected));y=max(0,(hh-nh)//2);work=work[y:y+nh,:]
    return cv2.resize(work,(640,896),interpolation=cv2.INTER_AREA)
def block(ch,w,h):return cv2.resize(ch,(w,h),interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
def descriptor(card):
    card=rectify(card);small=cv2.resize(card,(256,358),interpolation=cv2.INTER_AREA);gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY);hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV);f=[];f.extend((block(gray,8,8)/255).tolist());h8=cv2.resize(hsv,(8,8),interpolation=cv2.INTER_AREA).astype(np.float32);f.extend((h8[:,:,0]/179).reshape(-1).tolist());f.extend((h8[:,:,1]/255).reshape(-1).tolist());f.extend((h8[:,:,2]/255).reshape(-1).tolist());f.extend((block(cv2.Canny(gray,55,145),8,8)/255).tolist());gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3);mag,ang=cv2.cartToPolar(gx,gy,angleInDegrees=True);ch,cw=gray.shape[0]//4,gray.shape[1]//4
    for yy in range(4):
        for xx in range(4):
            m=mag[yy*ch:(yy+1)*ch,xx*cw:(xx+1)*cw];a=ang[yy*ch:(yy+1)*ch,xx*cw:(xx+1)*cw];hist=np.zeros(8,dtype=np.float32);bins=np.floor((a%180)/22.5).astype(np.int32)
            for b in range(8):hist[b]=float(m[bins==b].sum())
            s=float(hist.sum());hist=hist/s if s>1e-6 else hist;f.extend(hist.tolist())
    g=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32);g-=float(g.mean());d=cv2.dct(g)[:8,:8].reshape(-1);n=float(np.linalg.norm(d));d=d/n if n>1e-6 else d;f.extend(d.tolist());v=np.asarray(f,dtype=np.float32)
    if v.size!=512:raise RuntimeError('descriptor dimension mismatch')
    n=float(np.linalg.norm(v));return v/n if n>1e-8 else v
def signature(v):return ''.join('1' if x>=0 else '0' for x in (PROJ@v).tolist())
def hashes(card):
    pil=Image.fromarray(cv2.cvtColor(rectify(card),cv2.COLOR_BGR2RGB));return str(imagehash.phash(pil,hash_size=8)),str(imagehash.dhash(pil,hash_size=8))
def hsim(a,b):
    if not a or not b or len(a)!=len(b):return 0.0
    try:return max(0.0,1-bin(int(a,16)^int(b,16)).count('1')/(len(a)*4))
    except Exception:return 0.0
def orb(a,b):
    aa=cv2.cvtColor(rectify(a),cv2.COLOR_BGR2GRAY);bb=cv2.cvtColor(rectify(b),cv2.COLOR_BGR2GRAY);o=cv2.ORB_create(nfeatures=1400,scaleFactor=1.18,nlevels=8,edgeThreshold=15,fastThreshold=10);_,da=o.detectAndCompute(aa,None);_,db=o.detectAndCompute(bb,None)
    if da is None or db is None or len(da)<8 or len(db)<8:return 0.0
    pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2);good=sum(1 for p in pairs if len(p)==2 and p[0].distance<.72*p[1].distance);return min(1.0,good/max(25,min(len(da),len(db),220)))
def headers():return {'apikey':SUPABASE_KEY,'authorization':f'Bearer {SUPABASE_KEY}','content-type':'application/json','accept':'application/json'}
def candidates(sig,count,category):
    if not(SUPABASE_URL and SUPABASE_KEY):return []
    try:r=SESSION.post(f'{SUPABASE_URL}/rest/v1/rpc/cd_match_vision_signatures',headers=headers(),json={'query_signature':sig,'match_count':count,'category_filter':category},timeout=REQUEST_TIMEOUT);return r.json() if r.ok and isinstance(r.json(),list) else []
    except Exception:return []
def rerank(scan,qv,qp,qd,rows):
    out=[]
    for row in rows[:16]:
        url=str(row.get('reference_image_url') or '')
        if not url:continue
        try:ref=download(url);rv=descriptor(ref);rp,rd=hashes(ref);cos=max(0.0,min(1.0,float(np.dot(qv,rv))));ps=hsim(qp,str(row.get('phash') or rp));ds=hsim(qd,str(row.get('dhash') or rd));os=orb(scan,ref)
        except Exception:continue
        ss=max(0.0,min(1.0,float(row.get('signature_similarity') or 0)));score=.46*cos+.12*ss+.15*ps+.05*ds+.22*os;item=dict(row);item.update({'descriptor_similarity':cos,'phash_similarity':ps,'dhash_similarity':ds,'orb_similarity':os,'match_score':score});out.append(item)
    return sorted(out,key=lambda x:x['match_score'],reverse=True)
def ident(row,conf):
    k=str(row.get('category_key') or 'unknown');sport=k in {'soccer','basketball','baseball','american_football','ice_hockey','motorsport','tennis','golf','boxing','wrestling'};name=str(row.get('player') or row.get('name') or '')
    return {'categoryType':'sport' if sport else 'tcg','categoryKey':k,'categoryLabel':'Fußball' if k=='soccer' else k.replace('_',' ').title(),'subject':name,'game':'Fußball' if k=='soccer' else k.replace('_',' ').title(),'manufacturer':str(row.get('brand') or ''),'year':str(row.get('year') or ''),'setName':str(row.get('set_name') or ''),'cardNumber':str(row.get('card_number') or ''),'parallel':str(row.get('parallel') or ''),'rarity':'Rookie / RC' if row.get('rookie') else '','edition':str(row.get('insert_name') or ''),'serial':'','gradingCompany':'','grade':'','recognitionConfidence':conf,'needsReview':False,'searchQuery':' '.join(x for x in [str(row.get('year') or ''),str(row.get('set_name') or ''),name,str(row.get('card_number') or ''),str(row.get('parallel') or '')] if x),'recognitionMethod':'carddistrict-visual-ai-v3','scanText':''}
def card(row):return {'name':str(row.get('player') or row.get('name') or ''),'setName':str(row.get('set_name') or ''),'cardNumber':str(row.get('card_number') or ''),'rarity':'Rookie / RC' if row.get('rookie') else '','variant':str(row.get('parallel') or row.get('insert_name') or ''),'releaseDate':str(row.get('year') or ''),'source':'CardDistrict Visual AI','sourceUrl':str(row.get('source_url') or ''),'imageUrl':str(row.get('reference_image_url') or ''),'marketValue':0,'marketCurrency':''}
def parse_price(text):
    cur='EUR' if '€' in text else 'GBP' if '£' in text else 'USD' if '$' in text else '';m=re.search(r'(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)',text.replace('\xa0',' '))
    if not m:return 0.0,cur
    n=m.group(1);n=n.replace('.','').replace(',','.') if ',' in n and '.' in n and n.rfind(',')>n.rfind('.') else n.replace(',','') if ',' in n and '.' in n else n.replace(',','.') if ',' in n else n
    try:return float(n),cur
    except:return 0.0,cur
def ebay(query,sold=False,limit=60):
    params={'_nkw':query,'_ipg':'120'}
    if sold:params.update({'LH_Sold':'1','LH_Complete':'1'})
    try:r=SESSION.get('https://www.ebay.de/sch/i.html',params=params,timeout=REQUEST_TIMEOUT)
    except:return []
    if not r.ok:return []
    soup=BeautifulSoup(r.text,'html.parser');out=[]
    for it in soup.select('li.s-item'):
        t=it.select_one('.s-item__title');a=it.select_one('a.s-item__link');p=it.select_one('.s-item__price')
        if not(t and a and p):continue
        title=t.get_text(' ',strip=True);price,currency=parse_price(p.get_text(' ',strip=True))
        if not title or 'Shop on eBay' in title or price<=0:continue
        im=it.select_one('img.s-item__image-img');out.append({'title':title,'price':price,'currency':currency,'source':'eBay','sourceUrl':str(a.get('href') or ''),'buyUrl':str(a.get('href') or ''),'imageUrl':str((im.get('src') or im.get('data-src') or '') if im else ''),'status':'sold' if sold else 'active','kind':'completed sale' if sold else 'listing'})
        if len(out)>=limit:break
    return out
def exact(items,row):
    name=toks(str(row.get('player') or row.get('name') or ''));num=norm(str(row.get('card_number') or '')).replace(' ','');ins=toks(str(row.get('insert_name') or ''));par=toks(str(row.get('parallel') or ''));out=[]
    for x in items:
        text=norm(x['title']);words=text.split();compact=text.replace(' ','');nameok=not name or sum(1 for n in name if n in words)>=max(1,math.ceil(len(name)*.67));numok=not num or num in compact;insok=not ins or any(v in words for v in ins);parok=not par or any(v in words for v in par) or not str(row.get('parallel') or '').strip()
        if nameok and numok and insok and parok:out.append(x)
    return out

@app.get('/health')
def health():return {'ok':True,'service':'carddistrict-visual-search','version':APP_VERSION,'engine':'512D descriptor + 256bit SimHash + pHash/dHash + ORB','supabase':bool(SUPABASE_URL and SUPABASE_KEY)}
@app.post('/describe')
def describe(req:DescribeRequest):
    im=rectify(decode(req.image));v=descriptor(im);p,d=hashes(im);return {'embedding':v.tolist(),'signature':signature(v),'phash':p,'dhash':d,'dimension':512,'engine':'carddistrict-visual-ai-v3'}
@app.get('/describe-url')
def describe_url(url:str=Query(...,min_length=8)):
    im=rectify(download(url));v=descriptor(im);p,d=hashes(im);return {'embedding':v.tolist(),'signature':signature(v),'phash':p,'dhash':d,'dimension':512,'engine':'carddistrict-visual-ai-v3','url':url}
@app.get('/discover-ebay')
def discover_ebay(query:str=Query(...,min_length=2),sold:bool=False,limit:int=Query(40,ge=1,le=120)):return {'items':ebay(query,sold,limit),'query':query,'sold':sold}
@app.post('/recognize')
def recognize(req:RecognizeRequest):
    scan=rectify(decode(req.front));qv=descriptor(scan);sig=signature(qv);qp,qd=hashes(scan);rows=candidates(sig,max(30,req.max_candidates*4),req.category_key);ranked=rerank(scan,qv,qp,qd,rows);best=ranked[0] if ranked else None;second=ranked[1] if len(ranked)>1 else None;score=float(best.get('match_score') or 0) if best else 0.0;margin=score-(float(second.get('match_score') or 0) if second else 0.0);struct=max(float(best.get('phash_similarity') or 0),float(best.get('orb_similarity') or 0)) if best else 0.0;verified=bool(best and score>=.70 and margin>=.012 and struct>=.16);cand=[card(x)|{'visualScore':round(float(x.get('match_score') or 0)*100,1)} for x in ranked[:req.max_candidates]]
    if not verified:return {'identity':None,'verified':False,'card':None,'candidates':cand,'items':[],'sales':[],'checkedAt':now_iso(),'warning':'Noch kein sicherer visueller Treffer. OCR beeinflusst diese Entscheidung nicht.','engine':{'name':'CardDistrict Visual AI v3','score':round(score*100,1),'margin':round(margin*100,1),'indexedCandidates':len(rows)}}
    identity=ident(best,min(.999,max(.8,score)));resolved=card(best);items=exact(ebay(identity['searchQuery'],False,100)+ebay(identity['searchQuery'],True,100),best) if req.include_market else [];sales=[x for x in items if x['status']=='sold'];return {'identity':identity,'verified':True,'card':resolved,'candidates':[],'items':items,'sales':sales,'checkedAt':now_iso(),'warning':f'Visuell bestätigt: {score*100:.1f}% · Abstand {margin*100:.1f}%. Erst danach wurden Marktdaten geöffnet.','engine':{'name':'CardDistrict Visual AI v3','score':round(score*100,1),'margin':round(margin*100,1),'descriptor':round(float(best.get('descriptor_similarity') or 0)*100,1),'phash':round(float(best.get('phash_similarity') or 0)*100,1),'orb':round(float(best.get('orb_similarity') or 0)*100,1)}}
