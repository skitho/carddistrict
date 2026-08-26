from __future__ import annotations

import base64
import math
import os
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import cv2
import imagehash
import numpy as np
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

APP_VERSION='2.0.0'
SUPABASE_URL=os.getenv('SUPABASE_URL','').rstrip('/')
SUPABASE_KEY=os.getenv('SUPABASE_PUBLISHABLE_KEY','') or os.getenv('SUPABASE_ANON_KEY','')
REQUEST_TIMEOUT=float(os.getenv('HTTP_TIMEOUT','10'))
UA='Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1 CardDistrictVisual/2.0'
SESSION=requests.Session()
SESSION.headers.update({'user-agent':UA,'accept-language':'de-DE,de;q=0.9,en;q=0.8','accept':'text/html,application/xhtml+xml,application/json,image/avif,image/webp,image/apng,*/*;q=0.8'})

app=FastAPI(title='CardDistrict Visual Search AI',version=APP_VERSION)
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['GET','POST','OPTIONS'],allow_headers=['*'])

class ImagePayload(BaseModel):
    data:str
    mimeType:str='image/jpeg'

class RecognizeRequest(BaseModel):
    front:ImagePayload
    category_key:str|None=None
    max_candidates:int=Field(default=10,ge=1,le=30)
    include_market:bool=True

class DescribeRequest(BaseModel):
    image:ImagePayload


def now_iso()->str:
    return datetime.now(timezone.utc).isoformat()

def norm(s:str)->str:
    return re.sub(r'[^a-z0-9]+',' ',s.lower().replace('ü','u').replace('ö','o').replace('ä','a').replace('ß','ss')).strip()

def toks(s:str)->list[str]:
    stop={'card','cards','trading','rookie','rc','topps','chrome','panini','upper','deck','the','and','fc'}
    return [x for x in norm(s).split() if len(x)>1 and x not in stop]

def decode_image(payload:ImagePayload)->np.ndarray:
    try: raw=base64.b64decode(payload.data.split(',',1)[-1],validate=False)
    except Exception as exc: raise HTTPException(400,'Ungültige Bilddaten') from exc
    img=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
    if img is None or img.size==0: raise HTTPException(400,'Bild konnte nicht gelesen werden')
    return img

def download_image(url:str)->np.ndarray:
    if not re.match(r'^https?://',url,re.I): raise HTTPException(400,'Ungültige Bild-URL')
    try:
        r=SESSION.get(url,timeout=REQUEST_TIMEOUT,stream=True)
        r.raise_for_status(); raw=r.content
    except Exception as exc: raise HTTPException(502,'Referenzbild konnte nicht geladen werden') from exc
    img=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
    if img is None or img.size==0: raise HTTPException(502,'Referenzbild konnte nicht dekodiert werden')
    return img

def order_quad(pts:np.ndarray)->np.ndarray:
    pts=pts.astype(np.float32); s=pts.sum(axis=1); d=np.diff(pts,axis=1).reshape(-1)
    return np.array([pts[np.argmin(s)],pts[np.argmin(d)],pts[np.argmax(s)],pts[np.argmax(d)]],dtype=np.float32)

def rectify_card(img:np.ndarray)->np.ndarray:
    h,w=img.shape[:2]; scale=min(1.0,1400.0/max(h,w)); work=cv2.resize(img,(max(1,int(w*scale)),max(1,int(h*scale))),interpolation=cv2.INTER_AREA) if scale<1 else img.copy()
    gray=cv2.GaussianBlur(cv2.cvtColor(work,cv2.COLOR_BGR2GRAY),(5,5),0)
    edges=cv2.dilate(cv2.Canny(gray,45,135),np.ones((3,3),np.uint8),iterations=1)
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE); total=work.shape[0]*work.shape[1]; quad=None; best=0.0
    for c in sorted(contours,key=cv2.contourArea,reverse=True)[:35]:
        area=cv2.contourArea(c)
        if area<total*.16: continue
        approx=cv2.approxPolyDP(c,.024*cv2.arcLength(c,True),True)
        if len(approx)==4 and area>best: best=area; quad=approx.reshape(4,2)
    if quad is not None:
        q=order_quad(quad); tl,tr,br,bl=q
        mw=int(max(np.linalg.norm(br-bl),np.linalg.norm(tr-tl))); mh=int(max(np.linalg.norm(tr-br),np.linalg.norm(tl-bl)))
        if mw>180 and mh>250:
            dst=np.array([[0,0],[mw-1,0],[mw-1,mh-1],[0,mh-1]],dtype=np.float32)
            work=cv2.warpPerspective(work,cv2.getPerspectiveTransform(q,dst),(mw,mh))
    if work.shape[1]>work.shape[0]: work=cv2.rotate(work,cv2.ROTATE_90_CLOCKWISE)
    hh,ww=work.shape[:2]
    target_h=896; target_w=640
    ratio=ww/max(1,hh); expected=target_w/target_h
    if ratio>expected*1.14:
        nw=max(1,int(hh*expected)); x=max(0,(ww-nw)//2); work=work[:,x:x+nw]
    elif ratio<expected*.86:
        nh=max(1,int(ww/expected)); y=max(0,(hh-nh)//2); work=work[y:y+nh,:]
    return cv2.resize(work,(target_w,target_h),interpolation=cv2.INTER_AREA)

def _block_mean(channel:np.ndarray,w:int,h:int)->np.ndarray:
    return cv2.resize(channel,(w,h),interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)

def descriptor512(card:np.ndarray)->np.ndarray:
    card=rectify_card(card); small=cv2.resize(card,(256,358),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY); hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    f:list[float]=[]
    f.extend((_block_mean(gray,8,8)/255.0).tolist())
    hsv8=cv2.resize(hsv,(8,8),interpolation=cv2.INTER_AREA).astype(np.float32)
    hch=(hsv8[:,:,0]/179.0).reshape(-1); sch=(hsv8[:,:,1]/255.0).reshape(-1); vch=(hsv8[:,:,2]/255.0).reshape(-1)
    f.extend(hch.tolist()); f.extend(sch.tolist()); f.extend(vch.tolist())
    edge=cv2.Canny(gray,55,145)
    f.extend((_block_mean(edge,8,8)/255.0).tolist())
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    mag,ang=cv2.cartToPolar(gx,gy,angleInDegrees=True)
    ch,cw=gray.shape[0]//4,gray.shape[1]//4
    for yy in range(4):
        for xx in range(4):
            m=mag[yy*ch:(yy+1)*ch,xx*cw:(xx+1)*cw]; a=ang[yy*ch:(yy+1)*ch,xx*cw:(xx+1)*cw]
            hist=np.zeros(8,dtype=np.float32)
            bins=np.floor((a%180.0)/22.5).astype(np.int32)
            for b in range(8): hist[b]=float(m[bins==b].sum())
            s=float(hist.sum())
            if s>1e-6: hist/=s
            f.extend(hist.tolist())
    g32=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32); g32-=float(g32.mean())
    dct=cv2.dct(g32)[:8,:8].reshape(-1); scale=float(np.linalg.norm(dct))
    if scale>1e-6: dct/=scale
    f.extend(dct.tolist())
    arr=np.asarray(f,dtype=np.float32)
    if arr.size!=512: raise RuntimeError(f'visual descriptor dimension {arr.size}, expected 512')
    n=float(np.linalg.norm(arr))
    if n>1e-8: arr/=n
    return arr

def hashes(card:np.ndarray)->tuple[str,str]:
    rgb=cv2.cvtColor(rectify_card(card),cv2.COLOR_BGR2RGB); pil=Image.fromarray(rgb)
    return str(imagehash.phash(pil,hash_size=8)),str(imagehash.dhash(pil,hash_size=8))

def hsim(a:str,b:str)->float:
    if not a or not b or len(a)!=len(b): return 0.0
    try: dist=bin(int(a,16)^int(b,16)).count('1'); return max(0.0,1.0-dist/(len(a)*4.0))
    except Exception: return 0.0

def orb_score(a:np.ndarray,b:np.ndarray)->float:
    aa=cv2.cvtColor(rectify_card(a),cv2.COLOR_BGR2GRAY); bb=cv2.cvtColor(rectify_card(b),cv2.COLOR_BGR2GRAY)
    orb=cv2.ORB_create(nfeatures=1200,scaleFactor=1.18,nlevels=8,edgeThreshold=15,fastThreshold=10)
    _,da=orb.detectAndCompute(aa,None); _,db=orb.detectAndCompute(bb,None)
    if da is None or db is None or len(da)<8 or len(db)<8:return 0.0
    matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs=matcher.knnMatch(da,db,k=2); good=0
    for p in pairs:
        if len(p)==2 and p[0].distance<.72*p[1].distance: good+=1
    denom=max(25,min(len(da),len(db),220)); return min(1.0,good/denom)

def vector_literal(v:np.ndarray)->str:
    return '['+','.join(f'{float(x):.7f}' for x in v.tolist())+']'

def sb_headers()->dict[str,str]:
    return {'apikey':SUPABASE_KEY,'authorization':f'Bearer {SUPABASE_KEY}','content-type':'application/json','accept':'application/json'}

def match_vector(v:np.ndarray,count:int,category_key:str|None)->list[dict[str,Any]]:
    if not(SUPABASE_URL and SUPABASE_KEY): return []
    payload={'query_embedding':vector_literal(v),'match_count':count,'category_filter':category_key}
    try:
        r=SESSION.post(f'{SUPABASE_URL}/rest/v1/rpc/cd_match_vision_cards',headers=sb_headers(),json=payload,timeout=REQUEST_TIMEOUT)
        if not r.ok:return []
        data=r.json(); return data if isinstance(data,list) else []
    except Exception:return []

def reference_metrics(scan:np.ndarray,row:dict[str,Any],scan_phash:str,scan_dhash:str)->dict[str,float]:
    url=str(row.get('reference_image_url') or '')
    if not url:return {'phash':0.0,'dhash':0.0,'orb':0.0}
    try: ref=download_image(url)
    except Exception:return {'phash':0.0,'dhash':0.0,'orb':0.0}
    rp,rd=hashes(ref)
    return {'phash':hsim(scan_phash,str(row.get('phash') or rp)),'dhash':hsim(scan_dhash,str(row.get('dhash') or rd)),'orb':orb_score(scan,ref)}

def rank_rows(scan:np.ndarray,rows:list[dict[str,Any]],scan_phash:str,scan_dhash:str)->list[dict[str,Any]]:
    out=[]
    for row in rows[:12]:
        visual=max(0.0,min(1.0,float(row.get('similarity') or 0.0))); metrics=reference_metrics(scan,row,scan_phash,scan_dhash)
        score=.55*visual+.17*metrics['phash']+.06*metrics['dhash']+.22*metrics['orb']
        item=dict(row); item['vector_similarity']=round(visual,4); item['phash_similarity']=round(metrics['phash'],4); item['dhash_similarity']=round(metrics['dhash'],4); item['orb_similarity']=round(metrics['orb'],4); item['match_score']=round(score,4); out.append(item)
    return sorted(out,key=lambda x:float(x.get('match_score') or 0),reverse=True)

def identity_from_row(row:dict[str,Any],confidence:float)->dict[str,Any]:
    category=str(row.get('category_key') or 'unknown')
    sport=category in {'soccer','basketball','baseball','american_football','ice_hockey','motorsport','tennis','golf','boxing','wrestling'}
    return {'categoryType':'sport' if sport else 'tcg','categoryKey':category,'categoryLabel':'Fußball' if category=='soccer' else category.replace('_',' ').title(),'subject':str(row.get('player') or row.get('name') or ''),'game':'Fußball' if category=='soccer' else category.replace('_',' ').title(),'manufacturer':str(row.get('brand') or ''),'year':str(row.get('year') or ''),'setName':str(row.get('set_name') or ''),'cardNumber':str(row.get('card_number') or ''),'parallel':str(row.get('parallel') or ''),'rarity':'Rookie / RC' if bool(row.get('rookie')) else '','edition':str(row.get('insert_name') or ''),'serial':'','gradingCompany':'','grade':'','recognitionConfidence':confidence,'needsReview':False,'searchQuery':' '.join(x for x in [str(row.get('year') or ''),str(row.get('set_name') or ''),str(row.get('player') or row.get('name') or ''),str(row.get('card_number') or ''),str(row.get('parallel') or '')] if x),'recognitionMethod':'carddistrict-visual-vector-v2','scanText':''}

def candidate_from_row(row:dict[str,Any])->dict[str,Any]:
    return {'name':str(row.get('player') or row.get('name') or ''),'setName':str(row.get('set_name') or ''),'cardNumber':str(row.get('card_number') or ''),'rarity':'Rookie / RC' if bool(row.get('rookie')) else '','variant':str(row.get('parallel') or row.get('insert_name') or ''),'releaseDate':str(row.get('year') or ''),'source':'CardDistrict Visual Index','sourceUrl':str(row.get('source_url') or ''),'imageUrl':str(row.get('reference_image_url') or ''),'marketValue':0,'marketCurrency':''}

def parse_price(text:str)->tuple[float,str]:
    raw=text.replace('\xa0',' ').strip(); currency='EUR' if '€' in raw or re.search(r'\bEUR\b',raw,re.I) else 'GBP' if '£' in raw else 'USD' if '$' in raw else ''
    m=re.search(r'(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)',raw)
    if not m:return 0.0,currency
    n=m.group(1)
    if ',' in n and '.' in n:n=n.replace('.','').replace(',','.') if n.rfind(',')>n.rfind('.') else n.replace(',','')
    elif ',' in n:n=n.replace(',','.') if len(n.rsplit(',',1)[-1])<=2 else n.replace(',','')
    try:return float(n),currency
    except ValueError:return 0.0,currency

def search_ebay(query:str,sold:bool=False,limit:int=60)->list[dict[str,Any]]:
    params={'_nkw':query,'_ipg':'120'}
    if sold:params.update({'LH_Sold':'1','LH_Complete':'1'})
    try:r=SESSION.get('https://www.ebay.de/sch/i.html',params=params,timeout=REQUEST_TIMEOUT)
    except Exception:return []
    if not r.ok:return []
    soup=BeautifulSoup(r.text,'html.parser'); out=[]
    for item in soup.select('li.s-item'):
        t=item.select_one('.s-item__title'); a=item.select_one('a.s-item__link'); p=item.select_one('.s-item__price')
        if not(t and a and p):continue
        title=t.get_text(' ',strip=True)
        if not title or 'Shop on eBay' in title:continue
        price,currency=parse_price(p.get_text(' ',strip=True))
        if price<=0:continue
        img=item.select_one('img.s-item__image-img'); image_url=str((img.get('src') or img.get('data-src') or '') if img else '')
        out.append({'title':title,'price':price,'currency':currency,'source':'eBay','sourceUrl':str(a.get('href') or ''),'buyUrl':str(a.get('href') or ''),'imageUrl':image_url,'status':'sold' if sold else 'active','kind':'completed sale' if sold else 'listing'})
        if len(out)>=limit:break
    return out

def exact_market(items:list[dict[str,Any]],row:dict[str,Any])->list[dict[str,Any]]:
    name=toks(str(row.get('player') or row.get('name') or '')); num=norm(str(row.get('card_number') or '')).replace(' ',''); insert=toks(str(row.get('insert_name') or '')); parallel=toks(str(row.get('parallel') or ''))
    out=[]
    for x in items:
        text=norm(str(x.get('title') or '')); words=text.split(); compact=text.replace(' ','')
        name_ok=not name or sum(1 for n in name if n in words)>=max(1,math.ceil(len(name)*.67)); num_ok=not num or num in compact
        insert_ok=not insert or any(v in words for v in insert); parallel_ok=not parallel or any(v in words for v in parallel) or not str(row.get('parallel') or '').strip()
        if name_ok and num_ok and insert_ok and parallel_ok:out.append(x)
    return out

@app.get('/health')
def health():
    return {'ok':True,'service':'carddistrict-visual-search','version':APP_VERSION,'engine':'512D layout+color+edge+HOG+DCT / pHash+dHash / ORB','supabase':bool(SUPABASE_URL and SUPABASE_KEY)}

@app.post('/describe')
def describe(req:DescribeRequest):
    img=decode_image(req.image); card=rectify_card(img); v=descriptor512(card); p,d=hashes(card)
    return {'embedding':v.tolist(),'phash':p,'dhash':d,'dimension':512,'engine':'carddistrict-visual-descriptor-v1'}

@app.get('/describe-url')
def describe_url(url:str=Query(...,min_length=8)):
    img=download_image(url); card=rectify_card(img); v=descriptor512(card); p,d=hashes(card)
    return {'embedding':v.tolist(),'phash':p,'dhash':d,'dimension':512,'engine':'carddistrict-visual-descriptor-v1','url':url}

@app.get('/discover-ebay')
def discover_ebay(query:str=Query(...,min_length=2),sold:bool=False,limit:int=Query(40,ge=1,le=120)):
    return {'items':search_ebay(query,sold=sold,limit=limit),'query':query,'sold':sold}

@app.post('/recognize')
def recognize(req:RecognizeRequest):
    scan=decode_image(req.front); card=rectify_card(scan); v=descriptor512(card); p,d=hashes(card)
    rows=match_vector(v,max(18,req.max_candidates*2),req.category_key); ranked=rank_rows(card,rows,p,d)
    best=ranked[0] if ranked else None; second=ranked[1] if len(ranked)>1 else None
    score=float(best.get('match_score') or 0) if best else 0.0; margin=score-(float(second.get('match_score') or 0) if second else 0.0)
    structural=max(float(best.get('phash_similarity') or 0),float(best.get('orb_similarity') or 0)) if best else 0.0
    verified=bool(best and score>=.76 and margin>=.018 and structural>=.18)
    candidates=[candidate_from_row(x) | {'visualScore':round(float(x.get('match_score') or 0)*100,1)} for x in ranked[:req.max_candidates]]
    if not verified:
        return {'identity':None,'verified':False,'card':None,'candidates':candidates,'items':[],'sales':[],'checkedAt':now_iso(),'warning':'Noch kein sicherer visueller Treffer. CardDistrict vergleicht ausschließlich das Kartenbild mit dem eigenen Referenzindex.','engine':{'name':'CardDistrict Visual Search AI v2','score':round(score*100,1),'margin':round(margin*100,1),'indexedCandidates':len(rows),'phash':p,'dhash':d}}
    identity=identity_from_row(best,min(.999,max(.80,score))); resolved=candidate_from_row(best); items=[]
    if req.include_market:
        q=identity['searchQuery']; items=exact_market(search_ebay(q,False,100)+search_ebay(q,True,100),best)
    sales=[x for x in items if x.get('status')=='sold']
    return {'identity':identity,'verified':True,'card':resolved,'candidates':[],'items':items,'sales':sales,'checkedAt':now_iso(),'warning':f'Visueller Treffer {score*100:.1f}% · Abstand {margin*100:.1f}%. Markt wird erst nach diesem Match freigegeben.','engine':{'name':'CardDistrict Visual Search AI v2','score':round(score*100,1),'margin':round(margin*100,1),'indexedCandidates':len(rows),'vector':round(float(best.get('vector_similarity') or 0)*100,1),'phash':round(float(best.get('phash_similarity') or 0)*100,1),'orb':round(float(best.get('orb_similarity') or 0)*100,1)}}
