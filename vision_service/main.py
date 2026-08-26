from __future__ import annotations

import base64
import os
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import cv2
import imagehash
import numpy as np
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:
    RapidOCR = None

APP_VERSION = '1.0.0'
SUPABASE_URL = os.getenv('SUPABASE_URL', '').rstrip('/')
SUPABASE_KEY = os.getenv('SUPABASE_PUBLISHABLE_KEY', '') or os.getenv('SUPABASE_ANON_KEY', '')
REQUEST_TIMEOUT = float(os.getenv('HTTP_TIMEOUT', '8'))
MAX_MARKET_ITEMS = int(os.getenv('MAX_MARKET_ITEMS', '80'))
UA = 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1 CardDistrictVision/1.0'
SESSION = requests.Session()
SESSION.headers.update({'user-agent': UA, 'accept-language': 'de-DE,de;q=0.9,en;q=0.8', 'accept': 'text/html,application/xhtml+xml,application/json'})
OCR_ENGINE: Any = None

GENERIC = {'topps','chrome','rookie','rc','card','cards','trading','edition','series','collection','uefa','club','competitions','soccer','football','fc','the','and','stage','ultimate','base','insert','parallel','holo','refractor','bayern','munchen','muenchen','munich'}
TEAM_MARKERS = {'fc','cf','ac','afc','sc','club','team'}

app = FastAPI(title='CardDistrict Vision API', version=APP_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['GET','POST','OPTIONS'], allow_headers=['*'])

class ImagePayload(BaseModel):
    data: str
    mimeType: str = 'image/jpeg'

class RecognizeRequest(BaseModel):
    front: ImagePayload
    max_candidates: int = Field(default=8, ge=1, le=12)
    include_market: bool = True

@dataclass
class Candidate:
    name: str
    player: str = ''
    team: str = ''
    brand: str = ''
    year: str = ''
    setName: str = ''
    cardNumber: str = ''
    insertName: str = ''
    parallel: str = ''
    rarity: str = ''
    rookie: bool = False
    imageUrl: str = ''
    source: str = ''
    sourceUrl: str = ''
    title: str = ''
    textScore: float = 0.0
    visualScore: float = 0.0
    support: int = 0
    matchScore: float = 0.0
    marketValue: float = 0.0
    marketCurrency: str = ''

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def norm(value: str) -> str:
    value = value.lower().replace('ü','u').replace('ö','o').replace('ä','a').replace('ß','ss')
    return re.sub(r'[^a-z0-9]+',' ',value).strip()

def compact(value: str) -> str:
    return norm(value).replace(' ','')

def toks(value: str, keep_generic: bool=False) -> list[str]:
    values = [x for x in norm(value).split() if len(x)>1]
    return values if keep_generic else [x for x in values if x not in GENERIC]

def unique(values: list[str]) -> list[str]:
    seen:set[str]=set(); out:list[str]=[]
    for value in values:
        k=norm(value)
        if k and k not in seen:
            seen.add(k); out.append(value.strip())
    return out

def decode_image(payload: ImagePayload) -> np.ndarray:
    try:
        raw=base64.b64decode(payload.data.split(',',1)[-1],validate=False)
    except Exception as exc:
        raise HTTPException(400,'Ungültige Bilddaten') from exc
    img=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
    if img is None or img.size==0:
        raise HTTPException(400,'Bild konnte nicht gelesen werden')
    return img

def order_quad(pts: np.ndarray) -> np.ndarray:
    pts=pts.astype(np.float32); s=pts.sum(axis=1); d=np.diff(pts,axis=1).reshape(-1)
    return np.array([pts[np.argmin(s)],pts[np.argmin(d)],pts[np.argmax(s)],pts[np.argmax(d)]],dtype=np.float32)

def rectify_card(img: np.ndarray) -> np.ndarray:
    h,w=img.shape[:2]; scale=min(1.0,1200.0/max(h,w)); work=cv2.resize(img,(max(1,int(w*scale)),max(1,int(h*scale)))) if scale<1 else img.copy()
    gray=cv2.GaussianBlur(cv2.cvtColor(work,cv2.COLOR_BGR2GRAY),(5,5),0); edges=cv2.dilate(cv2.Canny(gray,50,150),np.ones((3,3),np.uint8),iterations=1)
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE); total=work.shape[0]*work.shape[1]; quad=None; best=0.0
    for c in sorted(contours,key=cv2.contourArea,reverse=True)[:25]:
        area=cv2.contourArea(c)
        if area<total*.18: continue
        approx=cv2.approxPolyDP(c,.025*cv2.arcLength(c,True),True)
        if len(approx)==4 and area>best: best=area; quad=approx.reshape(4,2)
    if quad is not None:
        q=order_quad(quad); tl,tr,br,bl=q; mw=int(max(np.linalg.norm(br-bl),np.linalg.norm(tr-tl))); mh=int(max(np.linalg.norm(tr-br),np.linalg.norm(tl-bl)))
        if mw>180 and mh>250:
            dst=np.array([[0,0],[mw-1,0],[mw-1,mh-1],[0,mh-1]],dtype=np.float32); work=cv2.warpPerspective(work,cv2.getPerspectiveTransform(q,dst),(mw,mh))
    if work.shape[1]>work.shape[0]: work=cv2.rotate(work,cv2.ROTATE_90_CLOCKWISE)
    hh,ww=work.shape[:2]
    if ww>900:
        k=900/ww; work=cv2.resize(work,(900,max(1,int(hh*k))),interpolation=cv2.INTER_AREA)
    return work

def scan_views(card: np.ndarray) -> list[tuple[str,np.ndarray]]:
    h=card.shape[0]
    return [('full',card),('top',card[:max(1,int(h*.40)),:]),('bottom',card[max(0,int(h*.58)):,:]),('middle',card[max(0,int(h*.18)):max(1,int(h*.82)),:])]

def get_ocr_engine() -> Any:
    global OCR_ENGINE
    if OCR_ENGINE is None and RapidOCR is not None: OCR_ENGINE=RapidOCR()
    return OCR_ENGINE

def run_ocr(card: np.ndarray) -> tuple[list[dict[str,Any]],str]:
    engine=get_ocr_engine()
    if engine is None: return [],''
    rows:list[dict[str,Any]]=[]
    for region,image in scan_views(card):
        try: result,_=engine(image)
        except Exception: result=None
        for item in result or []:
            if len(item)<3: continue
            text=str(item[1]).strip()
            try: conf=float(item[2])
            except Exception: conf=0.0
            if text and conf>=.35: rows.append({'region':region,'text':text,'confidence':round(conf,4)})
    best:dict[str,dict[str,Any]]={}
    for row in rows:
        k=norm(row['text'])
        if k and (k not in best or row['confidence']>best[k]['confidence']): best[k]=row
    lines=sorted(best.values(),key=lambda x:(0 if x['region']=='bottom' else 1 if x['region']=='top' else 2,-x['confidence']))
    return lines,' | '.join(x['text'] for x in lines)

def player_from_ocr(lines: list[dict[str,Any]]) -> str:
    choices:list[tuple[float,str]]=[]
    for row in lines:
        text=re.sub(r"[^A-Za-zÀ-ÿ .'-]+",' ',row['text']).strip(); parts=[p for p in text.split() if len(p)>1]
        if not 2<=len(parts)<=8: continue
        cut=len(parts)
        for i,p in enumerate(parts):
            if norm(p) in TEAM_MARKERS: cut=i; break
        parts=parts[:cut]
        if not 2<=len(parts)<=4: continue
        ns=[norm(p) for p in parts]
        if sum(1 for p in ns if p in GENERIC)>=max(1,len(parts)-1): continue
        score=float(row['confidence'])+(.35 if row['region']=='bottom' else 0)+min(.3,len(parts)*.05); choices.append((score,' '.join(parts)))
    return max(choices,default=(0,''),key=lambda x:x[0])[1]

def insert_from_ocr(lines: list[dict[str,Any]]) -> str:
    joined=norm(' '.join(x['text'] for x in lines))
    for value in ['ultimate stage','regency chrome','trophy chasers','roots','prizm','select','optic','mosaic']:
        if value in joined: return ' '.join(w.capitalize() for w in value.split())
    for row in lines:
        if row['region']=='bottom':
            parts=toks(row['text'],True)
            if 1<=len(parts)<=4 and any(p in GENERIC for p in parts): return row['text'].strip()
    return ''

def brand_from_text(text: str) -> str:
    n=norm(text)
    if 'topps chrome' in n: return 'Topps Chrome'
    if 'topps' in n: return 'Topps'
    if 'panini prizm' in n or 'prizm' in n: return 'Panini Prizm'
    if 'panini' in n: return 'Panini'
    if 'upper deck' in n: return 'Upper Deck'
    return ''

def query_signatures(lines: list[dict[str,Any]],ocr_text: str) -> list[str]:
    player=player_from_ocr(lines); insert=insert_from_ocr(lines); brand=brand_from_text(ocr_text); bottom=' '.join(x['text'] for x in lines if x['region']=='bottom')
    anchors=[x for x in toks(ocr_text,True) if len(x)>=4 and x not in {'munchen','munich'}]
    sigs=[' '.join(x for x in [player,insert,brand] if x),' '.join(x for x in [player,insert] if x),' '.join(x for x in [player,brand] if x),bottom,' '.join(unique(anchors)[:9])]
    return [x[:160] for x in unique([re.sub(r'\s+',' ',x).strip() for x in sigs]) if len(x)>=3][:5]

def sb_reference_candidates(ocr_text: str) -> list[Candidate]:
    if not(SUPABASE_URL and SUPABASE_KEY): return []
    anchors=sorted(unique([x for x in toks(ocr_text) if len(x)>=4]),key=len,reverse=True)[:5]; found:dict[str,Candidate]={}
    headers={'apikey':SUPABASE_KEY,'authorization':f'Bearer {SUPABASE_KEY}','accept':'application/json'}
    for anchor in anchors:
        safe=re.sub(r'[^a-zA-Z0-9À-ÿ-]+','',anchor)[:50]
        if len(safe)<3: continue
        params={'select':'id,category_key,name,player,team,brand,year,set_name,card_number,insert_name,parallel,rookie,reference_image_url,source_url','or':f'(player.ilike.*{safe}*,name.ilike.*{safe}*,set_name.ilike.*{safe}*,insert_name.ilike.*{safe}*,card_number.ilike.*{safe}*)','limit':'20'}
        try:
            r=SESSION.get(f'{SUPABASE_URL}/rest/v1/cd_vision_cards',params=params,headers=headers,timeout=REQUEST_TIMEOUT)
            if r.ok:
                for x in r.json():
                    c=Candidate(name=str(x.get('name') or x.get('player') or ''),player=str(x.get('player') or x.get('name') or ''),team=str(x.get('team') or ''),brand=str(x.get('brand') or ''),year=str(x.get('year') or ''),setName=str(x.get('set_name') or ''),cardNumber=str(x.get('card_number') or ''),insertName=str(x.get('insert_name') or ''),parallel=str(x.get('parallel') or ''),rookie=bool(x.get('rookie')),imageUrl=str(x.get('reference_image_url') or ''),source='CardDistrict Reference',sourceUrl=str(x.get('source_url') or ''),title=' '.join(str(x.get(k) or '') for k in ['year','brand','set_name','player','card_number','insert_name','parallel']))
                    found[str(x.get('id') or compact(c.title))]=c
        except Exception: pass
        if len(found)>=20: break
    return list(found.values())

def parse_price(text: str) -> tuple[float,str]:
    raw=text.replace('\xa0',' ').strip(); currency='EUR' if '€' in raw or re.search(r'\bEUR\b',raw,re.I) else 'GBP' if '£' in raw else 'USD' if '$' in raw or re.search(r'\bUSD\b',raw,re.I) else ''
    m=re.search(r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)',raw)
    if not m: return 0.0,currency
    n=m.group(1)
    if ',' in n and '.' in n: n=n.replace('.','').replace(',','.') if n.rfind(',')>n.rfind('.') else n.replace(',','')
    elif ',' in n: n=n.replace(',','.') if len(n.rsplit(',',1)[-1])<=2 else n.replace(',','')
    try: return float(n),currency
    except ValueError: return 0.0,currency

def extract_card_number(text: str) -> str:
    upper=text.upper()
    for p in [r'#\s*([A-Z]{1,6}-\d{1,4})\b',r'\b([A-Z]{1,6}-\d{1,4})\b',r'#\s*([A-Z]{1,4}\d{1,4})\b']:
        m=re.search(p,upper)
        if m:return m.group(1)
    return ''

def extract_year(text: str) -> str:
    m=re.search(r'\b(20\d{2})(?:[-/](\d{2}))?\b',text); return m.group(1) if m else ''

def search_ebay(query: str,sold: bool=False,limit: int=40) -> list[dict[str,Any]]:
    params={'_nkw':query,'_ipg':'120'}
    if sold: params.update({'LH_Sold':'1','LH_Complete':'1'})
    try:r=SESSION.get('https://www.ebay.de/sch/i.html',params=params,timeout=REQUEST_TIMEOUT)
    except Exception:return []
    if not r.ok:return []
    soup=BeautifulSoup(r.text,'html.parser');out=[]
    for item in soup.select('li.s-item'):
        t=item.select_one('.s-item__title');a=item.select_one('a.s-item__link');p=item.select_one('.s-item__price')
        if not t or not a or not p:continue
        title=t.get_text(' ',strip=True)
        if not title or 'Shop on eBay' in title:continue
        price,currency=parse_price(p.get_text(' ',strip=True))
        if price<=0:continue
        img=item.select_one('img.s-item__image-img'); image_url=str((img.get('src') or img.get('data-src') or '') if img else '')
        meta=item.get_text(' ',strip=True);date='';dm=re.search(r'(\d{1,2}\.\s*[A-Za-zÄÖÜäöü]{3,12}\.?\s*20\d{2})',meta)
        if dm:date=dm.group(1)
        out.append({'title':title,'price':price,'currency':currency or 'EUR','date':date,'source':'eBay','sourceUrl':str(a.get('href') or ''),'buyUrl':str(a.get('href') or ''),'status':'sold' if sold else 'active','kind':'Verkauft' if sold else 'Live Listing','imageUrl':image_url})
        if len(out)>=limit:break
    return out

def candidate_from_market(row: dict[str,Any],lines: list[dict[str,Any]],ocr_text: str) -> Candidate:
    title=str(row.get('title') or '');player=player_from_ocr(lines);insert=insert_from_ocr(lines)
    return Candidate(name=player or title[:90],player=player,brand=brand_from_text(f'{ocr_text} {title}'),year=extract_year(title),setName=title,cardNumber=extract_card_number(title),insertName=insert,rookie=bool(re.search(r'\bRC\b|rookie',title,re.I)),imageUrl=str(row.get('imageUrl') or ''),source=str(row.get('source') or 'eBay'),sourceUrl=str(row.get('sourceUrl') or ''),title=title)

def pil_for_hash(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

def fetch_image(url: str) -> np.ndarray|None:
    if not re.match(r'^https?://',url or '',re.I):return None
    try:
        r=SESSION.get(url,timeout=REQUEST_TIMEOUT,stream=True)
        if not r.ok:return None
        img=cv2.imdecode(np.frombuffer(r.content[:8_000_000],dtype=np.uint8),cv2.IMREAD_COLOR)
        return rectify_card(img) if img is not None else None
    except Exception:return None

def visual_similarity(scan: np.ndarray,ref: np.ndarray|None) -> float:
    if ref is None or scan.size==0 or ref.size==0:return 0.0
    a=cv2.resize(scan,(384,540),interpolation=cv2.INTER_AREA);b=cv2.resize(ref,(384,540),interpolation=cv2.INTER_AREA)
    try:
        ph=1-(imagehash.phash(pil_for_hash(a))-imagehash.phash(pil_for_hash(b)))/64;dh=1-(imagehash.dhash(pil_for_hash(a))-imagehash.dhash(pil_for_hash(b)))/64
    except Exception:ph=dh=0.0
    ga=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY);gb=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY);orb=cv2.ORB_create(nfeatures=1400,scaleFactor=1.2,nlevels=8);ka,da=orb.detectAndCompute(ga,None);kb,db=orb.detectAndCompute(gb,None);orb_score=0.0
    if da is not None and db is not None and len(ka or [])>=8 and len(kb or [])>=8:
        try:
            matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2);good=[m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance];orb_score=min(1.0,len(good)/55)
        except Exception:pass
    ha=cv2.calcHist([a],[0,1],None,[32,32],[0,256,0,256]);hb=cv2.calcHist([b],[0,1],None,[32,32],[0,256,0,256]);cv2.normalize(ha,ha);cv2.normalize(hb,hb);hist=max(0.0,min(1.0,1-cv2.compareHist(ha,hb,cv2.HISTCMP_BHATTACHARYYA)))
    return round(max(0.0,min(1.0,.33*ph+.12*dh+.38*orb_score+.17*hist)),4)

def text_similarity(ocr_text: str,c: Candidate) -> float:
    o=set(toks(ocr_text,True));ct=set(toks(' '.join([c.name,c.player,c.brand,c.year,c.setName,c.cardNumber,c.insertName,c.parallel,c.title]),True))
    if not o or not ct:return 0.0
    weights={x:(.35 if x in GENERIC else 1+min(.6,len(x)/15)) for x in o};base=sum(weights[x] for x in o if x in ct)/(sum(weights.values()) or 1);boost=0.0
    if c.player and norm(c.player) in norm(ocr_text):boost+=.22
    if c.insertName and norm(c.insertName) in norm(ocr_text):boost+=.14
    if c.brand and norm(c.brand) in norm(ocr_text):boost+=.08
    if c.cardNumber and compact(c.cardNumber) in compact(ocr_text):boost+=.18
    return round(min(1.0,base+boost),4)

def same_identity(a: Candidate,b: Candidate) -> bool:
    if a.cardNumber and b.cardNumber and compact(a.cardNumber)!=compact(b.cardNumber):return False
    ap=toks(a.player or a.name);bp=toks(b.player or b.name)
    return not(ap and bp and len(set(ap)&set(bp))<max(1,min(len(ap),len(bp))-1))

def rank_candidates(scan: np.ndarray,ocr_text: str,candidates: list[Candidate]) -> list[Candidate]:
    dedup:dict[str,Candidate]={}
    for c in candidates:
        k='|'.join([compact(c.player or c.name),compact(c.cardNumber),compact(c.insertName),compact(c.parallel),compact(c.sourceUrl)])
        if k not in dedup:dedup[k]=c
    items=list(dedup.values())[:35]
    for c in items[:20]:
        c.textScore=text_similarity(ocr_text,c);c.visualScore=visual_similarity(scan,fetch_image(c.imageUrl)) if c.imageUrl else 0.0
    for c in items:
        c.support=sum(1 for x in items if x is not c and same_identity(c,x));source_bonus=.06 if c.source=='CardDistrict Reference' else 0;support_bonus=min(.12,c.support*.035);c.matchScore=round(min(1.0,.57*c.textScore+.37*c.visualScore+source_bonus+support_bonus),4)
    return sorted(items,key=lambda c:(c.matchScore,c.visualScore,c.textScore),reverse=True)

def verified_candidate(ranked: list[Candidate]) -> Candidate|None:
    if not ranked:return None
    top=ranked[0];second=ranked[1] if len(ranked)>1 else None;margin=top.matchScore-(second.matchScore if second else 0)
    ref_rule=top.source=='CardDistrict Reference' and bool(top.cardNumber) and top.textScore>=.42 and top.visualScore>=.34 and top.matchScore>=.52
    market_rule=bool(top.cardNumber) and top.textScore>=.48 and top.visualScore>=.30 and top.matchScore>=.55 and (top.support>=1 or margin>=.08)
    visual_rule=bool(top.cardNumber) and top.visualScore>=.64 and top.textScore>=.32 and top.matchScore>=.58
    return top if ref_rule or market_rule or visual_rule else None

def exact_market_match(row: dict[str,Any],card: Candidate) -> bool:
    title=str(row.get('title') or '');words=norm(title).split();pt=toks(card.player or card.name)
    if pt and len([x for x in pt if x in words])<max(1,int(np.ceil(len(pt)*.67))):return False
    if card.cardNumber and compact(card.cardNumber) not in compact(title):return False
    ins=toks(card.insertName,True)
    if ins and len([x for x in ins if x in words])<max(1,int(np.ceil(len(ins)*.5))):return False
    return True

def priceguide_from_source(card: Candidate) -> tuple[float,str]:
    if not card.sourceUrl:return 0.0,''
    try:
        r=SESSION.get(card.sourceUrl,timeout=REQUEST_TIMEOUT)
        if r.ok:
            text=BeautifulSoup(r.text,'html.parser').get_text(' ',strip=True);m=re.search(r'\bRAW\b.{0,40}?([$€£]\s*[\d,.]+)',text,re.I)
            if m:return parse_price(m.group(1))
    except Exception:pass
    return 0.0,''

def market_for(card: Candidate) -> tuple[list[dict[str,Any]],float,str]:
    q=' '.join(x for x in [card.year,card.brand,card.player or card.name,card.cardNumber and f'#{card.cardNumber}',card.insertName,card.parallel] if x)
    active=[x for x in search_ebay(q,False,60) if exact_market_match(x,card)];sold=[x for x in search_ebay(q,True,60) if exact_market_match(x,card)];seen:set[str]=set();out=[]
    for x in active+sold:
        k=f"{x['status']}|{compact(x['title'])}|{x['price']}|{x['sourceUrl']}"
        if k not in seen:seen.add(k);out.append(x)
        if len(out)>=MAX_MARKET_ITEMS:break
    es=[x['price'] for x in sold if x.get('currency')=='EUR' and x.get('price',0)>0];ea=[x['price'] for x in active if x.get('currency')=='EUR' and x.get('price',0)>0]
    value=statistics.median(es) if es else round(statistics.median(ea)*.85,2) if ea else 0.0;currency='EUR' if value else ''
    if not value:value,currency=priceguide_from_source(card)
    return out,float(value or 0),currency

def candidate_ui(c: Candidate) -> dict[str,Any]:
    return {'name':c.player or c.name,'setName':c.setName or c.insertName,'cardNumber':c.cardNumber,'rarity':c.rarity,'variant':c.parallel or c.insertName,'releaseDate':c.year,'source':c.source,'sourceUrl':c.sourceUrl,'imageUrl':c.imageUrl,'marketValue':c.marketValue,'marketCurrency':c.marketCurrency,'matchScore':c.matchScore,'textScore':c.textScore,'visualScore':c.visualScore}

def identity_payload(card: Candidate|None,ocr_text: str,confidence: float,method: str) -> dict[str,Any]:
    c=card or Candidate(name='');category='soccer' if any(x in norm(' '.join([c.team,c.setName,ocr_text])) for x in ['bayern','uefa','soccer','football']) else 'unknown'
    return {'categoryType':'sport' if category=='soccer' else 'unknown','categoryKey':category,'categoryLabel':'Fußball' if category=='soccer' else 'Unbekannt','subject':c.player or c.name,'game':'Fußball' if category=='soccer' else '','manufacturer':c.brand,'year':c.year,'setName':c.setName,'cardNumber':c.cardNumber,'parallel':c.parallel or c.insertName,'rarity':c.rarity,'edition':'','serial':'','gradingCompany':'','grade':'','recognitionConfidence':round(max(0,min(1,confidence)),4),'needsReview':card is None,'searchQuery':' '.join(x for x in [c.year,c.brand,c.setName,c.player or c.name,c.cardNumber,c.insertName,c.parallel] if x),'recognitionMethod':method,'scanText':ocr_text[:1800]}

@app.get('/health')
def health() -> dict[str,Any]:
    return {'ok':True,'service':'carddistrict-vision','version':APP_VERSION,'ocr':RapidOCR is not None,'supabaseReferences':bool(SUPABASE_URL and SUPABASE_KEY),'persistence':'reference-only; scan images are not stored'}

@app.post('/recognize')
def recognize(req: RecognizeRequest) -> dict[str,Any]:
    started=time.time();card_img=rectify_card(decode_image(req.front));lines,ocr_text=run_ocr(card_img);sigs=query_signatures(lines,ocr_text);refs=sb_reference_candidates(ocr_text);market_candidates:list[Candidate]=[]
    for q in sigs[:3]:
        market_candidates.extend(candidate_from_market(row,lines,ocr_text) for row in search_ebay(q,False,24))
        if len(market_candidates)>=45:break
    ranked=rank_candidates(card_img,ocr_text,refs+market_candidates);top=verified_candidate(ranked)
    if top is None:
        conf=ranked[0].matchScore if ranked else 0;candidates=[candidate_ui(c) for c in ranked[:req.max_candidates] if c.textScore>=.2 or c.visualScore>=.2]
        return {'identity':identity_payload(None,ocr_text,conf,'carddistrict-vision-v1-unverified'),'verified':False,'card':None,'candidates':candidates,'items':[],'sales':[],'checkedAt':now_iso(),'warning':'CardDistrict Vision hat Merkmale gelesen, aber noch keinen sicheren Bild+Text-Match. Es werden bewusst noch keine Preise freigegeben.','verification':{'engine':'CardDistrict Vision v1','ocrLines':lines[:24],'queries':sigs,'candidateCount':len(ranked),'topScore':round(conf,4),'elapsedMs':int((time.time()-started)*1000)}}
    market=[];value=0.0;currency=''
    if req.include_market:market,value,currency=market_for(top)
    top.marketValue=value;top.marketCurrency=currency;identity=identity_payload(top,ocr_text,max(top.matchScore,.9 if top.source=='CardDistrict Reference' else top.matchScore),'carddistrict-vision-v1-exact');sold=[x for x in market if x.get('status')=='sold'];active=[x for x in market if x.get('status')=='active']
    return {'identity':identity,'verified':True,'card':candidate_ui(top),'candidates':[],'items':market,'sales':sold,'checkedAt':now_iso(),'warning':f'Exakter Match: {round(top.matchScore*100)}%. {len(active)} aktuelle eBay-Listings und {len(sold)} abgeschlossene eBay-Treffer gefunden.','verification':{'engine':'CardDistrict Vision v1','ocrLines':lines[:24],'queries':sigs,'textScore':top.textScore,'visualScore':top.visualScore,'support':top.support,'matchScore':top.matchScore,'source':top.source,'elapsedMs':int((time.time()-started)*1000)}}
