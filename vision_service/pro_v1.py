from __future__ import annotations
import base64, io, os, time
from pathlib import Path
from typing import Any
import cv2, imagehash, numpy as np, onnxruntime as ort, requests
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

app=FastAPI(title='CardDistrict Recognition Engine',version='1.0.0')
MODEL_PATH=Path(__file__).resolve().parent/'models'/'dinov2-small-int8.onnx'
SUPABASE_URL=os.getenv('SUPABASE_URL','').rstrip('/')
SUPABASE_KEY=os.getenv('SUPABASE_PUBLISHABLE_KEY','')
_session:ort.InferenceSession|None=None

class ImagePayload(BaseModel):
    data:str
    mimeType:str='image/jpeg'
class ShortlistRequest(BaseModel):
    front:ImagePayload
    category_key:str|None=None
    max_candidates:int=Field(default=12,ge=1,le=50)
class VerifyRequest(BaseModel):
    front:ImagePayload
    candidate:dict[str,Any]
class FingerprintUrlRequest(BaseModel):
    url:str

def now_ms():return int(time.time()*1000)
def decode(p:ImagePayload)->np.ndarray:
    raw=p.data.split(',',1)[-1]
    try:b=base64.b64decode(raw,validate=False)
    except Exception as e:raise HTTPException(400,f'invalid base64: {e}')
    arr=np.frombuffer(b,np.uint8);im=cv2.imdecode(arr,cv2.IMREAD_COLOR)
    if im is None:raise HTTPException(400,'invalid image')
    return im

def order_quad(pts:np.ndarray)->np.ndarray:
    p=pts.reshape(4,2).astype(np.float32);s=p.sum(1);d=np.diff(p,axis=1).reshape(-1)
    return np.array([p[np.argmin(s)],p[np.argmin(d)],p[np.argmax(s)],p[np.argmax(d)]],np.float32)

def rectify(im:np.ndarray)->np.ndarray:
    h,w=im.shape[:2];scale=min(1.0,1400/max(h,w))
    work=cv2.resize(im,(round(w*scale),round(h*scale))) if scale<1 else im.copy()
    gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY);gray=cv2.GaussianBlur(gray,(5,5),0)
    edges=cv2.Canny(gray,50,150);edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8),iterations=2)
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    area_img=work.shape[0]*work.shape[1];best=None;best_score=-1.0
    for c in sorted(contours,key=cv2.contourArea,reverse=True)[:80]:
        area=cv2.contourArea(c)
        if area<area_img*.025 or area>area_img*.98:continue
        peri=cv2.arcLength(c,True);poly=cv2.approxPolyDP(c,.02*peri,True)
        if len(poly)!=4 or not cv2.isContourConvex(poly):continue
        q=order_quad(poly);widths=[np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])];heights=[np.linalg.norm(q[3]-q[0]),np.linalg.norm(q[2]-q[1])]
        cw=max(widths);ch=max(heights)
        if min(cw,ch)<50:continue
        ratio=min(cw,ch)/max(cw,ch);ratio_score=max(0,1-abs(ratio-.714)/.28)
        score=(area/area_img)*2.2+ratio_score
        if score>best_score:best_score=score;best=q
    if best is None:
        # Conservative center crop fallback; keeps the whole central card-sized region.
        hh,ww=work.shape[:2];crop_h=int(hh*.78);crop_w=min(int(crop_h*.714),int(ww*.82));x=max(0,(ww-crop_w)//2);y=max(0,(hh-crop_h)//2)
        out=work[y:y+crop_h,x:x+crop_w]
    else:
        q=best;portrait=max(np.linalg.norm(q[3]-q[0]),np.linalg.norm(q[2]-q[1]))>=max(np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3]))
        ow,oh=(700,980) if portrait else (980,700);dst=np.array([[0,0],[ow-1,0],[ow-1,oh-1],[0,oh-1]],np.float32);m=cv2.getPerspectiveTransform(q,dst);out=cv2.warpPerspective(work,m,(ow,oh),flags=cv2.INTER_CUBIC)
    if out.shape[1]>out.shape[0]:out=cv2.rotate(out,cv2.ROTATE_90_CLOCKWISE)
    return out

def session()->ort.InferenceSession:
    global _session
    if _session is None:
        if not MODEL_PATH.exists():raise RuntimeError(f'model missing: {MODEL_PATH}')
        opts=ort.SessionOptions();opts.intra_op_num_threads=max(1,min(2,os.cpu_count() or 1));opts.inter_op_num_threads=1;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session=ort.InferenceSession(str(MODEL_PATH),sess_options=opts,providers=['CPUExecutionProvider'])
    return _session

def embedding(card:np.ndarray)->np.ndarray:
    rgb=cv2.cvtColor(card,cv2.COLOR_BGR2RGB);rgb=cv2.resize(rgb,(224,224),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    mean=np.array([.485,.456,.406],np.float32);std=np.array([.229,.224,.225],np.float32);x=((rgb-mean)/std).transpose(2,0,1)[None]
    s=session();inp=s.get_inputs()[0].name;outs=s.run(None,{inp:x})
    z=np.asarray(outs[0],dtype=np.float32)
    if z.ndim==3:z=z[:,0,:]
    elif z.ndim>2:z=z.reshape(z.shape[0],-1)
    v=z[0];v/=max(float(np.linalg.norm(v)),1e-8)
    if v.shape[0]!=384:raise RuntimeError(f'unexpected embedding dim {v.shape}')
    return v

def hashes(card:np.ndarray)->tuple[str,str]:
    pil=Image.fromarray(cv2.cvtColor(card,cv2.COLOR_BGR2RGB));return str(imagehash.phash(pil,hash_size=16)),str(imagehash.dhash(pil,hash_size=16))
def hsim(a:str,b:str)->float:
    if not a or not b or len(a)!=len(b):return 0.0
    try:return 1-(int(a,16)^int(b,16)).bit_count()/(len(a)*4)
    except:return 0.0

def orb_sim(a:np.ndarray,b:np.ndarray)->float:
    ga=cv2.cvtColor(cv2.resize(a,(420,588)),cv2.COLOR_BGR2GRAY);gb=cv2.cvtColor(cv2.resize(b,(420,588)),cv2.COLOR_BGR2GRAY);orb=cv2.ORB_create(nfeatures=1400,fastThreshold=9)
    ka,da=orb.detectAndCompute(ga,None);kb,db=orb.detectAndCompute(gb,None)
    if da is None or db is None or len(ka)<10 or len(kb)<10:return 0.0
    ms=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2);good=[m for m,n in ms if m.distance<.72*n.distance]
    return min(1.0,len(good)/max(22,min(len(ka),len(kb))*.18))

def download_image(url:str)->np.ndarray:
    r=requests.get(url,timeout=(4,12),headers={'User-Agent':'CardDistrictRecognition/1.0'});r.raise_for_status();arr=np.frombuffer(r.content,np.uint8);im=cv2.imdecode(arr,cv2.IMREAD_COLOR)
    if im is None:raise ValueError('invalid reference image')
    return rectify(im)

def rpc_match(v:np.ndarray,count:int,category:str|None)->list[dict[str,Any]]:
    if not SUPABASE_URL or not SUPABASE_KEY:raise RuntimeError('Supabase configuration missing')
    headers={'apikey':SUPABASE_KEY,'Authorization':f'Bearer {SUPABASE_KEY}','Content-Type':'application/json'}
    body={'query_embedding':v.tolist(),'match_count':count,'category_filter':category}
    r=requests.post(f'{SUPABASE_URL}/rest/v1/rpc/cd_match_dinov2_cards',headers=headers,json=body,timeout=(4,12));r.raise_for_status();return r.json()

def public_card(row:dict[str,Any],score:float|None=None)->dict[str,Any]:
    s=float(row.get('similarity') if score is None else score or 0)
    return {'id':str(row.get('id') or ''),'name':str(row.get('player') or row.get('name') or ''),'team':str(row.get('team') or ''),'brand':str(row.get('brand') or ''),'year':str(row.get('year') or ''),'setName':str(row.get('set_name') or ''),'cardNumber':str(row.get('card_number') or ''),'insertName':str(row.get('insert_name') or ''),'parallel':str(row.get('parallel') or ''),'rookie':bool(row.get('rookie')),'categoryKey':str(row.get('category_key') or ''),'referenceImageUrl':str(row.get('reference_image_url') or ''),'sourceUrl':str(row.get('source_url') or ''),'visualScore':round(s*100,2),'verificationPayload':row}

@app.get('/health')
def health():
    return {'ok':True,'engine':'CardDistrict DINOv2 Recognition v1','modelLoaded':_session is not None,'modelExists':MODEL_PATH.exists(),'supabaseConfigured':bool(SUPABASE_URL and SUPABASE_KEY),'time':now_ms()}

@app.post('/fingerprint-url')
def fingerprint_url(req:FingerprintUrlRequest):
    try:card=download_image(req.url);v=embedding(card);p,d=hashes(card);return {'embedding':v.tolist(),'phash':p,'dhash':d,'model':'dinov2-small-int8-v1','time':now_ms()}
    except Exception as e:raise HTTPException(502,str(e))

@app.post('/shortlist')
def shortlist(req:ShortlistRequest):
    t=time.time();card=rectify(decode(req.front));v=embedding(card);p,d=hashes(card);rows=rpc_match(v,req.max_candidates,req.category_key)
    return {'verified':False,'candidates':[public_card(dict(r)) for r in rows],'engine':{'name':'CardDistrict DINOv2 v1','ms':round((time.time()-t)*1000),'phash':p,'dhash':d,'count':len(rows)},'warning':'' if rows else 'DINOv2 reference index has no candidate yet.'}

@app.post('/verify')
def verify(req:VerifyRequest):
    row=dict(req.candidate.get('verificationPayload') or req.candidate or {});url=str(row.get('reference_image_url') or req.candidate.get('referenceImageUrl') or '')
    if not url:return {'verified':False,'warning':'reference image missing','card':public_card(row,0)}
    t=time.time();scan=rectify(decode(req.front));sv=embedding(scan);sp,sd=hashes(scan)
    try:ref=download_image(url)
    except Exception as e:return {'verified':False,'warning':f'reference unavailable: {e}','card':public_card(row,0)}
    rv=embedding(ref);rp,rd=hashes(ref);cos=float(np.dot(sv,rv));ps=hsim(sp,rp);ds=hsim(sd,rd);osim=orb_sim(scan,ref)
    score=.68*max(0,cos)+.12*ps+.05*ds+.15*osim
    verified=bool(cos>=.80 and score>=.78 and (ps>=.46 or osim>=.12))
    return {'verified':verified,'card':public_card(row,score),'identity':public_card(row,score) if verified else None,'engine':{'name':'CardDistrict DINOv2 Exact Verify v1','score':round(score*100,2),'dinov2':round(cos*100,2),'phash':round(ps*100,2),'dhash':round(ds*100,2),'orb':round(osim*100,2),'ms':round((time.time()-t)*1000)},'warning':'' if verified else 'Candidate is visually similar but not exact enough.'}
