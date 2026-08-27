from __future__ import annotations
import base64, os, time
from pathlib import Path
from typing import Any
import cv2, imagehash, numpy as np, onnxruntime as ort, requests
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

app=FastAPI(title='CardDistrict Recognition Engine',version='1.1.0')
MODEL_PATH=Path(__file__).resolve().parent/'models'/'dinov2-small-int8.onnx'
MODEL_ID='dinov2-small-int8-v1.1'
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
    h,w=im.shape[:2];scale=min(1.0,1600/max(h,w));work=cv2.resize(im,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA) if scale<1 else im.copy()
    gray=cv2.GaussianBlur(cv2.cvtColor(work,cv2.COLOR_BGR2GRAY),(5,5),0);edges=cv2.Canny(gray,42,138);edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8),iterations=2)
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);area_img=work.shape[0]*work.shape[1];best=None;best_score=-1.0;cx,cy=work.shape[1]/2,work.shape[0]/2
    for c in sorted(contours,key=cv2.contourArea,reverse=True)[:100]:
        area=cv2.contourArea(c)
        if area<area_img*.018 or area>area_img*.97:continue
        peri=cv2.arcLength(c,True);poly=cv2.approxPolyDP(c,.021*peri,True)
        if len(poly)!=4 or not cv2.isContourConvex(poly):continue
        q=order_quad(poly);widths=[np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])];heights=[np.linalg.norm(q[3]-q[0]),np.linalg.norm(q[2]-q[1])];cw=max(widths);ch=max(heights)
        if min(cw,ch)<45:continue
        ratio=min(cw,ch)/max(cw,ch);ratio_score=max(0.0,1-abs(ratio-.714)/.22);qc=q.mean(0);center_dist=np.linalg.norm(qc-np.array([cx,cy]))/max(work.shape[:2]);center_score=max(0.0,1-center_dist*1.8);rect=cv2.minAreaRect(c);box_area=max(1.0,rect[1][0]*rect[1][1]);fill=min(1.0,area/box_area)
        score=(area/area_img)*1.8+ratio_score*1.25+center_score*.45+fill*.35
        if score>best_score:best_score=score;best=q
    if best is None:
        hh,ww=work.shape[:2];crop_h=int(hh*.82);crop_w=min(int(crop_h*.714),int(ww*.88));x=max(0,(ww-crop_w)//2);y=max(0,(hh-crop_h)//2);out=work[y:y+crop_h,x:x+crop_w]
    else:
        q=best;portrait=max(np.linalg.norm(q[3]-q[0]),np.linalg.norm(q[2]-q[1]))>=max(np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3]));ow,oh=(700,980) if portrait else (980,700);dst=np.array([[0,0],[ow-1,0],[ow-1,oh-1],[0,oh-1]],np.float32);out=cv2.warpPerspective(work,cv2.getPerspectiveTransform(q,dst),(ow,oh),flags=cv2.INTER_CUBIC)
    if out.shape[1]>out.shape[0]:out=cv2.rotate(out,cv2.ROTATE_90_CLOCKWISE)
    return out

def session()->ort.InferenceSession:
    global _session
    if _session is None:
        if not MODEL_PATH.exists():raise RuntimeError(f'model missing: {MODEL_PATH}')
        opts=ort.SessionOptions();opts.intra_op_num_threads=max(1,min(2,os.cpu_count() or 1));opts.inter_op_num_threads=1;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session=ort.InferenceSession(str(MODEL_PATH),sess_options=opts,providers=['CPUExecutionProvider'])
    return _session

def embed_one(card:np.ndarray)->np.ndarray:
    rgb=cv2.cvtColor(card,cv2.COLOR_BGR2RGB);rgb=cv2.resize(rgb,(224,224),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0;mean=np.array([.485,.456,.406],np.float32);std=np.array([.229,.224,.225],np.float32);x=((rgb-mean)/std).transpose(2,0,1)[None]
    s=session();z=np.asarray(s.run(None,{s.get_inputs()[0].name:x})[0],dtype=np.float32)
    if z.ndim==3:z=z[:,0,:]
    elif z.ndim>2:z=z.reshape(z.shape[0],-1)
    v=z[0];v/=max(float(np.linalg.norm(v)),1e-8)
    if v.shape[0]!=384:raise RuntimeError(f'unexpected embedding dim {v.shape}')
    return v

def normalize_luma(card:np.ndarray)->np.ndarray:
    lab=cv2.cvtColor(card,cv2.COLOR_BGR2LAB);l,a,b=cv2.split(lab);l=cv2.createCLAHE(clipLimit=1.6,tileGridSize=(8,8)).apply(l);return cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)

def robust_embedding(card:np.ndarray)->np.ndarray:
    v1=embed_one(card);v2=embed_one(normalize_luma(card));v=v1+v2;return v/max(float(np.linalg.norm(v)),1e-8)

def hashes(card:np.ndarray)->tuple[str,str]:
    pil=Image.fromarray(cv2.cvtColor(card,cv2.COLOR_BGR2RGB));return str(imagehash.phash(pil,hash_size=16)),str(imagehash.dhash(pil,hash_size=16))
def hsim(a:str,b:str)->float:
    if not a or not b or len(a)!=len(b):return 0.0
    try:return 1-(int(a,16)^int(b,16)).bit_count()/(len(a)*4)
    except:return 0.0

def feature_geometry(a:np.ndarray,b:np.ndarray)->tuple[float,float,int]:
    ga=cv2.cvtColor(cv2.resize(a,(420,588)),cv2.COLOR_BGR2GRAY);gb=cv2.cvtColor(cv2.resize(b,(420,588)),cv2.COLOR_BGR2GRAY);orb=cv2.ORB_create(nfeatures=1800,fastThreshold=8,edgeThreshold=12)
    ka,da=orb.detectAndCompute(ga,None);kb,db=orb.detectAndCompute(gb,None)
    if da is None or db is None or len(ka)<10 or len(kb)<10:return 0.0,0.0,0
    raw=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2);good=[m for pair in raw if len(pair)==2 for m,n in [pair] if m.distance<.74*n.distance];orb_score=min(1.0,len(good)/max(20,min(len(ka),len(kb))*.16))
    if len(good)<8:return orb_score,0.0,len(good)
    src=np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1,1,2);dst=np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1,1,2);_,mask=cv2.findHomography(src,dst,cv2.RANSAC,4.0);geom=float(mask.mean()) if mask is not None else 0.0
    return orb_score,geom,len(good)

def download_image(url:str)->np.ndarray:
    r=requests.get(url,timeout=(4,14),headers={'User-Agent':'CardDistrictRecognition/1.1'});r.raise_for_status();arr=np.frombuffer(r.content,np.uint8);im=cv2.imdecode(arr,cv2.IMREAD_COLOR)
    if im is None:raise ValueError('invalid reference image')
    return rectify(im)

def rpc_match(v:np.ndarray,count:int,category:str|None)->list[dict[str,Any]]:
    if not SUPABASE_URL or not SUPABASE_KEY:raise RuntimeError('Supabase configuration missing')
    headers={'apikey':SUPABASE_KEY,'Authorization':f'Bearer {SUPABASE_KEY}','Content-Type':'application/json'};body={'query_embedding':v.tolist(),'match_count':count,'category_filter':category}
    r=requests.post(f'{SUPABASE_URL}/rest/v1/rpc/cd_match_dinov2_cards',headers=headers,json=body,timeout=(4,14));r.raise_for_status();return r.json()

def public_card(row:dict[str,Any],score:float|None=None)->dict[str,Any]:
    s=float(row.get('similarity') if score is None else score or 0)
    return {'id':str(row.get('id') or ''),'name':str(row.get('player') or row.get('name') or ''),'team':str(row.get('team') or ''),'brand':str(row.get('brand') or ''),'year':str(row.get('year') or ''),'setName':str(row.get('set_name') or ''),'cardNumber':str(row.get('card_number') or ''),'insertName':str(row.get('insert_name') or ''),'parallel':str(row.get('parallel') or ''),'rookie':bool(row.get('rookie')),'categoryKey':str(row.get('category_key') or ''),'referenceImageUrl':str(row.get('reference_image_url') or ''),'sourceUrl':str(row.get('source_url') or ''),'visualScore':round(s*100,2),'verificationPayload':row}

@app.on_event('startup')
def warm_model():
    s=session();dummy=np.zeros((1,3,224,224),np.float32);s.run(None,{s.get_inputs()[0].name:dummy})

@app.get('/health')
def health():return {'ok':True,'engine':'CardDistrict DINOv2 Recognition v1.1','model':MODEL_ID,'modelLoaded':_session is not None,'modelExists':MODEL_PATH.exists(),'supabaseConfigured':bool(SUPABASE_URL and SUPABASE_KEY),'time':now_ms()}

@app.post('/fingerprint-url')
def fingerprint_url(req:FingerprintUrlRequest):
    try:card=download_image(req.url);v=robust_embedding(card);p,d=hashes(card);return {'embedding':v.tolist(),'phash':p,'dhash':d,'model':MODEL_ID,'time':now_ms()}
    except Exception as e:raise HTTPException(502,str(e))

@app.post('/shortlist')
def shortlist(req:ShortlistRequest):
    t=time.time();card=rectify(decode(req.front));v=robust_embedding(card);p,d=hashes(card);rows=rpc_match(v,req.max_candidates,req.category_key);top1=float(rows[0].get('similarity') or 0) if rows else 0.0;top2=float(rows[1].get('similarity') or 0) if len(rows)>1 else 0.0;margin=top1-top2
    enriched=[]
    for i,r in enumerate(rows):rr=dict(r);rr['_retrieval_rank']=i+1;rr['_retrieval_margin']=margin if i==0 else 0.0;enriched.append(public_card(rr))
    ready=bool(rows and top1>=.62 and (len(rows)==1 or margin>=.012))
    return {'verified':False,'candidates':enriched,'decision':{'state':'candidate_ready' if ready else 'needs_more_evidence','top1':round(top1*100,2),'top2':round(top2*100,2),'margin':round(margin*100,2),'autoVerifyEligible':ready},'engine':{'name':'CardDistrict DINOv2 retrieval v1.1','model':MODEL_ID,'ms':round((time.time()-t)*1000),'phash':p,'dhash':d,'count':len(rows)},'warning':'' if rows else 'DINOv2 reference index has no candidate yet.'}

@app.post('/verify')
def verify(req:VerifyRequest):
    row=dict(req.candidate.get('verificationPayload') or req.candidate or {});url=str(row.get('reference_image_url') or req.candidate.get('referenceImageUrl') or '')
    if not url:return {'verified':False,'warning':'reference image missing','card':public_card(row,0)}
    t=time.time();scan=rectify(decode(req.front));sv=robust_embedding(scan);sp,sd=hashes(scan)
    try:ref=download_image(url)
    except Exception as e:return {'verified':False,'warning':f'reference unavailable: {e}','card':public_card(row,0)}
    rv=robust_embedding(ref);rp,rd=hashes(ref);cos=float(np.dot(sv,rv));ps=hsim(sp,rp);ds=hsim(sd,rd);osim,geom,good=feature_geometry(scan,ref);retrieval=float(row.get('similarity') or 0);margin=float(row.get('_retrieval_margin') or 0)
    score=.58*max(0,cos)+.11*ps+.04*ds+.12*osim+.15*geom;structural=bool(geom>=.18 or ps>=.52 or osim>=.22);retrieval_ok=bool(retrieval>=.60 or cos>=.91);ambiguity_ok=bool(margin>=.012 or retrieval>=.82 or (cos>=.91 and geom>=.18));verified=bool(cos>=.84 and score>=.80 and structural and retrieval_ok and ambiguity_ok)
    reasons=[]
    if cos<.84:reasons.append('deep_visual_similarity_low')
    if score<.80:reasons.append('combined_exact_score_low')
    if not structural:reasons.append('structural_evidence_low')
    if not retrieval_ok:reasons.append('retrieval_confidence_low')
    if not ambiguity_ok:reasons.append('top_candidate_not_separated')
    return {'verified':verified,'card':public_card(row,score),'identity':public_card(row,score) if verified else None,'decision':{'state':'exact_match' if verified else 'abstain','reasons':reasons,'failClosed':True},'engine':{'name':'CardDistrict DINOv2 Exact Verify v1.1','model':MODEL_ID,'score':round(score*100,2),'dinov2':round(cos*100,2),'retrieval':round(retrieval*100,2),'retrievalMargin':round(margin*100,2),'phash':round(ps*100,2),'dhash':round(ds*100,2),'orb':round(osim*100,2),'geometry':round(geom*100,2),'featureMatches':good,'ms':round((time.time()-t)*1000)},'warning':'' if verified else 'Not enough independent evidence for an exact match.'}
