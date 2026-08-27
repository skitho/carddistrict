from pathlib import Path
import hashlib
import requests

URL='https://huggingface.co/onnx-community/dinov2-small-ONNX/resolve/main/onnx/model_quantized.onnx?download=true'
TARGET=Path(__file__).resolve().parent/'models'/'dinov2-small-int8.onnx'
EXPECTED_MIN=20_000_000

def main():
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    if TARGET.exists() and TARGET.stat().st_size>=EXPECTED_MIN:
        print('DINOv2 model already present',TARGET.stat().st_size)
        return
    tmp=TARGET.with_suffix('.tmp')
    with requests.get(URL,stream=True,timeout=120) as r:
        r.raise_for_status()
        with tmp.open('wb') as f:
            for chunk in r.iter_content(1024*1024):
                if chunk:f.write(chunk)
    if tmp.stat().st_size<EXPECTED_MIN:
        raise RuntimeError(f'model download too small: {tmp.stat().st_size}')
    tmp.replace(TARGET)
    h=hashlib.sha256(TARGET.read_bytes()).hexdigest()
    print('Downloaded DINOv2 model',TARGET.stat().st_size,h)

if __name__=='__main__':main()
