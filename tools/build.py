#!/usr/bin/env python3
"""assets/ を元ファイルから作り直す。

index.html を file:// で直接開いても通信なしで動くよう、エンジンとモデルを
Base64 にして <script> で読める .js に包む。内容そのものは変更しない。

使い方:
    python3 tools/build.py <元ファイルを置いたフォルダ>

フォルダに置くもの(入手先は README.md の「assets の作り直し」を参照):
    onnxruntime-web-1.22.0.tgz
    ch_PP-OCRv5_det_mobile.onnx
    ch_PP-OCRv5_rec_mobile.onnx
    ppocrv5_dict.txt
"""
import base64
import hashlib
import pathlib
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / 'assets'

SHA256 = {
    'ch_PP-OCRv5_det_mobile.onnx': '4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae',
    'ch_PP-OCRv5_rec_mobile.onnx': '5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5',
}
ORT_TGZ = 'onnxruntime-web-1.22.0.tgz'
ORT_FILES = {
    'ort_js': 'package/dist/ort.wasm.min.js',
    'ort_mjs': 'package/dist/ort-wasm-simd-threaded.mjs',
    'ort_wasm': 'package/dist/ort-wasm-simd-threaded.wasm',
}


def wrap(name, data):
    """data を assets/<name>.js に書き出す。"""
    b64 = base64.b64encode(data).decode()
    (ASSETS / f'{name}.js').write_text(
        f'(window.OCR_ASSETS=window.OCR_ASSETS||{{}})["{name}"]="{b64}";\n')
    print(f'  assets/{name}.js  ({len(data):,} bytes)')


def jp_mask(dict_bytes):
    """認識モデルの出力番号ごとに、日本語(CP932)の文字かどうかの 0/1 を並べる。

    0 番は CTC の空白ラベル、末尾は半角スペース(src/ai-worker.js と同じ並び)。
    """
    text = dict_bytes.decode('utf-8').replace('\r', '')
    if text.endswith('\n'):
        text = text[:-1]
    chars = [''] + text.split('\n') + [' ']

    def ok(c):
        if c in ('', ' '):
            return True
        try:
            c.encode('cp932')
            return True
        except UnicodeEncodeError:
            return False

    return bytes(1 if ok(c) else 0 for c in chars)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    vendor = pathlib.Path(sys.argv[1])
    ASSETS.mkdir(exist_ok=True)

    with tarfile.open(vendor / ORT_TGZ) as tar:
        for name, member in ORT_FILES.items():
            wrap(name, tar.extractfile(member).read())

    for name, filename in (('det', 'ch_PP-OCRv5_det_mobile.onnx'), ('rec', 'ch_PP-OCRv5_rec_mobile.onnx')):
        data = (vendor / filename).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != SHA256[filename]:
            sys.exit(f'{filename}: SHA-256 が想定と違います ({digest})')
        wrap(name, data)

    dict_bytes = (vendor / 'ppocrv5_dict.txt').read_bytes()
    wrap('dict', dict_bytes)
    wrap('jpmask', jp_mask(dict_bytes))
    wrap('ai_worker', (ROOT / 'src' / 'ai-worker.js').read_bytes())
    print('done')


if __name__ == '__main__':
    main()
