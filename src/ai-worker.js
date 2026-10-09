/* AI-OCR ワーカー: PP-OCRv5 (文字位置の検出 + 文字認識) を ONNX Runtime Web で実行する。
   このファイルの前に ort.wasm.min.js が連結されて 1 つの Blob として起動される。通信は行わない。 */
'use strict';
var det = null, rec = null, chars = null, allow = {}, same = null;

self.onmessage = async function (e) {
  var m = e.data;
  try {
    if (m.type === 'init') {
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      ort.env.wasm.wasmPaths = {
        mjs: URL.createObjectURL(new Blob([m.mjs], { type: 'text/javascript' })),
        wasm: URL.createObjectURL(new Blob([m.wasm], { type: 'application/wasm' }))
      };
      var so = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
      det = await ort.InferenceSession.create(m.det, so);
      rec = await ort.InferenceSession.create(m.rec, so);
      // 0 番は CTC の空白ラベル、末尾は半角スペース
      chars = [''].concat(m.dict.replace(/\r/g, '').replace(/\n$/, '').split('\n'), [' ']);
      // 言語ごとに候補文字を絞る(日本語: JIS の文字のみ。中国語の簡体字への取り違えを防ぐ)
      var jp = [], en = [];
      for (var i = 1; i < chars.length; i++) {
        if (m.jpmask[i]) jp.push(i);
        var cc = chars[i].length === 1 ? chars[i].charCodeAt(0) : 0;
        if ((cc >= 32 && cc < 127) || (chars[i] && '£¥€°'.indexOf(chars[i]) >= 0)) en.push(i);
      }
      allow = { jp: Int32Array.from(jp), en: Int32Array.from(en) };
      [allow.jp, allow.en].forEach(function (l) { l.ok = new Uint8Array(chars.length); l.forEach(function (i) { l.ok[i] = 1; }); });
      // 全角と半角の同じ文字(% と ％ など)は同一視し、出力は半角にそろえる
      same = new Int32Array(chars.length); var first = {};
      for (var k = 0; k < chars.length; k++) {
        var z = chars[k];
        if (z.length === 1 && z >= '\uff01' && z <= '\uff5e') { z = String.fromCharCode(z.charCodeAt(0) - 0xFEE0); chars[k] = z; }
        if (z && first[z] === undefined) first[z] = k;
        same[k] = z ? first[z] : k;
      }
      self.postMessage({ type: 'ready', id: m.id });
    } else if (m.type === 'run') {
      var r = await run(m);
      self.postMessage({ type: 'done', id: m.id, result: r });
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: String((err && err.message) || err) });
  }
};

// ---- 文字位置の検出。傾きに合わせた矩形(中心 cx,cy・幅 w・高さ h・傾き a)を返す ----
async function detect(bmp, maxSide, unclip) {
  var W = bmp.width, H = bmp.height;
  var sc = Math.min(1, maxSide / Math.max(W, H));
  var dw = Math.max(32, Math.round(W * sc / 32) * 32), dh = Math.max(32, Math.round(H * sc / 32) * 32);
  var cv = new OffscreenCanvas(dw, dh), g = cv.getContext('2d', { willReadFrequently: true });
  g.fillStyle = '#fff'; g.fillRect(0, 0, dw, dh);
  g.drawImage(bmp, 0, 0, dw, dh);
  var d = g.getImageData(0, 0, dw, dh).data, n = dw * dh, x = new Float32Array(3 * n);
  for (var i = 0, j = 0; i < n; i++, j += 4) {      // BGR の順
    x[i] = (d[j + 2] / 255 - 0.485) / 0.229;
    x[n + i] = (d[j + 1] / 255 - 0.456) / 0.224;
    x[2 * n + i] = (d[j] / 255 - 0.406) / 0.225;
  }
  var feeds = {}; feeds[det.inputNames[0]] = new ort.Tensor('float32', x, [1, 3, dh, dw]);
  var out = (await det.run(feeds))[det.outputNames[0]];
  var p = out.data, oh = out.dims[2], ow = out.dims[3];

  // しきい値で二値化し、つながった領域ごとに画素を集める
  var seen = new Uint8Array(ow * oh), q = new Int32Array(ow * oh), tail = 0, comps = [];
  for (var s = 0; s < ow * oh; s++) {
    if (seen[s] || p[s] <= 0.3) continue;
    var start = tail, head = tail, sum = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    q[tail++] = s; seen[s] = 1;
    while (head < tail) {
      var k0 = q[head++], qx = k0 % ow, qy = (k0 - qx) / ow;
      sum += p[k0]; sx += qx; sy += qy; sxx += qx * qx; syy += qy * qy; sxy += qx * qy;
      for (var dy = -1; dy <= 1; dy++) for (var dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        var nx = qx + dx, ny = qy + dy;
        if (nx < 0 || ny < 0 || nx >= ow || ny >= oh) continue;
        var k = ny * ow + nx;
        if (!seen[k] && p[k] > 0.3) { seen[k] = 1; q[tail++] = k; }
      }
    }
    var cnt = tail - start;
    if (cnt < 9 || sum / cnt < 0.5) continue;
    // 主軸の向き(細長い領域だけ信頼する)
    var mx = sx / cnt, my = sy / cnt, a = sxx / cnt - mx * mx, c = syy / cnt - my * my, b = sxy / cnt - mx * my;
    var root = Math.sqrt((a - c) * (a - c) / 4 + b * b), l1 = (a + c) / 2 + root, l2 = Math.max(1e-6, (a + c) / 2 - root);
    var ang = null;
    if (cnt > 60 && l1 / l2 > 16) {
      ang = 0.5 * Math.atan2(2 * b, a - c);
      if (ang > Math.PI / 4) ang -= Math.PI / 2; else if (ang < -Math.PI / 4) ang += Math.PI / 2;   // 縦長の行は「直立からの傾き」に直す
    }
    comps.push({ start: start, end: tail, ang: ang });
  }
  // 全体の傾き(細長い領域の中央値)。短い領域にはこれを使う
  var angs = comps.filter(function (c) { return c.ang !== null; }).map(function (c) { return c.ang; }).sort(function (a, b) { return a - b; });
  var skew = angs.length ? angs[angs.length >> 1] : 0;
  var fx = W / ow, fy = H / oh, f = (fx + fy) / 2, boxes = [];
  comps.forEach(function (cp) {
    var A = cp.ang === null || Math.abs(cp.ang - skew) > 0.26 ? skew : cp.ang, co = Math.cos(A), si = Math.sin(A);
    var u0 = 1e9, u1 = -1e9, v0 = 1e9, v1 = -1e9;
    for (var i = cp.start; i < cp.end; i++) {
      var k = q[i], px = k % ow, py = (k - px) / ow, u = px * co + py * si, v = -px * si + py * co;
      if (u < u0) u0 = u; if (u > u1) u1 = u; if (v < v0) v0 = v; if (v > v1) v1 = v;
    }
    var bw = u1 - u0 + 1, bh = v1 - v0 + 1;
    if (Math.min(bw, bh) < 3) return;
    var pad = bw * bh * unclip / (2 * (bw + bh));       // 検出領域は文字より一回り小さいので広げる
    var uc = (u0 + u1 + 1) / 2, vc = (v0 + v1 + 1) / 2;
    boxes.push({ cx: (uc * co - vc * si) * fx, cy: (uc * si + vc * co) * fy, w: (bw + 2 * pad) * f, h: (bh + 2 * pad) * f, a: A, pad: pad * f });
  });
  return { boxes: boxes, skew: skew, scale: 1 / f };
}

// ---- 1 領域の文字認識 ----
async function recognize(bmp, b, rotate, idx) {
  var Hh = 48, ratio = rotate ? b.h / b.w : b.w / b.h;
  var tw = Math.max(8, Math.min(3200, Math.round(Hh * ratio)));
  var cv = new OffscreenCanvas(tw, Hh), g = cv.getContext('2d', { willReadFrequently: true });
  g.fillStyle = '#fff'; g.fillRect(0, 0, tw, Hh);
  g.imageSmoothingQuality = 'high';
  // 傾いた矩形をまっすぐに切り出す。縦長の行はさらに左に 90 度倒して読む
  g.translate(tw / 2, Hh / 2);
  if (rotate) { g.scale(tw / b.h, Hh / b.w); g.rotate(-Math.PI / 2); } else g.scale(tw / b.w, Hh / b.h);
  g.rotate(-b.a); g.translate(-b.cx, -b.cy);
  var r = Math.ceil(Math.hypot(b.w, b.h) / 2) + 2;
  var sx = Math.max(0, Math.floor(b.cx - r)), sy = Math.max(0, Math.floor(b.cy - r));
  var sw = Math.min(bmp.width, Math.ceil(b.cx + r)) - sx, sh = Math.min(bmp.height, Math.ceil(b.cy + r)) - sy;
  if (sw < 1 || sh < 1) return { text: '', conf: 0 };
  g.drawImage(bmp, sx, sy, sw, sh, sx, sy, sw, sh);
  g.setTransform(1, 0, 0, 1, 0, 0);
  var d = g.getImageData(0, 0, tw, Hh).data;
  var Wt = Math.max(tw, 320), n = Wt * Hh, x = new Float32Array(3 * n);   // 右側は 0 で埋める
  for (var yy = 0; yy < Hh; yy++) for (var xx = 0; xx < tw; xx++) {
    var j = (yy * tw + xx) * 4, i = yy * Wt + xx;
    x[i] = d[j + 2] / 127.5 - 1; x[n + i] = d[j + 1] / 127.5 - 1; x[2 * n + i] = d[j] / 127.5 - 1;
  }
  var feeds = {}; feeds[rec.inputNames[0]] = new ort.Tensor('float32', x, [1, 3, Hh, Wt]);
  var out = (await rec.run(feeds))[rec.outputNames[0]];
  var p = out.data, T = out.dims[1], C = out.dims[2], text = '', sum = 0, cnt = 0, prev = 0;
  for (var t = 0; t < T; t++) {
    var o = t * C, best = 0, bv = p[o];
    for (var c = 1; c < C; c++) if (p[o + c] > bv) { bv = p[o + c]; best = c; }
    if (best !== 0 && idx && !idx.ok[best]) {          // 対象外の文字が最有力なら、対象内の文字で最も近いものに置き換える
      best = idx[0]; bv = p[o + best];
      for (var a = 1; a < idx.length; a++) { var c2 = idx[a]; if (p[o + c2] > bv) { bv = p[o + c2]; best = c2; } }
    }
    best = same[best];
    if (best !== 0 && best !== prev) { text += chars[best] || ''; sum += bv; cnt++; }
    prev = best;
  }
  return { text: text, conf: cnt ? sum / cnt : 0 };
}

// 余白の取り方を変えて 2 回読み、良いほうを採る(ぼけた小さい文字で 1 文字抜ける誤りを減らす)
async function recognize2(bmp, b, rotate, idx) {
  var r1 = await recognize(bmp, b, rotate, idx);
  var wide = { cx: b.cx, cy: b.cy, a: b.a, w: b.w + 0.5 * b.pad, h: b.h + 0.5 * b.pad };
  var r2 = await recognize(bmp, wide, rotate, idx);
  var ok1 = r1.conf >= 0.5, ok2 = r2.conf >= 0.5;
  if (ok1 !== ok2) return ok1 ? r1 : r2;
  var n1 = r1.text.replace(/\s/g, '').length, n2 = r2.text.replace(/\s/g, '').length;
  if (n1 !== n2) return n1 > n2 ? r1 : r2;
  return r1.conf >= r2.conf ? r1 : r2;
}

// ---- 読む順に並べる(全体の傾きを打ち消した座標で比べる)----
function group(items, vertical, skew) {
  // 横書き: 縦位置の近いものを同じ行にまとめ、行内は左から右。縦書き: 列ごとに右から左、列内は上から下
  var co = Math.cos(skew), si = Math.sin(skew);
  items.forEach(function (b) { b.u = b.cx * co + b.cy * si; b.v = -b.cx * si + b.cy * co; });
  var c = vertical ? function (b) { return b.u; } : function (b) { return b.v; };
  var t = vertical ? function (b) { return b.w; } : function (b) { return b.h; };
  var s = vertical ? function (b) { return b.v; } : function (b) { return b.u; };
  items.sort(function (a, b) { return c(a) - c(b); });
  var rows = [];
  items.forEach(function (it) {
    var r = rows[rows.length - 1];
    if (r && Math.abs(c(it) - r.c) < 0.5 * Math.min(t(it), r.t)) {
      r.items.push(it); r.c = (r.c * (r.items.length - 1) + c(it)) / r.items.length; r.t = Math.max(r.t, t(it));
    } else rows.push({ c: c(it), t: t(it), items: [it] });
  });
  if (vertical) rows.reverse();
  return rows.map(function (r) {
    return r.items.sort(function (a, b) { return s(a) - s(b); }).map(function (i) { return i.text; });
  });
}

// 縦書きで、列のすぐ下に離れて検出された句読点などの小さな領域を、その列につなげる
function mergeTail(boxes) {
  var cols = boxes.filter(function (b) { return b.h >= 1.5 * b.w; });
  return boxes.filter(function (s) {
    if (s.h >= 1.5 * s.w) return true;
    for (var i = 0; i < cols.length; i++) {
      var c = cols[i], top = c.cy - c.h / 2, gap = (s.cy - s.h / 2) - (c.cy + c.h / 2);
      if (Math.abs(s.cx - c.cx) < 0.5 * c.w && gap > -0.5 * c.w && gap < c.w) {
        var bottom = Math.max(c.cy + c.h / 2, s.cy + s.h / 2);
        c.h = bottom - top; c.cy = top + c.h / 2;
        return false;
      }
    }
    return true;
  });
}

// 検出時の縮尺での文字の高さ(文字列の短い辺、余白を除く)の中央値
var SMALL = 12;
function textHeight(dres) {
  var hs = dres.boxes.map(function (b) { return (Math.min(b.w, b.h) - 2 * b.pad) * dres.scale; }).sort(function (a, b) { return a - b; });
  return hs.length ? hs[hs.length >> 1] : 0;
}

async function run(m) {
  var bmp = m.bitmap, t0 = Date.now();
  var dres = await detect(bmp, 1600, 1.6), zoomed = false;
  // 文字がごく小さいときは、大きくしてから検出・認識し直す(小さすぎると字形の情報が足りない)
  var th = textHeight(dres);
  if (dres.boxes.length && th < SMALL) {
    var side = Math.max(bmp.width, bmp.height);
    if (dres.scale < 0.99) { dres = await detect(bmp, Math.min(side, 3200), 1.6); zoomed = true; }
    else if (side * 2 <= 4000) {
      var big = new OffscreenCanvas(bmp.width * 2, bmp.height * 2), bg = big.getContext('2d');
      bg.imageSmoothingQuality = 'high'; bg.fillStyle = '#fff'; bg.fillRect(0, 0, big.width, big.height);
      bg.drawImage(bmp, 0, 0, big.width, big.height);
      if (bmp.close) bmp.close();
      bmp = big; dres = await detect(bmp, 3200, 1.6); zoomed = true;
    }
  }
  var boxes = dres.boxes, items = [], idx = allow[m.charset] || null;
  // 縦書きか横書きかの自動判定: 縦長の文字列と横長の文字列の長さの合計を比べる
  var vertical = m.vertical;
  if (vertical === 'auto') {
    var tall = 0, wide = 0;
    boxes.forEach(function (b) { if (b.h >= 2 * b.w) tall += b.h; else if (b.w >= 2 * b.h) wide += b.w; });
    vertical = tall > wide;
  }
  if (vertical) boxes = mergeTail(boxes);
  for (var i = 0; i < boxes.length; i++) {
    var b = boxes[i], r;
    if (b.h >= 1.5 * b.w) {
      r = await recognize2(bmp, b, true, idx);
      if (b.h < 3 * b.w) {                           // 縦長でも「1」のような横書き 1 文字のことがあるので両方試す
        var r2 = await recognize2(bmp, b, false, idx);
        if (r2.conf > r.conf) r = r2;
      }
    } else r = await recognize2(bmp, b, false, idx);
    b.text = r.text.trim(); b.conf = r.conf;
    if (b.text && b.conf >= 0.5) items.push(b);
    self.postMessage({ type: 'progress', id: m.id, p: (i + 1) / boxes.length });
  }
  if (bmp.close) bmp.close();
  var lines = group(items, vertical, dres.skew);
  return {
    lines: lines,
    raw: lines.map(function (c) { return c.join(vertical ? '' : '  '); }).join('\n'), vertical: vertical,
    boxes: boxes.length, textHeight: th, zoomed: zoomed, ms: Date.now() - t0
  };
}
