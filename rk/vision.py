"""Person mask and face boxes with OpenCV DNN only (MediaPipe selfie-segmentation and BlazeFace TFLite models).

No mediapipe / torch / onnx package is needed: cv2.dnn.readNetFromTFLite (OpenCV 4.8 or newer) runs both models.
"""
import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(os.path.dirname(HERE), 'assets', 'models')


def load_tflite(name):
    path = os.path.join(MODELS, name)
    if not hasattr(cv2, 'dnn') or not hasattr(cv2.dnn, 'readNetFromTFLite'):
        raise RuntimeError('this OpenCV (%s) cannot read TFLite models; run: pip install -U opencv-python-headless' % cv2.__version__)
    if hasattr(cv2.dnn, 'ENGINE_CLASSIC'):
        try:
            return cv2.dnn.readNetFromTFLite(path, engine=cv2.dnn.ENGINE_CLASSIC)
        except TypeError:
            pass
    return cv2.dnn.readNetFromTFLite(path)


def box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided(I, p, r, eps):
    mI, mp = box(I, r), box(p, r)
    cIp, cII = box(I * p, r), box(I * I, r)
    a = (cIp - mI * mp) / (cII - mI * mI + eps)
    b = mp - a * mI
    return box(a, r) * I + box(b, r)


class Segmenter:
    """Soft person alpha for a frame. Vertical/square footage → 256×256 model, landscape → 256×144 model."""

    def __init__(self, landscape=False):
        self.landscape = landscape
        self.net = load_tflite('selfie_segmentation_landscape.tflite' if landscape else 'selfie_segmentation.tflite')
        self.size = (256, 144) if landscape else (256, 256)
        self.prev = None

    def raw(self, rgb):
        blob = cv2.dnn.blobFromImage(rgb, 1 / 255.0, self.size, swapRB=False, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        return np.clip(out.reshape(self.size[1], self.size[0]).astype(np.float32), 0, 1)

    def alpha(self, rgb, temporal=True, work=540):
        """Full-resolution alpha (float32 0..1) with an edge refined on the frame (guided filter) and smoothed in time."""
        H, W = rgb.shape[:2]
        m = self.raw(rgb)
        s = work / float(min(H, W))
        ww, hh = max(64, int(round(W * s))), max(64, int(round(H * s)))
        small = cv2.resize(rgb, (ww, hh), interpolation=cv2.INTER_AREA)
        I = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        p = cv2.resize(m, (ww, hh), interpolation=cv2.INTER_LINEAR)
        q = guided(I, p, max(2, int(work / 90)), 2e-3)
        q = guided(I, np.clip(q, 0, 1), max(1, int(work / 270)), 1e-4)
        a = np.clip((q - 0.5) * 2.2 + 0.5, 0, 1)
        if temporal and self.prev is not None and self.prev.shape == a.shape:
            a = 0.68 * a + 0.32 * self.prev
        self.prev = a
        a = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
        return np.clip((a - 0.04) / 0.92, 0, 1)

    def reset(self):
        self.prev = None


def _anchors():
    out = []
    strides = [8, 16, 16, 16]
    i = 0
    while i < 4:
        s, n = strides[i], 0
        while i < 4 and strides[i] == s:
            n += 2
            i += 1
        g = 128 // s
        for y in range(g):
            for x in range(g):
                for _ in range(n):
                    out.append(((x + .5) / g, (y + .5) / g))
    return np.array(out, np.float32)


class FaceDetector:
    """BlazeFace (short range). detect(rgb) → {'box': (x0, y0, x1, y1), 'eyes', 'nose', 'mouth', 'score'} in pixels."""

    def __init__(self):
        self.net = load_tflite('face_detection_short_range.tflite')
        self.anchors = _anchors()
        names = list(self.net.getUnconnectedOutLayersNames())
        self.multi = len(names) >= 2

    def detect(self, rgb, thr=0.55):
        H, W = rgb.shape[:2]
        S = max(H, W)
        sq = np.zeros((S, S, 3), np.uint8)
        ox, oy = (S - W) // 2, (S - H) // 2
        sq[oy:oy + H, ox:ox + W] = rgb
        blob = cv2.dnn.blobFromImage(sq, 1 / 127.5, (128, 128), mean=(127.5, 127.5, 127.5), swapRB=False)
        self.net.setInput(blob)
        if self.multi:
            outs = self.net.forward(['regressors', 'classificators'])
        else:
            outs = [self.net.forward('regressors'), self.net.forward('classificators')]
        reg = next(o for o in outs if o.shape[-1] == 16).reshape(-1, 16)
        sc = next(o for o in outs if o.shape[-1] == 1).reshape(-1)
        p = 1 / (1 + np.exp(-np.clip(sc, -80, 80)))
        i = int(np.argmax(p))
        if p[i] < thr:
            return None
        # weighted average of the overlapping candidates (like MediaPipe's weighted NMS)
        A = self.anchors

        def dec(j):
            r = reg[j]
            cx, cy = r[0] / 128 + A[j, 0], r[1] / 128 + A[j, 1]
            w, h = r[2] / 128, r[3] / 128
            k = [(r[4 + 2 * q] / 128 + A[j, 0], r[5 + 2 * q] / 128 + A[j, 1]) for q in range(6)]
            return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] + [c for xy in k for c in xy], np.float64)
        best = dec(i)
        cand = np.where(p > thr)[0]
        acc, wsum = np.zeros_like(best), 0.0
        for j in cand:
            d = dec(j)
            ix = max(0, min(best[2], d[2]) - max(best[0], d[0])) * max(0, min(best[3], d[3]) - max(best[1], d[1]))
            u = (best[2] - best[0]) * (best[3] - best[1]) + (d[2] - d[0]) * (d[3] - d[1]) - ix
            if u > 0 and ix / u > 0.3:
                acc += d * p[j]
                wsum += p[j]
        if wsum > 0:
            best = acc / wsum
        X = lambda u: u * S - ox
        Y = lambda v: v * S - oy
        pts = [(X(best[4 + 2 * q]), Y(best[5 + 2 * q])) for q in range(6)]
        return {'box': (X(best[0]), Y(best[1]), X(best[2]), Y(best[3])), 'score': float(p[i]),
                'eyes': (pts[0], pts[1]), 'nose': pts[2], 'mouth': pts[3], 'ears': (pts[4], pts[5])}


def head_top(mask, x0, x1, thr=0.5):
    """Top-most row of the person mask inside the column span [x0, x1] (pixel coordinates of mask)."""
    h, w = mask.shape
    a, b = max(0, int(x0)), min(w, int(x1))
    if b <= a:
        return None
    rows = np.where((mask[:, a:b] > thr).mean(1) > 0.12)[0]
    return int(rows[0]) if len(rows) else None


def person_box(mask, thr=0.5):
    ys, xs = np.where(mask > thr)
    if len(xs) < 50:
        return None
    return (float(np.percentile(xs, 1)), float(np.percentile(ys, 0.5)), float(np.percentile(xs, 99)), float(ys.max()))
