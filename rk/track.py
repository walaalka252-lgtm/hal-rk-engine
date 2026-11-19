"""Analysis pass: where the face and head are over time (normalised 0..1 to the source frame).

Saved as track.json in the work folder so re-renders are instant.
"""
import json
import os
import time
import numpy as np
import cv2

from . import media
from .vision import Segmenter, FaceDetector, head_top, person_box


def _fill(a):
    """Linear interpolation over NaN rows, then hold at the ends."""
    a = np.array(a, np.float64)
    if a.ndim == 1:
        a = a[:, None]
    n = len(a)
    idx = np.arange(n)
    for c in range(a.shape[1]):
        col = a[:, c]
        ok = ~np.isnan(col)
        if ok.sum() == 0:
            continue
        a[:, c] = np.interp(idx, idx[ok], col[ok])
    return a


def _smooth(a, k=5, ema=0.35):
    from scipy.ndimage import median_filter  # scipy is optional: fall back below
    return median_filter(a, size=(k, 1), mode='nearest')


def smooth(a, k=5, ema=0.4):
    a = np.array(a, np.float64)
    try:
        a = _smooth(a, k)
    except Exception:
        pad = k // 2
        p = np.pad(a, ((pad, pad), (0, 0)), mode='edge')
        a = np.stack([np.median(p[i:i + k], 0) for i in range(len(a))])
    # zero-phase EMA (forward + backward)
    f = a.copy()
    for i in range(1, len(f)):
        f[i] = f[i - 1] + ema * (f[i] - f[i - 1])
    for i in range(len(f) - 2, -1, -1):
        f[i] = f[i + 1] + ema * (f[i] - f[i + 1])
    return f


class Track:
    def __init__(self, d):
        self.d = d
        self.t = np.array(d['t'], np.float64)
        self.face = np.array(d['face'], np.float64)      # x0 y0 x1 y1 (normalised)
        self.eyes = np.array(d['eyes'], np.float64)      # ex ey (mid point)
        self.mouth = np.array(d['mouth'], np.float64)    # mx my
        self.head = np.array(d['head'], np.float64)      # head top y
        self.person = np.array(d['person'], np.float64)  # x0 y0 x1 y1
        self.has_mask = bool(d.get('mask'))
        self.has_face = bool(d.get('faces'))
        self.aspect = d.get('aspect', 9 / 16)

    def _at(self, arr, t):
        if len(self.t) == 0:
            return arr[0]
        return np.array([np.interp(t, self.t, arr[:, c]) for c in range(arr.shape[1])])

    def at(self, t):
        f = self._at(self.face, t)
        return {'face': tuple(f), 'eyes': tuple(self._at(self.eyes, t)), 'mouth': tuple(self._at(self.mouth, t)),
                'head': float(self._at(self.head, t)[0]), 'person': tuple(self._at(self.person, t))}

    def span(self, t0, t1):
        """Face box that covers the whole interval (a little margin), head top = highest in the interval."""
        m = (self.t >= t0 - 0.15) & (self.t <= t1 + 0.15)
        if m.sum() == 0:
            a = self.at((t0 + t1) / 2)
            return a
        f = self.face[m]
        box = (f[:, 0].min(), f[:, 1].min(), f[:, 2].max(), f[:, 3].max())
        return {'face': box, 'eyes': tuple(self.eyes[m].mean(0)), 'mouth': tuple(self.mouth[m].mean(0)),
                'head': float(self.head[m].min()), 'person': tuple(np.r_[self.person[m][:, :2].min(0), self.person[m][:, 2:].max(0)])}

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.d, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def analyze(cls, path, info, fps=8.0, log=print, use_mask=True):
        w, h = info['w'], info['h']
        s = 512.0 / max(w, h)
        aw, ah = max(64, int(round(w * s / 2)) * 2), max(64, int(round(h * s / 2)) * 2)
        landscape = w > h * 1.15
        seg = face = None
        notes = []
        if use_mask:
            try:
                seg = Segmenter(landscape=landscape)
            except Exception as e:
                notes.append('person mask unavailable: %s' % e)
        try:
            face = FaceDetector()
        except Exception as e:
            notes.append('face detector unavailable: %s' % e)
        haar = None
        if face is None:
            try:
                xml = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
                haar = cv2.CascadeClassifier(xml) if os.path.exists(xml) else None
            except Exception:
                haar = None
        rd = media.Reader(path, aw, ah, fps)
        n = max(1, int(info['dur'] * fps + 0.5))
        T, F, E, M, HD, P = [], [], [], [], [], []
        nf = 0
        t_start = time.time()
        for i in range(n):
            fr = rd.read()
            t = i / fps
            fb = ey = mo = None
            if face is not None:
                d = face.detect(fr)
                if d:
                    x0, y0, x1, y1 = d['box']
                    fb = (x0 / aw, y0 / ah, x1 / aw, y1 / ah)
                    e1, e2 = d['eyes']
                    ey = ((e1[0] + e2[0]) / 2 / aw, (e1[1] + e2[1]) / 2 / ah)
                    mo = (d['mouth'][0] / aw, d['mouth'][1] / ah)
            elif haar is not None:
                g = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
                fs = haar.detectMultiScale(g, 1.15, 5, minSize=(int(aw * .12), int(aw * .12)))
                if len(fs):
                    x, y, ww, hh = max(fs, key=lambda r: r[2] * r[3])
                    fb = (x / aw, y / ah, (x + ww) / aw, (y + hh) / ah)
                    ey = ((x + ww / 2) / aw, (y + hh * .4) / ah)
                    mo = ((x + ww / 2) / aw, (y + hh * .78) / ah)
            ht = pb = None
            if seg is not None:
                m = seg.raw(fr)
                mm = cv2.resize(m, (aw, ah), interpolation=cv2.INTER_LINEAR)
                pb = person_box(mm)
                if pb:
                    pb = (pb[0] / aw, pb[1] / ah, pb[2] / aw, pb[3] / ah)
                    span = (fb[0] * aw, fb[2] * aw) if fb else (pb[0] * aw + (pb[2] - pb[0]) * aw * .3, pb[0] * aw + (pb[2] - pb[0]) * aw * .7)
                    ht = head_top(mm, *span)
                    ht = ht / ah if ht is not None else None
            if fb:
                nf += 1
            nan4 = (np.nan,) * 4
            T.append(t)
            F.append(fb or nan4)
            E.append(ey or (np.nan, np.nan))
            M.append(mo or (np.nan, np.nan))
            HD.append((ht,) if ht is not None else (np.nan,))
            P.append(pb or nan4)
            if log and i and i % int(fps * 15) == 0:
                log('  analysed %d/%d s (%.0f s)' % (i / fps, info['dur'], time.time() - t_start))
        rd.close()
        F, E, M, HD, P = (_fill(x) for x in (F, E, M, HD, P))
        # defaults where nothing was ever found: a centred talking head
        if np.isnan(F).all():
            F[:] = (0.3, 0.18, 0.7, 0.42) if h > w else (0.38, 0.18, 0.62, 0.55)
            E[:] = (0.5, 0.28 if h > w else 0.33)
            M[:] = (0.5, 0.37 if h > w else 0.47)
        if np.isnan(HD).all():
            HD[:, 0] = F[:, 1] - (F[:, 3] - F[:, 1]) * 0.35
        if np.isnan(P).all():
            P[:] = (max(0, F[0, 0] - 0.2), HD[0, 0], min(1, F[0, 2] + 0.2), 1.0)
        F, E, M, HD, P = (smooth(np.nan_to_num(x)) for x in (F, E, M, HD, P))
        d = {'t': T, 'face': F.tolist(), 'eyes': E.tolist(), 'mouth': M.tolist(), 'head': HD.tolist(), 'person': P.tolist(),
             'mask': seg is not None, 'faces': nf, 'samples': n, 'fps': fps, 'aspect': w / float(h), 'notes': notes}
        return cls(d)
