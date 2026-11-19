"""Render loop: footage → camera → grade → beats (behind / front / full-frame) → captions → transitions → encoder.

Renders in resumable chunks so a long video can be finished over several runs of the same command.
"""
import hashlib
import json
import math
import os
import shutil
import time

import numpy as np
import cv2

from . import media, draw as D, look as L, plan as P
from .track import Track
from .util import clamp, ease_in_out_sine, ease_out_cubic, ease_in_cubic, prog

FORMATS = {'9:16': (1080, 1920), '16:9': (1920, 1080), '1:1': (1080, 1080), '4:5': (1080, 1350)}


# ---------------------------------------------------------------- footage placement + camera
class Footage:
    def __init__(self, info, W, H, track, fit='auto'):
        self.W, self.H, self.track = W, H, track
        sw, sh = info['w'], info['h']
        src_ar, out_ar = sw / float(sh), W / float(H)
        if fit == 'auto':
            fit = 'fit' if (src_ar < 0.8 and out_ar > 1.2) else 'cover'
        self.mode = fit
        if fit == 'fit':
            self.fh = H
            self.fw = int(round(sw * H / sh / 2)) * 2
            self.cw, self.ch = self.fw, self.fh
        else:
            s = max(W / sw, H / sh)
            self.cw = max(W, int(math.ceil(sw * s / 2)) * 2)
            self.ch = max(H, int(math.ceil(sh * s / 2)) * 2)

    def offset(self, t):
        if self.mode == 'fit':
            return -(self.W - self.fw) / 2.0, 0.0
        sx, sy = self.cw - self.W, self.ch - self.H
        f = self.track.at(t)
        fx = (f['face'][0] + f['face'][2]) / 2 * self.cw
        fy = f['eyes'][1] * self.ch
        ox = clamp(fx - self.W / 2, 0, sx) if sx > 2 else sx / 2.0
        oy = clamp(fy - 0.34 * self.H, 0, sy) if sy > 2 else sy / 2.0
        return ox, oy

    def to_canvas(self, nx, ny, t, s=1.0, a=None):
        ox, oy = self.offset(t)
        x, y = nx * self.cw - ox, ny * self.ch - oy
        if a is not None and s != 1.0:
            x = a[0] + s * (x - a[0])
            y = a[1] + s * (y - a[1])
        return x, y

    def render(self, src, t, s, a):
        ox, oy = self.offset(t)
        W, H = self.W, self.H
        if self.mode == 'fit':
            bg = cv2.resize(src, (W, int(W * src.shape[0] / src.shape[1])), interpolation=cv2.INTER_LINEAR)
            y0 = (bg.shape[0] - H) // 2
            bg = bg[max(0, y0):max(0, y0) + H]
            bg = cv2.resize(bg, (W, H))
            bg = cv2.GaussianBlur(cv2.resize(bg, (W // 8, H // 8)), (0, 0), 3)
            bg = (cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR) * 0.72).astype(np.uint8)
            x0 = int(-ox)
            bg[:, x0:x0 + self.fw] = src[:, :min(self.fw, W - x0)]
            if abs(s - 1) < 1e-4:
                return bg
            M = np.float32([[s, 0, a[0] * (1 - s)], [0, s, a[1] * (1 - s)]])
            return cv2.warpAffine(bg, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if abs(s - 1) < 1e-4 and abs(ox - round(ox)) < 1e-3 and abs(oy - round(oy)) < 1e-3:
            x, y = int(round(ox)), int(round(oy))
            return src[y:y + H, x:x + W].copy()
        M = np.float32([[s, 0, a[0] - s * (ox + a[0])], [0, s, a[1] - s * (oy + a[1])]])
        return cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class Camera:
    """Two multiplied tracks: slow sentence zooms (easeInOutSine) and the effect moves (digital zoom, zoom-through…)."""

    def __init__(self, slow, fast):
        self.slow = sorted(slow)
        self.fast = sorted(fast)

    @staticmethod
    def _eval(keys, t, base=1.0):
        s = base
        for start, dur, a, b, curve in keys:
            if t >= start + dur:
                s = b
            elif t >= start:
                s = a + (b - a) * curve(clamp((t - start) / dur))
                break
            else:
                break
        return s

    def scale(self, t):
        return self._eval(self.slow, t) * self._eval(self.fast, t)


def slow_keys(cues, dur, avoid):
    """Slow in (4 s) → hold → slow out (3 s) → hold, starting on sentence starts, never inside effect camera moves."""
    keys = []
    starts = [c['t0'] for c in cues] or [0.5]
    t = starts[0]
    zoomed = False
    while t < dur - 3:
        nxt = next((s for s in starts if s >= t), None)
        if nxt is None:
            break
        t = nxt
        d = 4.0 if not zoomed else 3.0
        if any(a - 0.6 < t + d and t < b + 0.6 for a, b in avoid):
            t += 1.0
            continue
        if zoomed:
            keys.append((t, d, 1.075, 1.0, ease_in_out_sine))
        else:
            keys.append((t, d, 1.0, 1.075, ease_in_out_sine))
        zoomed = not zoomed
        t += d + 2.6
    return keys


def pushpull(img, valid, k=6.0):
    """Fill the holes (valid = 0) of an image smoothly from the coarse levels of a mask-weighted pyramid (no streaks)."""
    levels = []
    I = img * valid[..., None]
    Wt = valid.astype(np.float32)
    while min(I.shape[:2]) > 12:
        levels.append((I, Wt))
        I = cv2.pyrDown(I)
        Wt = cv2.pyrDown(Wt)
    C = I / np.maximum(Wt, 1e-4)[..., None]
    for I, Wt in reversed(levels):
        up = cv2.pyrUp(C, dstsize=(I.shape[1], I.shape[0]))
        known = I / np.maximum(Wt, 1e-4)[..., None]
        a = np.clip(Wt * k, 0, 1)[..., None]
        C = known * a + up * (1 - a)
    return C.astype(np.float32)


_LUT = None


def grade_u8(u8):
    """Clean light grade on uint8 (fast): a touch brighter, a little more contrast, slightly richer colour."""
    global _LUT
    if _LUT is None:
        x = np.arange(256, dtype=np.float32) / 255.0
        y = (np.power(x, 0.94) - 0.5) * 1.05 + 0.515
        _LUT = np.clip(y * 255 + 0.5, 0, 255).astype(np.uint8)
    g = cv2.LUT(u8, _LUT)
    gray = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(g, 1.06, gray3, -0.06, 0)


def grade(u8):
    return np.multiply(grade_u8(u8), 1 / 255.0, dtype=np.float32)


# ---------------------------------------------------------------- context shared with the beats
class Ctx:
    def __init__(self, src, info, W, H, fps, track, cues, plan, work, seg=None):
        self.src, self.info, self.W, self.H, self.fps = src, info, W, H, fps
        self.u = min(W, H) / 100.0
        self.vertical = H > W * 1.2
        self.track, self.cues, self.plan, self.work = track, cues, plan, work
        self.seg = seg
        self.has_mask = seg is not None
        self.footage = Footage(info, W, H, track, plan.get('fit', 'auto'))
        self.cam = None
        self.fi = -1
        self.t = 0.0
        self.F0 = None
        self._alpha = None
        self._alpha_fi = -99
        self._stills = {}
        self._plates = {}
        self.sounds = []
        self.caps = None
        self.cap_y = 0.765 * H if self.vertical else 0.86 * H
        self.top_safe = 0.06 * H if self.vertical else 0.05 * H

    # --- per frame
    def begin(self, fi, t, F0, U0=None):
        self.fi, self.t, self.F0, self.U0 = fi, t, F0, U0
        self._alpha = None

    def alpha(self):
        """Person alpha of the current footage frame (H, W) float32; zeros when no mask model is available."""
        if self._alpha is not None:
            return self._alpha
        if self.seg is None:
            self._alpha = np.zeros((self.H, self.W), np.float32)
            return self._alpha
        if self._alpha_fi != self.fi - 1:
            self.seg.reset()
        u8 = self.U0 if self.U0 is not None else D.to_u8(self.F0)
        self._alpha = self.seg.alpha(u8)
        self._alpha_fi = self.fi
        return self._alpha

    # --- geometry in canvas coordinates
    def cam_at(self, t):
        if self.cam is None:
            return 1.0, None
        s = self.cam.scale(t)
        f = self.track.at(t)
        a = self.footage.to_canvas(f['eyes'][0], f['eyes'][1], t)
        return s, a

    def face(self, t=None):
        t = self.t if t is None else t
        f = self.track.at(t)
        s, a = self.cam_at(t)
        tc = lambda x, y: self.footage.to_canvas(x, y, t, s, a)
        x0, y0 = tc(f['face'][0], f['face'][1])
        x1, y1 = tc(f['face'][2], f['face'][3])
        ex, ey = tc(*f['eyes'])
        mx, my = tc(*f['mouth'])
        _, hy = tc(0.5, f['head'])
        px0, py0 = tc(f['person'][0], f['person'][1])
        px1, py1 = tc(f['person'][2], f['person'][3])
        return {'box': (x0, y0, x1, y1), 'eyes': (ex, ey), 'mouth': (mx, my), 'head': hy, 'person': (px0, py0, px1, py1),
                'cx': (x0 + x1) / 2, 'w': x1 - x0, 'h': y1 - y0}

    def span(self, t0, t1, n=7):
        fs = [self.face(t0 + (t1 - t0) * i / (n - 1)) for i in range(n)]
        bx = (min(f['box'][0] for f in fs), min(f['box'][1] for f in fs), max(f['box'][2] for f in fs), max(f['box'][3] for f in fs))
        return {'box': bx, 'head': min(f['head'] for f in fs), 'eyes': fs[n // 2]['eyes'], 'mouth': fs[n // 2]['mouth'],
                'cx': (bx[0] + bx[2]) / 2, 'w': bx[2] - bx[0], 'h': bx[3] - bx[1],
                'person': (min(f['person'][0] for f in fs), min(f['person'][1] for f in fs), max(f['person'][2] for f in fs), max(f['person'][3] for f in fs))}

    # --- frames from other moments
    def still(self, at):
        key = round(float(at), 2)
        if key in self._stills:
            return D.to_f(self._stills[key])
        at = clamp(key, 0, max(0, self.info['dur'] - 0.1))
        rd = media.Reader(self.src, self.footage.cw, self.footage.ch, self.fps, t0=at, t1=at + 0.2)
        fr = rd.read()
        rd.close()
        U = grade_u8(self.footage.render(fr, at, 1.0, (self.W / 2, self.H / 2)))
        self._stills[key] = U
        if len(self._stills) > 16:
            self._stills.pop(next(k for k in self._stills if not isinstance(k, tuple)))
        return D.to_f(U)

    def still_alpha(self, at):
        key = ('a', round(float(at), 2))
        if key in self._stills:
            return self._stills[key]
        F = self.still(at)
        if self.seg is None:
            a = np.zeros(F.shape[:2], np.float32)
        else:
            self.seg.reset()
            a = self.seg.alpha(D.to_u8(F), temporal=False)
            self.seg.reset()
        self._stills[key] = a
        if sum(1 for k in self._stills if isinstance(k, tuple)) > 10:
            self._stills.pop(next(k for k in self._stills if isinstance(k, tuple)))
        return a

    def plate(self, at):
        """Background plate of the frame at `at` with the person painted out (for vertigo, clone, green screen, UI world)."""
        key = round(float(at), 1)
        if key in self._plates:
            return self._plates[key]
        F = self.still(at)
        a = self.still_alpha(at)
        q = 2
        small = cv2.resize(F, (self.W // q, self.H // q), interpolation=cv2.INTER_AREA)
        hole = cv2.resize(a, (self.W // q, self.H // q), interpolation=cv2.INTER_AREA)
        hole = cv2.dilate((hole > 0.12).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(np.float32)
        fill = pushpull(small, 1 - hole)
        big = cv2.resize(fill, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        m = cv2.GaussianBlur(cv2.resize(hole, (self.W, self.H), interpolation=cv2.INTER_LINEAR), (0, 0), 5)[..., None]
        P_ = F * (1 - m) + big * m
        self._plates[key] = P_
        return P_

    # --- words
    def word_time(self, phrase, t0, t1):
        return P.find_word(self.cues, phrase, t0, t1)

    def sfx(self, at, kind, gain=1.0):
        self.sounds.append((float(at), kind, float(gain)))


# ---------------------------------------------------------------- main render
def plan_hash(plan, extra=''):
    return hashlib.sha1((json.dumps(plan, sort_keys=True, ensure_ascii=False) + extra).encode('utf-8')).hexdigest()[:12]


class Job:
    def __init__(self, plan_path, out, log=print, stills_only=False):
        from . import fx as FX
        self.FX = FX
        self.log = log
        self.plan_path = plan_path
        self.base = os.path.dirname(os.path.abspath(plan_path))
        self.plan = P.load(plan_path)
        pl = self.plan
        self.src = self._path(pl.get('video') or pl.get('footage'))
        if not self.src or not os.path.exists(self.src):
            raise SystemExit('plan.json: "video" must name the footage file (not found: %s)' % self.src)
        self.out = out
        self.info = media.probe(self.src)
        fmt = pl.get('format', '9:16')
        self.W, self.H = FORMATS.get(fmt, FORMATS['9:16'])
        self.fps = float(pl.get('fps') or (30 if self.info['fps'] > 31 or self.info['fps'] < 23 else round(self.info['fps'])))
        self.dur = float(pl.get('duration') or self.info['dur'])
        srt = self._path(pl.get('srt'))
        self.cues = P.parse_srt(open(srt, encoding='utf-8', errors='replace').read()) if srt and os.path.exists(srt) else []
        # what is said together (a word-by-word SRT becomes phrases too)
        self.phrases = P.phrases(self.cues) if self.cues else []
        self.work = os.path.splitext(os.path.abspath(out))[0] + '.rkwork'
        os.makedirs(self.work, exist_ok=True)
        tp = os.path.join(self.work, 'track.json')
        if os.path.exists(tp):
            self.track = Track.load(tp)
        else:
            log('analysing the footage (face + person mask)…')
            self.track = Track.analyze(self.src, self.info, log=log, use_mask=pl.get('mask', True) is not False)
            self.track.save(tp)
            for n in self.track.d.get('notes', []):
                log('  note: ' + n)
        seg = None
        if self.track.has_mask:
            try:
                from .vision import Segmenter
                seg = Segmenter(landscape=self.W > self.H * 1.15)
            except Exception as e:
                log('  note: person mask unavailable (%s): effects that need it fall back' % e)
        self.ctx = Ctx(self.src, self.info, self.W, self.H, self.fps, self.track, self.cues, pl, self.work, seg)
        ctx = self.ctx
        ctx.phrases = self.phrases
        # beats
        self.beats = []
        for i, b in enumerate(pl.get('beats', [])):
            try:
                self.beats.append(FX.make(b, ctx, i))
            except Exception as e:
                log('  beat %d (%s) skipped: %s' % (i + 1, b.get('fx'), e))
        # the end card runs to the end of the video
        if self.beats and type(self.beats[-1]).__name__ == 'End' and self.beats[-1].t1 < self.dur:
            self.beats[-1].t1 = self.dur
        # camera
        fast, avoid = [], []
        for b in self.beats:
            ks = b.cam()
            fast += ks
            if ks or b.full_frame:
                avoid.append((b.t0 - 0.5, b.t1 + 0.5))
        ctx.cam = Camera(slow_keys(self.phrases or self.cues, self.dur, avoid) if pl.get('camera', 'auto') == 'auto' else [], fast)
        # every beat lays itself out once (needs the camera for face positions)
        for b in self.beats:
            try:
                b.ensure()
            except Exception as e:
                import traceback
                traceback.print_exc()
                log('  beat %d (%s) failed to prepare: %s' % (b.idx + 1, b.spec.get('fx'), e))
                b.on = lambda t: False
        # captions
        # captions: on by default; a lockup on screen is the subtitle itself ("captions": "always" shows both)
        cap_mode = pl.get('captions', True)
        if cap_mode and self.cues:
            ctx.caps = L.Captions(self.phrases or self.cues, self.W, self.H, y=ctx.cap_y)
            for b in self.beats:
                if cap_mode == 'always' and type(b).__name__ == 'Say':
                    continue
                for a, z in b.hide_captions():
                    ctx.caps.hide(a, z)
        # transitions + sounds
        self.trans = []
        for b in self.beats:
            tr = b.spec.get('transition')
            if tr and tr != 'none':
                self.trans.append((b.t0, tr, b.idx))
                ctx.sfx(b.t0 - 0.05, 'whoosh', 0.8)
        for b in self.beats:
            b.sounds()
        for b in self.beats:
            b.release()
        self._tags = {}
        self.n_frames = int(round(self.dur * self.fps))
        self.hash = plan_hash(pl, '%dx%d@%g' % (self.W, self.H, self.fps))

    def _path(self, p):
        if not p:
            return None
        if os.path.isabs(p) or os.path.exists(p):
            return p
        q = os.path.join(self.base, p)
        return q if os.path.exists(q) else p

    # one frame
    def frame(self, fi, src_u8):
        ctx = self.ctx
        t = fi / self.fps
        s, a = ctx.cam_at(t)
        canvas = ctx.footage.render(src_u8, t, s, a)
        F0 = grade(canvas)
        ctx.begin(fi, t, F0, canvas)
        F = F0.copy()
        act = [b for b in self.beats if b.on(t)]
        full = None
        for b in act:
            if b.full_frame:
                r = b.full(ctx, F, t)
                if r is not None:
                    F = r
                    full = b
        bgs = [b for b in act if b.behind and b.bg_on(t)] if full is None or full.allow_bg else []
        if bgs:
            B = F.copy()
            for b in bgs:
                b.bg(ctx, B, t)
            A = ctx.alpha()[..., None]
            F = B * (1 - A) + F * A
        for b in act:
            b.front(ctx, F, t)
        if ctx.caps is not None:
            ctx.caps.draw(F, t)
        for b in act:
            tg = b.spec.get('tag')
            if tg and b.t0 <= t <= b.t1:
                spr = self._tags.get(tg)
                if spr is None:
                    spr = self._tags[tg] = L.pill(str(tg), 2.6 * ctx.u, bg='#141416')
                a = min(1.0, (t - b.t0) / 0.2, (b.t1 - t) / 0.2)
                D.place(F, spr, 0.04 * self.W + spr.shape[1] / 2, ctx.top_safe + 0.005 * self.H, op=max(0.0, a))
        for t0, kind, idx in self.trans:
            if t0 - 0.45 <= t <= t0 + 0.6:
                apply_transition(F, t, t0, kind, ctx.u, idx)
        for b in self.beats:
            if b.ready and t > b.t1 + b.post + 0.2:
                b.release()
        return D.to_u8(F)

    def render(self, budget=None, chunk=6.0, crf=18, preset='veryfast'):
        t_start = time.time()
        n = self.n_frames
        per = int(round(chunk * self.fps))
        chunks = [(i, min(n, i + per)) for i in range(0, n, per)]
        state_p = os.path.join(self.work, 'chunks.json')
        state = {}
        if os.path.exists(state_p):
            try:
                state = json.load(open(state_p))
            except Exception:
                state = {}
        if state.get('hash') != self.hash:
            state = {'hash': self.hash, 'done': []}
        done = set(state['done'])
        for ci, (f0, f1) in enumerate(chunks):
            seg_p = os.path.join(self.work, 'seg_%04d.mp4' % ci)
            if ci in done and os.path.exists(seg_p):
                continue
            if budget and time.time() - t_start > budget:
                left = len(chunks) - len(done)
                self.log('RESUME: %d of %d parts done. Run the same command again to continue.' % (len(done), len(chunks)))
                return False
            rd = media.Reader(self.src, self.ctx.footage.cw, self.ctx.footage.ch, self.fps, t0=f0 / self.fps, t1=f1 / self.fps + 0.2)
            wr = media.Writer(seg_p, self.W, self.H, self.fps, crf=crf, preset=preset)
            if self.ctx.seg:
                self.ctx.seg.reset()
            tc = time.time()
            for fi in range(f0, f1):
                wr.write(self.frame(fi, rd.read()))
            rd.close()
            wr.close()
            done.add(ci)
            state['done'] = sorted(done)
            json.dump(state, open(state_p, 'w'))
            self.log('  part %d/%d rendered (%.1f s of video in %.0f s)' % (ci + 1, len(chunks), (f1 - f0) / self.fps, time.time() - tc))
        self.finish(chunks)
        return True

    def finish(self, chunks):
        from . import audio
        lst = os.path.join(self.work, 'list.txt')
        with open(lst, 'w') as f:
            for ci in range(len(chunks)):
                f.write("file '%s'\n" % os.path.join(self.work, 'seg_%04d.mp4' % ci))
        vid = os.path.join(self.work, 'video.mp4')
        media.run([media.FFMPEG, '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', vid])
        mix = os.path.join(self.work, 'mix.wav')
        audio.build(self.src, self.ctx.sounds, self.dur, mix, self.work, log=self.log, sfx_on=self.plan.get('sfx', True) is not False,
                    voice=self.plan.get('voice', True) is not False)
        media.run([media.FFMPEG, '-v', 'error', '-y', '-i', vid, '-i', mix, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac',
                   '-b:a', '192k', '-ar', '48000', '-shortest', '-movflags', '+faststart', self.out])
        self.log('DONE: %s' % self.out)

    def stills(self, times, out_dir, sheet=None, scale=0.5):
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for t in times:
            fi = int(round(t * self.fps))
            rd = media.Reader(self.src, self.ctx.footage.cw, self.ctx.footage.ch, self.fps, t0=max(0, fi / self.fps - 0.0), t1=fi / self.fps + 0.2)
            fr = rd.read()
            rd.close()
            if self.ctx.seg:
                self.ctx.seg.reset()
            img = self.frame(fi, fr)
            p = os.path.join(out_dir, 'still_%07.2f.jpg' % t)
            cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            paths.append((t, p, img))
        if sheet and paths:
            cols = 6 if self.H > self.W else 4
            tw = int(self.W * scale * (0.5 if len(paths) > 8 else 1))
            th = int(self.H * tw / self.W)
            rows = int(math.ceil(len(paths) / cols))
            S = np.full((rows * (th + 30), cols * tw, 3), 255, np.uint8)
            for i, (t, p, img) in enumerate(paths):
                x, y = (i % cols) * tw, (i // cols) * (th + 30)
                S[y + 30:y + 30 + th, x:x + tw] = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                cv2.putText(S, '%.2fs' % t, (x + 6, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.imwrite(sheet, cv2.cvtColor(S, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        return [p for _, p, _ in paths]


def apply_transition(F, t, t0, kind, u, seed):
    if kind == 'streak':
        L.streak(F, t, t0 - 0.2, u, seed=seed)
    elif kind == 'halftone':
        L.halftone(F, t, t0 - 0.05, u)
    elif kind == 'flash':
        L.flash(F, t, t0, 0.1)
    elif kind == 'rgb':
        L.rgb_split(F, t, t0, u, 0.1)
    elif kind == 'blurzoom':
        L.blur_zoom(F, t, t0, 0.32)
