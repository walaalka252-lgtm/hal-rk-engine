"""Beats of a Red Kinetic edit. Each beat type draws on three layers:
   full(ctx, F, t)  → replaces the whole frame (graphics-only beats, frames, freezes)
   bg(ctx, B, t)    → behind the person (the person mask is composited over it)
   front(ctx, F, t) → in front of everything (big words in front, cards, UI)
"""
import math
import re

import numpy as np
import cv2

from . import draw as D, look as L, plan as P
from .util import (clamp, prog, lerp, ease_out_cubic, ease_in_cubic, ease_in_out_cubic, ease_in_out_sine, ease_out_back,
                   ease_out_expo, hexrgb, stable_seed, rnd, window)

REG = {}
ALIASES = {}


def reg(name, need='', fallback='word', aliases=()):
    def deco(cls):
        cls.name, cls.need, cls.fallback = name, need, fallback
        REG[name] = cls
        for a in aliases:
            ALIASES[a] = name
        return cls
    return deco


def canon(name):
    n = re.sub(r'[\s_+]+', '-', str(name or 'word').strip().lower())
    n = n.replace('&', '').replace('--', '-')
    return ALIASES.get(n, n)


def make(spec, ctx, idx):
    name = canon(spec.get('fx', 'word'))
    cls = REG.get(name)
    if cls is None:
        raise ValueError('unknown fx "%s" (run: python3 -m rk fx)' % spec.get('fx'))
    if name == 'cards' and spec.get('layout') != 'full':
        t0 = float(spec.get('at', 0))
        lay = Lay(ctx, t0, float(spec.get('end', t0 + 2.5)))
        mode = word_spot(ctx, lay, (spec.get('word') or 'X').upper(), spec.get('place', 'behind'), idx)[0]
        if mode == 'chest':
            cls = REG['split']
    if cls.need == 'mask' and not ctx.has_mask:
        cls = REG[cls.fallback]
    return cls(spec, ctx, idx)


# ---------------------------------------------------------------- helpers
def img_from(ctx, ref, t_default, box=None):
    """An image (float RGB) from "frame:12.3", "frame:12.3@x,y,w,h" (normalised crop), a file path, or the frame at t_default."""
    if isinstance(ref, str) and ref and not ref.startswith('frame'):
        im = cv2.imread(ref)
        if im is None:
            im = cv2.imread(ctx.plan.get('_base', '.') + '/' + ref)
        if im is not None:
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ref = None
    at, crop = t_default, box
    if isinstance(ref, str) and ref.startswith('frame'):
        m = re.match(r'frame:?\s*([\d.]+)?(?:@([\d.,\s]+))?', ref)
        if m and m.group(1):
            at = float(m.group(1))
        if m and m.group(2):
            v = [float(x) for x in m.group(2).split(',')[:4]]
            if len(v) == 4:
                crop = v
    F = ctx.still(at)
    H, W = F.shape[:2]
    if crop:
        x, y, w, h = crop
        x0, y0 = int(clamp(x) * W), int(clamp(y) * H)
        x1, y1 = int(clamp(x + w) * W), int(clamp(y + h) * H)
        if x1 - x0 > 8 and y1 - y0 > 8:
            return F[y0:y1, x0:x1].copy()
    return F


def bg_crop(ctx, at, which=0, aspect=None):
    """The most interesting region of the frame at `at` that shows only the room (no person): product / room shots.
    Scores every window (integral images) by how much there is to see in it (colour + detail, not white walls), away
    from the person; `which` picks the 1st, 2nd, 3rd… of the distinct (non-overlapping) best regions."""
    W, H = ctx.W, ctx.H
    key = (round(float(at), 2), aspect)
    cache = ctx.__dict__.setdefault('_bgcrop', {})
    if key not in cache:
        cache[key] = _bg_regions(ctx, at, aspect)
    regs = cache[key]
    if not regs:
        return side_crop(ctx, at, which)
    return regs[which % len(regs)]


def _bg_regions(ctx, at, aspect):
    W, H = ctx.W, ctx.H
    A = ctx.still_alpha(at)
    u8 = D.to_u8(ctx.still(at))
    c = 20.0
    gw, gh = max(8, int(W / c)), max(8, int(H / c))
    g = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mag = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    edge = cv2.resize((mag > 60).astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0
    # colour that is neither near-black nor blown out = products, food, plants, posters
    col = sat * np.clip((val - 0.12) / 0.2, 0, 1)
    colc = cv2.resize(col, (gw, gh), interpolation=cv2.INTER_AREA)
    # local contrast (a flat or plain white wall has none)
    lc = cv2.resize(np.abs(g - cv2.blur(g, (9, 9))), (gw, gh), interpolation=cv2.INTER_AREA) / 40.0
    tex = 0.55 * edge + 1.3 * np.clip(colc * 2.2, 0, 1.2) + 0.5 * np.clip(lc, 0, 1)
    k = max(3, int(0.035 * W)) | 1
    busy = cv2.dilate((A > 0.15).astype(np.uint8), np.ones((k, k), np.uint8)).astype(np.float32)
    busy = cv2.resize(busy, (gw, gh), interpolation=cv2.INTER_AREA)
    I_b = np.pad(busy.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    I_t = np.pad(tex.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ar = aspect or 1.0
    cand = []
    lo = max(4, int(round((0.16 if ar < 1.1 else 0.2) * gw)))
    hi = max(lo, int(round(0.62 * gw)))
    for wc in range(lo, hi + 1, 2):
        for hk in (0.85, 1.0, 1.2):
            hc = int(round((wc * W / gw) / (ar * hk) / (H / gh)))
            if hc < 4 or hc > gh:
                continue
            Bs = I_b[hc:, wc:] - I_b[:-hc, wc:] - I_b[hc:, :-wc] + I_b[:-hc, :-wc]
            Ts = I_t[hc:, wc:] - I_t[:-hc, wc:] - I_t[hc:, :-wc] + I_t[:-hc, :-wc]
            area = float(wc * hc)
            ok = Bs <= 0.02 * area
            if not ok.any():
                continue
            # density of detail, with a gentle preference for bigger windows (sharper, more context)
            sc = np.where(ok, Ts / area * (area / (gw * gh)) ** 0.22 / (hk if hk > 1 else 1 / hk) ** 0.3, -1).ravel()
            m = min(24, sc.size)
            idx = np.argpartition(-sc, m - 1)[:m]
            ncol = Ts.shape[1]
            for j in idx:
                if sc[j] > 0:
                    iy, ix = divmod(int(j), ncol)
                    cand.append((float(sc[j]), ix, iy, wc, hc))
    cand.sort(key=lambda r: -r[0])
    picks = []
    for s_, ix, iy, wc, hc in cand:
        good = True
        for _, px, py, pw, ph in picks:
            ox = max(0, min(ix + wc, px + pw) - max(ix, px))
            oy = max(0, min(iy + hc, py + ph) - max(iy, py))
            if ox * oy > 0.3 * min(wc * hc, pw * ph):
                good = False
                break
        if good:
            picks.append((s_, ix, iy, wc, hc))
            if len(picks) >= 8:
                break
    if picks:
        top = picks[0][0]
        picks = [p for p in picks if p[0] >= 0.45 * top] or picks[:1]
    return [[ix / gw, iy / gh, wc / gw, hc / gh] for _, ix, iy, wc, hc in picks]


def texture(img):
    """How much there is to see in a picture (mean edge strength, 0 = a flat wall)."""
    g = cv2.cvtColor(D.to_u8(cv2.resize(img, (160, int(160 * img.shape[0] / max(1, img.shape[1]))))), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float((np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))).mean())


def side_crop(ctx, at, which=0):
    """A crop of the background beside the speaker (left/right of the face, top half) — good product/room shots."""
    f = ctx.face(at)
    W, H = ctx.W, ctx.H
    x0, y0, x1, y1 = f['box']
    cands = []
    if x0 > W * 0.16:
        cands.append((0.0, max(0, y0 - 0.1 * H), x0 / W, 0.26))
    if W - x1 > W * 0.16:
        cands.append((x1 / W, max(0, y0 - 0.1 * H), 1 - x1 / W, 0.26))
    cands.append((0.0, 0.0, 1.0, 0.16))
    c = cands[which % len(cands)]
    return [c[0], c[1] / H if c[1] > 1 else c[1], c[2], c[3]]


def pop_in(t, t0, dur=0.5, back=1.3):
    p = prog(t, t0, dur)
    return ease_out_back(p, back) if p < 1 else 1.0, ease_out_cubic(clamp(p * 1.6))


def typed(T, spr, frac):
    """Crop a text sprite after the first frac of its characters (typewriter)."""
    n = len(T.text)
    k = int(round(clamp(frac) * n))
    if k <= 0:
        return None
    if k >= n:
        return spr
    x = int(T.left + T.xs[k] - 1)
    out = spr.copy()
    out[:, max(0, x):] = 0
    return out


def boxes_overlap(a, b, m=0):
    return not (a[2] + m <= b[0] or b[2] + m <= a[0] or a[3] + m <= b[1] or b[3] + m <= a[1])


def place_masked(F, spr, cx, cy, A=None, occ=0.0, scale=1.0, rot=0.0, op=1.0, blur=0.0, anchor=(0.5, 0.5)):
    """Like draw.place, but the sprite is hidden behind the person by `occ` (0 = in front, 1 = fully behind)."""
    if op <= 0.002:
        return F
    if A is None or occ <= 0.001:
        return D.place(F, spr, cx, cy, scale=scale, rot=rot, op=op, blur=blur, anchor=anchor)
    H, W = F.shape[:2]
    h, w = spr.shape[:2]
    r = math.hypot(w, h) * scale / 2 + blur * 3 + 4
    x0, y0, x1, y1 = int(max(0, cx - r)), int(max(0, cy - r)), int(min(W, cx + r)), int(min(H, cy + r))
    if x1 <= x0 or y1 <= y0:
        return F
    buf = np.zeros((y1 - y0, x1 - x0, 4), np.float32)
    out = _place_rgba(buf, spr, cx - x0, cy - y0, scale, rot, blur, anchor)
    k = (1 - A[y0:y1, x0:x1] * occ)[..., None]
    out = out * k * op
    roi = F[y0:y1, x0:x1]
    roi *= (1 - out[..., 3:4])
    roi += out[..., :3]
    return F


def _place_rgba(buf, spr, cx, cy, scale, rot, blur, anchor):
    h, w = spr.shape[:2]
    ax, ay = anchor[0] * w, anchor[1] * h
    r = math.radians(rot)
    c, s = math.cos(r) * scale, math.sin(r) * scale
    M = np.float64([[c, -s, cx - (c * ax - s * ay)], [s, c, cy - (s * ax + c * ay)]])
    out = cv2.warpAffine(spr, M, (buf.shape[1], buf.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    if blur > 0.3:
        out = D.gblur(out, blur)
    return out


def scale_about(img, s, ax, ay, border=cv2.BORDER_CONSTANT, value=0):
    """Scale an image (or mask) by s around the point (ax, ay), same size out."""
    H, W = img.shape[:2]
    if abs(s - 1) < 1e-4:
        return img
    M = np.float32([[s, 0, ax * (1 - s)], [0, s, ay * (1 - s)]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=border, borderValue=value)


def contact_shadow(B, A, u, dy=1.4, blur=3.0, amt=0.42, dx=0.6):
    a = cv2.GaussianBlur(A, (0, 0), blur * u)
    M = np.float32([[1, 0, dx * u], [0, 1, dy * u]])
    a = cv2.warpAffine(a, M, (A.shape[1], A.shape[0]))
    B *= (1 - amt * a)[..., None]
    return B


def white_card(w, h, u, r=None, fill='#ffffff', a=1.0, shadow_a=0.24):
    c = D.rrect(int(w), int(h), (1.6 * u) if r is None else r, fill=fill, fill_a=a)
    c, p = D.shadow(c, blur=2.2 * u, dy=1.2 * u, color=(0.12, 0.02, 0.04), alpha=shadow_a)
    return c, p


def text_spr(text, key, size, color='#1a1a1a', tracking=0.0):
    T = D.Text(text, key, size, tracking=tracking, pad_px=int(size * 0.25))
    return T, T.sprite(color)


class Lay:
    """Where things can go during a beat (canvas pixels): around the face, never over it."""

    def __init__(self, ctx, t0, t1):
        sp = ctx.span(t0, t1)
        W, H = ctx.W, ctx.H
        self.W, self.H, self.u = W, H, ctx.u
        x0, y0, x1, y1 = sp['box']
        fh = y1 - y0
        # BlazeFace boxes stop near the mouth/chin: keep a margin so nothing touches the jaw or the forehead
        y0, y1 = y0 - 0.06 * fh, y1 + 0.1 * fh
        self.face = (x0, y0, x1, y1)
        self.face_pad = (x0 - 0.02 * W, y0 - 0.02 * H, x1 + 0.02 * W, y1 + 0.015 * H)
        self.head = min(sp['head'], y0)
        self.chin = y1
        self.cx = (x0 + x1) / 2
        self.eyes = sp['eyes']
        self.mouth = sp['mouth']
        self.person = sp['person']
        self.top = ctx.top_safe
        self.cap_top = ctx.cap_y - 0.04 * H
        self.chest = (y1 + 0.03 * H, self.cap_top - 0.012 * H)
        self.closeup = (fh > 0.2 * H) or (self.head - self.top < 0.1 * H)
        # the face itself (brows → chin): what nothing may ever cover
        ey, my = sp['eyes'][1], sp['mouth'][1]
        brow = ey - 0.55 * max(1.0, my - ey)
        self.core = (x0 + 0.06 * (x1 - x0), brow, x1 - 0.06 * (x1 - x0), y1)
        self.w = x1 - x0
        self.vertical = ctx.vertical
        # free side columns next to the face
        self.left = (0.03 * W, x0 - 0.025 * W)
        self.right = (x1 + 0.025 * W, 0.97 * W)

    def chest_h(self):
        return max(0.0, self.chest[1] - self.chest[0])


def word_spot(ctx, lay, text, place='auto', idx=0, max_w=0.86, cap_max=None):
    """(mode, font size, baseline y, centre x) for a big word that never covers the face."""
    W, H = ctx.W, ctx.H
    cap_max = cap_max or (0.098 if ctx.vertical else 0.16)
    size = L.big_size(text, W, H, max_w, cap_max)
    cap = size * 0.727
    opts = {}
    hd = lay.head
    # behind: the head covers ~34% of the letter height
    if ctx.has_mask:
        cb = min(cap, (hd - lay.top * 0.55) / 0.66)
        if cb >= 0.042 * H:
            opts['behind'] = (cb / 0.727, hd + 0.34 * cb)
    ca = min(cap, hd - 0.022 * H - lay.top)
    if ca >= 0.045 * H:
        opts['above'] = (ca / 0.727, hd - 0.022 * H)
    ch = lay.chest_h()
    cc = min(cap, ch * 0.82)
    if cc >= 0.034 * H:
        opts['chest'] = (cc / 0.727, lay.chest[0] + cc + 0.004 * H)
    if not ctx.vertical:
        # landscape: a column beside the head
        for side, (a, b) in (('left', lay.left), ('right', lay.right)):
            if b - a > 0.22 * W:
                s = L.big_size(text, b - a, H, 0.95, 0.14)
                opts[side] = (s, lay.eyes[1] + s * 0.36, (a + b) / 2)
    order = {'auto': ['behind', 'above', 'chest', 'left', 'right'], 'behind': ['behind', 'above', 'chest'],
             'above': ['above', 'behind', 'chest'], 'chest': ['chest', 'above', 'behind'], 'top': ['above', 'behind', 'chest'],
             'front': ['above', 'chest', 'behind'], 'left': ['left', 'chest', 'above'], 'right': ['right', 'chest', 'above']}.get(place, ['behind', 'above', 'chest'])
    for o in order + list(opts):
        if o in opts:
            v = opts[o]
            cx = v[2] if len(v) > 2 else W / 2
            return o, v[0], v[1], cx
    s = L.big_size(text, W, H, 0.7, 0.05)
    return 'chest', s, H * 0.62, W / 2


# ---------------------------------------------------------------- base class
class Beat:
    full_frame = False
    behind = False
    allow_bg = False
    pre = 0.0
    post = 0.3

    def __init__(self, spec, ctx, idx):
        self.spec, self.ctx, self.idx = spec, ctx, idx
        self.t0 = float(spec.get('at', spec.get('t', 0)))
        self.t1 = float(spec.get('end', self.t0 + 2.5))
        if self.t1 <= self.t0:
            self.t1 = self.t0 + 1.5
        self.word = str(spec.get('word') or '').strip()
        self.line = str(spec.get('line') or '').strip()
        self.items = [str(x.get('text', x) if isinstance(x, dict) else x) for x in (spec.get('items') or [])]
        self.item_times = [x.get('at') if isinstance(x, dict) else None for x in (spec.get('items') or [])]
        self.seed = stable_seed(idx, spec.get('fx', 'word'), self.word)
        self.R = rnd(self.seed)
        self.ready = False
        self.u = ctx.u
        self.W, self.H = ctx.W, ctx.H

    def on(self, t):
        return self.t0 - self.pre <= t <= self.t1 + self.post

    def bg_on(self, t):
        return True

    def ensure(self):
        if not self.ready:
            self.ready = True
            self.lay = Lay(self.ctx, self.t0, self.t1)
            self.prep()

    def prep(self):
        pass

    def release(self):
        """Free the big buffers of a beat that is over (prep runs again if it is needed later)."""
        if not self.ready:
            return
        for k, v in list(self.__dict__.items()):
            if isinstance(v, np.ndarray) and v.nbytes > 1500000:
                delattr(self, k)
            elif isinstance(v, (list, tuple)) and v and isinstance(v[0], (tuple, list)) and any(isinstance(x, np.ndarray) and x.nbytes > 1500000 for x in v[0]):
                delattr(self, k)
        self.ready = False

    def full(self, ctx, F, t):
        return None

    def bg(self, ctx, B, t):
        pass

    def front(self, ctx, F, t):
        pass

    def cam(self):
        return []

    def sounds(self):
        pass

    def hide_captions(self):
        return []

    # times of the beat's words
    def when(self, phrase, default):
        if not phrase:
            return default
        w = self.ctx.word_time(phrase, self.t0, self.t1)
        return default if w is None else max(self.t0, min(self.t1 - 0.2, w))

    def item_t(self, i, n=None):
        n = n or max(1, len(self.items))
        if i < len(self.item_times) and self.item_times[i] is not None:
            return float(self.item_times[i])
        even = self.t0 + 0.25 + (self.t1 - self.t0 - 0.6) * i / max(1, n)
        return self.when(self.items[i] if i < len(self.items) else '', even)


# ---------------------------------------------------------------- the normal Red Kinetic beat
class WordMixin:
    def prep_word(self, place=None, text=None, max_w=0.86, grad=L.GRAD, color=None, cap_max=None):
        text = (text or self.word or '').upper()
        if not text:
            self.bw = None
            return
        mode, size, base, cx = word_spot(self.ctx, self.lay, text, place or self.spec.get('place', 'auto'), self.idx, max_w, cap_max)
        self.mode, self.base_y, self.cx = mode, base, cx
        self.bw = L.BigWord(text, size, grad=grad, color=color)
        self.t_word = self.when(self.word, self.t0 + 0.05)
        self.word_box = (cx - self.bw.w / 2, base - self.bw.cap, cx + self.bw.w / 2, base)

    def draw_word(self, F, t, op=1.0, mode='slide'):
        if self.bw is None:
            return
        self.bw.draw(F, t, self.cx, self.base_y, self.t_word, self.t1 - 0.24, op=op, mode=mode)


@reg('word', aliases=('big-word', 'keyword', 'normal', 'name'))
class Word(Beat, WordMixin):
    behind = True

    def prep(self):
        self.prep_word()
        lay = self.lay
        self.thin = None
        if self.line and self.ctx.plan.get('captions', True) and not self.spec.get('keep_line'):
            # with subtitles on, a thin line that only repeats the spoken words is dropped (never the same words twice)
            spoken = set(P.norm(w['w']) for c in self.ctx.cues for w in c['words'] if self.t0 - 0.5 <= w['t0'] <= self.t1 + 0.5)
            lw = [P.norm(w) for w in self.line.split() if P.norm(w)]
            if lw and sum(1 for w in lw if w in spoken) >= len(lw) * 0.6:
                self.line = ''
        if self.line:
            red = self.spec.get('red')
            size = 4.3 * self.u if self.ctx.vertical else 3.2 * self.u
            # where the thin line goes: under the word when it is in the chest, else just under the chin
            y = None
            if self.bw is not None and self.mode == 'chest':
                above = self.word_box[1] - 0.022 * self.H
                below = self.word_box[3] + size * 1.25
                if above - size * 0.9 >= lay.chin + 0.004 * self.H:
                    y = above
                elif below + size * 0.35 <= lay.cap_top - (0.03 * self.H if self.ctx.caps is not None or self.ctx.plan.get('captions', True) else 0):
                    y = below
            elif self.bw is not None and self.mode in ('left', 'right'):
                y = self.word_box[3] + size * 1.3
            else:
                y = lay.chest[0] + size * 1.1
            if y is None:
                self.line = ''
                self.thin = None
                self.pill = None
                return
            dark = L.lum(self.ctx.still(self.t0), (self.W * 0.2, y - size, self.W * 0.8, y + size * 0.3)) > 0.62
            self.thin = L.thin_line(self.line, size, red, dark=dark)
            self.thin_xy = (self.cx if self.mode in ('left', 'right') else self.W / 2, y)
        self.pill = None
        if self.spec.get('pill'):
            self.pill = L.pill(str(self.spec['pill']), 3.9 * self.u)

    def bg_on(self, t):
        return self.bw is not None and self.mode == 'behind'

    def bg(self, ctx, B, t):
        self.ensure()
        if self.mode == 'behind':
            self.draw_word(B, t)

    def front(self, ctx, F, t):
        self.ensure()
        if self.bw is not None and self.mode != 'behind':
            self.draw_word(F, t)
        if self.bw is not None:
            L.pixels(F, t, self.t_word + 0.35, self.word_box, self.u, self.seed, n=4)
        tl = self.t_word + 0.18
        if self.thin is not None:
            a = ease_out_cubic(prog(t, tl, 0.45)) * (1 - prog(t, self.t1 - 0.22, 0.2))
            if a > 0:
                D.place(F, self.thin, self.thin_xy[0], self.thin_xy[1] + 8 * (1 - a), op=a, anchor=(0.5, 0.75))
        if self.pill is not None:
            s, a = pop_in(t, tl + 0.2, 0.45)
            a *= 1 - prog(t, self.t1 - 0.2, 0.18)
            if a > 0:
                D.place(F, self.pill, self.W * 0.5, self.thin_xy[1] + 6.2 * self.u if self.thin is not None else self.lay.chest[0] + 4 * self.u, scale=s, op=a, rot=-3)

    def sounds(self):
        self.ensure()
        if self.bw is not None:
            self.ctx.sfx(self.t_word, 'pop', 0.9)


@reg('punch')
class Punch(Word):
    def prep(self):
        self.spec.setdefault('place', 'chest')
        super().prep()

    def front(self, ctx, F, t):
        super().front(ctx, F, t)
        if self.bw is not None:
            L.halftone(F, t, self.t_word - 0.02, self.u, cx=self.cx, cy=self.base_y - self.bw.cap / 2)
            L.rgb_split(F, t, self.t_word, self.u, 0.08)

    def sounds(self):
        self.ensure()
        self.ctx.sfx(self.t_word, 'impact', 1.0)


@reg('cards', aliases=('card', 'photo-cards'))
class Cards(Word):
    """Big word above / behind the head and 1–3 red-bordered photo cards in the lower centre."""

    def prep(self):
        self.spec.setdefault('place', 'behind' if self.ctx.has_mask else 'above')
        self.line = ''
        super().prep()
        lay = self.lay
        cs = self.spec.get('cards') or [{}]
        if isinstance(cs, (str, dict)):
            cs = [cs]
        n = min(3, len(cs))
        zone_h = max(0.12 * self.H, lay.chest_h())
        ch = min(zone_h * 0.86, (0.26 if n == 1 else 0.2) * self.W * 1.0)
        cw = ch * (1.36 if n == 1 else 1.12)
        self.cards = []
        tilts = [-6, 5, -3]
        xs = {1: [0.5], 2: [0.33, 0.67], 3: [0.22, 0.5, 0.78]}[n]
        cy = lay.chest[0] + zone_h * 0.5
        for i, c in enumerate(cs[:n]):
            if isinstance(c, str):
                c = {'img': c}
            ref = c.get('img')
            img = img_from(self.ctx, ref, self.t0 + 0.2 * i, None if ref else bg_crop(self.ctx, self.t0 + 0.2 * i, i, aspect=1.3))
            lab = c.get('label')
            spr = L.card(img, cw, ch, label=lab, u=self.u)
            t_i = c.get('at') or self.when(lab.split()[0] if lab else '', self.t0 + 0.25 + 0.18 * i)
            self.cards.append((spr, self.W * xs[i], cy + (i % 2) * 0.012 * self.H, tilts[i % 3], float(t_i)))

    def front(self, ctx, F, t):
        super().front(ctx, F, t)
        for spr, x, y, rot, ti in self.cards:
            s, a = pop_in(t, ti, 0.55, 1.2)
            out = prog(t, self.t1 - 0.25, 0.22)
            a *= 1 - out
            if a > 0:
                D.place(F, spr, x, y, scale=(0.9 + 0.1 * s) * (1 - 0.1 * out), rot=rot * (0.6 + 0.4 * s), op=a)

    def sounds(self):
        super().sounds()
        for _, _, _, _, ti in self.cards[:1]:
            self.ctx.sfx(ti, 'tap', 0.7)


@reg('pills', aliases=('pill',))
class Pills(Word):
    def prep(self):
        self.line = ''
        super().prep()
        lay = self.lay
        self.pills = []
        n = len(self.items)
        if not n:
            return
        size = 4.0 * self.u
        sprs = [L.pill(x, size) for x in self.items[:4]]
        widths = [s.shape[1] for s in sprs]
        gap = 1.2 * self.u
        total = sum(widths) + gap * (len(sprs) - 1)
        y = lay.chest[0] + 6 * self.u if self.mode != 'chest' else self.word_box[3] + 6 * self.u
        x = self.W / 2 - total / 2
        for i, s in enumerate(sprs):
            self.pills.append((s, x + widths[i] / 2, y + (i % 2) * 1.2 * self.u, [-4, 3, -2, 4][i % 4], self.item_t(i, len(sprs))))
            x += widths[i] + gap

    def front(self, ctx, F, t):
        super().front(ctx, F, t)
        for spr, x, y, rot, ti in self.pills:
            s, a = pop_in(t, ti, 0.4, 1.6)
            a *= 1 - prog(t, self.t1 - 0.2, 0.18)
            if a > 0:
                D.place(F, spr, x, y, scale=s, rot=rot, op=a)


# ---------------------------------------------------------------- graphics-only beats
def icon(name, d, color='#ffffff'):
    """Simple icons drawn with shapes (d = size in px): percent, check, heart, star, play, clock, bolt, cart, tag."""
    from PIL import Image, ImageDraw
    S = 4
    n = max(8, int(d))
    im = Image.new('L', (n * S, n * S), 0)
    g = ImageDraw.Draw(im)
    k = n * S / 100.0
    P = lambda pts: [(x * k, y * k) for x, y in pts]
    w = int(9 * k)
    if name == 'check':
        g.line(P([(22, 52), (42, 72), (78, 30)]), fill=255, width=int(11 * k), joint='curve')
    elif name == 'heart':
        g.polygon(P(D.heart_pts(50, 54, 82)), fill=255)
    elif name == 'star':
        g.polygon(P(D.star_pts(50, 52, 46, 19, 5)), fill=255)
    elif name == 'play':
        g.polygon(P([(34, 22), (80, 50), (34, 78)]), fill=255)
    elif name == 'bolt':
        g.polygon(P([(58, 8), (22, 56), (48, 56), (40, 92), (78, 42), (52, 42)]), fill=255)
    elif name == 'clock':
        g.ellipse(P([(12, 12), (88, 88)]), outline=255, width=w)
        g.line(P([(50, 50), (50, 26)]), fill=255, width=w)
        g.line(P([(50, 50), (67, 60)]), fill=255, width=w)
    elif name == 'cart':
        g.line(P([(10, 22), (24, 22), (34, 64), (78, 64), (86, 34), (28, 34)]), fill=255, width=w, joint='curve')
        g.ellipse(P([(34, 72), (46, 84)]), fill=255)
        g.ellipse(P([(66, 72), (78, 84)]), fill=255)
    elif name == 'tag':
        g.polygon(P([(14, 46), (46, 14), (86, 14), (86, 54), (54, 86)]), fill=255)
        g.ellipse(P([(66, 26), (76, 36)]), fill=0)
    else:  # percent
        g.line(P([(26, 80), (74, 20)]), fill=255, width=int(10 * k))
        g.ellipse(P([(16, 16), (44, 44)]), outline=255, width=w)
        g.ellipse(P([(56, 56), (84, 84)]), outline=255, width=w)
    im = im.resize((n, n), Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255.0
    return D.to_sprite(a, hexrgb(color) if isinstance(color, str) else color)


def badge(name, d):
    base = D.rrect(d, d, d * 0.22, fill='#E0182D')
    g = D.grad_v(base.shape[0], base.shape[1], [(0, '#FF4A52'), (1, '#C40016')])
    base[..., :3] = g * base[..., 3:4]
    rim = D.rrect(d, d, d * 0.22, fill='#ffffff', border='#ffffff', bw=max(1.5, d * 0.045), fill_a=0)
    rim = rim[:base.shape[0], :base.shape[1]]
    base = rim * 0.6 + base * (1 - rim[..., 3:4] * 0.6)
    ic = icon(name, d * 0.62)
    D.blend_spr(base, ic, d * 0.19, d * 0.19)
    out, p = D.shadow(base, blur=d * 0.12, dy=d * 0.08, color=(0.35, 0, 0.04), alpha=0.35)
    return out


@reg('broll', aliases=('b-roll', 'phone-frame'))
class Broll(Beat):
    """Full-screen B-roll beat: the picture of what is said, a white outline phone frame with the text, a red badge."""
    full_frame = True
    pre = 0.0
    post = 0.0

    def prep(self):
        W, H, u = self.W, self.H, self.u
        ref = self.spec.get('img')
        self.img = img_from(self.ctx, ref, self.t0 + 0.1, None if ref else bg_crop(self.ctx, self.t0 + 0.1, 0, aspect=0.54))
        if self.ctx.vertical:
            self.pw, self.ph = 0.54 * W, 0.54 * W * 1.86
        else:
            self.pw, self.ph = 0.24 * W, 0.24 * W * 1.86
        self.pw, self.ph = int(self.pw), int(self.ph)
        r = 4.2 * u
        frame = D.rrect(self.pw, self.ph, r, fill='#ffffff', border='#ffffff', bw=0.42 * u, fill_a=0.0)
        glow = D.gblur(D.rrect(self.pw, self.ph, r, fill='#ff3341', border='#ff3341', bw=0.5 * u, fill_a=0), 1.2 * u)
        self.frame = glow * 0.8 + frame * 1.0
        sw, sh = self.pw - 2.2 * u, self.ph - 2.2 * u
        if ref or texture(self.img) > 9.0:
            screen = D.image_sprite(self.img, sw, sh, r=r - 1.1 * u)
            fade = D.grad_v(screen.shape[0], screen.shape[1], [(0, '#000000'), (0.6, '#000000'), (1, '#000000')])
            k = np.clip((np.arange(screen.shape[0], dtype=np.float32) / screen.shape[0] - 0.62) / 0.38, 0, 1)[:, None, None] * 0.55
            screen[..., :3] *= (1 - k)
        else:
            screen = self.designed(int(sw), int(sh), r - 1.1 * u)
        notch = D.rrect(8 * u, 1.6 * u, 0.8 * u, fill='#E0182D')
        D.blend_spr(screen, notch, screen.shape[1] / 2 - 4 * u, 1.4 * u)
        self.screen = screen
        self.bgimg = D.frame_blur(cv2.resize(self.img, (W, H)) if self.img.shape[:2] != (H, W) else self.img, 3.5 * u)
        txt = (self.word or '').upper()
        self.bw = L.BigWord(txt, L.big_size(txt, W, H, 0.84, 0.075)) if txt else None
        self.badge = badge(self.spec.get('icon', 'percent'), 9 * u)
        ln = self.line
        if ln and self.ctx.plan.get('captions', True) and not self.spec.get('keep_line'):
            spoken = set(P.norm(w['w']) for c in self.ctx.cues for w in c['words'] if self.t0 - 0.5 <= w['t0'] <= self.t1 + 0.5)
            lw = [P.norm(w) for w in ln.split() if P.norm(w)]
            if lw and sum(1 for w in lw if w in spoken) >= len(lw) * 0.6:
                ln = ''
        self.thin = L.thin_line(ln, 4 * u, self.spec.get('red'), dark=False) if ln else None

    def designed(self, w, h, r):
        """A phone screen built from the words when the footage has no picture of the thing: light UI, the key word, pills."""
        u = self.u
        scr = D.rrect(w, h, r, fill='#F4F4F7')
        g = D.grad_v(scr.shape[0], scr.shape[1], [(0, '#FFFFFF'), (1, '#E6E6EB')])
        scr[..., :3] = g * scr[..., 3:4]
        tile = badge(self.spec.get('icon', 'star'), w * 0.34)
        D.blend_spr(scr, tile, w / 2 - tile.shape[1] / 2, h * 0.16)
        txt = (self.word or '').title()
        size = min(w * 0.16, D.fit_size(txt, 'black_i', w * 0.82, w * 0.16))
        T, ts = text_spr(txt, 'black_i', size, L.INK)
        D.blend_spr(scr, ts, w / 2 - T.adv / 2 - T.left, h * 0.58 - T.base)
        words = [x for x in (self.items or P.content_words(self.spec.get('say') or self.line or ''))][:3]
        y = h * 0.66
        for i, wd in enumerate(words):
            pl = L.pill(wd, 3.0 * u, bg='#FFFFFF' if i % 2 else L.RED, fg=L.RED if i % 2 else '#FFFFFF', border=L.RED if i % 2 else None, bw=0.25 * u)
            D.blend_spr(scr, pl, w / 2 - pl.shape[1] / 2, y)
            y += pl.shape[0] + 0.6 * u
        return scr

    def full(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        p = ease_out_cubic(prog(t, self.t0, 0.45))
        z = 1.08 - 0.08 * ease_in_out_sine(prog(t, self.t0, self.t1 - self.t0))
        bg = self.bgimg
        if abs(z - 1) > 1e-3:
            M = np.float32([[z, 0, W / 2 * (1 - z)], [0, z, H / 2 * (1 - z)]])
            bg = cv2.warpAffine(bg, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        G = D.studio_bg(H, W, glow=1.0)
        out = bg * 0.55 + G * 0.45
        cx, cy = W / 2, H * (0.53 if ctx.vertical else 0.54)
        s = 0.92 + 0.08 * p
        D.place(out, self.screen, cx, cy, scale=s, op=p)
        D.place(out, self.frame, cx, cy, scale=s, op=p)
        D.place(out, self.badge, cx + self.pw * 0.5 * s - 1.5 * u, cy - self.ph * 0.5 * s + 1.5 * u, scale=ease_out_back(prog(t, self.t0 + 0.25, 0.4), 1.8), rot=8)
        if self.bw is not None:
            self.bw.draw(out, t, cx, cy - self.ph * 0.5 * s - 2.6 * u, self.t0 + 0.2, self.t1 - 0.2)
        if self.thin is not None:
            a = ease_out_cubic(prog(t, self.t0 + 0.45, 0.4))
            D.place(out, self.thin, cx, cy + self.ph * 0.2 * s, op=a)
        return out

    def sounds(self):
        self.ctx.sfx(self.t0 - 0.35, 'riser', 0.6)


@reg('end', aliases=('end-card', 'outro'))
class End(Beat):
    """End card: the channel / shop name built in red, a thin line and a Follow pill that gets tapped."""
    full_frame = True
    post = 0.0

    def prep(self):
        W, H, u = self.W, self.H, self.u
        name = (self.word or self.ctx.plan.get('brand') or 'Hal-abuur').upper()
        self.bw = L.BigWord(name, L.big_size(name, W, H, 0.78, 0.075))
        self.thin = L.thin_line(self.line or '', 4.2 * u, None, dark=True) if self.line and self.line.lower() not in ('follow', 'la soco') else None
        self.btn = L.pill(self.spec.get('button', 'Follow'), 4.4 * u)
        self.btn2 = L.pill(self.spec.get('done', 'Following'), 4.4 * u, bg='#1A1A1A')

    def full(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        p = ease_out_cubic(prog(t, self.t0, 0.4))
        G = D.studio_bg(H, W, glow=0.8)
        out = D.mix(D.frame_blur(F, 2.2 * u) * 0.9, G, 0.82 * p)
        y = H * 0.45
        self.bw.draw(out, t, W / 2, y, self.t0 + 0.1, self.t1 + 5)
        D.place(out, D.rrect(int(W * 0.16), max(2, int(0.25 * u)), 0.1 * u, fill='#E0182D'), W / 2, y + 3.4 * u, sx=ease_out_cubic(prog(t, self.t0 + 0.45, 0.4)), sy=1)
        if self.thin is not None:
            D.place(out, self.thin, W / 2, y + 8.5 * u, op=ease_out_cubic(prog(t, self.t0 + 0.55, 0.4)))
        tb = self.t0 + 0.75
        s, a = pop_in(t, tb, 0.4, 1.5)
        tap = prog(t, tb + 0.55, 0.25)
        by = y + (15 if self.thin is not None else 10) * u
        if tap < 0.5:
            D.place(out, self.btn, W / 2, by, scale=s * (1 - 0.08 * math.sin(math.pi * tap * 2)), op=a)
        else:
            D.place(out, self.btn2, W / 2, by, scale=1.0, op=a)
        if 0 < tap < 1:
            rr = D.ring(14 * u, 0.5 * u, '#E0182D', a=1 - tap)
            D.place(out, rr, W / 2, by, scale=0.4 + 0.8 * tap)
        return out

    def hide_captions(self):
        return [(self.t0 + 0.3, self.t1 + 1)]

    def sounds(self):
        self.ctx.sfx(self.t0 + 0.1, 'impact', 0.6)
        self.ctx.sfx(self.t0 + 1.32, 'click', 0.7)


@reg('split', aliases=('split-screen', 'panel'))
class Split(Beat, WordMixin):
    """Top: a light studio panel with the big word (and 1–2 tilted photo cards); bottom: my footage, reframed on the face."""
    full_frame = True
    post = 0.0

    def prep(self):
        W, H, u = self.W, self.H, self.u
        cs = self.spec.get('cards') or []
        if isinstance(cs, (str, dict)):
            cs = [cs]
        self.ph = int(H * (0.46 if cs else 0.4)) if self.ctx.vertical else W // 2
        txt = (self.word or '').upper()
        cap = 0.085 if cs else 0.11
        self.bw = L.BigWord(txt, L.big_size(txt, W, H, 0.84, cap)) if txt else None
        self.cards = []
        n = min(2, len(cs))
        for i, c in enumerate(cs[:n]):
            c = c if isinstance(c, dict) else {'img': c}
            ref = c.get('img')
            img = img_from(self.ctx, ref, self.t0 + 0.2 + i, None if ref else bg_crop(self.ctx, self.t0 + 0.2 + i, i, aspect=1.3))
            cw = 0.44 * W if n == 1 else 0.4 * W
            self.cards.append((L.card(img, cw, cw * 0.72, label=c.get('label'), u=u), (0.5 if n == 1 else (0.29 if i == 0 else 0.71)) * W, [-4, 5][i % 2],
                               c.get('at') or self.when((c.get('label') or '').split()[0] if c.get('label') else '', self.t0 + 0.35 + 0.15 * i)))
        self.t_word = self.when(self.word, self.t0 + 0.15)

    def full(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        p_in = ease_in_out_sine(prog(t, self.t0, 0.55))
        p_out = ease_in_out_sine(prog(t, self.t1 - 0.45, 0.45))
        k = p_in * (1 - p_out)
        if k <= 0:
            return None
        ph = int(self.ph * k)
        out = F.copy()
        if ctx.vertical:
            # footage: cover the lower part, face centred in it
            fh = H - ph
            f = ctx.face()
            s = max(1.0, fh / H * 1.18)
            cy = f['eyes'][1]
            src_y0 = clamp(cy - fh / s * 0.42, 0, H - fh / s)
            M = np.float32([[s, 0, -(f['cx'] - W / 2 / s) * s], [0, s, -src_y0 * s + ph]])
            M[0, 2] = W / 2 - f['cx'] * s
            low = cv2.warpAffine(F, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            out[ph:] = low[ph:]
            panel = D.studio_bg(self.ph, W, glow=1.0, cy=0.6)
            out[:ph] = panel[self.ph - ph:]
            out[max(0, ph - 2):ph + 2] = np.array(hexrgb('#E0182D'), np.float32)
            top = ph - self.ph
            if self.bw is not None:
                by = top + (self.ph * 0.34 if self.cards else self.ph * 0.6)
                by = max(by, top + ctx.top_safe + self.bw.cap)
                self.bw.draw(out, t, W / 2, by, self.t_word, self.t1 - 0.3)
            for spr, x, rot, ti in self.cards:
                s2, a = pop_in(t, float(ti), 0.5)
                D.place(out, spr, x, top + self.ph * 0.68, scale=0.9 + 0.1 * s2, op=a * (1 - p_out), rot=rot)
        return out

    def hide_captions(self):
        return []


from . import fx_mask, fx_ui  # noqa: E402,F401  (register the effect library)
