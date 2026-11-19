"""2D toolkit on float32 frames (H, W, 3) in 0..1 with premultiplied RGBA sprites (h, w, 4).

Everything is anti-aliased: shapes are drawn with Pillow at 2× and reduced, transforms use float affine warps.
"""
import math
import os
from functools import lru_cache

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .util import hexrgb, clamp

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(os.path.dirname(HERE), 'assets', 'fonts')
FONT_FILES = {
    'black_i': 'Inter-BlackItalic.otf', 'black': 'Inter-Black.otf', 'xb': 'Inter-ExtraBold.otf', 'xb_i': 'Inter-ExtraBoldItalic.otf',
    'bold': 'Inter-Bold.otf', 'bold_i': 'Inter-BoldItalic.otf', 'semi': 'Inter-SemiBold.otf', 'semi_i': 'Inter-SemiBoldItalic.otf',
    'reg': 'Inter-Regular.otf', 'reg_i': 'Inter-Italic.otf', 'med': 'Inter-Medium.otf',
    'serif': 'InstrumentSerif-Regular.ttf', 'serif_i': 'InstrumentSerif-Italic.ttf', 'mast': 'Gloock-Regular.ttf',
    'hand': 'NothingYouCouldDo-Regular.ttf',
}
try:
    _LAYOUT = ImageFont.Layout.RAQM if hasattr(ImageFont, 'Layout') and ImageFont.core.HAVE_RAQM else ImageFont.Layout.BASIC
except Exception:
    _LAYOUT = None


@lru_cache(maxsize=256)
def font(key, size):
    path = os.path.join(FONT_DIR, FONT_FILES.get(key, key))
    size = max(4, int(round(size)))
    try:
        if _LAYOUT is not None:
            return ImageFont.truetype(path, size, layout_engine=_LAYOUT)
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.truetype(path, size)


# ---------------------------------------------------------------- sprites
def to_sprite(alpha, rgb):
    """alpha (h, w) float + colour (3,) or (h, w, 3) → premultiplied sprite."""
    a = alpha[..., None].astype(np.float32)
    c = np.asarray(rgb, np.float32)
    if c.ndim == 1:
        c = c[None, None, :]
    return np.concatenate([c * a, a], axis=2)


def from_pil(im):
    a = np.asarray(im.convert('RGBA'), np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    return a


def empty(h, w):
    return np.zeros((max(1, int(h)), max(1, int(w)), 4), np.float32)


def pad(spr, p):
    p = int(math.ceil(p))
    return np.pad(spr, ((p, p), (p, p), (0, 0))) if p > 0 else spr


def opacity(spr, a):
    return spr * float(a)


def gblur(spr, sigma):
    if sigma <= 0.25:
        return spr
    k = int(sigma * 3) * 2 + 1
    return cv2.GaussianBlur(spr, (k, k), sigma, borderType=cv2.BORDER_CONSTANT)


def mblur(spr, dx, dy=0.0, grow=True):
    """Directional motion blur over a segment of dx/dy pixels (centred). grow=True pads the sprite by the blur length."""
    n = int(round(max(abs(dx), abs(dy)) / 2.0))
    if n < 1:
        return spr
    size = 2 * n + 1
    k = np.zeros((size, size), np.float32)
    cv2.line(k, (int(round(n - dx / 2)), int(round(n - dy / 2))), (int(round(n + dx / 2)), int(round(n + dy / 2))), 1.0, 1)
    k /= max(1e-6, k.sum())
    s = pad(spr, n) if grow else spr
    return cv2.filter2D(s, -1, k, borderType=cv2.BORDER_CONSTANT)


def shadow(spr, blur=8, dx=0, dy=6, color=(0, 0, 0), alpha=0.35, spread=0):
    """Sprite with a soft drop shadow under it (returns a larger sprite and the offset of the original)."""
    p = int(blur * 3 + max(abs(dx), abs(dy)) + spread + 2)
    s = pad(spr, p)
    a = s[..., 3].copy()
    if spread > 0:
        a = cv2.dilate(a, np.ones((int(spread) * 2 + 1,) * 2, np.uint8))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    a = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]))
    if blur > 0.3:
        k = int(blur * 3) * 2 + 1
        a = cv2.GaussianBlur(a, (k, k), blur)
    sh = to_sprite(a * alpha, color)
    out = s + sh * (1 - s[..., 3:4])
    return out, p


def over(dst, src):
    """Premultiplied src over dst (same size sprites)."""
    return src + dst * (1 - src[..., 3:4])


# ---------------------------------------------------------------- compositing onto frames
def blend(frame, spr, x, y, op=1.0):
    """Composite a premultiplied sprite at integer (x, y) (top-left) with clipping."""
    if op <= 0.002:
        return frame
    H, W = frame.shape[:2]
    h, w = spr.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return frame
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = frame[y0:y1, x0:x1]
    if op < 0.999:
        s = s * op
    k = 1.0 - s[..., 3]
    roi *= k[..., None]
    roi += s[..., :3]
    return frame


def place(frame, spr, cx, cy, scale=1.0, rot=0.0, op=1.0, anchor=(0.5, 0.5), blur=0.0, skew=0.0, sx=None, sy=None, mb=0.0):
    """Draw a sprite with its anchor at (cx, cy), scaled, rotated (degrees), skewed (degrees, x), blurred."""
    if op <= 0.002 or scale <= 0.001:
        return frame
    h, w = spr.shape[:2]
    if (sx is None and sy is None and abs(scale - 1) < 1e-3 and abs(rot) < 1e-3 and abs(skew) < 1e-3 and blur <= 0.3 and not mb):
        return blend(frame, spr, cx - anchor[0] * w, cy - anchor[1] * h, op)
    sx = scale if sx is None else sx
    sy = scale if sy is None else sy
    ax, ay = anchor[0] * w, anchor[1] * h
    r = math.radians(rot)
    c, s = math.cos(r), math.sin(r)
    k = math.tan(math.radians(skew))
    # p' = R · K · S · (p - a) + (cx, cy)
    A = np.array([[sx, k * sy], [0, sy]], np.float64)
    R = np.array([[c, -s], [s, c]], np.float64)
    L = R @ A
    t = np.array([cx, cy]) - L @ np.array([ax, ay])
    pts = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float64) @ L.T + t
    pd = int(blur * 3 + abs(mb) + 2)
    bx0, by0 = int(math.floor(pts[:, 0].min())) - pd, int(math.floor(pts[:, 1].min())) - pd
    bx1, by1 = int(math.ceil(pts[:, 0].max())) + pd, int(math.ceil(pts[:, 1].max())) + pd
    H, W = frame.shape[:2]
    cx0, cy0, cx1, cy1 = max(0, bx0), max(0, by0), min(W, bx1), min(H, by1)
    if cx1 <= cx0 or cy1 <= cy0:
        return frame
    M = np.hstack([L, (t - np.array([cx0, cy0]))[:, None]]).astype(np.float64)
    big = max(sx, sy) > 1.0
    out = cv2.warpAffine(spr, M, (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_CUBIC if big else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    if big:
        out = np.clip(out, 0, None)
        out[..., :3] = np.minimum(out[..., :3], out[..., 3:4])
    if mb:
        out = mblur(out, mb, 0.0, grow=False)
    if blur > 0.3:
        out = gblur(out, blur)
    return blend(frame, out, cx0, cy0, op)


def mix(a, b, t):
    return a * (1 - t) + b * t


# ---------------------------------------------------------------- colour / image ops on frames
def grad_v(h, w, stops):
    """Vertical gradient (h, w, 3). stops: [(pos 0..1, '#hex'), ...]."""
    ys = np.linspace(0, 1, max(2, h), dtype=np.float32)
    cols = np.zeros((len(ys), 3), np.float32)
    ps = [p for p, _ in stops]
    cs = [hexrgb(c) if isinstance(c, str) else c for _, c in stops]
    for ch in range(3):
        cols[:, ch] = np.interp(ys, ps, [c[ch] for c in cs])
    return np.repeat(cols[:, None, :], w, axis=1)


def radial(h, w, cx, cy, r, inner, outer, power=1.6):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(1.0, r)
    t = np.clip(d, 0, 1) ** power
    a = np.asarray(inner, np.float32)[None, None, :]
    b = np.asarray(outer, np.float32)[None, None, :]
    return a * (1 - t[..., None]) + b * t[..., None]


def studio_bg(H, W, glow=0.0, cx=0.5, cy=0.42):
    """Light studio grey of Red Kinetic (#f6f6f8 → #e6e6ea → #d9d9de) with an optional soft red glow."""
    bg = radial(H, W, W * cx, H * cy, max(W, H) * 0.75, hexrgb('#f7f7f9'), hexrgb('#d6d6dc'), 1.3)
    if glow > 0:
        g = radial(H, W, W * 0.5, H * 0.38, max(W, H) * 0.45, (1, 1, 1), (0, 0, 0), 1.0)[..., :1]
        red = np.array(hexrgb('#ff3341'), np.float32)
        bg = bg * (1 - g * glow * 0.22) + red * g * glow * 0.22
    return bg.astype(np.float32)


def vignette(frame, amount=0.25, power=2.2):
    H, W = frame.shape[:2]
    k = _vig(H, W, power)
    frame *= (1 - amount * k)[..., None]
    return frame


@lru_cache(maxsize=8)
def _vig(H, W, power):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
    return np.clip(d, 0, 1) ** power


def frame_blur(frame, sigma):
    if sigma < 0.4:
        return frame
    # blur at reduced size for speed (looks the same for large sigma)
    H, W = frame.shape[:2]
    f = 1
    while sigma / f > 6 and min(H, W) / (f * 2) > 64:
        f *= 2
    if f > 1:
        small = cv2.resize(frame, (W // f, H // f), interpolation=cv2.INTER_AREA)
        k = int(sigma / f * 3) * 2 + 1
        small = cv2.GaussianBlur(small, (k, k), sigma / f)
        return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    k = int(sigma * 3) * 2 + 1
    return cv2.GaussianBlur(frame, (k, k), sigma)


def desat(frame, amount):
    g = frame.mean(2, keepdims=True)
    return frame * (1 - amount) + g * amount


def to_u8(frame):
    return cv2.convertScaleAbs(frame, alpha=255.0)


def tight(spr, thr=0.002, m=1):
    """Crop a sprite to its visible pixels. Returns (sprite, ox, oy): the offset of the crop inside the original."""
    a = spr[..., 3]
    ys = np.where(a.max(1) > thr)[0]
    xs = np.where(a.max(0) > thr)[0]
    if not len(xs):
        return spr[:1, :1], 0, 0
    y0, y1 = max(0, ys[0] - m), min(spr.shape[0], ys[-1] + 1 + m)
    x0, x1 = max(0, xs[0] - m), min(spr.shape[1], xs[-1] + 1 + m)
    return np.ascontiguousarray(spr[y0:y1, x0:x1]), x0, y0


def to_f(img):
    return img.astype(np.float32) * (1 / 255.0)


# ---------------------------------------------------------------- shapes (2× supersampled)
SS = 2


def _canvas(w, h):
    return Image.new('L', (max(1, int(math.ceil(w * SS))), max(1, int(math.ceil(h * SS)))), 0)


def _done(im, w, h):
    im = im.resize((max(1, int(math.ceil(w))), max(1, int(math.ceil(h)))), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255.0


def rrect_alpha(w, h, r, inset=0.0):
    im = _canvas(w, h)
    d = ImageDraw.Draw(im)
    i = inset * SS
    d.rounded_rectangle([i, i, w * SS - 1 - i, h * SS - 1 - i], radius=max(0, (r - inset) * SS), fill=255)
    return _done(im, w, h)


def rrect(w, h, r, fill='#ffffff', border=None, bw=0.0, fill_a=1.0):
    """Rounded rectangle sprite with an optional border (border drawn inside the edge)."""
    outer = rrect_alpha(w, h, r)
    if border and bw > 0:
        inner = rrect_alpha(w, h, r, inset=bw)
        f = to_sprite(inner * fill_a, hexrgb(fill) if isinstance(fill, str) else fill)
        b = to_sprite(np.clip(outer - inner, 0, 1), hexrgb(border) if isinstance(border, str) else border)
        return f + b * (1 - f[..., 3:4])
    return to_sprite(outer * fill_a, hexrgb(fill) if isinstance(fill, str) else fill)


def ellipse_alpha(w, h):
    im = _canvas(w, h)
    ImageDraw.Draw(im).ellipse([0, 0, w * SS - 1, h * SS - 1], fill=255)
    return _done(im, w, h)


def circle(d, color='#ffffff', a=1.0):
    return to_sprite(ellipse_alpha(d, d) * a, hexrgb(color) if isinstance(color, str) else color)


def ring(d, width, color='#ffffff', a=1.0):
    im = _canvas(d, d)
    dr = ImageDraw.Draw(im)
    dr.ellipse([0, 0, d * SS - 1, d * SS - 1], outline=255, width=max(1, int(width * SS)))
    return to_sprite(_done(im, d, d) * a, hexrgb(color) if isinstance(color, str) else color)


def poly_alpha(w, h, pts, width=None, closed=True, joint='curve'):
    """pts in pixel coordinates of a w×h canvas. width=None → filled polygon, else a stroked line."""
    im = _canvas(w, h)
    d = ImageDraw.Draw(im)
    P = [(x * SS, y * SS) for x, y in pts]
    if width is None:
        d.polygon(P, fill=255)
    else:
        if closed:
            P = P + [P[0]]
        d.line(P, fill=255, width=max(1, int(round(width * SS))), joint=joint)
        r = width * SS / 2
        for x, y in (P[0], P[-1]):
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    return _done(im, w, h)


def line_sprite(w, h, pts, width, color='#ffffff', a=1.0):
    return to_sprite(poly_alpha(w, h, pts, width=width, closed=False) * a, hexrgb(color) if isinstance(color, str) else color)


def star_pts(cx, cy, r1, r2, n=4, rot=0):
    pts = []
    for i in range(n * 2):
        r = r1 if i % 2 == 0 else r2
        a = math.pi * i / n + math.radians(rot) - math.pi / 2
        pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
    return pts


def heart_pts(cx, cy, s, n=60):
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((cx + x * s / 32, cy + y * s / 32))
    return pts


# ---------------------------------------------------------------- text
class Text:
    """Rendered text: alpha mask, baseline, per-letter x offsets."""

    def __init__(self, text, key, size, tracking=0.0, pad_px=None):
        self.text, self.key, self.size = text, key, size
        f = font(key, size)
        asc, desc = f.getmetrics()
        self.asc, self.desc = asc, desc
        tr = tracking * size
        self.xs = []
        x = 0.0
        if tracking:
            for i, ch in enumerate(text):
                self.xs.append(x)
                x = f.getlength(text[:i + 1]) + tr * (i + 1)
            self.adv = x - tr if text else 0
        else:
            for i in range(len(text)):
                self.xs.append(f.getlength(text[:i]))
            self.adv = f.getlength(text)
        bb = f.getbbox('H')
        self.cap = bb[3] - bb[1]
        p = int(size * 0.35) if pad_px is None else int(pad_px)
        self.pad = p
        W = int(math.ceil(self.adv + size * 0.6)) + 2 * p
        H = int(asc + desc) + 2 * p
        im = Image.new('L', (W, H), 0)
        d = ImageDraw.Draw(im)
        if tracking:
            for ch, xx in zip(text, self.xs):
                d.text((p + xx, p + asc), ch, font=f, fill=255, anchor='ls')
        else:
            d.text((p, p + asc), text, font=f, fill=255, anchor='ls')
        self.alpha = np.asarray(im, np.float32) / 255.0
        # tight bbox
        ys, xs_ = np.where(self.alpha > 0.01)
        self.bbox = (int(xs_.min()), int(ys.min()), int(xs_.max()) + 1, int(ys.max()) + 1) if len(xs_) else (0, 0, W, H)
        self.base = p + asc              # baseline y inside alpha
        self.left = p

    def sprite(self, color='#1a1a1a', grad=None, a=1.0):
        if grad:
            top = self.base - self.cap * 1.08
            h, w = self.alpha.shape
            g = grad_v(h, w, [(clamp((top + (self.base - top) * p) / h), c) for p, c in grad])
            return to_sprite(self.alpha * a, g)
        return to_sprite(self.alpha * a, hexrgb(color) if isinstance(color, str) else color)

    def crop(self, spr):
        x0, y0, x1, y1 = self.bbox
        return spr[y0:y1, x0:x1]


def measure(text, key, size, tracking=0.0):
    f = font(key, size)
    if not tracking:
        return f.getlength(text)
    return f.getlength(text) + tracking * size * max(0, len(text) - 1)


def fit_size(text, key, max_w, max_size, min_size=8, tracking=0.0):
    s = max_size
    w = measure(text, key, s, tracking)
    if w > max_w:
        s = max(min_size, s * max_w / max(1.0, w))
    return s


def wrap(text, key, size, max_w, tracking=0.0, max_lines=3):
    words = text.split()
    lines, cur = [], ''
    for w in words:
        t = (cur + ' ' + w).strip()
        if cur and measure(t, key, size, tracking) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = t
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1] + [' '.join(lines[max_lines - 1:])]
    return lines


def label(text, key, size, color='#ffffff', bg=None, padx=0.55, pady=0.28, r=None, tracking=0.0, bw=0, border=None, shadow_a=0.0):
    """A text chip: text on an optional rounded background. Returns a sprite."""
    T = Text(text, key, size, tracking=tracking, pad_px=0)
    x0, y0, x1, y1 = T.bbox
    tw, th = T.adv, T.cap
    px, py = size * padx, size * pady
    W, H = int(math.ceil(tw + 2 * px)), int(math.ceil(th + 2 * py + size * 0.12))
    if bg:
        base = rrect(W, H, size * 0.36 if r is None else r, fill=bg, border=border, bw=bw)
    else:
        base = empty(H, W)
    ts = T.sprite(color)
    # put baseline so that caps are vertically centred
    ox = px - T.left
    oy = (H - th) / 2 + th - T.base - size * 0.02
    out = base.copy()
    blend_spr(out, ts, ox, oy)
    if shadow_a > 0:
        out, p = shadow(out, blur=size * 0.35, dy=size * 0.18, color=(0.3, 0, 0.03), alpha=shadow_a)
        return out, p
    return out


def blend_spr(dst, spr, x, y):
    """Composite sprite onto sprite (premultiplied) at float offset (integer rounded)."""
    H, W = dst.shape[:2]
    h, w = spr.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return dst
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    d = dst[y0:y1, x0:x1]
    d *= (1 - s[..., 3:4])
    d += s
    return dst


def image_sprite(rgb, w, h, r=0.0):
    """Cover-fit an RGB float/uint8 image into w×h with rounded corners → sprite."""
    img = rgb if rgb.dtype == np.float32 else rgb.astype(np.float32) / 255.0
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    nw, nh = int(math.ceil(iw * s)), int(math.ceil(ih * s))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    x0, y0 = (nw - int(w)) // 2, (nh - int(h)) // 2
    im = np.clip(im[y0:y0 + int(h), x0:x0 + int(w)], 0, 1)
    a = rrect_alpha(int(w), int(h), r) if r > 0 else np.ones((int(h), int(w)), np.float32)
    a = a[:im.shape[0], :im.shape[1]]
    return np.concatenate([im * a[..., None], a[..., None]], axis=2)
