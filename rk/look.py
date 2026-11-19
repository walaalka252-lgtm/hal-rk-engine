"""The Red Kinetic look: big red-gradient words built letter by letter, thin italic lines, red pills,
red-bordered photo cards, pixel blocks, the template captions and the transitions."""
import math
import numpy as np
import cv2

from . import draw as D
from .util import clamp, prog, ease_out_cubic, ease_in_cubic, ease_out_back, ease_in_out_sine, hexrgb, ease_out_expo, stable_seed, rnd

RED = '#E0182D'
RED2 = '#FF3341'
DRED = '#8E0010'
INK = '#1A1A1A'
MUTED = '#8A8A90'
GREY = '#E8E8EC'
WHITE = '#FFFFFF'
GRAD = [(0.0, '#FF4A52'), (0.45, '#E0182D'), (1.0, '#A3000F')]
WGRAD = [(0.0, '#FFFFFF'), (1.0, '#F1F1F4')]


def lum(frame, box):
    """Mean luminance of a frame region (x0, y0, x1, y1)."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.5
    r = frame[y0:y1:4, x0:x1:4]
    return float((r[..., 0] * .3 + r[..., 1] * .59 + r[..., 2] * .11).mean())


# ---------------------------------------------------------------- big word
class BigWord:
    """Huge Inter Black Italic word with the red gradient; letters slide in from the left with a blur."""

    def __init__(self, text, size, grad=GRAD, color=None, tracking=-0.055, shadow=True, key='black_i', outline=None):
        self.text = text
        self.size = size
        T = D.Text(text, key, size, tracking=tracking)
        self.T = T
        self.w = T.adv
        self.cap = T.cap
        self.letters = []
        f = D.font(key, size)
        for i, ch in enumerate(text):
            if ch == ' ':
                self.letters.append(None)
                continue
            L = D.Text(ch, key, size)
            spr = L.sprite(color or INK, grad=None if color else grad)
            if outline:
                a = L.alpha
                k = max(1, int(size * 0.035))
                dil = cv2.dilate(a, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
                ring = D.to_sprite(np.clip(dil - a, 0, 1), hexrgb(outline))
                spr = spr + ring * (1 - spr[..., 3:4])
            if shadow:
                spr, p = D.shadow(spr, blur=size * 0.045, dy=size * 0.035, color=(0.24, 0, 0.03), alpha=0.42)
                spr2, p2 = D.shadow(spr, blur=size * 0.012, dy=size * 0.01, color=(0.15, 0, 0.02), alpha=0.25)
                spr, p = spr2, p + p2
            else:
                p = 0
            spr, ox, oy = D.tight(spr)
            # letter origin = baseline-left of the glyph inside its (cropped) sprite
            self.letters.append((spr, T.xs[i], L.left + p - ox, L.base + p - oy))

    def draw(self, frame, t, x, y, t_in, t_out, stagger=0.055, dur=0.26, anchor_x=0.5, op=1.0, mode='slide', scale=1.0):
        """x, y = position of the baseline; anchor_x: 0 left … 0.5 centre … 1 right."""
        if t < t_in - 0.01 or t > t_out + 0.6:
            return
        ox = x - self.w * anchor_x * scale
        n = sum(1 for L in self.letters if L)
        k = 0
        for i, L in enumerate(self.letters):
            if not L:
                continue
            spr, lx, px, py = L
            a = prog(t, t_in + k * stagger, dur)
            e = ease_out_cubic(a)
            b = prog(t, t_out + k * 0.022, 0.2)
            eb = ease_in_cubic(b)
            k += 1
            if a <= 0 or eb >= 1:
                continue
            dx = -0.25 * self.size * (1 - e) + 0.35 * self.size * eb
            blur = 0.06 * self.size * (1 - e) * 0.35 + 0.05 * self.size * eb * 0.35
            mb = (0.18 * self.size * (1 - e) + 0.3 * self.size * eb) if mode == 'slide' else 0
            o = op * e * (1 - eb)
            if mode == 'blur':
                s = scale * (1.08 - 0.08 * e) * (1 + 0.06 * eb)
                cx = ox + lx * scale + (spr.shape[1] / 2 - px) * scale
                cy = y + (spr.shape[0] / 2 - py) * scale
                D.place(frame, spr, cx, cy, scale=s, op=o, blur=0.07 * self.size * scale * max(1 - e, eb))
                continue
            cx = ox + lx * scale + (spr.shape[1] / 2 - px) * scale + dx * scale
            cy = y + (spr.shape[0] / 2 - py) * scale
            D.place(frame, spr, cx, cy, scale=scale, op=o, blur=blur * scale, mb=mb * scale)

    def width(self, scale=1.0):
        return self.w * scale


def big_size(text, W, H, max_w=0.86, max_cap=0.105, min_size=40):
    """Font size so the word fills ~max_w of the width without its cap height passing max_cap of the height."""
    s = 400.0
    w = D.measure(text, 'black_i', s, -0.055)
    size = s * (W * max_w) / max(1.0, w)
    cap = size * 0.727
    if cap > H * max_cap:
        size = H * max_cap / 0.727
    return max(min_size, size)


# ---------------------------------------------------------------- thin line, pills, tags
def thin_line(text, size, red_word=None, dark=True):
    """Thin italic line (Inter Italic 400): ink or white, one word in red. Returns sprite."""
    words = text.split()
    key = 'reg_i'
    sp = D.measure(' ', key, size)
    parts = []
    x = 0.0
    for w in words:
        is_red = red_word and w.strip('.,!?').lower() == red_word.lower()
        col = RED if is_red else (INK if dark else WHITE)
        T = D.Text(w, key, size, pad_px=int(size * 0.3))
        parts.append((T, col, x))
        x += T.adv + sp
    total = x - sp
    T0 = parts[0][0]
    Wd = int(total + size * 1.0)
    Hd = T0.alpha.shape[0]
    out = D.empty(Hd, Wd)
    for T, col, xx in parts:
        D.blend_spr(out, T.sprite(col), xx + size * 0.2 - T.left + T.pad - T.pad, 0)
    if not dark:
        out, p = D.shadow(out, blur=size * 0.18, dy=size * 0.06, color=(0, 0, 0), alpha=0.55)
    else:
        out, p = D.shadow(out, blur=size * 0.2, dy=size * 0.04, color=(1, 1, 1), alpha=0.45)
    return D.tight(out)[0]


def pill(text, size, bg=RED, fg=WHITE, key='semi_i', border=None, bw=0):
    spr = D.label(text, key, size, color=fg, bg=bg, padx=0.62, pady=0.36, r=size * 0.42, border=border, bw=bw)
    spr, p = D.shadow(spr, blur=size * 0.35, dy=size * 0.2, color=(0.35, 0, 0.04), alpha=0.32)
    return D.tight(spr)[0]


def tag(text, size):
    """Small uppercase label tag (white ExtraBold on red) for cards."""
    spr = D.label(text.upper(), 'xb', size, color=WHITE, bg=RED, padx=0.55, pady=0.32, r=size * 0.32, tracking=0.04)
    return spr


# ---------------------------------------------------------------- photo cards
def card(img, w, h, label=None, u=10.8, border=RED, white_mat=False):
    """Red-bordered rounded photo card with a soft shadow and an optional red label tag."""
    w, h = int(w), int(h)
    r = 1.45 * u
    bw = max(3.0, 0.36 * u)
    base = D.rrect(w, h, r, fill=WHITE)
    if img is not None:
        inset = bw + (0.45 * u if white_mat else 0)
        ph = D.image_sprite(img, w - 2 * inset, h - 2 * inset, r=max(0, r - inset))
        D.blend_spr(base, ph, inset, inset)
    ring = D.rrect(w, h, r, fill=WHITE, border=border, bw=bw, fill_a=0.0)
    base = ring + base * (1 - ring[..., 3:4])
    if label:
        tg = tag(label, max(14, 2.1 * u))
        D.blend_spr(base, tg, 0.9 * u, h - tg.shape[0] - 0.9 * u)
    spr, p = D.shadow(base, blur=2.2 * u, dy=1.2 * u, color=(0.3, 0, 0.04), alpha=0.3)
    return D.tight(spr)[0]


# ---------------------------------------------------------------- pixel blocks
def pixels(frame, t, t0, box, u, seed, n=5, dur=0.75):
    """Small red/black square blocks flickering in steps around a box (x0, y0, x1, y1)."""
    if t < t0 or t > t0 + dur:
        return
    R = rnd(seed)
    x0, y0, x1, y1 = box
    step = int((t - t0) * 12)
    for i in range(n):
        side = R.choice([0, 1, 2, 3])
        if side == 0:
            px, py = R.uniform(x0, x1), y0 - R.uniform(0.6, 2.4) * u
        elif side == 1:
            px, py = R.uniform(x0, x1), y1 + R.uniform(0.6, 2.4) * u
        elif side == 2:
            px, py = x0 - R.uniform(0.5, 2.0) * u, R.uniform(y0, y1)
        else:
            px, py = x1 + R.uniform(0.5, 2.0) * u, R.uniform(y0, y1)
        s = R.choice([1.5, 1.9, 2.4, 1.3]) * u
        col = R.choice([RED2, RED2, RED, INK])
        pattern = R.getrandbits(9) | 1
        if (pattern >> (step % 9)) & 1:
            sq = D.to_sprite(np.ones((max(2, int(s)), max(2, int(s))), np.float32), hexrgb(col))
            D.blend(frame, sq, px, py)


# ---------------------------------------------------------------- captions
class Captions:
    """Bottom captions in the Red Kinetic design: Inter ExtraBold Italic, white with a dark edge,
    2–4 words at a time, and a red rounded box that slides onto the word being spoken."""

    def __init__(self, cues, W, H, words_per=3, max_chars=22, y=None, size=None):
        self.W, self.H = W, H
        u = min(W, H) / 100.0
        self.u = u
        self.size = size or (5.4 * u if H >= W else 4.2 * u)
        self.y = y if y is not None else (0.765 * H if H > W * 1.2 else 0.86 * H)
        self.chunks = []
        for c in cues:
            words = c['words']
            # group words into chunks of ≤ words_per words and ≤ max_chars characters
            cur = []
            for w in words:
                txt = ' '.join(x['w'] for x in cur + [w])
                if cur and (len(cur) >= words_per or len(txt) > max_chars):
                    self.chunks.append(cur)
                    cur = []
                cur.append(w)
            if cur:
                self.chunks.append(cur)
        # chunk time spans: from first word start to next chunk start (or cue end)
        for i, ch in enumerate(self.chunks):
            t0 = ch[0]['t0']
            t1 = ch[-1]['t1']
            if i + 1 < len(self.chunks):
                nxt = self.chunks[i + 1][0]['t0']
                if nxt - t1 < 0.6:
                    t1 = nxt
            ch_info = {'t0': t0, 't1': t1 + 0.12}
            self.chunks[i] = (ch, ch_info)
        # never two chunks at once: a chunk ends where the next one starts
        for i in range(len(self.chunks) - 1):
            nxt = self.chunks[i + 1][1]['t0']
            if self.chunks[i][1]['t1'] > nxt - 0.02:
                self.chunks[i][1]['t1'] = nxt - 0.02
        self.cache = {}
        self.boxes = {}
        self.hidden = []   # (t0, t1) spans where captions are hidden

    def _render(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        words, info = self.chunks[idx]
        key = 'xb_i'
        size = self.size
        maxw = self.W * 0.84
        txt = [w['w'] for w in words]
        sp = D.measure(' ', key, size) * 1.05
        widths = [D.measure(w, key, size) for w in txt]
        total = sum(widths) + sp * (len(txt) - 1)
        if total > maxw:
            size *= maxw / total
            widths = [D.measure(w, key, size) for w in txt]
            sp = D.measure(' ', key, size) * 1.05
            total = sum(widths) + sp * (len(txt) - 1)
        sprites = []
        x = -total / 2
        for w, ww in zip(txt, widths):
            T = D.Text(w, key, size, pad_px=int(size * 0.4))
            a = T.alpha
            k = max(2, int(size * 0.075))
            edge = cv2.dilate(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
            edge = cv2.GaussianBlur(edge, (0, 0), size * 0.03)
            fill = D.to_sprite(a, (1, 1, 1))
            ring = D.to_sprite(np.clip(edge, 0, 1) * 0.82, hexrgb('#141416'))
            spr = fill + ring * (1 - fill[..., 3:4])
            spr, p = D.shadow(spr, blur=size * 0.22, dy=size * 0.08, color=(0, 0, 0), alpha=0.5)
            spr, ox, oy = D.tight(spr)
            sprites.append({'spr': spr, 'base': T.base + p - oy, 'x': x, 'w': ww, 'cap': T.cap, 'left': T.left + p - ox})
            x += ww + sp
        out = {'size': size, 'items': sprites}
        self.cache[idx] = out
        if len(self.cache) > 40:
            self.cache.pop(next(iter(self.cache)))
        return out

    def hide(self, t0, t1):
        self.hidden.append((t0, t1))

    def draw(self, frame, t, y=None):
        if any(a <= t <= b for a, b in self.hidden):
            return
        y = self.y if y is None else y
        for idx, (words, info) in enumerate(self.chunks):
            if not (info['t0'] - 0.05 <= t <= info['t1']):
                continue
            R = self._render(idx)
            size = R['size']
            a_in = ease_out_cubic(prog(t, info['t0'] - 0.05, 0.16))
            a_out = 1 - prog(t, info['t1'] - 0.08, 0.08)
            op = a_in * a_out
            if op <= 0:
                continue
            s = 0.92 + 0.08 * ease_out_back(prog(t, info['t0'] - 0.05, 0.2), 1.6)
            cx = self.W / 2
            # active word → red box sliding between words
            act = 0
            for i, w in enumerate(words):
                if t >= w['t0'] - 0.03:
                    act = i
            it = R['items']
            prev = max(0, act - 1)
            w_act = words[act]
            k = ease_out_cubic(prog(t, w_act['t0'] - 0.03, 0.1)) if act > 0 else 1.0
            bx = it[prev]['x'] + (it[act]['x'] - it[prev]['x']) * k
            bw = it[prev]['w'] + (it[act]['w'] - it[prev]['w']) * k
            padx, pady = size * 0.2, size * 0.17
            cap = it[act]['cap']
            bh = cap + 2 * pady + size * 0.1
            bkey = (int(bw + 2 * padx) // 2 * 2, int(bh))
            box = self.boxes.get(bkey)
            if box is None:
                box = D.rrect(bkey[0], bkey[1], size * 0.2, fill=RED)
                box = D.tight(D.shadow(box, blur=size * 0.25, dy=size * 0.1, color=(0.25, 0, 0.02), alpha=0.35)[0])[0]
                self.boxes[bkey] = box
                if len(self.boxes) > 200:
                    self.boxes.clear()
            D.place(frame, box, cx + (bx + bw / 2) * s, y - cap / 2 * s + size * 0.02, scale=s, op=op, skew=-8)
            for i, item in enumerate(it):
                spr = item['spr']
                ccx = cx + (item['x'] + (spr.shape[1] / 2 - item['left'])) * s
                ccy = y + (spr.shape[0] / 2 - item['base']) * s
                D.place(frame, spr, ccx, ccy, scale=s, op=op)


# ---------------------------------------------------------------- transitions (drawn over the frame)
def streak(frame, t, t0, u, seed=1, dur=0.42):
    """Red neon light streak (white core, red glow) that draws across the frame."""
    p = prog(t, t0, dur)
    if p <= 0 or p >= 1:
        return
    H, W = frame.shape[:2]
    R = rnd(seed)
    flip = R.random() < 0.5
    pts = []
    for i in range(24):
        q = i / 23
        x = W * (-0.1 + 1.2 * q)
        y = H * (0.72 - 0.5 * q) + math.sin(q * 3.1 + seed) * H * 0.04
        if flip:
            x = W - x
        pts.append((x, y))
    head = ease_out_cubic(clamp(p / 0.6))
    tail = ease_in_cubic(clamp((p - 0.35) / 0.65))
    n0, n1 = int(tail * 23), max(1, int(head * 23))
    if n1 <= n0:
        return
    seg = pts[n0:n1 + 1]
    sh = 4
    small = np.zeros((H // sh, W // sh), np.float32)
    P = np.array([(x / sh, y / sh) for x, y in seg], np.int32)
    cv2.polylines(small, [P], False, 1.0, thickness=max(2, int(1.6 * u / sh)), lineType=cv2.LINE_AA)
    glow = cv2.GaussianBlur(small, (0, 0), 2.2 * u / sh) * 2.2
    glow = cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR)
    core = np.zeros((H, W), np.float32)
    cv2.polylines(core, [np.array(seg, np.int32)], False, 1.0, thickness=max(2, int(0.45 * u)), lineType=cv2.LINE_AA)
    core = cv2.GaussianBlur(core, (0, 0), 0.25 * u)
    fade = 1 - ease_in_cubic(clamp((p - 0.7) / 0.3))
    g = np.clip(glow, 0, 1)[..., None] * fade
    red = np.array(hexrgb('#FF2A3A'), np.float32)
    frame[:] = frame * (1 - g * 0.85) + red * g * 0.95
    c = np.clip(core * 1.4, 0, 1)[..., None] * fade
    frame[:] = frame * (1 - c) + c


def halftone(frame, t, t0, u, cx=None, cy=None, dur=0.55):
    """Ring of red halftone dots bursting outward from a point."""
    p = prog(t, t0, dur)
    if p <= 0 or p >= 1:
        return
    H, W = frame.shape[:2]
    cx = W / 2 if cx is None else cx
    cy = H * 0.5 if cy is None else cy
    e = ease_out_cubic(p)
    r0 = (0.05 + 0.9 * e) * max(W, H) * 0.62
    thick = (0.16 + 0.1 * (1 - e)) * max(W, H) * 0.3
    sh = 2
    h2, w2 = H // sh, W // sh
    yy, xx = np.mgrid[0:h2, 0:w2].astype(np.float32)
    d = np.sqrt((xx - cx / sh) ** 2 + (yy - cy / sh) ** 2)
    band = np.clip(1 - np.abs(d - r0 / sh) / (thick / sh), 0, 1)
    cell = 1.3 * u / sh
    gx = (xx % cell) - cell / 2
    gy = (yy % cell) - cell / 2
    dot_r = cell * 0.45 * band * (1 - p * 0.6)
    dots = np.clip(dot_r - np.sqrt(gx ** 2 + gy ** 2) + 0.5, 0, 1)
    a = cv2.resize(dots, (W, H), interpolation=cv2.INTER_LINEAR)[..., None] * (1 - ease_in_cubic(clamp((p - 0.6) / 0.4)))
    red = np.array(hexrgb(RED2), np.float32)
    frame[:] = frame * (1 - a) + red * a


def flash(frame, t, t0, dur=0.1):
    p = prog(t, t0, dur)
    if p <= 0 or p >= 1:
        return
    a = (1 - p) * 0.75
    pink = np.array(hexrgb('#FFE3E8'), np.float32)
    frame[:] = frame * (1 - a) + pink * a


def rgb_split(frame, t, t0, u, dur=0.1):
    p = prog(t, t0, dur)
    if p <= 0 or p >= 1:
        return
    d = int(1.3 * u * (1 - p * 0.5))
    r = np.roll(frame[..., 0], d, axis=1)
    b = np.roll(frame[..., 2], -d, axis=1)
    frame[..., 0] = r
    frame[..., 2] = b


def blur_zoom(frame, t, tc, dur=0.32):
    """Zoom blur centred on the cut time tc (peak at tc)."""
    p = 1 - abs(t - tc) / (dur / 2)
    if p <= 0:
        return
    H, W = frame.shape[:2]
    acc = frame.copy()
    n = 5
    amt = 0.09 * ease_out_cubic(p)
    for i in range(1, n):
        s = 1 + amt * i / (n - 1)
        M = np.float32([[s, 0, W / 2 * (1 - s)], [0, s, H / 2 * (1 - s)]])
        acc += cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    frame[:] = acc / n
