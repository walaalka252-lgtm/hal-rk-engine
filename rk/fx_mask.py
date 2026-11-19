"""Effects that use the person mask (my person layer composited over a changed background)."""
import math

import numpy as np
import cv2

from . import draw as D, look as L
from .fx import (reg, Beat, WordMixin, word_spot, img_from, side_crop, bg_crop, pop_in, typed, place_masked, scale_about,
                 contact_shadow, white_card, text_spr, badge, icon)
from .util import (clamp, prog, lerp, ease_out_cubic, ease_in_cubic, ease_in_out_cubic, ease_in_out_sine, ease_out_back,
                   hexrgb, rnd, window, stable_seed)


def env(t, t0, t1, fin=0.3, fout=0.25):
    return window(t, t0, t1, fin, fout)


def text_card(text, w, h, u, size=None):
    """White card with a red border and one word / short phrase (for orbit and walls when no picture is given)."""
    base = D.rrect(int(w), int(h), 1.2 * u, fill='#ffffff', border=L.RED, bw=max(3, 0.34 * u))
    size = size or min(h * 0.22, D.fit_size(text, 'xb_i', w * 0.8, h * 0.24))
    lines = D.wrap(text, 'xb_i', size, w * 0.8, max_lines=3)
    y = h / 2 - (len(lines) - 1) * size * 0.55
    for ln in lines:
        T, s = text_spr(ln, 'xb_i', size, L.INK)
        D.blend_spr(base, s, w / 2 - T.adv / 2 - T.left, y + T.cap / 2 - T.base)
        y += size * 1.1
    bar = D.rrect(int(w * 0.3), max(3, int(0.5 * u)), 0.25 * u, fill=L.RED)
    D.blend_spr(base, bar, w * 0.35, h * 0.12)
    out, p = D.shadow(base, blur=1.8 * u, dy=1.0 * u, color=(0.3, 0, 0.04), alpha=0.3)
    return out


# ---------------------------------------------------------------- Orbit
@reg('orbit', need='mask', fallback='headline-wall')
class Orbit(Beat):
    """4–6 small red-bordered cards travel on a tilted ellipse around me: behind me on the far side, in front on the near side."""
    fields = 'items (the things I name) and/or cards [{img, label}]'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        cards = self.spec.get('cards') or []
        n = max(4, min(6, len(cards) or len(self.items) or 5))
        cw = (0.24 if self.ctx.vertical else 0.12) * W
        ch = cw * 1.3
        self.sprs = []
        dur = self.ctx.info['dur']
        for i in range(n):
            if i < len(cards):
                c = cards[i] if isinstance(cards[i], dict) else {'img': cards[i]}
                img = img_from(self.ctx, c.get('img'), self.t0 + 0.3 * i)
                self.sprs.append(L.card(img, cw, ch, label=c.get('label'), u=u))
            else:
                at = clamp(self.t0 - 5 + 2.3 * i, 0, max(0, dur - 0.3))
                img = img_from(self.ctx, None, at, bg_crop(self.ctx, at, i, aspect=0.77))
                lab = self.items[i % len(self.items)] if self.items else None
                self.sprs.append(L.card(img, cw, ch, label=lab, u=u))
        self.n = n
        if lay.closeup:
            self.rx, self.ry = 0.47 * W, 0.05 * H
            # the near cards stay clear of the captions (their bottom edge above the caption box)
            self.c = (lay.cx, min(lay.chin + 0.06 * H, lay.chest[1] - self.ry - ch * 0.5 + 0.02 * H))
        else:
            self.c = (lay.cx, lay.eyes[1] + 0.02 * H)
            self.rx, self.ry = max(0.3 * W, lay.w * 1.1), 0.055 * H
        self.period = float(self.spec.get('period', 3.6))

    def state(self, i, t):
        e_in = ease_out_cubic(prog(t, self.t0, 0.55))
        e_out = ease_in_cubic(prog(t, self.t1 - 0.4, 0.4))
        th = 2 * math.pi * ((t - self.t0) / self.period + i / self.n) + 0.6
        r = 0.3 + 0.7 * e_in + 0.7 * e_out
        x = self.c[0] + self.rx * r * math.cos(th)
        y = self.c[1] + self.ry * r * math.sin(th)
        tilt = math.radians(-7)
        dx, dy = x - self.c[0], y - self.c[1]
        x, y = self.c[0] + dx * math.cos(tilt) - dy * math.sin(tilt), self.c[1] + dx * math.sin(tilt) + dy * math.cos(tilt)
        z = math.sin(th)
        return x, y, z, e_in * (1 - e_out)

    def front(self, ctx, F, t):
        self.ensure()
        A = ctx.alpha()
        st = sorted([(self.state(i, t), i) for i in range(self.n)], key=lambda s: s[0][2])
        for (x, y, z, op), i in st:
            if op <= 0:
                continue
            s = 0.72 + 0.28 * (z + 1) / 2
            occ = clamp(0.5 - z * 2.5)      # behind me on the far half, smooth switch at the sides
            place_masked(F, self.sprs[i], x, y, A=A, occ=occ, scale=s, rot=-6 * math.cos(z), op=op, blur=1.4 * self.u * clamp(-z))

    def sounds(self):
        self.ctx.sfx(self.t0, 'whoosh', 0.8)


# ---------------------------------------------------------------- Couch wall
@reg('couch-wall', need='mask', fallback='headline-wall')
class CouchWall(Beat):
    """The wall behind me becomes a collage of tilted red-bordered cards and headline scraps, drifting sideways; I stay in front."""
    behind = True
    fields = 'items (headline words), word'

    def prep(self):
        W, H, u = self.W, self.H, self.u
        words = self.items or [w for w in (self.line or self.word or '').split() if len(w) > 2] or [self.word or 'Hal-abuur']
        cols, rows = (3, 5) if self.ctx.vertical else (5, 3)
        self.tiles = []
        k = 0
        for r in range(rows):
            for c in range(cols):
                photo = (r + c) % 2 == 0
                tw = W / cols * self.R.uniform(0.82, 0.98)
                th = H / rows * self.R.uniform(0.62, 0.8) if photo else H / rows * 0.32
                if photo:
                    at = clamp(self.t0 - 6 + k * 1.7, 0, max(0, self.ctx.info['dur'] - 0.5))
                    img = img_from(self.ctx, None, at)
                    spr = L.card(img, tw, th, u=u * 0.8)
                else:
                    txt = ' '.join(words[(k + j) % len(words)] for j in range(2))
                    spr = self.scrap(txt, tw, th)
                x = (c + 0.5) * W / cols + self.R.uniform(-0.05, 0.05) * W
                y = (r + 0.5) * H / rows + self.R.uniform(-0.04, 0.04) * H
                self.tiles.append((spr, x, y, self.R.uniform(-7, 7), self.t0 + 0.04 * k, self.R.choice([-1, 1])))
                k += 1

    def scrap(self, txt, w, h):
        u = self.u
        base = D.rrect(int(w), int(h), 0.4 * u, fill='#F6F4EF' if self.R.random() < 0.5 else '#FFFFFF')
        size = min(h * 0.42, D.fit_size(txt.upper(), 'black_i', w * 0.84, h * 0.5))
        T, s = text_spr(txt.upper(), 'black_i', size, L.INK if self.R.random() < 0.6 else L.RED)
        D.blend_spr(base, s, w * 0.08 - T.left, h / 2 + T.cap / 2 - T.base)
        out, p = D.shadow(base, blur=1.2 * u, dy=0.8 * u, color=(0, 0, 0), alpha=0.35)
        return out

    def bg(self, ctx, B, t):
        self.ensure()
        e = env(t, self.t0, self.t1, 0.25, 0.25)
        if e <= 0:
            return
        B *= (1 - 0.55 * e)
        drift = 0.03 * self.W * (t - self.t0) / max(0.5, self.t1 - self.t0)
        for spr, x, y, rot, ti, d in self.tiles:
            s, a = pop_in(t, ti, 0.35, 1.4)
            D.place(B, spr, x + d * drift, y, scale=0.85 + 0.15 * s, rot=rot, op=a * e)
        contact_shadow(B, ctx.alpha(), self.u, amt=0.5 * e)

    def sounds(self):
        self.ctx.sfx(self.t0, 'whoosh', 0.7)


# ---------------------------------------------------------------- UI world
@reg('ui-world', need='mask', fallback='prompt-box')
class UIWorld(Beat):
    """The room behind me turns into a clean app world; a white chat box at the top types my sentence; I stay in front."""
    full_frame = True
    fields = 'line (the sentence typed in the chat box), label'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        self.text = self.line or self.spec.get('say') or self.word
        self.bw_ = int(W * (0.9 if self.ctx.vertical else 0.56))
        size = 4.3 * u
        self.lines = D.wrap(self.text, 'semi', size, self.bw_ * 0.8, max_lines=2)
        self.size = size
        self.bh = int(size * (2.2 + 1.35 * len(self.lines)) + 6 * u)
        self.box, self.bp = white_card(self.bw_, self.bh, u, r=2.2 * u, shadow_a=0.18)
        self.Ts = [text_spr(ln, 'semi', size, L.INK) for ln in self.lines]
        lab = self.spec.get('label') or 'Hal-abuur'
        self.lab = text_spr(lab, 'xb', 1.9 * u, '#8A8A90', tracking=0.08)
        self.send = D.circle(5.4 * u, L.RED)
        arr = D.line_sprite(int(5.4 * u), int(5.4 * u), [(2.7 * u, 3.9 * u), (2.7 * u, 1.5 * u)], 0.55 * u)
        arr2 = D.line_sprite(int(5.4 * u), int(5.4 * u), [(1.7 * u, 2.5 * u), (2.7 * u, 1.5 * u), (3.7 * u, 2.5 * u)], 0.55 * u)
        D.blend_spr(self.send, arr, 0, 0)
        D.blend_spr(self.send, arr2, 0, 0)
        self.sp = 0.8 if lay.closeup else 0.92
        self.uibg = D.radial(H, W, W / 2, H * 0.35, max(W, H) * 0.8, hexrgb('#FFFFFF'), hexrgb('#DCDCE2'), 1.4).astype(np.float32)
        self.type_t = (self.t0 + 0.35, max(0.6, min(self.t1 - self.t0 - 0.8, 0.045 * len(self.text))))
        head_s = H + self.sp * (lay.head - H)
        self.box_y = clamp(head_s - 0.04 * H - self.bh / 2, self.ctx.top_safe + self.bh / 2, H * 0.4)
        chips = self.items[:3]
        self.chips = [L.pill(c, 3.0 * u, bg='#FFFFFF', fg=L.INK, border='#DADAE0', bw=0.18 * u) for c in chips]
        self.title = text_spr(self.spec.get('title') or (self.word or '').title(), 'serif', 7.5 * u, L.INK) if (self.spec.get('title') or self.word) else None

    def full(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        e = env(t, self.t0, self.t1, 0.35, 0.3)
        if e <= 0:
            return None
        A = ctx.alpha()
        s = lerp(1.0, self.sp, ease_in_out_cubic(e))
        plate = ctx.plate(self.t0)
        bgm = plate * (1 - e) + self.uibg * e
        Fp = scale_about(F, s, W / 2, H)
        Ap = scale_about(A, s, W / 2, H)
        contact_shadow(bgm, Ap, u, amt=0.25 * e)
        out = bgm * (1 - Ap[..., None]) + Fp * Ap[..., None]
        # chat box drops in from the top
        y = lerp(-self.bh, self.box_y - self.bh / 2, ease_out_cubic(e)) + self.bh / 2
        x = W / 2
        if self.title is not None:
            TT, ts = self.title
            D.place(out, ts, W / 2, y - self.bh / 2 - 3.6 * u, op=ease_out_cubic(prog(t, self.t0 + 0.2, 0.4)) * e)
        D.place(out, self.box, x, y, op=e)
        bx0, by0 = x - self.bw_ / 2, y - self.bh / 2
        LT, ls = self.lab
        D.place(out, ls, bx0 + 3.2 * u + ls.shape[1] / 2 - LT.left, by0 + 3.0 * u, op=e)
        D.place(out, D.circle(1.1 * u, L.RED), bx0 + 2.4 * u, by0 + 3.0 * u, op=e)
        frac = prog(t, *self.type_t)
        n_all = sum(len(T.text) for T, _ in self.Ts)
        left = frac * n_all
        yy = by0 + 3.2 * u + self.size * 1.6
        cur = None
        for T, sp in self.Ts:
            k = clamp(left / max(1, len(T.text)))
            left -= len(T.text)
            spr = typed(T, sp, k)
            if spr is not None:
                D.place(out, spr, bx0 + 3.2 * u + spr.shape[1] / 2 - T.left, yy - T.base + spr.shape[0] / 2, op=e)
                cur = (bx0 + 3.2 * u + T.xs[min(len(T.text) - 1, int(k * len(T.text)))] + (T.adv / len(T.text) if k > 0 else 0), yy)
            yy += self.size * 1.35
        if cur and (int(t * 2.5) % 2 == 0 or frac < 1):
            D.place(out, D.rrect(max(2, int(0.3 * u)), int(self.size * 1.1), 0.1 * u, fill=L.RED), cur[0] + 0.4 * u, cur[1] - self.size * 0.38, op=e)
        if self.chips:
            tw = sum(c.shape[1] for c in self.chips) + 1.0 * u * (len(self.chips) - 1)
            cx0 = W / 2 - tw / 2
            for i, c in enumerate(self.chips):
                sc, a = pop_in(t, self.t0 + 0.5 + 0.12 * i, 0.35, 1.5)
                D.place(out, c, cx0 + c.shape[1] / 2, y + self.bh / 2 + 3.4 * u, scale=sc, op=a * e)
                cx0 += c.shape[1] + 1.0 * u
        tap = prog(t, self.type_t[0] + self.type_t[1] + 0.15, 0.3)
        D.place(out, self.send, bx0 + self.bw_ - 4.6 * u, by0 + self.bh - 4.6 * u, scale=1 - 0.12 * math.sin(math.pi * tap), op=e)
        if 0 < tap < 1:
            D.place(out, D.ring(9 * u, 0.4 * u, L.RED, a=1 - tap), bx0 + self.bw_ - 4.6 * u, by0 + self.bh - 4.6 * u, scale=0.5 + 0.9 * tap)
        return out

    def sounds(self):
        self.ctx.sfx(self.t0, 'whoosh', 0.7)
        self.ctx.sfx(self.type_t[0] + self.type_t[1] + 0.15, 'click', 0.8)


# ---------------------------------------------------------------- Frame break
@reg('frame-break', need='mask', fallback='pull-back')
class FrameBreak(Beat):
    """My footage shrinks into a rounded red-bordered portrait frame on a dark blurred copy of itself; my head breaks out over the top edge."""
    full_frame = True

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        self.k = 0.74 if self.ctx.vertical else 0.62
        fw = W * (0.74 if self.ctx.vertical else 0.4)
        fh = H * (0.6 if self.ctx.vertical else 0.74)
        # pick the shrink anchor so that the frame's top edge crosses my head about 10% of the height below its top
        py = H * 0.62
        head_screen = py + self.k * (lay.head - py)
        top_screen = head_screen + (0.11 if self.ctx.vertical else 0.14) * H
        self.py = py
        self.target = (W / 2 - fw / 2, top_screen, W / 2 + fw / 2, min(H * 0.95, top_screen + fh))

    def full(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        e = ease_in_out_cubic(prog(t, self.t0, 0.5)) * (1 - ease_in_out_cubic(prog(t, self.t1 - 0.45, 0.45)))
        if e <= 0:
            return None
        A = ctx.alpha()
        k = lerp(1.0, self.k, e)
        S = scale_about(F, k, W / 2, self.py, border=cv2.BORDER_REFLECT)
        SA = scale_about(A, k, W / 2, self.py)
        bg = D.frame_blur(F, 3.5 * u) * (1 - 0.6 * e)
        x0, y0, x1, y1 = [lerp(a, b, e) for a, b in zip((0, 0, W, H), self.target)]
        r = lerp(0, 2.4 * u, e)
        rw, rh = max(2, int(x1 - x0)), max(2, int(y1 - y0))
        m = np.zeros((H, W), np.float32)
        ra = D.rrect_alpha(rw, rh, r)
        xi, yi = int(x0), int(y0)
        hh, ww = min(H - yi, ra.shape[0]), min(W - xi, ra.shape[1])
        m[yi:yi + hh, xi:xi + ww] = ra[:hh, :ww]
        sh = cv2.GaussianBlur(m, (0, 0), 2.5 * u)
        sh = np.roll(sh, int(1.4 * u), axis=0)
        out = bg * (1 - 0.5 * sh[..., None] * e)
        out = out * (1 - m[..., None]) + S * m[..., None]
        bw = max(3, int(0.34 * u))
        ring = D.rrect(rw, rh, r, fill='#ffffff', border=L.RED, bw=bw, fill_a=0.0)
        D.blend(out, ring, xi, yi, e)
        # the head breaks out: my person layer above the frame's top edge (and over the border)
        yy = np.arange(H, dtype=np.float32)[:, None]
        above = np.clip((y0 + bw * 1.5 - yy) / (0.006 * H) + 0.5, 0, 1)
        bo = (SA * above)[..., None] * e
        out = out * (1 - bo) + S * bo
        return out

    def sounds(self):
        self.ctx.sfx(self.t0, 'whoosh', 0.7)


# ---------------------------------------------------------------- Green screen
@reg('green-screen', need='mask', fallback='broll')
class GreenScreen(Beat):
    """My person layer is lifted onto a new clean background (light studio with a red glow, or a dark stage) with a soft contact shadow; a card above shows a picture."""
    full_frame = True
    fields = 'bg: light|dark, img (picture for the card), word'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        self.dark = self.spec.get('bg', 'light' if self.idx % 2 == 0 else 'dark') == 'dark'
        if self.dark:
            self.newbg = (D.radial(H, W, W / 2, H * 0.55, max(W, H) * 0.6, hexrgb('#3a3a42'), hexrgb('#0e0e11'), 1.2)).astype(np.float32)
            glow = D.radial(H, W, lay.cx, H * 0.5, 0.6 * W, (1.0, 0.25, 0.3), (0, 0, 0), 1.3)
            self.newbg = np.clip(self.newbg + glow * 0.22, 0, 1).astype(np.float32)
        else:
            self.newbg = D.studio_bg(H, W, glow=1.0)
        self.sp = 0.82 if lay.closeup else 0.92
        ref = self.spec.get('img')
        img = img_from(self.ctx, ref, self.t0, None if ref else None)
        cw = 0.46 * W if self.ctx.vertical else 0.28 * W
        self.card = L.card(img, cw, cw * 0.62, label=self.spec.get('label') or (self.word.title() if self.word else None), u=u)
        head_s = H + self.sp * (lay.head - H)
        self.card_y = max(self.ctx.top_safe + cw * 0.31 + 1.5 * u, min(head_s - cw * 0.35, H * 0.22))
        self.card_x = W * (0.72 if lay.cx < W / 2 else 0.28) if lay.closeup else W / 2

    def full(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        e = env(t, self.t0, self.t1, 0.3, 0.25)
        if e <= 0:
            return None
        A = ctx.alpha()
        s = lerp(1, self.sp, ease_in_out_cubic(e))
        bg = ctx.plate(self.t0) * (1 - e) + self.newbg * e
        Fp = scale_about(F, s, W / 2, H)
        Ap = scale_about(A, s, W / 2, H)
        contact_shadow(bg, Ap, u, amt=0.45 * e, blur=3.5)
        out = bg * (1 - Ap[..., None]) + Fp * Ap[..., None]
        sc, a = pop_in(t, self.t0 + 0.3, 0.5)
        D.place(out, self.card, self.card_x, self.card_y, scale=0.9 + 0.1 * sc, rot=-4, op=a * e)
        return out

    def sounds(self):
        self.ctx.sfx(self.t0, 'whoosh', 0.7)


# ---------------------------------------------------------------- Paper cut
def paper_texture(H, W, seed):
    R = np.random.default_rng(seed)
    n = cv2.resize(R.random((H // 8, W // 8)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    fine = R.random((H, W)).astype(np.float32)
    kraft = np.array(hexrgb('#C9A87C'), np.float32)
    base = kraft[None, None, :] * (0.9 + 0.12 * n[..., None]) * (0.96 + 0.05 * fine[..., None])
    return base.astype(np.float32)


def torn_poly(cx, cy, w, h, R, jag=0.03):
    pts = []
    for side in range(4):
        for i in range(14):
            q = i / 14
            if side == 0:
                x, y = cx - w / 2 + w * q, cy - h / 2
            elif side == 1:
                x, y = cx + w / 2, cy - h / 2 + h * q
            elif side == 2:
                x, y = cx + w / 2 - w * q, cy + h / 2
            else:
                x, y = cx - w / 2, cy + h / 2 - h * q
            x += R.uniform(-1, 1) * w * jag
            y += R.uniform(-1, 1) * h * jag
            pts.append((x, y))
    return pts


@reg('paper-cut', need='mask', fallback='word')
class PaperCut(Beat):
    """A scrapbook collage: torn kraft and grid paper, I become a paper cut-out with a white border, the key word lands as ransom-note letters, crayon scribbles draw on (12 fps stop motion)."""
    full_frame = True

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        R = self.R
        bg = paper_texture(H, W, self.seed % 1000)
        im = np.zeros((H, W), np.uint8)
        from PIL import Image, ImageDraw
        layers = []
        for i, (col, cx, cy, w, h) in enumerate([('#FBFBF7', W * 0.32, H * 0.3, W * 0.8, H * 0.34), ('#F2E7B5', W * 0.78, H * 0.55, W * 0.6, H * 0.3),
                                                  ('#FFFFFF', W * 0.25, H * 0.78, W * 0.7, H * 0.32)]):
            pts = torn_poly(cx, cy, w, h, R)
            a = D.poly_alpha(W, H, pts)
            c = np.array(hexrgb(col), np.float32)
            if i == 0:
                # grid paper
                g = np.ones((H, W), np.float32)
                step = int(2.6 * u)
                g[::step, :] = 0.82
                g[:, ::step] = 0.82
                c = c[None, None, :] * g[..., None]
            sh = cv2.GaussianBlur(a, (0, 0), 1.2 * u)
            bg = bg * (1 - 0.25 * np.roll(sh, int(0.6 * u), axis=0)[..., None])
            bg = bg * (1 - a[..., None]) + c * a[..., None]
        self.bgimg = bg.astype(np.float32)
        self.sp = 0.8 if lay.closeup else 0.9
        word = (self.word or 'HAL-ABUUR').upper()[:12]
        cols = [('#E0182D', '#FFFFFF'), ('#1A1A1A', '#FFFFFF'), ('#FFFFFF', '#1A1A1A'), ('#F4EBD0', '#E0182D')]
        fonts = ['black', 'mast', 'serif', 'black_i', 'xb']
        size = min(0.16 * W, 0.9 * W / max(4, len(word)))
        self.letters = []
        x = W / 2 - size * 0.86 * len(word) / 2
        for i, ch in enumerate(word):
            if ch == ' ':
                x += size * 0.5
                continue
            bgc, fg = cols[R.randrange(len(cols))]
            fk = fonts[R.randrange(len(fonts))]
            T, s = text_spr(ch, fk, size * R.uniform(0.82, 1.0), fg)
            x0, y0, x1, y1 = T.bbox
            gw, gh = x1 - x0, y1 - y0
            pw, ph = gw + size * 0.36, gh + size * 0.36
            scrap = D.to_sprite(D.poly_alpha(pw, ph, torn_poly(pw / 2, ph / 2, pw * 0.96, ph * 0.96, R, 0.02)), hexrgb(bgc))
            D.blend_spr(scrap, s[y0:y1, x0:x1], size * 0.18, size * 0.18)
            scrap, p = D.shadow(scrap, blur=0.5 * u, dy=0.4 * u, alpha=0.35)
            self.letters.append((scrap, x + pw / 2, R.uniform(-0.012, 0.012) * H, R.uniform(-9, 9)))
            x += size * 0.86
        self.word_y = max(self.ctx.top_safe + size * 0.7, min(H * 0.17, H + self.sp * (lay.head - H) - size * 0.2))
        # scribbles
        self.scribbles = []
        for i in range(3):
            cx, cy = R.uniform(0.15, 0.85) * W, R.uniform(0.3, 0.6) * H
            pts = [(cx + (j / 12 - 0.5) * 0.3 * W, cy + (0.04 if j % 2 else -0.04) * H + R.uniform(-0.01, 0.01) * H) for j in range(13)]
            col = ['#3A57C8', '#E0182D', '#8A8A90'][i]
            self.scribbles.append((pts, col))
        self.xmark = (W * (0.84 if lay.cx < W / 2 else 0.16), H * 0.42)

    def full(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        e = env(t, self.t0, self.t1, 0.12, 0.15)
        if e <= 0:
            return None
        tq = math.floor((t - self.t0) * 12) / 12.0 + self.t0     # stop motion
        A = ctx.alpha()
        Fp = scale_about(F, self.sp, W / 2, H)
        Ap = scale_about(A, self.sp, W / 2, H)
        out = self.bgimg.copy()
        # scribbles drawing on
        for i, (pts, col) in enumerate(self.scribbles):
            f = prog(tq, self.t0 + 0.2 + i * 0.25, 0.5)
            if f > 0:
                n = max(2, int(f * len(pts)))
                al = np.zeros((H, W), np.float32)
                cv2.polylines(al, [np.array(pts[:n], np.int32)], False, 1.0, int(1.1 * u), cv2.LINE_AA)
                tex = (np.random.default_rng(int(tq * 12) + i).random((H, W)) > 0.35).astype(np.float32)
                al = al * tex * 0.85
                c = np.array(hexrgb(col), np.float32)
                out = out * (1 - al[..., None]) + c * al[..., None]
        # paper cut-out of me: white border + shadow
        k = max(3, int(1.3 * u))
        border = cv2.dilate((Ap > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))).astype(np.float32)
        border = cv2.GaussianBlur(border, (0, 0), 0.6)
        sh = cv2.GaussianBlur(border, (0, 0), 1.2 * u)
        sh = np.roll(np.roll(sh, int(0.8 * u), axis=0), int(0.5 * u), axis=1)
        out *= (1 - 0.35 * sh)[..., None]
        out = out * (1 - border[..., None]) + border[..., None]
        out = out * (1 - Ap[..., None]) + Fp * Ap[..., None]
        # ransom letters, one by one
        for i, (spr, x, dy, rot) in enumerate(self.letters):
            ti = self.t0 + 0.25 + i * 0.09
            if tq >= ti:
                j = rnd(int(tq * 12) * 31 + i)
                D.place(out, spr, x + j.uniform(-1, 1) * 0.15 * u, self.word_y + dy + j.uniform(-1, 1) * 0.15 * u, rot=rot + j.uniform(-1.5, 1.5))
        # red underline + X mark
        f = prog(tq, self.t0 + 0.25 + len(self.letters) * 0.09, 0.3)
        if f > 0 and self.letters:
            xs = [l[1] for l in self.letters]
            y = self.word_y + 0.06 * H
            al = np.zeros((H, W), np.float32)
            cv2.line(al, (int(xs[0] - 2 * u), int(y)), (int(xs[0] - 2 * u + (xs[-1] - xs[0] + 4 * u) * f), int(y + 0.4 * u)), 1.0, int(0.9 * u), cv2.LINE_AA)
            red = np.array(hexrgb(L.RED), np.float32)
            out = out * (1 - al[..., None]) + red * al[..., None]
        f = prog(tq, self.t0 + 0.5, 0.25)
        if f > 0:
            x, y = self.xmark
            s = 3.2 * u
            al = np.zeros((H, W), np.float32)
            cv2.line(al, (int(x - s), int(y - s)), (int(x - s + 2 * s * min(1, f * 2)), int(y - s + 2 * s * min(1, f * 2))), 1.0, int(0.8 * u), cv2.LINE_AA)
            if f > 0.5:
                g = (f - 0.5) * 2
                cv2.line(al, (int(x + s), int(y - s)), (int(x + s - 2 * s * g), int(y - s + 2 * s * g)), 1.0, int(0.8 * u), cv2.LINE_AA)
            red = np.array(hexrgb(L.RED), np.float32)
            out = out * (1 - al[..., None]) + red * al[..., None]
        return F * (1 - e) + out * e

    def sounds(self):
        self.ctx.sfx(self.t0, 'paper', 1.0)


# ---------------------------------------------------------------- Doodle boil
@reg('doodle-boil', need='mask', fallback='word')
class DoodleBoil(Beat):
    """A hand-drawn white chalk outline traces my silhouette and is redrawn every 3 frames with a wobble; a red heart and sparkles draw on near my head."""

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        side = 1 if lay.cx < W / 2 else -1
        hx = lay.cx + side * (lay.w * 0.75 + 0.06 * W)
        self.heart = (clamp(lay.cx - side * (lay.w * 0.72 + 0.05 * W), 0.12 * W, 0.88 * W), clamp(lay.head + 0.02 * H, self.ctx.top_safe + 0.04 * H, H))
        self.stars = [(clamp(hx, 0.1 * W, 0.9 * W), clamp(lay.head + 0.05 * H, self.ctx.top_safe + 0.05 * H, H), 6.0 * u),
                      (clamp(hx + side * 0.08 * W, 0.08 * W, 0.92 * W), clamp(lay.head + 0.0 * H, self.ctx.top_safe + 0.02 * H, H), 3.4 * u)]
        self.heart = (self.heart[0], self.heart[1] + 0.04 * H)
        self.col = None

    def front(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        e = env(t, self.t0, self.t1, 0.12, 0.2)
        if e <= 0:
            return
        A = ctx.alpha()
        q = 2
        small = cv2.resize(A, (W // q, H // q), interpolation=cv2.INTER_AREA)
        k = max(2, int(1.15 * u / q))
        m = cv2.dilate((small > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            return
        c = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float32) * q
        step = max(1, len(c) // 260)
        c = c[::step]
        boil = int((t - self.t0) * self.ctx.fps / 3)
        R = np.random.default_rng(self.seed + boil)
        c = c + R.normal(0, 0.28 * u, c.shape)
        # do not draw along the frame edges
        keep = (c[:, 0] > 2) & (c[:, 0] < W - 3) & (c[:, 1] > 2) & (c[:, 1] < H - 3)
        al = np.zeros((H, W), np.float32)
        seg = []
        for p, kk in zip(c, keep):
            if kk:
                seg.append(p)
            elif len(seg) > 2:
                cv2.polylines(al, [np.array(seg, np.int32)], False, 1.0, max(2, int(0.75 * u)), cv2.LINE_AA)
                seg = []
            else:
                seg = []
        if len(seg) > 2:
            cv2.polylines(al, [np.array(seg, np.int32)], False, 1.0, max(2, int(0.75 * u)), cv2.LINE_AA)
        tex = 0.7 + 0.3 * (R.random((H // 4, W // 4)).astype(np.float32))
        al *= cv2.resize(tex, (W, H), interpolation=cv2.INTER_NEAREST)
        col = self.col
        if col is None:
            ring = cv2.dilate(al, np.ones((9, 9), np.uint8)) > 0.2
            lum = float((F[ring] @ np.array([0.3, 0.59, 0.11], np.float32)).mean()) if ring.any() else 0.3
            col = self.col = np.array(hexrgb(L.RED2 if lum > 0.62 else '#FFFFFF'), np.float32)
        sh = cv2.GaussianBlur(al, (0, 0), 0.5 * u)
        F *= (1 - 0.35 * sh * e)[..., None]
        F[:] = F * (1 - al[..., None] * e) + col * al[..., None] * e
        # doodles drawing on stroke by stroke
        hx, hy = self.heart
        f = prog(t, self.t0 + 0.25, 0.45)
        if f > 0:
            pts = D.heart_pts(hx, hy, 13 * u, 40)
            n = max(2, int(f * len(pts)))
            jr = np.random.default_rng(boil + 5)
            P = np.array([(x + jr.normal(0, 0.15 * u), y + jr.normal(0, 0.15 * u)) for x, y in pts[:n]], np.int32)
            al = np.zeros((H, W), np.float32)
            cv2.polylines(al, [P], False, 1.0, max(2, int(0.5 * u)), cv2.LINE_AA)
            if f >= 1:
                fill = np.zeros((H, W), np.float32)
                cv2.fillPoly(fill, [P], 1.0, cv2.LINE_AA)
                hatch = np.zeros((H, W), np.float32)
                for i in range(-16, 16):
                    x0 = int(hx + i * 0.9 * u)
                    cv2.line(hatch, (x0 - int(7 * u), int(hy + 7 * u)), (x0 + int(7 * u), int(hy - 7 * u)), 1.0, max(1, int(0.35 * u)), cv2.LINE_AA)
                al = np.maximum(al, fill * hatch * 0.85)
            red = np.array(hexrgb(L.RED2), np.float32)
            F[:] = F * (1 - al[..., None] * e) + red * al[..., None] * e
        for i, (sx, sy, r) in enumerate(self.stars):
            f = prog(t, self.t0 + 0.4 + i * 0.18, 0.3)
            if f <= 0:
                continue
            pts = D.star_pts(sx, sy, r * ease_out_back(f, 1.8), r * 0.32, 4, rot=boil * 3 % 12)
            al = np.zeros((H, W), np.float32)
            cv2.polylines(al, [np.array(pts, np.int32)], True, 1.0, max(2, int(0.45 * u)), cv2.LINE_AA)
            F[:] = F * (1 - al[..., None] * e) + al[..., None] * e

    def sounds(self):
        self.ctx.sfx(self.t0 + 0.25, 'tap', 0.6)


# ---------------------------------------------------------------- Vertigo
@reg('vertigo', need='mask', fallback='digital-zoom')
class Vertigo(Beat):
    """A dolly zoom: I keep my size and place while the room behind me zooms (100% → 118% or back)."""
    full_frame = True

    def full(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        e = env(t, self.t0, self.t1, 0.12, 0.15)
        if e <= 0:
            return None
        p = ease_in_out_sine(prog(t, self.t0, self.t1 - self.t0))
        z = 1 + 0.18 * (p if self.spec.get('dir', 'in') == 'in' else 1 - p)
        plate = ctx.plate(self.t0)
        lay = self.lay
        bgz = scale_about(plate, z, lay.cx, lay.head + 0.1 * self.H, border=cv2.BORDER_REFLECT)
        bgz = D.frame_blur(bgz, 0.25 * u * (z - 1) / 0.18)
        A = ctx.alpha()
        out = bgz * (1 - A[..., None]) + F * A[..., None]
        D.vignette(out, 0.18 * e)
        return F * (1 - e) + out * e

    def sounds(self):
        self.ctx.sfx(self.t0, 'swell', 0.8)


# ---------------------------------------------------------------- Cover story
@reg('cover-story', need='mask', fallback='serif-flow')
class CoverStory(Beat):
    """A magazine cover: a huge serif masthead rises in behind my head, a thin white frame line, cover lines and an italic serif line."""
    behind = True
    fields = 'word (masthead), items (cover lines), line (italic serif line), kicker'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        title = (self.spec.get('title') or self.word or 'HAL-ABUUR').upper()
        size = D.fit_size(title, 'mast', W * 0.92, H * 0.17)
        T, s = text_spr(title, 'mast', size, '#FFFFFF')
        self.mast, self.mpad = D.shadow(s, blur=0.6 * u, dy=0.3 * u, alpha=0.25)
        self.mT = T
        cap = T.cap
        base = clamp(lay.head + 0.3 * cap, 0.03 * H + cap, H * 0.3)
        self.mast_xy = (W / 2, base)
        lines = self.items or [w.upper() for w in (self.line or '').split()[:4]]
        self.cover = []
        for i, ln in enumerate(lines[:4]):
            T2, s2 = text_spr(ln.upper(), 'xb', 2.9 * u, '#FFFFFF', tracking=0.1)
            self.cover.append((T2, D.shadow(s2, blur=0.4 * u, dy=0.2 * u, alpha=0.5)[0]))
        k = (self.spec.get('kicker') or '').upper()
        self.kick = text_spr(k, 'xb', 2.3 * u, L.RED2, tracking=0.16) if k else None
        bc = np.zeros((int(5 * u), int(11 * u)), np.float32)
        x = 0
        while x < bc.shape[1] - 2:
            w_ = self.R.choice([1, 1, 2, 3])
            bc[:, x:x + w_] = 1
            x += w_ + self.R.choice([1, 2, 2, 3])
        img = np.ones((bc.shape[0] + 12, bc.shape[1] + 12, 3), np.float32)
        img[6:-6, 6:-6][bc > 0] = 0.08
        self.barcode = np.concatenate([img, np.ones(img.shape[:2] + (1,), np.float32)], 2)
        ln = self.line or self.spec.get('say') or ''
        self.serif = [text_spr(w, 'serif_i', 6.4 * u, '#FFFFFF') for w in ln.split()[:6]]

    def bg(self, ctx, B, t):
        self.ensure()
        e = env(t, self.t0, self.t1, 0.25, 0.25)
        if e <= 0:
            return
        g = B.mean(2, keepdims=True)
        B[:] = (B * 0.8 + g * 0.2) * (1 - 0.18 * e) + B * 0.0
        r = ease_out_cubic(prog(t, self.t0, 0.55))
        x, y = self.mast_xy
        D.place(B, self.mast, x, y - (self.mT.base + self.mpad) + self.mast.shape[0] / 2 + (1 - r) * 0.05 * self.H, op=r * e)

    def front(self, ctx, F, t):
        self.ensure()
        W, H, u, lay = self.W, self.H, self.u, self.lay
        e = env(t, self.t0, self.t1, 0.25, 0.25)
        if e <= 0:
            return
        ins = int(0.035 * W)
        fr = D.rrect(W - 2 * ins, H - 2 * ins, 0, fill='#ffffff', border='#ffffff', bw=max(2, 0.22 * u), fill_a=0)
        D.blend(F, fr, ins, ins, e * 0.9)
        # cover lines in the chest zone, left and right
        y = lay.chest[0] + 2 * u
        for i, (T2, s2) in enumerate(self.cover):
            a = ease_out_cubic(prog(t, self.t0 + 0.4 + 0.12 * i, 0.35)) * e
            left = i % 2 == 0
            x = ins + 3 * u + s2.shape[1] / 2 if left else W - ins - 3 * u - s2.shape[1] / 2
            D.place(F, s2, x, y + (i // 2) * 3.2 * u, op=a)
        # kicker + italic serif line writing on at the lower left
        yk = ctx.cap_y - 0.095 * H
        if self.kick is not None:
            KT, ks = self.kick
            D.place(F, ks, ins + 3 * u + ks.shape[1] / 2 - KT.left, yk, op=ease_out_cubic(prog(t, self.t0 + 0.5, 0.3)) * e)
        D.place(F, self.barcode, W - ins - 3 * u - self.barcode.shape[1] / 2, ctx.cap_y - 0.12 * H, op=0.9 * e)
        x = ins + 3 * u
        for i, (T3, s3) in enumerate(self.serif):
            a = ease_out_cubic(prog(t, self.t0 + 0.65 + i * 0.1, 0.3)) * e
            if x + T3.adv > W - ins - 3 * u:
                break
            D.place(F, s3, x + s3.shape[1] / 2 - T3.left, yk + 5.2 * u, op=a)
            x += T3.adv + 1.3 * u

    def sounds(self):
        self.ctx.sfx(self.t0, 'whoosh', 0.7)


# ---------------------------------------------------------------- Type wall
@reg('type-wall', need='mask', fallback='word')
class TypeWall(Beat):
    """3–5 rows of my key phrase in huge Inter Black Italic caps fill the wall behind me (solid red / red outline rows), sliding in opposite directions."""
    behind = True

    def prep(self):
        W, H, u = self.W, self.H, self.u
        phrase = (self.word or self.line or 'HAL-ABUUR').upper()
        n = 5 if self.ctx.vertical else 3
        rh = H / n
        size = rh * 0.78 / 0.727
        rep = ' ' + phrase + ' ·'
        txt = (rep * max(2, int(3 * W / max(1, D.measure(rep, 'black_i', size))) + 2)).strip()
        self.rows = []
        for i in range(n):
            T = D.Text(txt, 'black_i', size, tracking=-0.03, pad_px=int(size * 0.1))
            a = T.alpha
            if i % 2:
                k = max(2, int(size * 0.03))
                er = cv2.erode(a, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
                a = np.clip(a - er, 0, 1)
            spr = D.to_sprite(a, hexrgb(L.RED))
            self.rows.append((spr, T, (i + 0.5) * rh, 1 if i % 2 else -1))
        self.wall = np.array(hexrgb('#EDEDF1'), np.float32)

    def bg(self, ctx, B, t):
        self.ensure()
        e = env(t, self.t0, self.t1, 0.2, 0.2)
        if e <= 0:
            return
        B[:] = B * (1 - 0.92 * e) + self.wall * 0.92 * e
        sp = 0.06 * self.W
        for i, (spr, T, y, d) in enumerate(self.rows):
            off = d * sp * (t - self.t0) - spr.shape[1] * 0.25
            a = ease_out_cubic(prog(t, self.t0 + 0.05 * i, 0.3)) * e
            D.place(B, spr, self.W / 2 + off, y - T.base + spr.shape[0] / 2 + T.cap / 2, op=a)
        contact_shadow(B, ctx.alpha(), self.u, amt=0.35 * e)

    def sounds(self):
        self.ctx.sfx(self.t0, 'whoosh', 0.7)


# ---------------------------------------------------------------- Depth scatter
@reg('depth-scatter', need='mask', fallback='chip-row')
class DepthScatter(Beat):
    """Each word appears where and when I say it, scattered at different depths: far words small, grey and soft behind me; near words big and red in front."""
    fields = 'items (3–5 words of the line)'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        words = self.items or [w for w in (self.line or self.spec.get('say') or self.word).split()][:5]
        A0 = self.ctx.still_alpha(self.t0 + 0.2)
        self.ws = []
        R = self.R
        far_spots = []
        for yy in np.linspace(self.ctx.top_safe + 0.03 * H, lay.chin, 6):
            for xx in np.linspace(0.12 * W, 0.88 * W, 5):
                if A0[int(min(H - 1, yy)), int(xx)] < 0.15:
                    far_spots.append((xx, yy))
        R.shuffle(far_spots)
        near_y = lay.chest[0] + 0.03 * H
        for i, w in enumerate(words):
            near = i % 2 == 1 or not far_spots
            if near:
                size = min(8.5 * u, D.fit_size(w.upper(), 'black_i', W * 0.5, 9 * u))
                bw = L.BigWord(w.upper(), size)
                x = W * (0.32 if (i // 2) % 2 == 0 else 0.68)
                y = near_y + size * 0.7 + (i // 2) * 0.012 * H
                self.ws.append(('near', bw, x, min(y, lay.chest[1]), self.item_t(i, len(words)) if self.items else self.when(w, self.t0 + 0.3 * i)))
            else:
                x, y = far_spots.pop()
                bgl = L.lum(self.ctx.still(self.t0), (x - 6 * u, y - 3 * u, x + 6 * u, y + 3 * u))
                T, s = text_spr(w, 'xb_i', 4.6 * u, '#55555C' if bgl > 0.6 else '#E2E2E6')
                self.ws.append(('far', (T, s), x, y, self.item_t(i, len(words)) if self.items else self.when(w, self.t0 + 0.3 * i)))

    def front(self, ctx, F, t):
        self.ensure()
        A = ctx.alpha()
        e = env(t, self.t0, self.t1, 0.01, 0.25)
        for kind, obj, x, y, ti in self.ws:
            if kind == 'far':
                T, s = obj
                a = ease_out_cubic(prog(t, ti, 0.35)) * e
                place_masked(F, s, x, y, A=A, occ=1.0, op=a * 0.92, blur=0.45 * self.u + 1.5 * self.u * (1 - a))
            else:
                obj.draw(F, t, x, y, ti, self.t1 - 0.24, mode='blur')

    def sounds(self):
        self.ensure()
        for kind, obj, x, y, ti in self.ws[:2]:
            self.ctx.sfx(ti, 'pop', 0.6)


# ---------------------------------------------------------------- Scene text
@reg('scene-text', need='mask', fallback='word')
class SceneText(Beat):
    """1–2 words painted on the real wall behind me: red Inter Black Italic, multiplied into the wall texture, plus a faint pattern of a small repeated word."""
    behind = True

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        txt = (self.word or 'HAL-ABUUR').upper()
        A0 = self.ctx.still_alpha(self.t0 + 0.1)
        top = self.ctx.top_safe
        cands = []
        # band above my head
        if lay.head - top > 0.07 * H:
            cands.append(('top', (0.05 * W, top, 0.95 * W, lay.head - 0.005 * H)))
        # columns beside my head (the wall that shows left and right of me)
        for name, x0, x1 in (('left', 0.02 * W, lay.face_pad[0]), ('right', lay.face_pad[2], 0.98 * W)):
            if x1 - x0 > 0.09 * W:
                ys = np.arange(int(top), int(lay.chin))
                col = A0[int(top):int(lay.chin), int(x0):int(x1)]
                free = (col < 0.2).mean(1) > 0.7 if col.size else np.zeros(0, bool)
                if free.any():
                    # longest free vertical run
                    best, cur, st, bs = 0, 0, 0, 0
                    for i, f in enumerate(free):
                        cur = cur + 1 if f else 0
                        if cur == 1:
                            st = i
                        if cur > best:
                            best, bs = cur, st
                    if best > 0.15 * H:
                        cands.append((name, (x0, top + bs, x1, top + bs + best)))
        if not cands:
            cands.append(('top', (0.05 * W, top, 0.95 * W, top + 0.08 * H)))
        name, (x0, y0, x1, y1) = max(cands, key=lambda c: (c[1][2] - c[1][0]) * (c[1][3] - c[1][1]))
        self.where = name
        if name == 'top':
            size = min(D.fit_size(txt, 'black_i', (x1 - x0) * 0.95, 400), (y1 - y0) * 0.9 / 0.727)
            T = D.Text(txt, 'black_i', size, tracking=-0.04)
            a = T.alpha
            self.pos = ((x0 + x1) / 2 - a.shape[1] / 2, y1 - T.base - (y1 - y0) * 0.05)
        else:
            size = min(D.fit_size(txt, 'black_i', (y1 - y0) * 0.95, 400), (x1 - x0) * 0.92 / 0.727)
            T = D.Text(txt, 'black_i', size, tracking=-0.04)
            a = np.ascontiguousarray(np.rot90(T.alpha, 1 if name == 'left' else -1))
            self.pos = ((x0 + x1) / 2 - a.shape[1] / 2, (y0 + y1) / 2 - a.shape[0] / 2)
        self.T, self.a = T, a
        small = (self.spec.get('pattern') or txt.split()[0]).upper()
        pt = D.Text(small, 'xb', 1.8 * u, tracking=0.2, pad_px=int(2.6 * u))
        tile = pt.alpha
        reps_x = int(W / max(1, tile.shape[1])) + 2
        reps_y = int(H / max(1, tile.shape[0])) + 2
        self.pat = np.tile(tile, (reps_y, reps_x))[:H, :W].astype(np.float32)

    def bg(self, ctx, B, t):
        self.ensure()
        e = env(t, self.t0, self.t1, 0.3, 0.25)
        if e <= 0:
            return
        red = np.array(hexrgb(L.RED), np.float32)
        k = 0.1 * e
        B[:] = B * (1 - self.pat[..., None] * k) + B * red * self.pat[..., None] * k
        h, w = self.a.shape
        x0, y0 = int(self.pos[0]), int(self.pos[1])
        H, W = self.H, self.W
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if xb <= xa or yb <= ya:
            return
        a = self.a[ya - y0:yb - y0, xa - x0:xb - x0][..., None] * 0.9 * ease_out_cubic(prog(t, self.t0, 0.5)) * e
        roi = B[ya:yb, xa:xb]
        roi[:] = roi * (1 - a) + roi * red * a * 1.15

    def sounds(self):
        self.ctx.sfx(self.t0, 'pop', 0.6)


# ---------------------------------------------------------------- Freeze sticker
def sunburst(H, W, cx, cy, rot, n=24):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ang = (np.arctan2(yy - cy, xx - cx) + math.radians(rot)) % (2 * math.pi / n * 2)
    ray = (ang < 2 * math.pi / n).astype(np.float32)
    ray = cv2.GaussianBlur(ray, (0, 0), 1.0)
    a = np.array(hexrgb('#E0182D'), np.float32)
    b = np.array(hexrgb('#C4091F'), np.float32)
    img = a * ray[..., None] + b * (1 - ray[..., None])
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(W, H)
    img *= (1 - 0.35 * np.clip(d * 1.4, 0, 1) ** 1.5)[..., None]
    return img.astype(np.float32)


@reg('freeze-sticker', need='mask', fallback='freeze-label')
class FreezeSticker(Beat):
    """One frame freezes and I become a sticker (thick white outline, drop shadow) on a red sunburst, with a stamp label and a hand-written line with an arrow."""
    full_frame = True
    fields = 'word (stamp label), line (hand-written line)'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        at = self.t0 + 0.04
        Ff, Af = self.ctx.still(at), self.ctx.still_alpha(at)
        k = max(4, int(1.7 * u))
        ol = cv2.dilate((Af > 0.45).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))).astype(np.float32)
        ol = cv2.GaussianBlur(ol, (0, 0), 0.8)
        spr = np.concatenate([np.ones_like(Ff) * ol[..., None], ol[..., None]], 2)
        person = np.concatenate([Ff * Af[..., None], Af[..., None]], 2)
        spr = person + spr * (1 - person[..., 3:4])
        ys, xs = np.where(ol > 0.05)
        if len(xs):
            y0, y1, x0, x1 = max(0, ys.min() - 4), min(H, ys.max() + 4), max(0, xs.min() - 4), min(W, xs.max() + 4)
            spr = spr[y0:y1, x0:x1]
            self.off = ((x0 + x1) / 2 - W / 2, (y0 + y1) / 2 - H / 2)
        else:
            self.off = (0, 0)
        self.sticker, p = D.shadow(spr, blur=2.0 * u, dy=1.4 * u, alpha=0.45)
        self.s = 0.66 if lay.closeup else 0.8
        txt = (self.word or 'HAL-ABUUR').upper()
        T, ts = text_spr(txt, 'black', 4.2 * u, '#FFFFFF', tracking=0.06)
        wd, ht = int(T.adv + 4 * u), int(T.cap + 3.2 * u)
        box = D.rrect(wd, ht, 1.0 * u, fill='#ffffff', border='#ffffff', bw=0.4 * u, fill_a=0)
        D.blend_spr(box, ts, 2 * u - T.left, ht / 2 + T.cap / 2 - T.base)
        self.stamp = box
        hand = self.line or ''
        self.hand = text_spr(hand, 'hand', 6.2 * u, '#FFFFFF') if hand else None
        self.cy = H * 0.58

    def full(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        e = env(t, self.t0, self.t1, 0.06, 0.08)
        if e <= 0:
            return None
        out = sunburst(H, W, W / 2, self.cy, (t - self.t0) * 12)
        bob = math.sin((t - self.t0) * 2 * math.pi / 1.3)
        s = self.s * (1 + 0.015 * bob) * (0.94 + 0.06 * ease_out_back(prog(t, self.t0, 0.3), 1.6))
        D.place(out, self.sticker, W / 2 + self.off[0] * s, self.cy + self.off[1] * s, scale=s, rot=1.8 * bob)
        sc, a = pop_in(t, self.t0 + 0.25, 0.35, 1.8)
        D.place(out, self.stamp, W * 0.3, ctx.top_safe + 0.06 * H, scale=sc, rot=-7, op=a)
        if self.hand:
            T, hs = self.hand
            f = prog(t, self.t0 + 0.45, 0.6)
            sp = typed(T, hs, f)
            if sp is not None:
                D.place(out, sp, W * 0.3 + sp.shape[1] / 2 - T.left - 6 * u, ctx.top_safe + 0.13 * H)
            g = prog(t, self.t0 + 0.8, 0.4)
            if g > 0:
                al = np.zeros((H, W), np.float32)
                pts = [(W * (0.62 + 0.18 * q), ctx.top_safe + 0.12 * H + math.sin(q * math.pi) * -0.03 * H + q * 0.12 * H) for q in np.linspace(0, g, 16)]
                cv2.polylines(al, [np.array(pts, np.int32)], False, 1.0, max(2, int(0.5 * u)), cv2.LINE_AA)
                if g >= 1:
                    x, y = pts[-1]
                    cv2.line(al, (int(x), int(y)), (int(x - 2.4 * u), int(y - 1.2 * u)), 1.0, max(2, int(0.5 * u)), cv2.LINE_AA)
                    cv2.line(al, (int(x), int(y)), (int(x + 0.4 * u), int(y - 2.6 * u)), 1.0, max(2, int(0.5 * u)), cv2.LINE_AA)
                out = out * (1 - al[..., None]) + al[..., None]
        return out

    def sounds(self):
        self.ctx.sfx(self.t0, 'impact', 0.6)


# ---------------------------------------------------------------- Clone
@reg('clone', need='mask', fallback='word')
class Clone(Beat):
    """For one beat two more copies of me (my person layer from other moments of the take) sit left and right of me in the same room."""
    full_frame = True

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        dur = self.ctx.info['dur']
        ta = clamp(self.t0 - 3.0, 0, dur - 0.2) if self.t0 > 3.2 else clamp(self.t1 + 2.5, 0, dur - 0.2)
        tb = clamp(self.t1 + 1.2, 0, dur - 0.2) if self.t1 + 1.4 < dur else clamp(self.t0 - 1.5, 0, dur - 0.2)
        self.cl = []
        sc = 0.62 if lay.closeup else 0.8
        for at, x in ((ta, 0.19), (tb, 0.81)):
            Fc, Ac = self.ctx.still(at), self.ctx.still_alpha(at)
            fx_c = self.ctx.face(at)['cx']
            M = np.float32([[sc, 0, x * W - sc * fx_c], [0, sc, H * (1 - sc)]])
            Fs = cv2.warpAffine(Fc, M, (W, H), flags=cv2.INTER_LINEAR)
            As = cv2.warpAffine(Ac, M, (W, H), flags=cv2.INTER_LINEAR)
            self.cl.append((Fs, As))
        self.sm = 1.0

    def full(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        e = env(t, self.t0, self.t1, 0.15, 0.2)
        if e <= 0:
            return None
        A = ctx.alpha()
        out = ctx.plate(self.t0).copy()
        for Fs, As in self.cl:
            a = (As * e)[..., None]
            contact_shadow(out, As * e, u, amt=0.3)
            out = out * (1 - a) + Fs * a
        s = lerp(1, self.sm, ease_in_out_cubic(e))
        Fp = scale_about(F, s, W / 2, H)
        Ap = scale_about(A, s, W / 2, H)
        contact_shadow(out, Ap, u, amt=0.3)
        out = out * (1 - Ap[..., None]) + Fp * Ap[..., None]
        return out

    def sounds(self):
        self.ctx.sfx(self.t0, 'whoosh', 0.6)


# ---------------------------------------------------------------- Blackout / Spotlight
@reg('blackout', need='mask', fallback='punch')
class Blackout(Beat, WordMixin):
    """The room goes almost black around me (I stay lit) and one big red word sits behind my head; at most 1.5 s."""
    behind = True

    def prep(self):
        self.t1 = min(self.t1, self.t0 + 1.6)
        self.prep_word(place='behind')

    def bg(self, ctx, B, t):
        self.ensure()
        e = env(t, self.t0, self.t1, 0.12, 0.22)
        if e <= 0:
            return
        B *= (1 - 0.9 * e)
        if self.bw is not None and self.mode == 'behind':
            self.draw_word(B, t)

    def front(self, ctx, F, t):
        self.ensure()
        if self.bw is not None and self.mode != 'behind':
            self.draw_word(F, t)

    def on(self, t):
        return self.t0 <= t <= self.t1 + 0.05

    def sounds(self):
        self.ensure()
        self.ctx.sfx(self.t0, 'swell', 0.9)


@reg('spotlight', need='mask', fallback='punch')
class Spotlight(Beat, WordMixin):
    """The room darkens to deep charcoal while a soft spotlight glows behind me; a big white word behind my head, the next word in red in front of my body."""
    behind = True
    fields = 'word (white, behind), word2 (red, in front)'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        self.prep_word(place='behind', grad=L.WGRAD)
        w2 = (self.spec.get('word2') or '').upper()
        self.w2 = None
        if w2:
            mode, size, base, cx = word_spot(self.ctx, lay, w2, 'chest', self.idx)
            if self.bw is not None and self.mode != 'behind':
                # both words in front: stack them in the chest zone (white on top, red under it)
                room = lay.chest[1] - self.base_y
                size = min(size, room * 0.85 / 0.727)
                base = self.base_y + size * 0.727 + 0.012 * H
                if size < 0.03 * H / 0.727:
                    size = 0
            if size:
                self.w2 = (L.BigWord(w2, size), cx, base, self.when(w2, self.t0 + 0.6))
        self.glow = D.radial(H, W, lay.cx, (lay.head + lay.chin) / 2, 0.55 * W, (1, 0.93, 0.86), (0, 0, 0), 1.0)
        self.char = np.array(hexrgb('#141418'), np.float32)

    def bg(self, ctx, B, t):
        self.ensure()
        e = env(t, self.t0, self.t1, 0.25, 0.25)
        if e <= 0:
            return
        B[:] = B * (1 - 0.86 * e) + self.char * 0.86 * e
        B += self.glow * 0.42 * e
        if self.bw is not None and self.mode == 'behind':
            self.draw_word(B, t)

    def front(self, ctx, F, t):
        self.ensure()
        if self.bw is not None and self.mode != 'behind':
            self.draw_word(F, t)
        if self.w2:
            bw, cx, base, tw = self.w2
            bw.draw(F, t, cx, base, tw, self.t1 - 0.24)

    def sounds(self):
        self.ensure()
        self.ctx.sfx(self.t0, 'swell', 0.8)
