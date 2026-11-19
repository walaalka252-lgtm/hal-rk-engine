"""Graphic, UI and camera effects (work with or without the person mask)."""
import math
import re

import numpy as np
import cv2

from . import draw as D, look as L
from .fx import (reg, Beat, Word, WordMixin, word_spot, img_from, side_crop, pop_in, typed, place_masked, scale_about,
                 contact_shadow, white_card, text_spr, badge, icon, boxes_overlap)
from .util import (clamp, prog, lerp, ease_out_cubic, ease_in_cubic, ease_in_out_cubic, ease_in_out_sine, ease_out_back,
                   ease_out_expo, hexrgb, rnd, window)


def env(t, t0, t1, fin=0.3, fout=0.25):
    return window(t, t0, t1, fin, fout)


def frost(F, box, u, r, tint=0.2, dark=False):
    """Frosted glass panel: blur what is behind the box, tint it, white 1 px edge and a soft shadow."""
    H, W = F.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    sh = D.rrect(x1 - x0, y1 - y0, r, fill='#000000')
    sh = D.gblur(D.pad(sh, int(3 * u)), 2.0 * u)
    D.blend(F, sh * 0.35, x0 - int(3 * u), y0 - int(3 * u) + int(1.2 * u))
    roi = F[y0:y1, x0:x1]
    k = int(2.2 * u) * 2 + 1
    bl = cv2.GaussianBlur(roi, (k, k), 2.0 * u)
    col = np.array(hexrgb('#141418' if dark else '#FFFFFF'), np.float32)
    bl = bl * (1 - tint) + col * tint
    a = D.rrect_alpha(x1 - x0, y1 - y0, r)[:roi.shape[0], :roi.shape[1]][..., None]
    roi[:] = roi * (1 - a) + bl * a
    edge = D.rrect(x1 - x0, y1 - y0, r, fill='#ffffff', border='#ffffff', bw=max(1.0, 0.12 * u), fill_a=0)
    D.blend(F, edge, x0, y0, 0.55)


def chest_box(lay, w, h, x=None):
    """A w×h box centred in the chest zone (below the face, above the captions)."""
    cx = lay.W / 2 if x is None else x
    y0 = lay.chest[0] + max(0, (lay.chest_h() - h) * 0.35)
    return (cx - w / 2, y0, cx + w / 2, y0 + h)


# ---------------------------------------------------------------- camera effects
@reg('digital-zoom', need='footage', aliases=('punch-in', 'zoom'))
class DigitalZoom(Word):
    """A fast but smooth punch-in on my face (100% → 134% in 0.45 s), held through the word, then eased back out."""

    def __init__(self, spec, ctx, idx):
        super().__init__(spec, ctx, idx)
        spec.setdefault('place', 'chest')
        self.tz = ctx.word_time(self.word, self.t0, self.t1) if self.word else None
        self.tz = self.t0 + 0.05 if self.tz is None else max(self.t0, min(self.t1 - 0.5, self.tz - 0.05))

    def cam(self):
        z = float(self.spec.get('zoom', 1.34))
        back = max(self.tz + 0.9, self.t1 - 0.15)
        return [(self.tz, 0.45, 1.0, z, ease_out_cubic), (back, 0.9, z, 1.0, ease_in_out_sine)]

    def sounds(self):
        self.ctx.sfx(self.tz, 'whoosh', 0.7)


@reg('zoom-through', need='footage', aliases=('zoom-through-cut',))
class ZoomThrough(Word):
    """Push into one point of the frame (100% → 260% in 0.25 s) with a zoom blur, cut, and come out of it (0.3 s)."""
    pre = 0.3

    def __init__(self, spec, ctx, idx):
        super().__init__(spec, ctx, idx)
        spec.setdefault('place', 'chest')

    def cam(self):
        return [(self.t0 - 0.25, 0.25, 1.0, 2.6, ease_in_cubic), (self.t0, 0.32, 2.6, 1.0, ease_out_cubic)]

    def front(self, ctx, F, t):
        L.blur_zoom(F, t, self.t0, 0.5)
        if t >= self.t0:
            super().front(ctx, F, t)

    def bg(self, ctx, B, t):
        if t >= self.t0:
            super().bg(ctx, B, t)

    def sounds(self):
        self.ctx.sfx(self.t0 - 0.25, 'whoosh', 0.9)


# ---------------------------------------------------------------- Phone mockup
@reg('phone', aliases=('phone-mockup',))
class Phone(Beat):
    """A black phone (thin bezel, 3D tilt) slides in next to me with an app screen built from my words; rows tick red; a button gets tapped."""
    fields = 'word (screen title), items (2–3 rows), button, done'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        ph = int((0.46 if self.ctx.vertical else 0.72) * H)
        pw = int(ph * 0.49)
        self.pw, self.ph = pw, ph
        self.side = 1 if lay.cx <= W / 2 else -1
        cx = W - pw * 0.62 if self.side > 0 else pw * 0.62
        cy = clamp(lay.chin + ph * 0.5 - 0.02 * H, H * 0.3, H - ph * 0.5 - 0.02 * H)
        self.xy = (cx, cy)
        self.title = (self.word or '').title() or 'Hal-abuur'
        self.rows = self.items[:3] or [w for w in (self.line or '').split()[:3]] or ['…']
        self.btn = self.spec.get('button', 'OK')
        self.done = self.spec.get('done', '✓')
        self.cache = {}
        self.rt = [self.item_t(i, len(self.rows)) for i in range(len(self.rows))]
        self.tap = min(self.t1 - 0.5, (self.rt[-1] if self.rt else self.t0) + 0.5)
        if boxes_overlap((cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2), (0, self.ctx.cap_y - 0.03 * H, W, self.ctx.cap_y + 0.03 * H)):
            self.cap_hide = True
        else:
            self.cap_hide = False

    def screen(self, state):
        if state in self.cache:
            return self.cache[state]
        u = self.u
        pw, ph = self.pw, self.ph
        body = D.rrect(pw, ph, pw * 0.16, fill='#0B0B0D')
        rim = D.rrect(pw, ph, pw * 0.16, fill='#ffffff', border='#3a3a40', bw=max(2, 0.3 * u), fill_a=0)
        body = rim + body * (1 - rim[..., 3:4])
        bz = max(4, int(0.022 * pw * 1.6))
        sw, sh = pw - 2 * bz, ph - 2 * bz
        scr = D.rrect(sw, sh, pw * 0.13, fill='#ffffff')
        s = pw / 100.0
        T, ts = text_spr(self.title, 'xb', 8.5 * s, L.INK)
        D.blend_spr(scr, ts, 7 * s - T.left, 22 * s - T.base)
        bar = D.rrect(int(sw * 0.86), max(2, int(1.2 * s)), 0.6 * s, fill='#EDEDF1')
        D.blend_spr(scr, bar, 7 * s, 27 * s)
        nt, tap = state
        y = 38 * s
        for i, row in enumerate(self.rows):
            on = i < nt
            card = D.rrect(int(sw - 14 * s), int(15 * s), 3 * s, fill='#FFF1F2' if on else '#F5F5F8')
            D.blend_spr(scr, card, 7 * s, y)
            c = D.circle(7 * s, L.RED) if on else D.ring(7 * s, 1.0 * s, '#C9C9CF')
            D.blend_spr(scr, c, 10 * s, y + 4 * s)
            if on:
                ck = icon('check', 6 * s)
                D.blend_spr(scr, ck, 10.5 * s, y + 4.5 * s)
            T2, t2 = text_spr(row, 'semi', 6.2 * s, L.RED if on else L.INK)
            D.blend_spr(scr, t2, 21 * s - T2.left, y + 7.5 * s + T2.cap / 2 - T2.base)
            y += 19 * s
        b = D.rrect(int(sw - 14 * s), int(14 * s), 7 * s, fill=L.RED if tap else '#ffffff', border=L.RED, bw=1.0 * s)
        T3, t3 = text_spr(self.done if tap else self.btn, 'bold', 6 * s, '#ffffff' if tap else L.RED)
        D.blend_spr(b, t3, b.shape[1] / 2 - T3.adv / 2 - T3.left, b.shape[0] / 2 + T3.cap / 2 - T3.base)
        D.blend_spr(scr, b, 7 * s, sh - 24 * s)
        isl = D.rrect(int(26 * s), int(6.5 * s), 3.2 * s, fill='#0B0B0D')
        D.blend_spr(scr, isl, sw / 2 - 13 * s, 4 * s)
        D.blend_spr(body, scr, bz, bz)
        # 3D tilt: a slight perspective turn towards me
        h, w = body.shape[:2]
        k = 0.06 if self.side > 0 else -0.06
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        if k > 0:
            dst = np.float32([[0, h * k], [w * 0.93, 0], [w * 0.93, h], [0, h * (1 - k)]])
        else:
            dst = np.float32([[w * 0.07, 0], [w, -h * k], [w, h * (1 + k)], [w * 0.07, h]])
        M = cv2.getPerspectiveTransform(src, dst)
        body = cv2.warpPerspective(body, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        body, p = D.shadow(body, blur=2.4 * u, dy=1.6 * u, alpha=0.45)
        self.cache[state] = body
        return body

    def front(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        e_in = ease_out_cubic(prog(t, self.t0, 0.6))
        e_out = ease_in_cubic(prog(t, self.t1 - 0.35, 0.35))
        if e_in <= 0 or e_out >= 1:
            return
        nt = sum(1 for ti in self.rt if t >= ti)
        tapped = t >= self.tap + 0.12
        spr = self.screen((nt, tapped))
        x = self.xy[0] + self.side * (1 - e_in + e_out) * (self.pw * 1.3)
        D.place(F, spr, x, self.xy[1], rot=self.side * -3 * (1 - e_in))
        p = prog(t, self.tap, 0.35)
        if 0 < p < 1:
            s = self.pw / 100.0
            bx = x
            by = self.xy[1] + self.ph / 2 - 20 * s
            D.place(F, D.ring(18 * s, 1.2 * s, L.RED, a=1 - p), bx, by, scale=0.3 + 1.0 * p)

    def hide_captions(self):
        self.ensure()
        return [(self.t0, self.t1)] if self.cap_hide else []

    def sounds(self):
        self.ensure()
        self.ctx.sfx(self.t0, 'whoosh', 0.6)
        self.ctx.sfx(self.tap, 'click', 0.8)


# ---------------------------------------------------------------- Checklist
@reg('checklist')
class Checklist(Beat):
    """A small card: a header with a counter and a thin red progress bar; the items type on and each gets a red tick on its spoken word."""
    fields = 'title, items (2–4; {text, at} for exact times)'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        rows = self.items[:4] or [w for w in (self.line or '').split()[:3]]
        self.rows = rows
        self.title = (self.spec.get('title') or self.word or '').upper()
        rs = 3.4 * u
        cw = int(max(0.56 * W if self.ctx.vertical else 0.3 * W, max(D.measure(r, 'semi', rs) for r in rows) + 10 * u if rows else 0))
        cw = min(cw, int(0.9 * W))
        chh = int(6.6 * u + 1.6 * u + len(rows) * 5.4 * u + 1.2 * u)
        self.cw, self.ch = cw, chh
        # top-left beside my head if it does not touch my face, else in the chest zone
        box = (0.04 * W, self.ctx.top_safe + 0.005 * H, 0.04 * W + cw, self.ctx.top_safe + 0.005 * H + chh)
        if boxes_overlap(box, lay.face_pad) or box[3] > lay.head + 0.02 * H and box[2] > lay.person[0] and lay.closeup:
            box = chest_box(lay, cw, chh)
            if box[3] > self.ctx.cap_y - 0.03 * H:
                self.cap_low = True
        self.box = box
        self.card, self.cp = white_card(cw, chh, u, r=1.8 * u, shadow_a=0.22)
        self.hT = text_spr(self.title, 'xb', 2.2 * u, L.INK, tracking=0.08) if self.title else None
        self.rT = [text_spr(r, 'semi', rs, L.INK) for r in rows]
        self.rTr = [text_spr(r, 'semi', rs, L.RED) for r in rows]
        self.times = [self.item_t(i, len(rows)) for i in range(len(rows))]

    def front(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        s, a = pop_in(t, self.t0, 0.45, 1.2)
        a *= 1 - prog(t, self.t1 - 0.25, 0.25)
        if a <= 0:
            return
        x0, y0, x1, y1 = self.box
        D.place(F, self.card, (x0 + x1) / 2, (y0 + y1) / 2 + (1 - a) * 1.2 * u, scale=0.96 + 0.04 * s, op=a)
        n = len(self.rows)
        done = sum(1 for ti in self.times if t >= ti)
        yh = y0 + 3.6 * u
        D.place(F, D.circle(1.2 * u, L.RED), x0 + 3.0 * u, yh, op=a)
        if self.hT:
            T, sp = self.hT
            D.place(F, sp, x0 + 4.8 * u + sp.shape[1] / 2 - T.left, yh + sp.shape[0] / 2 - T.base + T.cap / 2, op=a)
        T, cs = text_spr('%d/%d' % (done, n), 'xb', 2.3 * u, L.RED)
        D.place(F, cs, x1 - 2.8 * u - cs.shape[1] / 2 + T.left, yh + cs.shape[0] / 2 - T.base + T.cap / 2, op=a)
        by = y0 + 6.8 * u
        bw = self.cw - 4.8 * u
        D.place(F, D.rrect(int(bw), max(3, int(0.5 * u)), 0.25 * u, fill='#ECECF0'), x0 + 2.4 * u + bw / 2, by, op=a)
        fr = 0.0
        for ti in self.times:
            fr += ease_out_cubic(prog(t, ti, 0.35))
        if fr > 0:
            w = max(3, int(bw * fr / max(1, n)))
            D.place(F, D.rrect(w, max(3, int(0.5 * u)), 0.25 * u, fill=L.RED), x0 + 2.4 * u + w / 2, by, op=a)
        y = by + 4.6 * u
        for i in range(n):
            ti = self.times[i]
            ty = prog(t, ti - 0.6, 0.45)
            on = t >= ti
            T, sp = (self.rTr if on else self.rT)[i]
            spr = typed(T, sp, ty)
            cx = x0 + 4.0 * u
            if ty > 0:
                if on:
                    k = ease_out_back(prog(t, ti, 0.3), 2.0)
                    D.place(F, D.circle(3.0 * u, L.RED), cx, y, scale=k, op=a)
                    D.place(F, icon('check', 2.4 * u), cx, y, scale=clamp(prog(t, ti + 0.08, 0.2)), op=a)
                else:
                    D.place(F, D.ring(3.0 * u, 0.3 * u, '#C9C9CF'), cx, y, op=a * ty)
            if spr is not None:
                D.place(F, spr, cx + 2.8 * u + spr.shape[1] / 2 - T.left, y + spr.shape[0] / 2 - T.base + T.cap / 2, op=a)
            y += 5.4 * u

    def hide_captions(self):
        self.ensure()
        return [(self.t0, self.t1)] if getattr(self, 'cap_low', False) else []

    def sounds(self):
        self.ensure()
        self.ctx.sfx(self.t0, 'pop', 0.6)
        for ti in self.times[:3]:
            self.ctx.sfx(ti, 'tick', 0.8)


# ---------------------------------------------------------------- Headline wall
@reg('headline-wall', aliases=('headlines', 'news'))
class HeadlineWall(Beat):
    """4–6 small cards pop in around my head: newspaper scraps with a serif headline and a red NEWS tag, a post card with an avatar, a search pill that types."""
    fields = 'items (headlines), word'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        heads = self.items or [self.line or self.spec.get('say') or self.word]
        fp = lay.core
        cands = [(0.24, 0.0), (0.76, 0.0), (0.12, 0.5), (0.88, 0.5), (0.26, 1.0), (0.74, 1.0)]
        self.cards = []
        k = 0
        for i, (cx, ry) in enumerate(cands):
            kind = ['news', 'post', 'search', 'news', 'post', 'news'][i]
            cw = (0.46 if self.ctx.vertical else 0.26) * W
            spr = self.make(kind, heads[k % len(heads)], cw)
            k += 1
            h, w = spr.shape[:2]
            if ry == 0.0:
                y = self.ctx.top_safe + h / 2
            elif ry == 0.5:
                y = (fp[1] + fp[3]) / 2
            else:
                y = lay.chest[0] + h / 2
            x = cx * W
            box = (x - w * 0.42, y - h * 0.42, x + w * 0.42, y + h * 0.42)
            if boxes_overlap(box, fp):
                # push it sideways out of the face, or skip it
                if x < lay.cx:
                    x = fp[0] - w * 0.42
                else:
                    x = fp[2] + w * 0.42
                if x - w * 0.3 < 0 or x + w * 0.3 > W:
                    continue
            self.cards.append((spr, x, y, self.R.uniform(-6, 6), self.t0 + 0.2 * len(self.cards)))

    def make(self, kind, text, cw):
        u = self.u
        words = text.split()
        if kind == 'news':
            size = 3.8 * u
            lines = D.wrap(text, 'serif', size, cw * 0.84, max_lines=3)
            h = int(size * 1.15 * len(lines) + 7 * u)
            base = D.rrect(int(cw), h, 0.5 * u, fill='#F5EFE2')
            tg = L.tag('NEWS', 1.4 * u)
            D.blend_spr(base, tg, 2 * u, 1.4 * u)
            y = 5.2 * u + size * 0.8
            for ln in lines:
                T, s = text_spr(ln, 'serif', size, L.INK)
                D.blend_spr(base, s, 2 * u - T.left, y - T.base)
                y += size * 1.15
        elif kind == 'post':
            h = int(11 * u)
            base = D.rrect(int(cw), h, 1.6 * u, fill='#FFFFFF')
            av = D.circle(6.4 * u, L.RED)
            D.blend_spr(base, av, 1.8 * u, 2.2 * u)
            T, s = text_spr((self.ctx.plan.get('brand') or 'Hal-abuur'), 'bold', 2.5 * u, L.INK)
            D.blend_spr(base, s, 9.6 * u - T.left, 3.8 * u - T.base + T.cap)
            ln = ' '.join(words[:4])
            T2, s2 = text_spr(ln, 'reg', 2.5 * u, '#4A4A50')
            D.blend_spr(base, s2[:, :int(cw - 10.5 * u)], 9.6 * u - T2.left, 7.6 * u - T2.base + T2.cap)
        else:
            h = int(7.6 * u)
            base = D.rrect(int(cw), h, h / 2, fill='#FFFFFF', border=L.RED, bw=0.3 * u)
            ic = D.ring(3.2 * u, 0.42 * u, L.RED)
            D.blend_spr(base, ic, 2.2 * u, h / 2 - 1.8 * u)
            T, s = text_spr(' '.join(words[:3]) + '?', 'semi', 2.7 * u, L.INK)
            D.blend_spr(base, s[:, :int(cw - 7.5 * u)], 6.4 * u - T.left, h / 2 + T.cap / 2 - T.base)
        out, p = D.shadow(base, blur=1.6 * u, dy=1.0 * u, alpha=0.3)
        return out

    def front(self, ctx, F, t):
        self.ensure()
        out = prog(t, self.t1 - 0.25, 0.25)
        drift = (t - self.t0) * 0.004 * self.W
        for i, (spr, x, y, rot, ti) in enumerate(self.cards):
            s, a = pop_in(t, ti, 0.4, 1.5)
            a *= 1 - out
            if a > 0:
                D.place(F, spr, x + (drift if i % 2 else -drift), y, scale=0.9 + 0.1 * s, rot=rot, op=a)

    def sounds(self):
        self.ensure()
        self.ctx.sfx(self.t0, 'pop', 0.6)


# ---------------------------------------------------------------- Chart grow
@reg('chart', aliases=('chart-grow', 'graph'))
class Chart(Beat):
    """The picture dims; a faint grid; a white line with round nodes draws left to right, then the last segment shoots up in red to a glowing node with the number."""
    fields = 'number (label at the top, e.g. "+3 h" or "16$"), word'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        if lay.closeup or not self.ctx.vertical:
            top, bot = lay.chest[0] + 0.01 * H, lay.chest[1] - 0.005 * H
        else:
            top, bot = self.ctx.top_safe + 0.02 * H, lay.head - 0.02 * H
        if bot - top < 0.12 * H:
            top, bot = lay.chest[0], lay.chest[0] + 0.16 * H
        x0, x1 = 0.1 * W, 0.86 * W
        self.box = (x0, top, x1, bot)
        vals = [0.18, 0.32, 0.28, 0.44, 0.4, 0.95]
        self.pts = [(x0 + (x1 - x0) * i / (len(vals) - 1), bot - (bot - top) * v) for i, v in enumerate(vals)]
        num = str(self.spec.get('number') or self.word or '').upper()
        self.num = L.BigWord(num, min(12 * u, D.fit_size(num, 'black_i', 0.42 * W, 12 * u))) if num else None
        self.t_up = self.when(self.spec.get('number') or self.word, self.t0 + 0.9)

    def front(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        e = env(t, self.t0, self.t1, 0.3, 0.25)
        if e <= 0:
            return
        F *= (1 - 0.32 * e)
        x0, y0, x1, y1 = self.box
        g = np.zeros((H, W), np.float32)
        for i in range(5):
            y = int(y0 + (y1 - y0) * i / 4)
            cv2.line(g, (int(x0), y), (int(x1), y), 1.0, 1, cv2.LINE_AA)
        for i in range(6):
            x = int(x0 + (x1 - x0) * i / 5)
            cv2.line(g, (x, int(y0)), (x, int(y1)), 1.0, 1, cv2.LINE_AA)
        F[:] = F * (1 - g[..., None] * 0.28 * e) + g[..., None] * 0.28 * e
        p = prog(t, self.t0 + 0.2, max(0.4, self.t_up - self.t0 - 0.2))
        pts = self.pts[:-1]
        n = len(pts) - 1
        pos = p * n
        al = np.zeros((H, W), np.float32)
        drawn = [pts[0]]
        for i in range(n):
            if pos >= i + 1:
                drawn.append(pts[i + 1])
            elif pos > i:
                f = pos - i
                drawn.append((pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f))
                break
        if len(drawn) > 1:
            cv2.polylines(al, [np.array(drawn, np.int32)], False, 1.0, max(3, int(0.7 * u)), cv2.LINE_AA)
            F[:] = F * (1 - al[..., None] * e) + al[..., None] * e
        for i, (x, y) in enumerate(pts):
            if pos >= i:
                k = ease_out_back(clamp((pos - i) * 2), 2.0)
                D.place(F, D.circle(2.4 * u, '#FFFFFF'), x, y, scale=k, op=e)
        q = ease_out_expo(prog(t, self.t_up, 0.45))
        if q > 0:
            a, b = self.pts[-2], self.pts[-1]
            c = (a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q)
            r = np.zeros((H, W), np.float32)
            cv2.line(r, (int(a[0]), int(a[1])), (int(c[0]), int(c[1])), 1.0, max(3, int(0.65 * u)), cv2.LINE_AA)
            glow = cv2.GaussianBlur(r, (0, 0), 1.2 * u) * 1.6
            red = np.array(hexrgb(L.RED2), np.float32)
            m = np.clip(np.maximum(r, glow * 0.6), 0, 1)[..., None] * e
            F[:] = F * (1 - m) + red * m
            if q >= 1:
                pulse = (t - self.t_up - 0.45) % 1.2 / 1.2
                D.place(F, D.circle(5 * u, L.RED2, a=0.3), b[0], b[1], op=e)
                D.place(F, D.circle(2.4 * u, L.RED2), b[0], b[1], op=e)
                D.place(F, D.ring(6 * u, 0.3 * u, L.RED2, a=1 - pulse), b[0], b[1], scale=0.6 + 0.9 * pulse, op=e)
        if self.num is not None:
            bx, by = self.pts[-1]
            self.num.draw(F, t, min(W - self.num.w / 2 - 2 * u, bx), by - 3 * u, self.t_up + 0.2, self.t1 - 0.25, anchor_x=0.5)

    def sounds(self):
        self.ensure()
        self.ctx.sfx(self.t_up, 'whoosh', 0.6)


# ---------------------------------------------------------------- Pull-back reveal
@reg('pull-back', need='footage', aliases=('pull-back-reveal', 'pullback'))
class PullBack(Beat):
    """The camera pulls back until my footage is a window in an app on a light grey desk: a window bar, a side panel whose steps tick red, a red progress bar."""
    full_frame = True
    fields = 'title, items (steps)'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        if self.ctx.vertical:
            ww, wh = 0.72 * W, 0.56 * H
            cx, cy = W * 0.5, H * 0.43
        else:
            ww, wh = 0.42 * W, 0.66 * H
            cx, cy = W * 0.44, H * 0.48
        self.win = (cx - ww / 2, cy - wh / 2, cx + ww / 2, cy + wh / 2)
        self.k = ww / W
        self.bar_h = 3.6 * u
        title = self.spec.get('title') or self.ctx.plan.get('brand') or 'Hal-abuur'
        self.tT = text_spr(title, 'semi', 2.2 * u, '#6A6A70')
        steps = self.items[:4] or [w for w in (self.line or self.spec.get('say') or '').split() if len(w) > 3][:4] or [self.word]
        self.steps = steps
        pw = int(0.5 * W if self.ctx.vertical else 0.24 * W)
        phh = int(5.6 * u + len(steps) * 5.0 * u + 1.5 * u)
        self.panel, _ = white_card(pw, phh, u, r=1.6 * u)
        self.pwh = (pw, phh)
        self.sT = [(text_spr(s, 'semi', 3.0 * u, L.INK), text_spr(s, 'semi', 3.0 * u, L.RED)) for s in steps]
        self.pT = text_spr((self.spec.get('panel') or self.word or title).upper(), 'xb', 2.2 * u, L.INK, tracking=0.08)
        self.desk = D.studio_bg(H, W, glow=0.0)
        self.stt = [self.t0 + 1.0 + i * max(0.25, (self.t1 - self.t0 - 1.8) / max(1, len(steps))) for i in range(len(steps))]

    def full(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        e = ease_in_out_cubic(prog(t, self.t0, 1.0)) * (1 - ease_in_out_cubic(prog(t, self.t1 - 0.6, 0.6)))
        if e <= 0:
            return None
        x0, y0, x1, y1 = [lerp(a, b, e) for a, b in zip((0, -self.bar_h, W, H), self.win)]
        bar = self.bar_h * e
        out = self.desk.copy()
        # footage scaled into the window (cover)
        ww, wh = x1 - x0, y1 - y0 - bar
        s = max(ww / W, wh / H)
        lay = self.lay
        M = np.float32([[s, 0, (x0 + x1) / 2 - s * lay.cx], [0, s, y0 + bar + wh * 0.42 - s * lay.eyes[1]]])
        M[1, 2] = clamp(M[1, 2], y0 + bar + wh - s * H, y0 + bar)
        S = cv2.warpAffine(F, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        rw, rh = max(2, int(ww)), max(2, int(y1 - y0))
        r = 1.8 * u * e
        a = D.rrect_alpha(rw, rh, r)
        m = np.zeros((H, W), np.float32)
        xi, yi = int(x0), int(y0)
        ya, xa = max(0, yi), max(0, xi)
        hh, wwi = min(H, yi + a.shape[0]) - ya, min(W, xi + a.shape[1]) - xa
        if hh > 0 and wwi > 0:
            m[ya:ya + hh, xa:xa + wwi] = a[ya - yi:ya - yi + hh, xa - xi:xa - xi + wwi]
        sh = np.roll(cv2.GaussianBlur(m, (0, 0), 2.6 * u), int(1.4 * u), axis=0)
        out *= (1 - 0.3 * sh * e)[..., None]
        out = out * (1 - m[..., None]) + S * m[..., None]
        if bar > 1:
            bm = m.copy()
            bm[int(y0 + bar):] = 0
            out = out * (1 - bm[..., None]) + np.array(hexrgb('#F7F7F9'), np.float32) * bm[..., None]
            for i in range(3):
                D.place(out, D.circle(1.0 * u, '#C9C9CF'), x0 + (2.2 + 1.6 * i) * u, y0 + bar / 2, op=e)
            T, ts = self.tT
            D.place(out, ts, x1 - 2 * u - ts.shape[1] / 2 + T.left, y0 + bar / 2 + ts.shape[0] / 2 - T.base + T.cap / 2, op=e)
        if e > 0.85:
            k = (e - 0.85) / 0.15
            pw, phh = self.pwh
            px = x1 - pw * 0.22 if self.ctx.vertical else x1 + pw * 0.45
            py = y0 + (y1 - y0) * 0.62
            D.place(out, self.panel, px, py, op=k)
            T, ts = self.pT
            D.place(out, D.circle(1.1 * u, L.RED), px - pw / 2 + 3.0 * u, py - phh / 2 + 3.2 * u, op=k)
            D.place(out, ts, px - pw / 2 + 4.8 * u + ts.shape[1] / 2 - T.left, py - phh / 2 + 3.2 * u + ts.shape[0] / 2 - T.base + T.cap / 2, op=k)
            yy = py - phh / 2 + 8.2 * u
            for i, ((A1, s1), (A2, s2)) in enumerate(self.sT):
                on = t >= self.stt[i]
                c = D.circle(2.8 * u, L.RED) if on else D.ring(2.8 * u, 0.3 * u, '#C9C9CF')
                D.place(out, c, px - pw / 2 + 4.0 * u, yy, op=k)
                if on:
                    D.place(out, icon('check', 2.2 * u), px - pw / 2 + 4.0 * u, yy, op=k)
                TT, ss = (A2, s2) if on else (A1, s1)
                D.place(out, ss, px - pw / 2 + 7.2 * u + ss.shape[1] / 2 - TT.left, yy + ss.shape[0] / 2 - TT.base + TT.cap / 2, op=k)
                yy += 5.0 * u
            # red progress bar along the bottom of the window
            fr = clamp((t - self.t0 - 1.0) / max(0.5, self.t1 - self.t0 - 1.6))
            bw = (x1 - x0) - 4 * u
            D.place(out, D.rrect(int(bw), max(3, int(0.6 * u)), 0.3 * u, fill='#ECECF0'), (x0 + x1) / 2, y1 - 2.4 * u, op=k)
            if fr > 0:
                w = max(3, int(bw * fr))
                D.place(out, D.rrect(w, max(3, int(0.6 * u)), 0.3 * u, fill=L.RED), x0 + 2 * u + w / 2, y1 - 2.4 * u, op=k)
        return out

    def sounds(self):
        self.ctx.sfx(self.t0, 'whoosh', 0.7)


# ---------------------------------------------------------------- Glass type
@reg('glass-type', aliases=('glass',))
class GlassType(Beat):
    """One key word as big 3D glass letters: the picture refracts inside them, a white specular edge, a thin red rim light, a soft shadow."""
    behind = True

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        txt = (self.word or '').upper()
        mode, size, base, cx = word_spot(self.ctx, lay, txt, self.spec.get('place', 'auto'), self.idx, 0.9)
        self.mode = mode
        T = D.Text(txt, 'black', size * 1.04, tracking=-0.03)
        a = T.alpha
        self.T, self.a = T, a
        g = cv2.GaussianBlur(a, (0, 0), size * 0.04)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
        self.dx = ((gx / mag.max()) * float(size) * 0.12).astype(np.float32)
        self.dy = ((gy / mag.max()) * float(size) * 0.12).astype(np.float32)
        edge = np.clip(mag / mag.max() * 3, 0, 1) * a
        spec = np.clip((-gx - gy) / (mag + 1e-3), 0, 1) ** 2 * edge
        rim = np.clip((gx + gy) / (mag + 1e-3), 0, 1) ** 2 * edge
        self.spec_a, self.rim_a = spec, rim
        self.pos = (cx - T.adv / 2 - T.left, base - T.base)
        self.t_in = self.when(self.word, self.t0 + 0.05)

    def _draw(self, F, t):
        e = ease_out_cubic(prog(t, self.t_in, 0.5)) * (1 - ease_in_cubic(prog(t, self.t1 - 0.25, 0.25)))
        if e <= 0:
            return
        H, W = F.shape[:2]
        h, w = self.a.shape
        x0 = int(self.pos[0])
        y0 = int(self.pos[1] + (1 - e) * 0.04 * H)
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if xb <= xa or yb <= ya:
            return
        sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
        roi = F[ya:yb, xa:xb]
        k = int(self.u * 1.2) * 2 + 1
        bl = cv2.GaussianBlur(roi, (k, k), self.u * 1.2)
        yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]].astype(np.float32)
        ref = cv2.remap(bl, (xx + self.dx[sl]).astype(np.float32), (yy + self.dy[sl]).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        a = self.a[sl][..., None] * e
        sh = cv2.GaussianBlur(self.a[sl], (0, 0), self.u * 0.9)
        sh = np.roll(sh, int(self.u * 0.8), axis=0)[..., None] * e
        roi *= (1 - 0.28 * sh)
        glass = ref * 0.86 + 0.14
        roi[:] = roi * (1 - a) + glass * a
        roi[:] = roi + self.spec_a[sl][..., None] * 0.9 * e
        red = np.array(hexrgb(L.RED2), np.float32)
        r = self.rim_a[sl][..., None] * 0.85 * e
        roi[:] = roi * (1 - r) + red * r

    def bg_on(self, t):
        return self.mode == 'behind'

    def bg(self, ctx, B, t):
        self.ensure()
        if self.mode == 'behind':
            self._draw(B, t)

    def front(self, ctx, F, t):
        self.ensure()
        if self.mode != 'behind':
            self._draw(F, t)

    def sounds(self):
        self.ensure()
        self.ctx.sfx(self.t_in, 'pop', 0.8)


# ---------------------------------------------------------------- The stack
def num_of(s):
    m = re.search(r'(\d+(?:[.,]\d+)?)', str(s))
    return float(m.group(1).replace(',', '.')) if m else None


@reg('stack', need='footage', aliases=('the-stack',))
class Stack(Beat):
    """The frame becomes 2–3 rounded panels stacked top to bottom: red bars that grow to the numbers I say, one line of my words, and my footage with small subtitles."""
    full_frame = True
    fields = 'items (bars: "16$", "14$" or "label 40%"), line, word'

    def prep(self):
        W, H, u = self.W, self.H, self.u
        self.bg_ = D.studio_bg(H, W, glow=0.3)
        bars = [x for x in self.items if num_of(x) is not None][:3]
        self.bars = bars
        pw = 0.88 * W
        if self.ctx.vertical:
            self.p1 = (W / 2 - pw / 2, 0.07 * H, W / 2 + pw / 2, 0.07 * H + (0.2 * H if bars else 0))
            y = self.p1[3] + (0.015 * H if bars else 0)
            self.p2 = (W / 2 - pw / 2, y, W / 2 + pw / 2, y + 0.1 * H)
            self.p3 = (W / 2 - pw / 2, self.p2[3] + 0.015 * H, W / 2 + pw / 2, 0.9 * H)
        else:
            self.p1 = (0.05 * W, 0.08 * H, 0.47 * W, 0.08 * H + (0.38 * H if bars else 0))
            self.p2 = (0.05 * W, self.p1[3] + 0.03 * H, 0.47 * W, 0.92 * H)
            self.p3 = (0.5 * W, 0.08 * H, 0.95 * W, 0.92 * H)
        ln = self.line or self.word or ''
        size = min(4.6 * u, D.fit_size(ln, 'xb_i', (self.p2[2] - self.p2[0]) * 0.86, 4.6 * u))
        self.lT = text_spr(ln, 'xb_i', size, L.INK)
        self.tT = text_spr((self.word or '').upper(), 'xb', 2.6 * u, L.RED, tracking=0.1)
        vals = [num_of(b) for b in bars]
        mx = max(vals) if vals else 1
        self.bvals = [v / mx for v in vals]
        self.bT = [text_spr(b, 'black_i', 3.6 * u, L.INK) for b in bars]

    def panel(self, out, box, k, r):
        x0, y0, x1, y1 = box
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        c, p = white_card(x1 - x0, y1 - y0, self.u, r=r, shadow_a=0.18)
        D.place(out, c, (x0 + x1) / 2, (y0 + y1) / 2 + (1 - k) * 0.06 * self.H, op=k)

    def full(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        e = env(t, self.t0, self.t1, 0.3, 0.25)
        if e <= 0:
            return None
        out = F * (1 - e) + self.bg_ * e
        k1 = ease_out_cubic(prog(t, self.t0, 0.45)) * e
        k2 = ease_out_cubic(prog(t, self.t0 + 0.12, 0.45)) * e
        k3 = ease_out_cubic(prog(t, self.t0 + 0.24, 0.45)) * e
        r = 2.2 * u
        if self.bars:
            self.panel(out, self.p1, k1, r)
            x0, y0, x1, y1 = self.p1
            T, s = self.tT
            D.place(out, s, x0 + 3 * u + s.shape[1] / 2 - T.left, y0 + 3.4 * u, op=k1)
            n = len(self.bars)
            bh = (y1 - y0 - 8 * u) / max(1, n)
            for i, (v, (BT, bs)) in enumerate(zip(self.bvals, self.bT)):
                g = ease_out_cubic(prog(t, self.t0 + 0.5 + 0.2 * i, 0.6))
                by = y0 + 6.5 * u + bh * (i + 0.5)
                bw = (x1 - x0 - 22 * u) * v * g
                D.place(out, D.rrect(int(x1 - x0 - 22 * u), max(4, int(bh * 0.5)), bh * 0.25, fill='#F0F0F3'), x0 + 3 * u + (x1 - x0 - 22 * u) / 2, by, op=k1)
                if bw > 4:
                    D.place(out, D.rrect(int(bw), max(4, int(bh * 0.5)), bh * 0.25, fill=L.RED), x0 + 3 * u + bw / 2, by, op=k1)
                D.place(out, bs, x1 - 3 * u - bs.shape[1] / 2 + BT.left, by + bs.shape[0] / 2 - BT.base + BT.cap / 2, op=k1 * g)
        self.panel(out, self.p2, k2, r)
        x0, y0, x1, y1 = self.p2
        T, s = self.lT
        D.place(out, s, (x0 + x1) / 2 - T.adv / 2 + s.shape[1] / 2 - T.left, (y0 + y1) / 2 + s.shape[0] / 2 - T.base + T.cap / 2, op=k2)
        # my footage in the last panel
        x0, y0, x1, y1 = self.p3
        if k3 > 0:
            pw, ph = int(x1 - x0), int(y1 - y0)
            lay = self.lay
            s = max(pw / W, ph / H) * 1.05
            M = np.float32([[s, 0, pw / 2 - s * lay.cx], [0, s, ph * 0.4 - s * lay.eyes[1]]])
            M[1, 2] = clamp(M[1, 2], ph - s * H, 0)
            crop = cv2.warpAffine(F, M, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            spr = np.concatenate([crop, np.ones((ph, pw, 1), np.float32)], 2)
            a = D.rrect_alpha(pw, ph, r)[:ph, :pw][..., None]
            spr = spr * a
            spr, p = D.shadow(spr, blur=2.0 * u, dy=1.2 * u, alpha=0.22)
            D.place(out, spr, (x0 + x1) / 2, (y0 + y1) / 2 + (1 - k3) * 0.06 * H, op=k3)
            if ctx.caps is not None:
                txt = ''
                for words, info in ctx.caps.chunks:
                    if info['t0'] <= t <= info['t1']:
                        txt = ' '.join(w['w'] for w in words).upper()
                if txt:
                    TT, ts = text_spr(txt, 'bold', 2.6 * u, '#FFFFFF', tracking=0.12)
                    ts = D.shadow(ts, blur=0.5 * u, dy=0.2 * u, alpha=0.6)[0]
                    D.place(out, ts, (x0 + x1) / 2, y1 - 4 * u, op=k3)
        return out

    def hide_captions(self):
        return [(self.t0 + 0.2, self.t1 - 0.1)]

    def sounds(self):
        self.ctx.sfx(self.t0, 'whoosh', 0.7)


# ---------------------------------------------------------------- Rack focus
@reg('rack-focus', need='footage')
class RackFocus(Beat):
    """A focus pull: soft red bokeh in the foreground, then I blur while one word in a large italic serif comes into focus in front; then back to me."""
    fields = 'word, kicker'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        R = self.R
        self.bokeh = []
        for i in range(7):
            r = R.uniform(6, 15) * u
            x = R.choice([R.uniform(-0.05, 0.25), R.uniform(0.75, 1.05)]) * W
            y = R.uniform(0.05, 0.95) * H
            self.bokeh.append((D.gblur(D.pad(D.circle(r * 2, R.choice([L.RED2, '#FF6B75', L.RED])), int(3 * u)), 1.6 * u), x, y, R.uniform(0.25, 0.5)))
        txt = self.word.lower() if self.word else ''
        size = min(13 * u, D.fit_size(txt, 'serif_i', 0.86 * W, 13 * u))
        T, s = text_spr(txt, 'serif_i', size, '#FFFFFF')
        self.wT = (T, D.shadow(s, blur=1.0 * u, dy=0.4 * u, alpha=0.45)[0])
        k = (self.spec.get('kicker') or '').upper()
        self.kT = text_spr(k, 'xb', 1.8 * u, '#FFFFFF', tracking=0.2) if k else None
        self.pos = (W / 2, min(lay.chest[0] + size * 0.95, lay.chest[1]))

    def front(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        e = env(t, self.t0, self.t1, 0.25, 0.25)
        if e <= 0:
            return
        f = ease_in_out_sine(prog(t, self.t0 + 0.25, 0.45)) * (1 - ease_in_out_sine(prog(t, self.t1 - 0.55, 0.45)))
        if f > 0.02:
            F[:] = D.frame_blur(F, 1.0 * u * f) * (1 - 0.15 * f)
        for spr, x, y, a in self.bokeh:
            D.place(F, spr, x, y, op=a * e, blur=1.2 * u * f)
        T, s = self.wT
        D.place(F, s, self.pos[0], self.pos[1] - T.base + s.shape[0] / 2, op=f, blur=1.4 * u * (1 - f))
        if self.kT:
            KT, ks = self.kT
            D.place(F, ks, self.pos[0], self.pos[1] - T.cap * 1.25, op=f)

    def sounds(self):
        self.ctx.sfx(self.t0 + 0.25, 'swell', 0.6)


# ---------------------------------------------------------------- Spatial glass
@reg('spatial-glass', need='footage')
class SpatialGlass(Beat):
    """Frosted glass panels float in my room: a search bar at chest height typing my phrase, a small widget on the wall behind me."""
    fields = 'line (the phrase), word (widget)'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        self.text = self.line or self.spec.get('say') or self.word
        bw, bh = 0.86 * W if self.ctx.vertical else 0.44 * W, 9.0 * u
        y = lay.chest[0] + bh / 2 + 0.01 * H
        self.bar = (W / 2 - bw / 2, y - bh / 2, W / 2 + bw / 2, y + bh / 2)
        size = 3.4 * u
        txt = self.text
        while D.measure(txt, 'semi', size) > bw - 12 * u and len(txt) > 4:
            txt = txt[:-2]
        self.tT = text_spr(txt, 'semi', size, '#FFFFFF')
        ws = 0.24 * W
        side = -1 if lay.cx > W / 2 else 1
        wx = W * (0.16 if side < 0 else 0.84)
        wy = self.ctx.top_safe + ws * 0.45
        if boxes_overlap((wx - ws / 2, wy - ws * 0.35, wx + ws / 2, wy + ws * 0.35), lay.core):
            wy = lay.chest[1] - ws * 0.35
            self.widget_ok = wy - ws * 0.35 > self.bar[3] + 0.01 * H
        else:
            self.widget_ok = True
        self.widget = (wx - ws / 2, wy - ws * 0.35, wx + ws / 2, wy + ws * 0.35)
        self.wT = text_spr((self.word or '').upper(), 'xb_i', 3.6 * u, '#FFFFFF')

    def cam_shift(self, ctx, t, box):
        s1, a = ctx.cam_at(t)
        s0, _ = ctx.cam_at(self.t0)
        if a is None:
            return box
        k = s1 / max(1e-3, s0)
        x0, y0, x1, y1 = box
        f = lambda x, y: (a[0] + k * (x - a[0]), a[1] + k * (y - a[1]))
        p0, p1 = f(x0, y0), f(x1, y1)
        return (p0[0], p0[1], p1[0], p1[1])

    def front(self, ctx, F, t):
        self.ensure()
        u = self.u
        e = env(t, self.t0, self.t1, 0.35, 0.25)
        if e <= 0:
            return
        if self.widget_ok:
            k = ease_out_cubic(prog(t, self.t0 + 0.15, 0.4)) * e
            if k > 0.05:
                bx = self.cam_shift(ctx, t, self.widget)
                frost(F, bx, u, 1.8 * u, tint=0.18 * k)
                D.place(F, badge('bolt', 4.4 * u), bx[0] + 3.6 * u, (bx[1] + bx[3]) / 2, op=k)
                T, s = self.wT
                D.place(F, s, bx[0] + 6.6 * u + s.shape[1] / 2 - T.left, (bx[1] + bx[3]) / 2 + s.shape[0] / 2 - T.base + T.cap / 2, op=k)
        k = ease_out_cubic(prog(t, self.t0, 0.4)) * e
        if k > 0.05:
            bx = self.cam_shift(ctx, t, self.bar)
            frost(F, bx, u, (bx[3] - bx[1]) / 2, tint=0.5 * k, dark=True)
            D.place(F, D.ring(3.2 * u, 0.4 * u, '#FFFFFF'), bx[0] + 4.0 * u, (bx[1] + bx[3]) / 2, op=k)
            T, s = self.tT
            f = prog(t, self.t0 + 0.4, max(0.5, min(self.t1 - self.t0 - 0.9, 0.05 * len(T.text))))
            sp = typed(T, s, f)
            if sp is not None:
                D.place(F, sp, bx[0] + 7.4 * u + sp.shape[1] / 2 - T.left, (bx[1] + bx[3]) / 2 + sp.shape[0] / 2 - T.base + T.cap / 2, op=k)

    def hide_captions(self):
        return [(self.t0 + 0.3, self.t1)]

    def sounds(self):
        self.ctx.sfx(self.t0, 'whoosh', 0.6)


# ---------------------------------------------------------------- Sticky text
@reg('sticky-text', need='footage', aliases=('sticky',))
class StickyText(Beat):
    """A small red label (white bold caps, a tiny kicker above) sticks next to my head and follows it with a slight lag."""
    fields = 'word (label), kicker'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        txt = (self.word or '').upper()
        T, s = text_spr(txt, 'black', 4.2 * u, '#FFFFFF', tracking=0.02)
        w, h = int(T.adv + 4.4 * u), int(T.cap + 3.6 * u)
        box = D.rrect(w, h, 0.9 * u, fill=L.RED)
        D.blend_spr(box, s, 2.2 * u - T.left, h / 2 + T.cap / 2 - T.base)
        k = (self.spec.get('kicker') or '').upper()
        if k:
            KT, ks = text_spr(k[:22], 'xb', 2.1 * u, '#FFFFFF', tracking=0.12)
            ks = D.shadow(ks, blur=0.4 * u, dy=0.2 * u, alpha=0.6)[0]
            out = D.empty(h + ks.shape[0], max(w, ks.shape[1]))
            D.blend_spr(out, ks, 0, 0)
            D.blend_spr(out, box, 0, ks.shape[0] - 0.6 * u)
            box = out
        self.label, _ = D.shadow(box, blur=1.0 * u, dy=0.6 * u, alpha=0.35)
        lw = self.label.shape[1]
        self.right = (W - lay.face_pad[2]) > lw * 0.95 or lay.cx < W / 2

    def front(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        s, a = pop_in(t, self.t0, 0.35, 1.6)
        a *= 1 - prog(t, self.t1 - 0.2, 0.2)
        if a <= 0:
            return
        f = ctx.face(max(self.t0, t - 0.09))
        f2 = ctx.face(max(self.t0, t - 0.2))
        lw, lh = self.label.shape[1], self.label.shape[0]
        x0, y0, x1, y1 = f['box']
        if self.right:
            x = min(W - lw / 2 - 1.5 * u, x1 + lw / 2 + 0.01 * W)
        else:
            x = max(lw / 2 + 1.5 * u, x0 - lw / 2 - 0.01 * W)
        y = max(ctx.top_safe + lh / 2, y0 + (y1 - y0) * 0.18)
        vx = (f['cx'] - f2['cx']) / max(1.0, u)
        D.place(F, self.label, x, y, scale=s, rot=clamp(-vx * 0.8, -6, 6) - 3, op=a)

    def sounds(self):
        self.ctx.sfx(self.t0, 'pop', 0.6)


# ---------------------------------------------------------------- Freeze + label
@reg('freeze-label', need='footage', aliases=('freeze-+-label', 'freeze-and-label', 'callouts'))
class FreezeLabel(Beat):
    """The frame freezes and numbered callouts type on one by one: a white pill with a red number badge, joined by a thin line to a dot on me."""
    full_frame = True
    fields = 'items (1–3 callout texts), freeze: true|false'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        self.freeze = self.spec.get('freeze', True)
        self.still = self.ctx.still(self.t0 + 0.04) if self.freeze else None
        items = self.items[:3] or [self.word or '…']
        fx0, fy0, fx1, fy1 = lay.face
        hy = lay.head + max(0.025 * H, (lay.core[1] - lay.head) * 0.45)
        targets = [(lay.cx, hy), (min(W - 0.06 * W, fx1 + 0.04 * W), lay.chin + 0.02 * H), (lay.cx - 0.12 * W, lay.chin + 0.14 * H)]
        spots = [(0.26 * W, self.ctx.top_safe + 0.035 * H), (0.76 * W, lay.chin + 0.1 * H), (0.3 * W, lay.chest[1] - 0.01 * H)]
        self.calls = []
        for i, txt in enumerate(items):
            T, s = text_spr(txt, 'bold', 3.4 * u, L.INK)
            w, h = int(T.adv + 9.4 * u), int(T.cap + 3.8 * u)
            pill = D.rrect(w, h, h / 2, fill='#FFFFFF')
            b = D.rrect(int(4.2 * u), int(3.0 * u), 0.8 * u, fill=L.RED)
            NT, ns = text_spr('%02d' % (i + 1), 'xb', 1.9 * u, '#FFFFFF')
            D.blend_spr(b, ns, 2.1 * u - NT.adv / 2 - NT.left, 1.5 * u + NT.cap / 2 - NT.base)
            D.blend_spr(pill, b, 1.2 * u, h / 2 - 1.5 * u)
            pill, _ = D.shadow(pill, blur=1.0 * u, dy=0.5 * u, alpha=0.3)
            sx, sy = spots[i]
            sx = clamp(sx, w / 2 + 1.5 * u, W - w / 2 - 1.5 * u)
            self.calls.append((T, s, pill, w, (sx, sy), targets[i], self.item_t(i, len(items)) if self.items else self.t0 + 0.2))

    def full(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        e = env(t, self.t0, self.t1, 0.06, 0.12)
        if e <= 0:
            return None
        base = (self.still if self.still is not None else F).copy()
        if self.freeze:
            base = D.desat(base, 0.12 * e)
        for T, s, pill, w, (sx, sy), (tx, ty), ti in self.calls:
            p = prog(t, ti, 0.35)
            if p <= 0:
                continue
            al = np.zeros((H, W), np.float32)
            ex, ey = lerp(tx, sx, ease_out_cubic(p)), lerp(ty, sy, ease_out_cubic(p))
            cv2.line(al, (int(tx), int(ty)), (int(ex), int(ey)), 1.0, max(1, int(0.22 * u)), cv2.LINE_AA)
            base = base * (1 - al[..., None]) + al[..., None]
            D.place(base, D.circle(2.6 * u, '#FFFFFF'), tx, ty, scale=ease_out_back(clamp(p * 2), 2.0))
            D.place(base, D.circle(1.3 * u, L.RED), tx, ty, scale=ease_out_back(clamp(p * 2), 2.0))
            if p >= 0.6:
                k = prog(t, ti + 0.2, 0.4)
                D.place(base, pill, sx, sy, scale=ease_out_back(clamp(k * 1.5), 1.6))
                sp = typed(T, s, k)
                if sp is not None:
                    D.place(base, sp, sx - w / 2 + 6.4 * u + sp.shape[1] / 2 - T.left, sy + sp.shape[0] / 2 - T.base + T.cap / 2)
        return base

    def sounds(self):
        self.ensure()
        for c in self.calls[:2]:
            self.ctx.sfx(c[-1], 'tap', 0.7)


# ---------------------------------------------------------------- Shatter
@reg('shatter', aliases=('shatter-3d',))
class Shatter(Beat):
    """The frame cracks from one point, splits into shards that separate in 3D depth with thin white borders, and fall away into the next beat."""
    full_frame = True
    post = 0.0

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        self.t1 = min(self.t1, self.t0 + 1.2)
        R = self.R
        cx, cy = lay.cx + R.uniform(-0.05, 0.05) * W, (lay.chin + lay.chest[1]) / 2
        self.c = (cx, cy)
        img = self.ctx.still(self.t0)
        angs = sorted([R.uniform(0, 2 * math.pi) for _ in range(7)] + [i * 2 * math.pi / 7 + R.uniform(-0.2, 0.2) for i in range(0)])
        angs = sorted(a % (2 * math.pi) for a in angs)
        rings = [0.0, 0.2 * H, 2.5 * max(W, H)]
        self.shards = []
        q = 2
        h2, w2 = H // q, W // q
        yy, xx = np.mgrid[0:h2, 0:w2].astype(np.float32) * q
        ang = np.arctan2(yy - cy, xx - cx) % (2 * math.pi)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        self.cracks = []
        for i in range(len(angs)):
            a0, a1 = angs[i], angs[(i + 1) % len(angs)]
            if a1 <= a0:
                sel_a = (ang >= a0) | (ang < a1)
            else:
                sel_a = (ang >= a0) & (ang < a1)
            for j in range(2):
                r0, r1 = rings[j], rings[j + 1]
                m = (sel_a & (dist >= r0) & (dist < r1)).astype(np.uint8)
                if m.sum() < 50:
                    continue
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                ys, xs = np.where(m > 0)
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                mm = m[y0:y1, x0:x1].astype(np.float32)
                edge = mm - cv2.erode(mm, np.ones((max(3, int(0.3 * u)),) * 2, np.uint8))
                spr = np.concatenate([img[y0:y1, x0:x1] * mm[..., None], mm[..., None]], 2)
                spr[..., :3] = spr[..., :3] * (1 - edge[..., None]) + edge[..., None] * mm[..., None]
                mid = ((a0 + a1) / 2 + (math.pi if a1 < a0 else 0)) % (2 * math.pi)
                self.shards.append((spr, (x0 + x1) / 2, (y0 + y1) / 2, mid, R.uniform(-1, 1), j))
            self.cracks.append(a0)

    def full(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        tt = t - self.t0
        if tt < 0:
            return None
        cx, cy = self.c
        if tt < 0.22:
            out = ctx.still(self.t0).copy()
            f = ease_out_cubic(tt / 0.22)
            al = np.zeros((H, W), np.float32)
            for a in self.cracks:
                L_ = f * max(W, H)
                cv2.line(al, (int(cx), int(cy)), (int(cx + math.cos(a) * L_), int(cy + math.sin(a) * L_)), 1.0, max(2, int(0.3 * u)), cv2.LINE_AA)
            cv2.circle(al, (int(cx), int(cy)), int(0.2 * H * f), 1.0, max(2, int(0.25 * u)), cv2.LINE_AA)
            return out * (1 - al[..., None]) + al[..., None]
        p = (tt - 0.22) / max(0.3, self.t1 - self.t0 - 0.22)
        out = F.copy()
        for spr, x, y, mid, rnd_, ring in sorted(self.shards, key=lambda s: s[4]):
            sep = ease_out_cubic(clamp(p * 2.2)) * (0.05 + 0.04 * ring) * W
            fall = (clamp(p - 0.35) ** 2) * 1.6 * H
            z = 1 + rnd_ * 0.16 * ease_out_cubic(clamp(p * 2))
            dx = math.cos(mid) * sep
            dy = math.sin(mid) * sep + fall
            a = 1 - clamp((p - 0.6) / 0.4)
            D.place(out, spr, x + dx, y + dy, scale=z, rot=rnd_ * 25 * clamp(p - 0.2), op=a)
        return out

    def sounds(self):
        self.ctx.sfx(self.t0, 'glass', 1.0)


# ---------------------------------------------------------------- Chip row
@reg('chip-row', aliases=('chips', 'tags'))
class ChipRow(Beat):
    """Small pills (white with a red outline, or red with white text) appear one by one in a row at chest height, each on its spoken word."""
    fields = 'items (short words)'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        items = self.items or [w for w in (self.line or '').split()][:5]
        size = 4.4 * u
        sprs = []
        for i, it in enumerate(items[:6]):
            if i % 2 == 0:
                sprs.append(L.pill(it, size, bg='#FFFFFF', fg=L.RED, border=L.RED, bw=0.3 * u))
            else:
                sprs.append(L.pill(it, size))
        maxw = 0.9 * W
        rows, cur, cw = [], [], 0
        for s in sprs:
            if cur and cw + s.shape[1] > maxw:
                rows.append(cur)
                cur, cw = [], 0
            cur.append(s)
            cw += s.shape[1] + 0.4 * u
        if cur:
            rows.append(cur)
        self.chips = []
        y = lay.chest[0] + 3.4 * u
        k = 0
        for r in rows:
            tw = sum(s.shape[1] for s in r) + 0.4 * u * (len(r) - 1)
            x = W / 2 - tw / 2
            for s in r:
                self.chips.append((s, x + s.shape[1] / 2, y, self.item_t(k, len(sprs)) if self.items else self.when(items[k], self.t0 + 0.3 * k)))
                x += s.shape[1] + 0.4 * u
                k += 1
            y += sprs[0].shape[0] * 0.85

    def front(self, ctx, F, t):
        self.ensure()
        out = prog(t, self.t1 - 0.2, 0.2)
        for s, x, y, ti in self.chips:
            sc, a = pop_in(t, ti, 0.35, 1.7)
            a *= 1 - out
            if a > 0:
                D.place(F, s, x, y, scale=sc, op=a)

    def sounds(self):
        self.ensure()
        for c in self.chips[:2]:
            self.ctx.sfx(c[3], 'pop', 0.55)


# ---------------------------------------------------------------- Serif flow
@reg('serif-flow', aliases=('serif', 'slate'))
class SerifFlow(Beat):
    """A full-screen slate (near-black or deep red): a tiny caps kicker in brackets, one elegant italic serif line, a thin vertical line, then bold words adding one by one."""
    full_frame = True
    post = 0.0
    fields = 'kicker, line (serif line), items (bold words added one by one), red: true'

    def prep(self):
        W, H, u = self.W, self.H, self.u
        red = self.spec.get('red', self.idx % 2 == 1)
        top, bot = ('#4A0010', '#2A0008') if red else ('#1B1B1F', '#0E0E11')
        self.bg_ = D.grad_v(H, W, [(0, top), (1, bot)])
        k = (self.spec.get('kicker') or self.word or '').upper()
        self.kT = text_spr('[ %s ]' % k if k else '', 'xb', 2.3 * u, '#B9B9C0' if not red else '#FF9AA2', tracking=0.2)
        ln = self.line or self.spec.get('say') or ''
        size = min(9.6 * u, D.fit_size(ln, 'serif_i', 0.88 * W, 9.6 * u)) if len(ln) < 22 else 8.2 * u
        self.lines = D.wrap(ln, 'serif_i', size, 0.88 * W, max_lines=2)
        self.size = size
        self.sT = [[text_spr(w, 'serif_i', size, '#FFFFFF') for w in l.split()] for l in self.lines]
        self.items_ = self.items[:3]
        self.iT = [text_spr(x, 'bold', 5.6 * u, '#FFFFFF') for x in self.items_]
        self.plus = text_spr('+', 'bold', 5.6 * u, L.RED2)

    def full(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        e = env(t, self.t0, self.t1, 0.25, 0.2)
        if e <= 0:
            return None
        out = F * (1 - e) + self.bg_ * e
        y = H * 0.36
        KT, ks = self.kT
        if KT.text:
            D.place(out, ks, W / 2, y, op=ease_out_cubic(prog(t, self.t0 + 0.12, 0.3)) * e)
        y += self.size * 1.25
        i = 0
        for ln, ws in zip(self.lines, self.sT):
            tw = sum(T.adv for T, _ in ws) + self.size * 0.25 * (len(ws) - 1)
            x = W / 2 - tw / 2
            for T, s in ws:
                a = ease_out_cubic(prog(t, self.t0 + 0.25 + i * 0.09, 0.35)) * e
                D.place(out, s, x + s.shape[1] / 2 - T.left, y - T.base + s.shape[0] / 2 + (1 - a) * 0.6 * u, op=a)
                x += T.adv + self.size * 0.25
                i += 1
            y += self.size * 1.15
        g = ease_out_cubic(prog(t, self.t0 + 0.5, 0.35))
        lh = 6 * u * g
        if lh > 1:
            D.place(out, D.rrect(max(2, int(0.18 * u)), int(lh), 0.1 * u, fill='#FFFFFF'), W / 2, y + lh / 2 - 1 * u, op=0.7 * e)
        y += 9.5 * u
        if self.iT:
            n = sum(1 for j in range(len(self.iT)) if t >= self.item_t(j, len(self.iT)))
            n = max(n, 1 if t >= self.t0 + 0.8 else 0)
            parts = []
            for j in range(n):
                if j:
                    parts.append(self.plus)
                parts.append(self.iT[j])
            tw = sum(T.adv for T, _ in parts) + 1.4 * u * (len(parts) - 1)
            x = W / 2 - tw / 2
            for T, s in parts:
                D.place(out, s, x + s.shape[1] / 2 - T.left, y - T.base + s.shape[0] / 2, op=e)
                x += T.adv + 1.4 * u
        return out

    def hide_captions(self):
        return [(self.t0 + 0.15, self.t1)]

    def sounds(self):
        self.ctx.sfx(self.t0, 'whoosh', 0.6)


# ---------------------------------------------------------------- Prompt box
@reg('prompt-box', aliases=('prompt',))
class PromptBox(Beat):
    """A translucent dark rounded input box with a tiny label above at chest height; my sentence types on with a blinking cursor, then a red send button is tapped."""
    fields = 'line (sentence), label'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        self.text = self.line or self.spec.get('say') or self.word
        bw = 0.88 * W if self.ctx.vertical else 0.5 * W
        size = 3.6 * u
        self.lines = D.wrap(self.text, 'semi', size, bw - 12 * u, max_lines=3)
        bh = 5 * u + len(self.lines) * size * 1.35
        x0, y0, x1, y1 = chest_box(lay, bw, bh)
        self.box = (x0, y0, x1, y1)
        self.size = size
        self.Ts = [text_spr(l, 'semi', size, '#FFFFFF') for l in self.lines]
        self.lab = text_spr((self.spec.get('label') or 'prompt').lower(), 'semi', 2.3 * u, '#FFFFFF', tracking=0.04)
        n = sum(len(l) for l in self.lines)
        self.type = (self.t0 + 0.3, max(0.6, min(self.t1 - self.t0 - 0.9, 0.045 * n)))
        self.tap = self.type[0] + self.type[1] + 0.15

    def front(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        e = env(t, self.t0, self.t1, 0.3, 0.25)
        if e <= 0:
            return
        x0, y0, x1, y1 = self.box
        y0e, y1e = y0 + (1 - e) * 1.5 * u, y1 + (1 - e) * 1.5 * u
        frost(F, (x0, y0e, x1, y1e), u, 2.2 * u, tint=0.62 * e, dark=True)
        T, s = self.lab
        D.place(F, s, x0 + 1.6 * u + s.shape[1] / 2 - T.left, y0e - 1.8 * u, op=e * 0.85)
        frac = prog(t, *self.type)
        left = frac * sum(len(T.text) for T, _ in self.Ts)
        y = y0e + 2.8 * u + self.size * 0.8
        cur = None
        for T, sp in self.Ts:
            k = clamp(left / max(1, len(T.text)))
            left -= len(T.text)
            spr = typed(T, sp, k)
            if spr is not None:
                D.place(F, spr, x0 + 2.6 * u + spr.shape[1] / 2 - T.left, y - T.base + spr.shape[0] / 2, op=e)
                cur = (x0 + 2.6 * u + T.xs[min(len(T.text) - 1, int(k * len(T.text)))] + T.adv / max(1, len(T.text)), y)
            y += self.size * 1.35
        if cur and (frac < 1 or int(t * 2.6) % 2 == 0):
            D.place(F, D.rrect(max(2, int(0.28 * u)), int(self.size * 1.1), 0.1 * u, fill=L.RED2), cur[0] + 0.3 * u, cur[1] - self.size * 0.36, op=e)
        tp = prog(t, self.tap, 0.3)
        bx, by = x1 - 3.4 * u, y1e - 3.4 * u
        D.place(F, D.circle(4.6 * u, L.RED), bx, by, scale=1 - 0.12 * math.sin(math.pi * tp), op=e)
        D.place(F, icon('play', 2.6 * u), bx + 0.2 * u, by, op=e)
        if 0 < tp < 1:
            D.place(F, D.ring(9 * u, 0.4 * u, L.RED2, a=1 - tp), bx, by, scale=0.4 + 0.9 * tp)

    def hide_captions(self):
        return [(self.t0 + 0.2, self.t1)]

    def sounds(self):
        self.ensure()
        self.ctx.sfx(self.tap, 'click', 0.8)


# ---------------------------------------------------------------- Counter list
@reg('counter-list', aliases=('counter', 'count'))
class CounterList(Beat):
    """When I count: a small red counter "01/03" above the item's name, item by item (and, if room, a compact list card)."""
    fields = 'items (the things counted)'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        items = self.items or [self.word]
        self.its = []
        n = len(items)
        for i, it in enumerate(items):
            txt = it.upper()
            mode, size, base, cx = word_spot(self.ctx, lay, txt, 'chest', self.idx, 0.8, cap_max=0.06)
            self.its.append((L.BigWord(txt, size), cx, base, self.item_t(i, n) if self.items else self.t0 + 0.1))
        self.cT = [text_spr('%02d/%02d' % (i + 1, n), 'xb', 3.3 * u, L.RED2, tracking=0.16) for i in range(n)]

    def front(self, ctx, F, t):
        self.ensure()
        u = self.u
        for i, (bw, cx, base, ti) in enumerate(self.its):
            te = self.its[i + 1][3] - 0.2 if i + 1 < len(self.its) else self.t1 - 0.25
            if t < ti - 0.05 or t > te + 0.5:
                continue
            bw.draw(F, t, cx, base, ti, te)
            T, s = self.cT[i]
            a = ease_out_cubic(prog(t, ti, 0.3)) * (1 - prog(t, te, 0.2))
            s = D.shadow(s, blur=0.5 * u, dy=0.2 * u, alpha=0.5)[0]
            D.place(F, s, cx, base - bw.cap - 2.6 * u, op=a)

    def sounds(self):
        self.ensure()
        for it in self.its[:3]:
            self.ctx.sfx(it[3], 'pop', 0.7)


# ---------------------------------------------------------------- Comment CTA
@reg('comment-cta', aliases=('comment', 'follow', 'cta'))
class CommentCTA(Beat):
    """A profile card and a comments sheet slide up; the word I ask for types into the comment box and posts with a red heart (or the Follow button turns into Following)."""
    fields = 'word (the comment word), name, mode: comment|follow'

    def prep(self):
        W, H, u, lay = self.W, self.H, self.u, self.lay
        self.mode = self.spec.get('mode') or ('follow' if re.search(r'follow|soco', (self.word + ' ' + self.line).lower()) else 'comment')
        self.name = self.spec.get('name') or self.ctx.plan.get('brand') or 'Hal-abuur'
        sh = int((0.34 if self.ctx.vertical else 0.44) * H)
        sw = int(W if self.ctx.vertical else 0.46 * W)
        self.sheet = D.rrect(sw, sh + int(3 * u), 3 * u, fill='#FFFFFF')
        self.sw, self.sh = sw, sh
        # avatar from my face
        f = self.ctx.face(self.t0)
        x0, y0, x1, y1 = [int(v) for v in f['box']]
        im = self.ctx.still(self.t0)
        crop = im[max(0, y0):max(1, y1), max(0, x0):max(1, x1)]
        self.av = D.image_sprite(crop, 9 * u, 9 * u, r=4.5 * u) if crop.size else D.circle(9 * u, L.RED)
        self.nT = text_spr(self.name, 'bold', 3.3 * u, L.INK)
        self.hT = text_spr(self.spec.get('title', 'Comments') if self.mode == 'comment' else '', 'xb', 2.9 * u, L.INK)
        self.wT = text_spr(self.word or '…', 'semi', 3.6 * u, L.INK)
        self.btn = L.pill(self.spec.get('button', 'Follow'), 4.2 * u)
        self.btn2 = L.pill(self.spec.get('done', 'Following'), 4.2 * u, bg='#1A1A1A')
        self.type = (self.t0 + 0.6, 0.5)
        self.post_t = self.t0 + 1.25

    def front(self, ctx, F, t):
        self.ensure()
        W, H, u = self.W, self.H, self.u
        e = ease_out_cubic(prog(t, self.t0, 0.45)) * (1 - ease_in_cubic(prog(t, self.t1 - 0.3, 0.3)))
        if e <= 0:
            return
        sx = W / 2 if self.ctx.vertical else W * 0.72
        top = H - self.sh * e
        D.place(F, D.shadow(self.sheet, blur=2.4 * u, dy=-0.6 * u, alpha=0.25)[0], sx, top + self.sheet.shape[0] / 2 - 1.5 * u)
        x0 = sx - self.sw / 2 + 5 * u
        y = top + 5.0 * u
        if self.mode == 'comment':
            T, s = self.hT
            D.place(F, s, sx, y)
            y += 8.0 * u
            if t >= self.post_t:
                k = ease_out_back(prog(t, self.post_t, 0.35), 1.6)
                D.place(F, self.av, x0 + 4.5 * u, y + 2.0 * u, scale=k)
                T, s = self.nT
                D.place(F, s, x0 + 11 * u + s.shape[1] / 2 - T.left, y, op=k)
                T2, s2 = self.wT
                D.place(F, s2, x0 + 11 * u + s2.shape[1] / 2 - T2.left, y + 4.4 * u, op=k)
                D.place(F, icon('heart', 4.4 * u, L.RED), x0 + self.sw - 14 * u, y + 2.0 * u, scale=ease_out_back(prog(t, self.post_t + 0.25, 0.35), 2.2))
            by = top + self.sh - 9 * u
            box = D.rrect(int(self.sw - 10 * u), int(7.4 * u), 3.7 * u, fill='#F1F1F4')
            D.place(F, box, sx, by)
            if t < self.post_t:
                T, s = self.wT
                sp = typed(T, s, prog(t, *self.type))
                if sp is not None:
                    D.place(F, sp, x0 + 3 * u + sp.shape[1] / 2 - T.left, by + sp.shape[0] / 2 - T.base + T.cap / 2)
            D.place(F, D.circle(5.4 * u, L.RED), x0 + self.sw - 15 * u, by, scale=1 - 0.12 * math.sin(math.pi * prog(t, self.post_t - 0.15, 0.3)))
            D.place(F, icon('play', 2.9 * u), x0 + self.sw - 14.7 * u, by)
        else:
            D.place(F, self.av, x0 + 4.5 * u, y + 4 * u)
            T, s = self.nT
            D.place(F, s, x0 + 10.5 * u + s.shape[1] / 2 - T.left, y + 4 * u + s.shape[0] / 2 - T.base + T.cap / 2)
            tap = prog(t, self.post_t, 0.25)
            b = self.btn if tap < 0.5 else self.btn2
            bx = x0 + self.sw - 8 * u - b.shape[1] / 2
            D.place(F, b, bx, y + 4 * u, scale=1 - 0.08 * math.sin(math.pi * tap))
            if 0 < tap < 1:
                D.place(F, D.ring(12 * u, 0.4 * u, L.RED, a=1 - tap), bx, y + 4 * u, scale=0.4 + 0.8 * tap)

    def hide_captions(self):
        return [(self.t0 + 0.1, self.t1)]

    def sounds(self):
        self.ensure()
        self.ctx.sfx(self.t0, 'whoosh', 0.6)
        self.ctx.sfx(self.post_t, 'pop', 0.8)


# ---------------------------------------------------------------- Blur-in words
@reg('blur-in', aliases=('blur-in-words', 'blurin'))
class BlurIn(Word):
    """A big word arrives with a blur-in (blur → 0, 108% → 100%) and leaves the same way, under a thin line (two-line caption)."""

    def __init__(self, spec, ctx, idx):
        super().__init__(spec, ctx, idx)
        spec.setdefault('place', 'chest')

    def draw_word(self, F, t, op=1.0, mode='blur'):
        if self.bw is None:
            return
        self.bw.draw(F, t, self.cx, self.base_y, self.t_word, self.t1 - 0.3, op=op, mode='blur', stagger=0.0, dur=0.32)
