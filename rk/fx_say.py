"""The core Red Kinetic beat: the spoken phrase as a typographic lockup built word by word as it is said
(small italic words around one or two huge red-gradient key words), with an optional picture or graphic that shows
what is said (a product card, the old price struck out, a calendar date, a list of chips, an icon).
While a lockup is on screen it is the subtitle: the bottom captions wait."""
import math
import re

import numpy as np

from . import draw as D
from . import look as L
from . import plan as P
from .util import clamp, prog, ease_out_cubic, ease_in_cubic, ease_out_back, hexrgb
from .fx import reg, Beat, img_from, bg_crop, bg_score, icon, badge

PUNCT = '.,!?;:…"“”«»()'
SMALL_KEY = 'semi_i'
SMALL_DARK = '#8E1020'
SMALL_DARK2 = '#C42A36'
SMALL_LIGHT = '#FFFFFF'
INK = '#1A1A1A'
GREY_GRAD = [(0.0, '#9A9AA2'), (1.0, '#5E5E66')]


def clean(w):
    return w.strip(PUNCT)


def _as_list(v):
    if v is None or v == '':
        return []
    return v if isinstance(v, list) else [v]


class _Small:
    """A small italic word of the lockup (its sprite, where its baseline-left goes, when it appears)."""

    def __init__(self, text, size, color, t):
        T = D.Text(text, SMALL_KEY, size, pad_px=int(size * 0.35))
        spr = T.sprite(color)
        if color == SMALL_LIGHT:
            spr, p = D.shadow(spr, blur=size * 0.22, dy=size * 0.07, color=(0, 0, 0), alpha=0.6)
        else:
            spr, p = D.shadow(spr, blur=size * 0.2, dy=size * 0.05, color=(1, 1, 1), alpha=0.5)
        spr, ox, oy = D.tight(spr)
        self.spr, self.size, self.t = spr, size, t
        self.adv = T.adv
        self.cap = T.cap
        self.ox = T.left + p - ox     # baseline-left inside the sprite
        self.oy = T.base + p - oy
        self.x = self.y = 0.0


@reg('say', aliases=('lockup', 'phrase', 'line', 'sentence', 'talk'))
class Say(Beat):
    """The spoken phrase, word by word: small italic words + 1–2 huge red key words (`big`), never over the face;
    optional `visual` (card / cards / chips / date / icon) that shows what is said; `big` items with "strike": true
    are an old price or a word crossed out."""
    fields = 'big (key word(s) shown huge, as said or as "16$"), visual {type: card|cards|chips|date|icon, …}, place'
    behind = True
    post = 0.2

    # ------------------------------------------------------------ words and roles
    def _words(self):
        ctx = self.ctx
        ws = [dict(w) for c in ctx.cues for w in c['words'] if self.t0 - 0.15 <= w['t0'] < self.t1 - 0.06]
        txt = str(self.spec.get('text') or '').split()
        if txt and ws and len(txt) == len(ws):
            for w, d in zip(ws, txt):
                w['w'] = d
        elif txt:
            n = len(txt)
            span = max(0.4, self.t1 - self.t0 - 0.3)
            ws = [{'w': d, 't0': self.t0 + 0.05 + span * i / n, 't1': self.t0 + 0.05 + span * (i + 0.9) / n} for i, d in enumerate(txt)]
        return [w for w in ws if clean(w['w'])]

    def _bigs(self, ws):
        """[(display text, first word index, last word index, strike)] sorted by word index."""
        specs = _as_list(self.spec.get('big', self.spec.get('word')))
        norms = [P.norm(clean(w['w'])) for w in ws]
        out = []
        used = set()
        for b in specs:
            if isinstance(b, dict):
                text, on, strike = str(b.get('text', '')), b.get('on'), bool(b.get('strike'))
            else:
                text, on, strike = str(b), None, False
            text = text.strip()
            if not text:
                continue
            span = self._find(norms, on or text, used)
            if span is None and re.search(r'\d', text):
                # "16$" shown where the number is said ("lix iyo toban doolar")
                for d, i0, i1, cur in P.numbers(' '.join(clean(w['w']) for w in ws)):
                    if i0 not in used and (P.norm(d) == P.norm(text) or P.norm(d).rstrip('$%') == P.norm(text).rstrip('$%')):
                        span = (i0, i1)
                        break
                if span is None:
                    nums = [x for x in P.numbers(' '.join(clean(w['w']) for w in ws)) if x[1] not in used]
                    if nums:
                        span = (nums[0][1], nums[0][2])
            if span is None:
                span = (None, None)
            out.append((text, span[0], span[1], strike))
            if span[0] is not None:
                used.update(range(span[0], span[1] + 1))
        if not out and ws:
            k = P.key_word(' '.join(clean(w['w']) for w in ws))
            span = self._find(norms, k, set())
            out.append((k, span[0] if span else None, span[1] if span else None, False))
        out.sort(key=lambda b: (b[1] if b[1] is not None else 1e9))
        out = out[:2]
        if len(out) == 1:
            text, i0, i1, strike = out[0]
            parts = text.split()
            if len(parts) == 2 and len(text) > 9 and min(len(parts[0]), len(parts[1])) >= 3:
                if i0 is not None and i1 is not None and i1 > i0:
                    out = [(parts[0], i0, i0, strike), (parts[1], i0 + 1, i1, strike)]
                else:
                    out = [(parts[0], i0, i1, strike), (parts[1], i0, i1, strike)]
        return out

    @staticmethod
    def _find(norms, phrase, used):
        toks = [P.norm(clean(x)) for x in str(phrase).split() if P.norm(clean(x))]
        if not toks:
            return None
        for i in range(len(norms)):
            if i in used:
                continue
            ok = True
            for j, tk in enumerate(toks):
                if i + j >= len(norms):
                    ok = False
                    break
                n = norms[i + j]
                if not (n == tk or (len(tk) >= 3 and len(n) >= 3 and (n.startswith(tk) or tk.startswith(n)))):
                    ok = False
                    break
            if ok:
                return (i, i + len(toks) - 1)
        return None

    # ------------------------------------------------------------ layout
    def prep(self):
        ctx, lay, W, H, u = self.ctx, self.lay, self.W, self.H, self.u
        ws = self._words()
        self.ws = ws
        bigs = self._bigs(ws)
        vis = self.spec.get('visual') or {}
        if isinstance(vis, str):
            vis = {'type': vis}
        self.vis_type = str(vis.get('type', '')).lower()
        chip_items = [str(x) for x in (vis.get('items') or [])] if self.vis_type == 'chips' else []
        covered = set()
        for _, i0, i1, _ in bigs:
            if i0 is not None:
                covered.update(range(i0, i1 + 1))
        chip_words = {}
        norms = [P.norm(clean(w['w'])) for w in ws]
        for k, it in enumerate(chip_items):
            sp = self._find(norms, it, covered | set(chip_words))
            if sp:
                for i in range(sp[0], sp[1] + 1):
                    chip_words[i] = k
        self.t_first = ws[0]['t0'] if ws else self.t0 + 0.05
        self.t_out = self.t1 - 0.16
        # small-word groups: before the first big, between the bigs, after the last big
        groups = {'pre': [], 'mid': [], 'post': []}
        bi = [b[1] for b in bigs if b[1] is not None]
        for i, w in enumerate(ws):
            if i in covered or i in chip_words:
                continue
            if not bi or i < bi[0]:
                g = 'pre'
            elif len(bi) > 1 and i < bi[1]:
                g = 'mid'
            else:
                g = 'post'
            groups[g].append(w)
        cap_max = (0.094 if ctx.vertical else 0.15) * (0.86 if len(bigs) > 1 else 1.0)
        has_vis = self.vis_type in ('card', 'cards', 'date')
        max_w = 0.84 if not (has_vis and not ctx.vertical) else 0.56
        longest = max([b[0] for b in bigs], key=len) if bigs else 'A'
        size0 = L.big_size(longest.upper(), W, H, max_w, cap_max)
        # where it can go
        zones = self._zones()
        place = str(self.spec.get('place', 'auto')).lower()
        order = {'behind': ['behind', 'above', 'chest'], 'above': ['above', 'behind', 'chest'], 'top': ['above', 'behind', 'chest'],
                 'chest': ['chest', 'above', 'behind'], 'front': ['chest', 'above', 'behind']}.get(place, ['behind', 'above', 'chest'])
        best = None
        vis_h = self._vis_h() if has_vis else 0.0
        for z in order:
            if z not in zones:
                continue
            y0, y1 = zones[z]
            comp = self._compose(bigs, groups, chip_items, chip_words, size0, ws)
            need = comp['h'] + (vis_h + 0.02 * H if has_vis and z == 'chest' else 0)
            room = y1 - y0
            sc = min(1.0, room / max(1.0, need), 0.92 * W / max(1.0, comp['w']))
            if z == 'behind':
                sc = min(sc, 1.0)
            if sc >= 0.62 or z == order[-1] and best is None:
                best = (z, sc)
                break
            if best is None or sc > best[1]:
                best = (z, sc)
        if best is None:
            best = ('chest', 0.7)
        self.mode, sc = best
        comp = self._compose(bigs, groups, chip_items, chip_words, size0 * sc, ws)
        self.comp = comp
        # position the block
        y0, y1 = zones.get(self.mode, (lay.chest[0], lay.cap_top))
        cx = W / 2 if ctx.vertical else self._cx(comp)
        if self.mode == 'behind':
            # the head covers the lower ~30% of the first big word
            base0 = lay.head + 0.3 * comp['cap0']
            dy = base0 - comp['base0']
        elif self.mode == 'above':
            dy = (lay.head - 0.02 * H) - comp['y1']
        else:
            extra = (vis_h + 0.02 * H) if has_vis else 0.0
            room = y1 - y0
            free = room - comp['h'] - extra
            top = y0 + max(0.0, min(free * 0.35, 0.03 * H))
            dy = top - comp['y0']
        dx = cx - (comp['x0'] + comp['x1']) / 2
        for it in comp['items']:
            it['x'] += dx
            it['y'] += dy
        self.box = (comp['x0'] + dx, comp['y0'] + dy, comp['x1'] + dx, comp['y1'] + dy)
        # small-word colour from what is behind them
        self._recolor()
        # the visual
        self.visual = None
        if has_vis or self.vis_type == 'icon':
            self._prep_visual(vis, zones)

    def _zones(self):
        ctx, lay, H = self.ctx, self.lay, self.H
        z = {}
        bottom = (0.845 if ctx.vertical else 0.9) * H
        top = lay.top + 0.01 * H
        if ctx.has_mask and lay.head - top > 0.075 * H:
            z['behind'] = (top, lay.head + 0.035 * H)
        if lay.head - top > 0.08 * H:
            z['above'] = (top, lay.head - 0.02 * H)
        c0 = lay.chin + 0.022 * H
        if bottom - c0 > 0.06 * H:
            z['chest'] = (c0, bottom)
        return z

    def _cx(self, comp):
        lay, W = self.lay, self.W
        if self.mode in ('behind', 'above', 'chest'):
            return clamp(lay.cx, comp['w'] / 2 + 0.04 * W, W - comp['w'] / 2 - 0.04 * W)
        return W / 2

    def _compose(self, bigs, groups, chip_items, chip_words, size, ws):
        """Builds every sprite at `size` and returns the items in block coordinates (first big baseline = 0)."""
        W, u = self.W, self.u
        items = []
        bw_objs = []
        n_big = max(1, len(bigs))
        for k, (text, i0, i1, strike) in enumerate(bigs):
            s = size * (0.92 if n_big > 1 and k == 1 else 1.0)
            T = text.upper() if not re.search(r'\d', text) else text
            bw = L.BigWord(T, s, grad=GREY_GRAD if strike else L.GRAD)
            t = ws[i0]['t0'] if i0 is not None and i0 < len(ws) else self.t_first + 0.12 * (k + 1)
            bw_objs.append((bw, t, strike, s))
        small = max((4.7 if self.ctx.vertical else 3.6) * u, min(6.4 * u, size * 0.3))
        if not bw_objs:
            return {'items': [], 'w': 1, 'h': 1, 'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1, 'cap0': 1, 'base0': 0}
        BW = max(b[0].w for b in bw_objs)
        y = 0.0
        x_big = []
        for k, (bw, t, strike, s) in enumerate(bw_objs):
            if k == 0:
                bx = 0.0
            else:
                gap_mid = groups['mid']
                y += (small * 1.55 + bw.cap * 1.06) if gap_mid else bw.cap * 1.1
                bx = (BW - bw.w) * 0.72
            items.append({'kind': 'big', 'bw': bw, 'x': bx, 'y': y, 't': t, 'strike': strike, 'cap': bw.cap})
            x_big.append((bx, y, bw))
        base_last = y
        cap0 = bw_objs[0][0].cap

        def line(words, x_anchor, base, align, color):
            if not words:
                return base
            maxw = max(BW * 1.05, 0.5 * W)
            rows, cur, cw = [], [], 0.0
            sp = D.measure(' ', SMALL_KEY, small) * 1.1
            for w in words:
                ww = D.measure(clean(w['w']), SMALL_KEY, small)
                if cur and cw + sp + ww > maxw:
                    rows.append(cur)
                    cur, cw = [], 0.0
                cur.append((w, ww))
                cw += (sp if len(cur) > 1 else 0) + ww
            if cur:
                rows.append(cur)
            return rows, sp

        # before the first big: right above it, left aligned (the last row sits on the big word)
        pre = groups['pre']
        if pre:
            rows, sp = line(pre, 0, 0, 'left', SMALL_DARK)
            by = -cap0 - small * 0.42
            for r in reversed(rows):
                x = x_big[0][0] + BW * 0.02
                for w, ww in r:
                    items.append({'kind': 'small', 'text': clean(w['w']), 'x': x, 'y': by, 't': w['t0'], 'col': 'dark'})
                    x += ww + sp
                by -= small * 1.28
        # between two bigs: one line, left aligned under the first
        mid = groups['mid']
        if mid and len(x_big) > 1:
            rows, sp = line(mid, 0, 0, 'left', SMALL_DARK)
            by = x_big[0][1] + small * 1.4
            for r in rows[:1]:
                x = x_big[0][0] + BW * 0.04
                for w, ww in r:
                    items.append({'kind': 'small', 'text': clean(w['w']), 'x': x, 'y': by, 't': w['t0'], 'col': 'dark'})
                    x += ww + sp
        # after the last big: right aligned under it
        post = groups['post']
        y_end = base_last
        if post:
            rows, sp = line(post, 0, 0, 'right', SMALL_DARK2)
            by = base_last + small * 1.55
            bx, _, bwl = x_big[-1]
            right = bx + bwl.w
            for r in rows:
                tot = sum(ww for _, ww in r) + sp * (len(r) - 1)
                x = right - tot if tot < bwl.w * 1.25 else bx
                for w, ww in r:
                    items.append({'kind': 'small', 'text': clean(w['w']), 'x': x, 'y': by, 't': w['t0'], 'col': 'dark2'})
                    x += ww + sp
                y_end = by
                by += small * 1.28
        # chips under everything
        if chip_items:
            cs = max(3.4 * u, small * 1.02)
            sprs = [L.pill(c, cs) for c in chip_items[:5]]
            gap = 1.1 * u
            rows, cur, cw = [], [], 0.0
            for k, sp_ in enumerate(sprs):
                if cur and cw + gap + sp_.shape[1] > 0.9 * W:
                    rows.append(cur)
                    cur, cw = [], 0.0
                cur.append((k, sp_))
                cw += (gap if len(cur) > 1 else 0) + sp_.shape[1]
            if cur:
                rows.append(cur)
            cy = y_end + cs * 1.9
            times = {}
            for i, k in chip_words.items():
                times.setdefault(k, ws[i]['t0'])
            for r in rows:
                tot = sum(s_.shape[1] for _, s_ in r) + gap * (len(r) - 1)
                x = BW / 2 - tot / 2
                for k, s_ in r:
                    t = times.get(k, self.t_first + 0.25 + 0.3 * k)
                    items.append({'kind': 'chip', 'spr': s_, 'x': x + s_.shape[1] / 2, 'y': cy, 't': t, 'rot': [-4, 3, -2, 4, -3][k % 5]})
                    x += s_.shape[1] + gap
                y_end = cy + s_.shape[0] * 0.5
                cy += s_.shape[0] * 1.25
        # small sprites (colour is set later from the background)
        for it in items:
            if it['kind'] == 'small':
                it['size'] = small
        # bounding box
        x0, y0, x1, y1 = 1e9, 1e9, -1e9, -1e9
        for it in items:
            if it['kind'] == 'big':
                bw = it['bw']
                a, b, c, d = it['x'], it['y'] - bw.cap * 1.08, it['x'] + bw.w, it['y'] + bw.size * 0.08
            elif it['kind'] == 'small':
                w = D.measure(it['text'], SMALL_KEY, small)
                a, b, c, d = it['x'], it['y'] - small * 0.8, it['x'] + w, it['y'] + small * 0.25
            else:
                s_ = it['spr']
                a, b, c, d = it['x'] - s_.shape[1] / 2, it['y'] - s_.shape[0] / 2, it['x'] + s_.shape[1] / 2, it['y'] + s_.shape[0] / 2
            x0, y0, x1, y1 = min(x0, a), min(y0, b), max(x1, c), max(y1, d)
        return {'items': items, 'w': x1 - x0, 'h': y1 - y0, 'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1, 'cap0': cap0, 'base0': 0.0, 'small': small}

    def _recolor(self):
        """Dark red small words on a light wall, white ones on anything darker (checked over the whole phrase)."""
        ctx = self.ctx
        times = [self.t0 + 0.1, (self.t0 + self.t1) / 2, max(self.t0 + 0.1, self.t1 - 0.25)]
        frames = [ctx.still(t) for t in times]
        for it in self.comp['items']:
            if it['kind'] != 'small':
                continue
            s = it['size']
            w = D.measure(it['text'], SMALL_KEY, s)
            box = (it['x'], it['y'] - s * 0.75, it['x'] + w, it['y'])
            bright = min(L.lum(F, box) for F in frames)
            col = (SMALL_DARK if it['col'] == 'dark' else SMALL_DARK2) if bright > 0.62 else SMALL_LIGHT
            it['sm'] = _Small(it['text'], s, col, it['t'])

    # ------------------------------------------------------------ the visual
    def _vis_h(self):
        H = self.H
        t = self.vis_type
        close = self.lay.closeup and self.ctx.vertical
        if t == 'card':
            return (0.125 if close else 0.17) * H
        if t == 'cards':
            return (0.12 if close else 0.15) * H
        if t == 'date':
            return (0.1 if close else 0.12) * H
        return 0.0

    def _prep_visual(self, vis, zones):
        ctx, lay, W, H, u = self.ctx, self.lay, self.W, self.H, self.u
        t = self.vis_type
        x0, y0, x1, y1 = self.box
        on = vis.get('on') or vis.get('label') or ''
        t_on = self.when(on, None) if on else None
        if t_on is None:
            t_on = min(self.t1 - 0.5, self.t_first + 0.35)
        if t == 'icon':
            name = str(vis.get('name', 'tag'))
            d = max(5.5 * u, min(9 * u, self.comp['cap0'] * 0.7))
            spr = badge(name, d)
            big = next((it for it in self.comp['items'] if it['kind'] == 'big'), None)
            if big is None:
                return
            bx, by = big['x'] + big['bw'].w, big['y'] - big['cap'] * 1.0
            self.visual = {'kind': 'icon', 'items': [(spr, bx + d * 0.2, by, 8, t_on + 0.1)]}
            return
        # the free space for it: under the lockup in the chest, else under the lockup and above the head
        if self.mode == 'chest':
            top, bottom = y1 + 0.018 * H, zones.get('chest', (0, 0.845 * H))[1]
        else:
            top, bottom = y1 + 0.015 * H, (lay.head - 0.012 * H) if self.mode == 'above' else lay.chin + 0.25 * H
        room = bottom - top
        side = None
        if t in ('card', 'date') and room < (0.13 if t == 'card' else 0.09) * H:
            side = self._side_box()
            if side is None and room < 0.06 * H:
                self.visual = None
                return
        if side is not None:
            sx0, sy0, sx1, sy1 = side
            if t == 'card':
                ref = vis.get('img')
                at = self.t0 + 0.2
                asp = float(vis.get('aspect') or (0.8 if ref else max((0.75, 1.0), key=lambda a: (bg_crop(ctx, at, 0, aspect=a), bg_score(ctx, at, a))[1])))
                cw = min(sx1 - sx0, 0.34 * W)
                ch = min(sy1 - sy0, cw / asp)
                cw = ch * asp
                img = img_from(ctx, ref, at, None if ref else bg_crop(ctx, at, int(vis.get('pick', 0)), aspect=cw / ch))
                spr = L.card(img, cw, ch, label=vis.get('label'), u=u)
                cx, cy = (sx0 + sx1) / 2, sy0 + ch / 2
                self.visual = {'kind': 'card', 'items': [(spr, cx, cy, 4 if cx > W / 2 else -4, t_on)], 'halftone': (cx, cy, t_on)}
            else:
                cw = min(sx1 - sx0, 0.24 * W)
                ch = cw / 0.92
                spr = self._calendar(cw, ch, str(vis.get('top', 'BISHA')), str(vis.get('day', '1')))
                cx = (sx0 + sx1) / 2
                self.visual = {'kind': 'date', 'items': [(spr, cx, sy0 + ch / 2, 5 if cx > W / 2 else -5, t_on)]}
            return
        cy = top + min(room, self._vis_h()) / 2
        if t == 'card':
            ch = min(room, (0.15 if lay.closeup and ctx.vertical else 0.19) * H)
            ref = vis.get('img')
            asp = vis.get('aspect')
            if not ref and not asp:
                # the shape the products have in this footage (a tall shelf or a wide one)
                at = self.t0 + 0.2
                asp = max((0.75, 1.0, 1.35), key=lambda a: (bg_crop(ctx, at, 0, aspect=a), bg_score(ctx, at, a))[1])
            asp = float(asp or 1.3)
            cw = min(0.62 * W, ch * asp)
            img = img_from(ctx, ref, self.t0 + 0.2, None if ref else bg_crop(ctx, self.t0 + 0.2, int(vis.get('pick', 0)), aspect=cw / ch))
            spr = L.card(img, cw, ch, label=vis.get('label'), u=u)
            self.visual = {'kind': 'card', 'items': [(spr, W / 2 if ctx.vertical else (x0 + x1) / 2, top + ch / 2, -3, t_on)], 'halftone': (W / 2, top + ch / 2, t_on)}
        elif t == 'cards':
            refs = _as_list(vis.get('imgs')) or [None, None, None]
            labels = _as_list(vis.get('labels'))
            n = max(2, min(3, len(refs)))
            ch = min(room, (0.13 if lay.closeup and ctx.vertical else 0.15) * H)
            cw = ch * 0.78
            xs = {2: [0.36, 0.64], 3: [0.24, 0.5, 0.76]}[n]
            tilts = [-7, 4, -3]
            out = []
            for k in range(n):
                ref = refs[k] if k < len(refs) else None
                img = img_from(ctx, ref, self.t0 + 0.2, None if ref else bg_crop(ctx, self.t0 + 0.2, k, aspect=cw / ch))
                lab = labels[k] if k < len(labels) else None
                spr = L.card(img, cw, ch, label=lab, u=u)
                tk = self.when(lab.split()[0], None) if lab else None
                out.append((spr, W * xs[k], top + ch / 2 + (k % 2) * 0.01 * H, tilts[k % 3], tk if tk is not None else t_on + 0.16 * k))
            self.visual = {'kind': 'cards', 'items': out}
        elif t == 'date':
            ch = min(room, 0.12 * H)
            cw = ch * 0.92
            spr = self._calendar(cw, ch, str(vis.get('top', 'BISHA')), str(vis.get('day', '1')))
            self.visual = {'kind': 'date', 'items': [(spr, W / 2, top + ch / 2, -4, t_on)]}

    def _side_box(self):
        """A free column beside the head (left or right of the face, above the chin): for a card in a close-up."""
        lay, W, H = self.lay, self.W, self.H
        fx0, fy0, fx1, fy1 = lay.face_pad
        top = lay.top + 0.03 * H
        bottom = min(lay.chin, top + 0.3 * H)
        best = None
        for a, b in ((0.03 * W, fx0 - 0.01 * W), (fx1 + 0.01 * W, 0.97 * W)):
            if b - a >= 0.19 * W and bottom - top >= 0.14 * H:
                if best is None or b - a > best[2] - best[0]:
                    best = (a, top, b, bottom)
        return best

    def _calendar(self, w, h, top, day):
        u = self.u
        w, h = int(w), int(h)
        r = 0.12 * w
        base = D.rrect(w, h, r, fill='#FFFFFF')
        hb = int(h * 0.3)
        band = D.rrect(w, hb + int(r), r, fill=L.RED)
        band = band[:hb]
        D.blend_spr(base, band, 0, 0)
        lab = D.label(top.upper(), 'xb', hb * 0.42, color='#FFFFFF', padx=0.2, pady=0.1)
        D.blend_spr(base, lab, (w - lab.shape[1]) / 2, (hb - lab.shape[0]) / 2)
        T = D.Text(day, 'black_i', h * 0.5, pad_px=0)
        num = T.sprite(INK)
        num, ox, oy = D.tight(num)
        D.blend_spr(base, num, (w - num.shape[1]) / 2, hb + (h - hb - num.shape[0]) / 2)
        for k in (0.3, 0.7):
            ring = D.rrect(int(0.06 * w), int(0.16 * h), 0.03 * w, fill='#3A3A40')
            D.blend_spr(base, ring, w * k - 0.03 * w, -0.06 * h)
        spr, p = D.shadow(base, blur=1.6 * u, dy=0.9 * u, color=(0.3, 0, 0.04), alpha=0.32)
        return D.tight(spr)[0]

    # ------------------------------------------------------------ drawing
    def bg_on(self, t):
        return getattr(self, 'mode', '') == 'behind'

    def bg(self, ctx, B, t):
        self.ensure()
        if self.mode == 'behind':
            self._draw_text(B, t)

    def front(self, ctx, F, t):
        self.ensure()
        if self.mode != 'behind':
            self._draw_text(F, t)
        self._draw_visual(F, t)
        for it in self.comp['items']:
            if it['kind'] == 'big':
                bw = it['bw']
                L.pixels(F, t, it['t'] + 0.05, (it['x'], it['y'] - bw.cap, it['x'] + bw.w, it['y']), self.u, self.seed + int(it['x']), n=4, dur=0.6)

    def _draw_text(self, F, t):
        out = prog(t, self.t_out, 0.16)
        drift = -0.006 * self.H * prog(t, self.t_first, max(0.5, self.t1 - self.t_first))
        for it in self.comp['items']:
            k = it['kind']
            if k == 'big':
                bw = it['bw']
                bw.draw(F, t, it['x'], it['y'] + drift, it['t'] - 0.02, self.t_out, anchor_x=0.0, stagger=0.04, dur=0.22)
                if it['strike']:
                    a = ease_out_cubic(prog(t, it['t'] + 0.28, 0.24)) * (1 - out)
                    if a > 0:
                        lw = max(3, int(bw.size * 0.07))
                        ln = int(bw.w * 1.08 * a)
                        if ln > 4:
                            bar = D.rrect(ln, lw, lw / 2, fill=L.RED)
                            D.place(F, bar, it['x'] - bw.w * 0.04, it['y'] - bw.cap * 0.42 + drift, rot=-7, anchor=(0.0, 0.5))
            elif k == 'small':
                sm = it.get('sm')
                if sm is None:
                    continue
                a = ease_out_cubic(prog(t, sm.t - 0.04, 0.18))
                if a <= 0:
                    continue
                a *= 1 - out
                if a <= 0:
                    continue
                dy = sm.size * 0.45 * (1 - a) - sm.size * 0.4 * out
                cx = it['x'] - sm.ox + sm.spr.shape[1] / 2
                cy = it['y'] - sm.oy + sm.spr.shape[0] / 2 + dy + drift
                D.place(F, sm.spr, cx, cy, op=a, blur=sm.size * 0.12 * (1 - a))
            elif k == 'chip':
                s = ease_out_back(prog(t, it['t'] - 0.03, 0.32), 1.7)
                a = clamp(prog(t, it['t'] - 0.03, 0.12)) * (1 - out)
                if a > 0:
                    D.place(F, it['spr'], it['x'], it['y'] + drift, scale=0.6 + 0.4 * s, rot=it['rot'] * s, op=a)

    def _draw_visual(self, F, t):
        V = self.visual
        if not V:
            return
        out = prog(t, self.t_out - 0.04, 0.2)
        if V.get('halftone'):
            hx, hy, ht = V['halftone']
            if ht - 0.05 <= t <= ht + 0.55:
                L.halftone(F, t, ht - 0.05, self.u, cx=hx, cy=hy, dur=0.55)
        for spr, x, y, rot, ti in V['items']:
            s = ease_out_back(prog(t, ti, 0.42), 1.3)
            a = clamp(prog(t, ti, 0.16)) * (1 - out)
            if a <= 0:
                continue
            D.place(F, spr, x, y + (1 - s) * 0.02 * self.H, scale=(0.82 + 0.18 * s) * (1 - 0.08 * out), rot=rot * (0.5 + 0.5 * s), op=a)

    def hide_captions(self):
        return [(self.t0 - 0.05, self.t1 + 0.02)]

    def sounds(self):
        self.ensure()
        for it in self.comp['items']:
            if it['kind'] == 'big':
                self.ctx.sfx(it['t'], 'pop', 0.85)
            elif it['kind'] == 'chip':
                self.ctx.sfx(it['t'], 'tap', 0.6)
        if self.visual:
            ti = self.visual['items'][0][4]
            self.ctx.sfx(ti, 'swell' if self.visual['kind'] == 'card' else 'tap', 0.7)
