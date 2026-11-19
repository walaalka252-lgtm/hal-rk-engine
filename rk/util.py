"""Small helpers: easing, colours, timing, deterministic random."""
import math
import random

def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x

def lerp(a, b, t):
    return a + (b - a) * t

def prog(t, t0, dur):
    """0..1 progress of t inside [t0, t0 + dur]."""
    if dur <= 0:
        return 1.0 if t >= t0 else 0.0
    return clamp((t - t0) / dur)

def ease_out_cubic(x):
    return 1 - (1 - x) ** 3

def ease_in_cubic(x):
    return x ** 3

def ease_in_out_cubic(x):
    return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2

def ease_in_out_sine(x):
    return -(math.cos(math.pi * x) - 1) / 2

def ease_out_expo(x):
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)

def ease_in_expo(x):
    return 0.0 if x <= 0 else 2 ** (10 * x - 10)

def ease_out_back(x, s=1.4):
    x -= 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2

def ease_out_quart(x):
    return 1 - (1 - x) ** 4

def window(t, t0, t1, fin=0.3, fout=0.25):
    """Opacity envelope: fades in over fin after t0, out over fout before t1."""
    if t < t0 or t > t1:
        return 0.0
    a = 1.0
    if fin > 0:
        a = min(a, ease_out_cubic(prog(t, t0, fin)))
    if fout > 0:
        a = min(a, 1 - ease_in_cubic(prog(t, t1 - fout, fout)))
    return clamp(a)

def hexrgb(h, a=None):
    h = h.lstrip('#')
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    return (r, g, b) if a is None else (r, g, b, a)

def rnd(seed):
    return random.Random(hash(str(seed)) & 0xffffffff if not isinstance(seed, int) else seed)

def stable_seed(*parts):
    s = '|'.join(str(p) for p in parts)
    h = 2166136261
    for ch in s.encode('utf-8'):
        h = ((h ^ ch) * 16777619) & 0xffffffff
    return h

def fmt_tc(sec):
    sec = max(0.0, float(sec))
    m, s = divmod(sec, 60)
    return '%02d:%05.2f' % (m, s)
