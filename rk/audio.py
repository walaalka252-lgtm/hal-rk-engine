"""Soft synthesized sound effects (few, warm, low-passed) mixed under the original voice, then −14 LUFS."""
import json
import math
import os
import re

import numpy as np

from . import media

SR = 48000
PRIORITY = {'impact': 5, 'whoosh': 4, 'riser': 4, 'swell': 4, 'glass': 3, 'paper': 3, 'pop': 3, 'tap': 2, 'click': 2, 'tick': 1}
PEAK_DB = {'pop': -21, 'tap': -25, 'click': -25, 'tick': -28, 'whoosh': -25, 'impact': -19, 'riser': -27, 'swell': -25, 'glass': -25, 'paper': -28}


def lowpass(x, fc=6000.0, order=4):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    X *= 1.0 / np.sqrt(1.0 + (f / fc) ** (2 * order))
    return np.fft.irfft(X, n=len(x))


def bandpass(x, lo, hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    X *= (1.0 / np.sqrt(1.0 + (f / hi) ** 8)) * (1.0 / np.sqrt(1.0 + (lo / np.maximum(f, 1)) ** 8))
    return np.fft.irfft(X, n=len(x))


def env(n, a=0.003, d=0.1, shape=4.0):
    t = np.arange(n) / SR
    e = np.minimum(1.0, t / max(1e-4, a)) * np.exp(-np.maximum(0, t - a) * shape / max(1e-3, d))
    fade = np.minimum(1.0, (n - np.arange(n)) / (0.004 * SR))
    return e * fade


def synth(kind, R):
    if kind == 'pop':
        n = int(0.12 * SR)
        t = np.arange(n) / SR
        f = 260 + 300 * np.exp(-t * 38)
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.003, 0.06, 3.2)
        x += lowpass(R.standard_normal(n), 2500) * env(n, 0.001, 0.01, 4) * 0.15
    elif kind == 'tap':
        n = int(0.08 * SR)
        t = np.arange(n) / SR
        f = 520 + 420 * np.exp(-t * 60)
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.002, 0.035, 3.5)
    elif kind == 'click':
        n = int(0.04 * SR)
        x = lowpass(R.standard_normal(n), 3000) * env(n, 0.001, 0.008, 3) + np.sin(2 * np.pi * 1100 * np.arange(n) / SR) * env(n, 0.001, 0.012, 3) * 0.5
    elif kind == 'tick':
        n = int(0.025 * SR)
        x = np.sin(2 * np.pi * 1800 * np.arange(n) / SR) * env(n, 0.001, 0.008, 3)
    elif kind == 'whoosh':
        n = int(0.38 * SR)
        t = np.arange(n) / SR
        noise = R.standard_normal(n)
        # sweep the band by mixing two filtered copies
        lo, hi = bandpass(noise, 300, 900), bandpass(noise, 900, 2600)
        k = t / t[-1]
        x = lo * (1 - k) + hi * k
        e = np.sin(np.pi * np.clip(k, 0, 1)) ** 1.6
        x *= e
    elif kind == 'impact':
        n = int(0.55 * SR)
        t = np.arange(n) / SR
        f = 48 + 50 * np.exp(-t * 12)
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.004, 0.32, 3.0)
        x += lowpass(R.standard_normal(n), 1600) * env(n, 0.002, 0.08, 4) * 0.35
    elif kind == 'riser':
        n = int(0.75 * SR)
        t = np.arange(n) / SR
        k = t / t[-1]
        noise = R.standard_normal(n)
        x = bandpass(noise, 400, 1200) * (1 - k) + bandpass(noise, 1200, 3500) * k
        x *= k ** 2.2 * np.minimum(1, (n - np.arange(n)) / (0.03 * SR))
    elif kind == 'swell':
        n = int(0.9 * SR)
        t = np.arange(n) / SR
        k = t / t[-1]
        x = (np.sin(2 * np.pi * 82 * t) + 0.5 * np.sin(2 * np.pi * 123 * t)) * np.sin(np.pi * k) ** 2
        x += bandpass(R.standard_normal(n), 200, 900) * np.sin(np.pi * k) ** 3 * 0.3
    elif kind == 'glass':
        n = int(0.3 * SR)
        x = np.zeros(n)
        for _ in range(7):
            p = int(R.uniform(0, 0.12) * SR)
            m = int(0.06 * SR)
            b = bandpass(R.standard_normal(m), 2200, 5800) * env(m, 0.0005, 0.02, 4)
            x[p:p + m] += b[:n - p] * R.uniform(0.4, 1)
    elif kind == 'paper':
        n = int(0.32 * SR)
        noise = bandpass(R.standard_normal(n), 900, 4200)
        am = np.repeat(R.uniform(0.2, 1, n // 480 + 1), 480)[:n]
        x = noise * am * np.sin(np.pi * np.arange(n) / n)
    else:
        return np.zeros(1)
    x = lowpass(x, 6000)
    x /= max(1e-6, np.abs(x).max())
    return x * 10 ** (PEAK_DB.get(kind, -24) / 20.0)


def budget(sounds):
    """About one sound per beat: never two within 0.45 s (the stronger one wins) and never more than 2 in any second."""
    s = sorted(sounds, key=lambda x: x[0])
    kept = []
    for at, kind, g in s:
        if kept and at - kept[-1][0] < 0.45:
            if PRIORITY.get(kind, 1) > PRIORITY.get(kept[-1][1], 1):
                kept[-1] = (at, kind, g)
            continue
        if len([k for k in kept if at - k[0] < 1.0]) >= 2:
            continue
        kept.append((at, kind, g))
    return kept


def loudnorm(src, dst):
    r = media.run([media.FFMPEG, '-v', 'info', '-y', '-i', src, '-af', 'loudnorm=I=-14:TP=-1:LRA=11:print_format=json', '-f', 'null', '-'])
    m = re.search(r'\{[^{}]*"input_i"[^{}]*\}', r.stderr, re.S)
    if not m:
        media.run([media.FFMPEG, '-v', 'error', '-y', '-i', src, '-af', 'loudnorm=I=-14:TP=-1:LRA=11', '-ar', str(SR), dst])
        return
    j = json.loads(m.group(0))
    af = ('loudnorm=I=-14:TP=-1:LRA=11:measured_I=%s:measured_TP=%s:measured_LRA=%s:measured_thresh=%s:offset=%s:linear=true'
          % (j['input_i'], j['input_tp'], j['input_lra'], j['input_thresh'], j['target_offset']))
    media.run([media.FFMPEG, '-v', 'error', '-y', '-i', src, '-af', af, '-ar', str(SR), dst])


def build(src, sounds, dur, out, work, log=print, sfx_on=True, voice=True):
    voice_p = os.path.join(work, 'voice.wav')
    n = int(dur * SR) + 1
    if voice and media.extract_audio(src, voice_p, SR):
        v, sr = media.read_wav(voice_p)
        if v.shape[0] < n:
            v = np.vstack([v, np.zeros((n - v.shape[0], v.shape[1]), np.float32)])
        v = v[:n]
    else:
        v = np.zeros((n, 2), np.float32)
    fx = np.zeros(n, np.float64)
    kept = budget(sounds) if sfx_on else []
    R = np.random.default_rng(7)
    cache = {}
    for at, kind, g in kept:
        if kind not in cache:
            cache[kind] = synth(kind, R)
        x = cache[kind] * g
        i = int(max(0, at) * SR)
        if i >= n:
            continue
        m = min(len(x), n - i)
        fx[i:i + m] += x[:m]
    mix = v.astype(np.float64) + fx[:, None]
    tmp = os.path.join(work, 'premix.wav')
    media.write_wav(tmp, mix.astype(np.float32), SR)
    if voice:
        loudnorm(tmp, out)
    else:
        media.write_wav(out, (mix * 2.0).astype(np.float32), SR)
    with open(os.path.join(work, 'sfx.txt'), 'w') as f:
        for at, kind, g in kept:
            f.write('%.2f %s\n' % (at, kind))
    log('  sound effects: %d (list in %s)' % (len(kept), os.path.join(work, 'sfx.txt')))
