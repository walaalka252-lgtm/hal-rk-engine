"""Command line: python3 run.py <command> …   (or: cd engine && python3 -m rk <command> …)

  doctor                                   check ffmpeg, OpenCV (TFLite person mask + face), fonts
  auto VIDEO SRT [-o plan.json] [--format 9:16|16:9|1:1] [--brand NAME]
                                           analyse the footage and write a complete draft plan
  fx                                       list every beat type (effect) and its fields
  check plan.json                          list problems in a plan
  stills plan.json [--at 1.5,3.2] [-o stills]   render still frames (default: the middle of every beat) + sheet.jpg
  render plan.json -o out.mp4 [--budget 300]    render (resumable: run again after RESUME)
"""
import argparse
import json
import os
import sys
import time


def log(*a):
    print(*a, flush=True)


FIELDS = {
    'say': ('the normal beat, one per phrase: the words build as they are said. big: "DISCOUNT" | ["TAYADA", "QIIMAHA"] | '
            '[{"text": "16$", "on": "lix iyo toban doolar"}] | [{"text": "18$", "strike": true}] (old price); '
            'visual: {"type": "card", "img": "frame:12.3@x,y,w,h" | "photo.jpg", "label": "Buskud"} | '
            '{"type": "chips", "items": ["Timir", "Saliid"]} | {"type": "date", "top": "BISHA", "day": "1"} | '
            '{"type": "icon", "name": "cart|heart|star|tag|clock|bolt|check|play|percent"}; place: auto|behind|above|chest'),
    'word': 'word, line (thin line, ≤4 words), red (the red word of the line), place: auto|behind|above|chest, pill',
    'cards': 'word, cards: [{img: "frame:12.3" | "frame:12.3@x,y,w,h" | "photo.jpg", label, at}] (1–3), place',
    'pills': 'word, items: ["of", "my"] (red pills on their spoken words)',
    'punch': 'word (shouted word: halftone burst + RGB split + impact)',
    'broll': 'word, line, img (picture of what is said), icon: percent|check|heart|star|play|clock|bolt',
    'split': 'word, cards (top panel with the big word, footage below)',
    'end': 'word (channel / shop name), line, button, done',
}


def main(argv=None):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser(prog='rk', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd')
    ap.add_argument('args', nargs='*')
    ap.add_argument('-o', '--out')
    ap.add_argument('--format', default=None)
    ap.add_argument('--brand', default=None)
    ap.add_argument('--at', default=None)
    ap.add_argument('--budget', type=float, default=None)
    ap.add_argument('--crf', type=int, default=18)
    ap.add_argument('--preset', default='veryfast')
    ap.add_argument('--chunk', type=float, default=6.0)
    a = ap.parse_args(argv)
    from rk import media, plan as P
    if a.cmd == 'doctor':
        return doctor()
    if a.cmd == 'fx':
        from rk import fx as FX
        from rk import fx_mask, fx_ui, fx_say  # noqa: F401 (register)
        for name, cls in sorted(FX.REG.items(), key=lambda kv: (kv[0] != 'say', kv[0])):
            doc = (cls.__doc__ or '').strip().split('\n')[0]
            need = {'mask': ' [needs the person mask]', 'footage': ' [needs footage]'}.get(cls.need, '')
            log('%-15s%s %s' % (name, need, doc))
            if name in FIELDS or getattr(cls, 'fields', None):
                log('%15s fields: %s' % ('', FIELDS.get(name) or cls.fields))
        log('\ncommon fields on every beat: at, end (seconds), transition: streak|halftone|blurzoom|flash|rgb|none')
        return 0
    if a.cmd == 'auto':
        if len(a.args) < 2:
            ap.error('auto VIDEO SRT')
        video, srt = a.args[0], a.args[1]
        from rk.track import Track
        info = media.probe(video)
        fmt = a.format or ('9:16' if info['h'] >= info['w'] else '16:9')
        out = a.out or 'plan.json'
        work = os.path.splitext(os.path.abspath(out))[0].replace('plan', 'edit') + '.rkwork'
        cues = P.parse_srt(open(srt, encoding='utf-8', errors='replace').read())
        log('footage %dx%d %.1f fps %.1f s · %d subtitle lines' % (info['w'], info['h'], info['fps'], info['dur'], len(cues)))
        tr = Track.analyze(video, info, log=log)
        for n in tr.d.get('notes', []):
            log('  note: ' + n)
        log('person mask: %s · face found in %d of %d samples' % ('yes' if tr.has_mask else 'NO', tr.d['faces'], tr.d['samples']))
        # a close-up (the face fills the frame): no room for photo cards around the head, the words carry it
        closeup = False
        try:
            fs = [tr.at(info['dur'] * k / 8)['face'] for k in range(1, 8)]
            hs = sorted(f[3] - f[1] for f in fs)
            closeup = hs[len(hs) // 2] > 0.17          # face boxes are 0–1 of the frame
        except Exception:
            closeup = False
        pl = P.auto(P.parse_srt(open(srt, encoding='utf-8', errors='replace').read()), info['dur'], has_mask=tr.has_mask, fmt=fmt, brand=a.brand, closeup=closeup)
        pl = dict({'video': video, 'srt': srt}, **pl)
        P.save(pl, out)
        log('wrote %s with %d beats (one "say" beat per phrase%s). Read every phrase ("say"): fix its big word(s), the visual that shows '
            'what is said, and the effects that fit; then: check → stills → render' % (out, len(pl['beats']), ', close-up' if closeup else ''))
        return 0
    if a.cmd == 'check':
        pl = P.load(a.args[0])
        info = media.probe(pl['video'])
        probs = P.check(pl, info['dur'], True)
        log('\n'.join(probs) if probs else 'plan looks fine')
        return 0
    if a.cmd in ('stills', 'render'):
        from rk.render import Job
        from rk import fx_mask, fx_ui, fx_say  # noqa: F401
        out = a.out or ('stills' if a.cmd == 'stills' else 'edit.mp4')
        job = Job(a.args[0], out if a.cmd == 'render' else os.path.join(out, 'x.mp4'), log=log)
        if a.cmd == 'stills':
            if a.at:
                times = [float(x) for x in a.at.split(',') if x.strip()]
            else:
                times = []
                for b in job.beats:
                    m = (b.t0 + b.t1) / 2
                    times.append(round(min(b.t1 - 0.1, max(b.t0 + 0.55, m)), 2))
            sheet = os.path.join(out, 'sheet.jpg')
            t0 = time.time()
            job.stills(times, out, sheet=sheet)
            log('stills: %d frames in %.0f s → %s' % (len(times), time.time() - t0, sheet))
            return 0
        t0 = time.time()
        ok = job.render(budget=a.budget, chunk=a.chunk, crf=a.crf, preset=a.preset)
        log('time %.0f s' % (time.time() - t0))
        return 0 if ok else 3
    ap.error('unknown command ' + a.cmd)


def doctor():
    from rk import media
    ok = True
    log('python', sys.version.split()[0])
    log('ffmpeg:', 'ok' if media.have_ffmpeg() else 'MISSING (install ffmpeg)')
    ok &= media.have_ffmpeg()
    try:
        import cv2
        log('opencv', cv2.__version__)
        from rk.vision import Segmenter, FaceDetector
        import numpy as np
        img = np.zeros((512, 288, 3), np.uint8) + 128
        Segmenter().raw(img)
        FaceDetector().detect(img)
        log('person mask + face models: ok (OpenCV DNN TFLite)')
    except Exception as e:
        log('person mask / face: NOT available (%s). Fix: pip install -U opencv-python-headless' % e)
        log('  without it the engine still renders, but effects that need the mask use a fallback')
    try:
        from PIL import Image, features
        log('pillow ok, raqm:', features.check('raqm'))
    except Exception as e:
        log('pillow MISSING', e)
        ok = False
    from rk.draw import FONT_DIR, FONT_FILES
    miss = [f for f in FONT_FILES.values() if not os.path.exists(os.path.join(FONT_DIR, f))]
    log('fonts:', 'ok' if not miss else 'missing ' + ', '.join(miss))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
