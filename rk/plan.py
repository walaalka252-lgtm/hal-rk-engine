"""SRT parsing, estimated word timings, the draft plan (auto) and plan checks."""
import json
import re
import unicodedata

TIME = re.compile(r'(?:(\d{1,3}):)?(\d{1,2}):(\d{1,2})[,.](\d{1,3})')

STOP = set('''
aan aad ah ahaa ahaan ahayd ama ay ayaa ayaan ayay ayey ayuu baa bay buu dhan ee haa hadda haddii hadii hal hore in inaan inay inuu iney
iyo ka kala kale kan kaas kama ku kuma kuu kuwa la laakiin laga lagu lama loo ma marka markaas markaad maxaa maxay mid midka oo
sababtoo si sida sidaas soo sii ta taas tan u ugu uga uu waa waad waan wax waxa waxaa waxaad waxaan waxay waxba weeye yahay yihiin tahay
iska isagu iyada iyaga anigu adiga idinku idin ii ina naga nala noo iyadoo kaliya mar aya hadana hadaba kuwaas kuwan halkan halkaas
kaga kugu idinka idinkaga noogu loogu laguu kuu ku kuma uma lagu kaa kala jira jirta jiro tahay ahaa yahay doono doonaa doontaa
wuxuu waxuu ayna ayan baan baad haddaba sidoo waxaynu waxaanu waanu ayaynu ayaanu ugu ula ula ugala ugaga lagaga ah
the a an and or but of to in on at for with is are was were be been it this that these those you your we our they their i my me he she
his her its as by from so if then than not no yes do does did have has had will would can could just very really also all any
'''.split())

FOLLOW = re.compile(r'\b(follow|subscribe|comment|faallo|la soco|lasoco|nala soco|like|share|wadaag|qor)\b', re.I)
QUESTION = re.compile(r'\?\s*$')


def to_sec(m):
    ms = (m.group(4) + '00')[:3]
    return int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(ms) / 1000.0


def parse_srt(text):
    text = text.replace('﻿', '').replace('\r\n', '\n').replace('\r', '\n').strip()
    cues = []
    for block in re.split(r'\n[ \t]*\n+', text):
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        ti = next((i for i, l in enumerate(lines) if '-->' in l), -1)
        if ti < 0:
            continue
        a, b = lines[ti].split('-->')[:2]
        ma, mb = TIME.search(a), TIME.search(b)
        if not ma or not mb:
            continue
        t0, t1 = to_sec(ma), to_sec(mb)
        body = re.sub(r'<[^>]+>|\{\\[^}]*\}', '', ' '.join(lines[ti + 1:]))
        body = re.sub(r'\s+', ' ', body).strip()
        if body and t1 > t0:
            cues.append({'t0': t0, 't1': t1, 'text': body})
    cues.sort(key=lambda c: c['t0'])
    for c in cues:
        c['words'] = word_times(c)
    return cues


def word_times(cue):
    """Estimate when each word is spoken inside a cue: proportional to its length, pauses after punctuation."""
    toks = cue['text'].split()
    if not toks:
        return []
    weights = []
    for w in toks:
        core = re.sub(r'[^\w]', '', w, flags=re.U)
        wt = 0.6 + len(core) * 0.38
        if re.search(r'[.,!?;:…]$', w):
            wt += 1.2
        weights.append(wt)
    dur = max(0.2, cue['t1'] - cue['t0'])
    total = sum(weights)
    out, t = [], cue['t0']
    for w, wt in zip(toks, weights):
        d = dur * wt / total
        out.append({'w': w, 't0': round(t, 3), 't1': round(t + d * 0.92, 3)})
        t += d
    return out


def norm(s):
    s = unicodedata.normalize('NFKD', s.lower())
    return re.sub(r'[^\w$%]', '', ''.join(c for c in s if not unicodedata.combining(c)), flags=re.U)


def all_words(cues):
    return [w for c in cues for w in c['words']]


def find_word(cues, phrase, t0=None, t1=None):
    """Time (start) at which `phrase` is spoken, searching around [t0, t1]; None when not found."""
    toks = [norm(x) for x in str(phrase).split() if norm(x)]
    if not toks:
        return None
    W = all_words(cues)
    best = None
    for i, w in enumerate(W):
        if t0 is not None and w['t0'] < t0 - 0.35:
            continue
        if t1 is not None and w['t0'] > t1 - 0.15:
            break
        n = norm(w['w'])
        if n and (n == toks[0] or (len(toks[0]) >= 3 and (n.startswith(toks[0]) or toks[0].startswith(n) and len(n) >= 3))):
            best = w['t0']
            break
    return best


def key_word(text, prev=None):
    toks = [re.sub(r'^[^\w$]+|[^\w%$]+$', '', w, flags=re.U) for w in text.split()]
    best, score = None, -99
    for i, w in enumerate(toks):
        if not w:
            continue
        n = norm(w)
        s = 0.0
        if n in STOP:
            s -= 10
        if re.search(r'\d', w):
            s += 5
        if len(w) > 1 and w.isupper():
            s += 3
        if i > 0 and w[:1].isupper():
            s += 2
        s += min(len(n), 11) * 0.32
        s -= 2.2 * (i / max(1, len(toks) - 1)) if len(toks) > 3 else 0
        if prev and n == norm(prev):
            s -= 3
        if len(n) <= 2:
            s -= 4
        if s > score:
            best, score = w, s
    return (best or (toks[0] if toks else '')).upper()


def rest_line(text, key, maxw=4):
    toks = text.split()
    k = norm(key)
    idx = next((i for i, w in enumerate(toks) if norm(w) == k), None)
    if idx is None:
        return ' '.join(toks[:maxw]), None
    left = toks[max(0, idx - 2):idx]
    right = toks[idx + 1:idx + 1 + max(1, maxw - len(left))]
    line = left + right
    red = None
    for w in line:
        if norm(w) not in STOP and len(norm(w)) >= 4:
            red = w.strip('.,!?')
    return ' '.join(line), red


# effects a talking-head plan can rotate through (strongest first)
POOL_MASK = ['cover-story', 'freeze-sticker', 'doodle-boil', 'blackout', 'frame-break', 'type-wall', 'paper-cut', 'couch-wall',
             'spotlight', 'depth-scatter', 'scene-text', 'green-screen', 'ui-world', 'vertigo', 'orbit', 'clone']
POOL_FOOT = ['digital-zoom', 'rack-focus', 'stack', 'pull-back', 'glass-type', 'headline-wall', 'spatial-glass', 'shatter',
             'sticky-text', 'freeze-label', 'serif-flow', 'zoom-through', 'phone', 'blur-in']
BG_CHANGE = {'couch-wall', 'ui-world', 'green-screen', 'paper-cut', 'freeze-sticker', 'type-wall', 'blackout', 'spotlight', 'cover-story'}
# how long each effect beat lasts at most (the rest of the line goes back to the normal look)
MAXDUR = {'blackout': 1.4, 'spotlight': 2.2, 'freeze-sticker': 1.8, 'clone': 1.4, 'shatter': 1.1, 'vertigo': 1.6, 'paper-cut': 2.5,
          'doodle-boil': 2.2, 'frame-break': 2.0, 'cover-story': 2.6, 'type-wall': 2.4, 'couch-wall': 2.6, 'ui-world': 2.8,
          'green-screen': 2.4, 'orbit': 3.2, 'scene-text': 2.6, 'glass-type': 2.4, 'rack-focus': 2.2, 'spatial-glass': 2.8,
          'headline-wall': 2.6, 'pull-back': 3.0, 'stack': 3.0, 'sticky-text': 2.2, 'freeze-label': 2.4, 'serif-flow': 2.6,
          'phone': 3.0, 'comment-cta': 3.0, 'broll': 2.6, 'digital-zoom': 2.6, 'zoom-through': 2.4, 'blur-in': 2.4}
SAYS_LINE = {'prompt-box', 'ui-world', 'serif-flow', 'spatial-glass', 'cover-story', 'phone', 'freeze-sticker', 'stack'}
LISTS = {'chip-row', 'checklist', 'orbit', 'headline-wall', 'counter-list', 'depth-scatter', 'stack', 'phone', 'pull-back', 'freeze-label', 'serif-flow', 'ui-world'}


def content_words(text, n=5):
    out = []
    for w in text.split():
        c = w.strip('.,!?;:')
        if norm(c) not in STOP and len(norm(c)) >= 3 and c not in out:
            out.append(c)
    return out[:n]


def auto(cues, dur, has_mask=True, fmt='9:16', brand=None, captions=True):
    """A complete draft plan: one beat per line, an effect every ~5 s (capped to its own length), a B-roll beat every
    ~10 s and an end card. Claude then fixes the key words and picks effects that fit what is said."""
    if not cues:
        return {'beats': []}
    lines = []
    for c in cues:
        if lines and (lines[-1]['t1'] - lines[-1]['t0'] < 1.3) and c['t0'] - lines[-1]['t1'] < 0.5:
            lines[-1] = {'t0': lines[-1]['t0'], 't1': c['t1'], 'text': lines[-1]['text'] + ' ' + c['text']}
        else:
            lines.append({'t0': c['t0'], 't1': c['t1'], 'text': c['text']})
    beats = []
    prev_key = None
    last_fx_t, last_bg_t, last_broll = -99.0, -99.0, -99.0
    used = {}
    if has_mask:
        pool = [x for pair in zip(POOL_MASK, POOL_FOOT) for x in pair] + POOL_MASK[len(POOL_FOOT):]
    else:
        pool = POOL_FOOT[:]
    pi = 0
    trans = ['streak', 'halftone', 'blurzoom', 'flash']
    ti = 0
    place_cycle = ['behind', 'chest'] if has_mask else ['above', 'chest']
    pc = 0
    prev_fx = None
    last_trans = -99.0
    for i, L in enumerate(lines):
        t0 = L['t0']
        t1 = lines[i + 1]['t0'] if i + 1 < len(lines) else min(dur, L['t1'] + 0.6)
        text = L['text']
        words = text.split()
        key = key_word(text, prev_key)
        prev_key = key
        is_last = i == len(lines) - 1
        fx = None
        if FOLLOW.search(text) and (is_last or i >= len(lines) - 2) and not used.get('comment-cta'):
            fx = 'comment-cta'
        elif QUESTION.search(text) and t0 - last_fx_t > 4 and used.get('prompt-box', 0) < 1:
            fx = 'prompt-box'
        elif t0 - last_fx_t >= 4.5 and t0 > 1.5 and not is_last:
            commas = text.count(',') + sum(1 for w in words if norm(w) in ('iyo', 'ama', 'and', 'or'))
            if re.search(r'\d', text) and used.get('chart', 0) < 1 and re.search(r'(%|\$|doolar|dollar|kor|growth|koror)', text, re.I):
                fx = 'chart'
            elif re.search(r'\b(koowaad|labaad|saddexaad|first|second|third|1\.|2\.|3\.)', text, re.I) and used.get('counter-list', 0) < 1:
                fx = 'counter-list'
            elif commas >= 2 and len(content_words(text)) >= 3:
                fx = 'chip-row' if used.get('chip-row', 0) < 1 else ('checklist' if used.get('checklist', 0) < 1 else None)
            if fx is None:
                for _ in range(len(pool)):
                    cand = pool[pi % len(pool)]
                    pi += 1
                    if used.get(cand, 0) >= 2 or cand == prev_fx:
                        continue
                    if cand in BG_CHANGE and t0 - last_bg_t < 3.2:
                        continue
                    fx = cand
                    break
        elif t0 - last_broll >= 9.5 and t0 > 6 and not is_last and t0 - last_fx_t >= 2.5:
            fx = 'broll'
        if fx:
            used[fx] = used.get(fx, 0) + 1
            prev_fx = fx
            end = min(t1, t0 + MAXDUR[fx]) if fx in MAXDUR else t1
            b = {'at': round(t0, 2), 'end': round(end, 2), 'fx': fx, 'say': text, 'word': key}
            if fx == 'broll':
                last_broll = t0
            else:
                last_fx_t = t0
            if fx in BG_CHANGE:
                last_bg_t = end
            if fx in LISTS:
                b['items'] = content_words(text) or [key.title()]
            if fx in SAYS_LINE:
                b['line'] = text
            if fx == 'chart':
                m = re.search(r'[\d.,]+\s*(%|\$|doolar|dollar)?', text)
                b['number'] = m.group(0).strip() if m else key
            if t0 - last_trans >= 3.0 and i > 0:
                b['transition'] = trans[ti % len(trans)]
                ti += 1
                last_trans = t0
            beats.append(b)
            # the rest of the line returns to the normal look with the next strongest word
            if t1 - end >= 1.3:
                k2 = key_word(' '.join(w for w in words if norm(w) != norm(key)), key)
                beats.append({'at': round(end, 2), 'end': round(t1, 2), 'fx': 'word', 'word': k2, 'place': place_cycle[pc % 2], 'say': text})
                pc += 1
        else:
            b = {'at': round(t0, 2), 'end': round(t1, 2), 'fx': 'word', 'word': key, 'place': place_cycle[pc % 2], 'say': text}
            pc += 1
            if not captions:
                line, red = rest_line(text, key)
                if line and len(line.split()) <= 4:
                    b['line'] = line
                    if red:
                        b['red'] = red
            if i > 0 and t0 - last_trans >= 3.0 and i % 3 == 0:
                b['transition'] = trans[ti % len(trans)]
                ti += 1
                last_trans = t0
            beats.append(b)
    end_t = max(beats[-1]['at'] + 0.8, dur - 1.6)
    beats.append({'at': round(end_t, 2), 'end': round(dur, 2), 'fx': 'end', 'word': (brand or 'Hal-abuur'), 'button': 'Follow', 'done': 'Following'})
    for b in beats[:-1]:
        if b['end'] > end_t:
            b['end'] = round(max(b['at'] + 0.5, end_t), 2)
    beats = [b for b in beats if b['end'] - b['at'] >= 0.45 or b['fx'] == 'end']
    return {'format': fmt, 'captions': captions, 'beats': beats}


def check(plan, dur, has_mask):
    """Problems a human editor would flag. Returns a list of strings (empty = fine)."""
    out = []
    beats = plan.get('beats', [])
    last = None
    count = {}
    prev_bg_end = -99
    for i, b in enumerate(beats):
        fx = b.get('fx', 'word')
        if b['end'] <= b['at']:
            out.append('beat %d (%s) ends before it starts' % (i + 1, fx))
        if b['at'] > dur + 0.5:
            out.append('beat %d (%s) starts after the video ends' % (i + 1, fx))
        if fx == last and fx not in ('word', 'cards', 'pills'):
            out.append('beat %d: %s twice in a row' % (i + 1, fx))
        count[fx] = count.get(fx, 0) + 1
        if fx in BG_CHANGE:
            if b['at'] - prev_bg_end < 2.5:
                out.append('beat %d: %s comes %.1f s after the last background effect (keep 3 s of the normal look)' % (i + 1, fx, b['at'] - prev_bg_end))
            prev_bg_end = b['end']
        last = fx
    for fx, n in count.items():
        if fx not in ('word', 'cards', 'pills', 'broll', 'end') and n > 2:
            out.append('%s is used %d times (at most twice)' % (fx, n))
    return out


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save(plan, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)
