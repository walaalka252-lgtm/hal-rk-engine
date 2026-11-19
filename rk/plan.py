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


STRONG = re.compile(r'^(discount|diskoon\w*|dhimis\w*|qiim\w*|bilaash\w*|free|jaban\w*|rakhiis\w*|cusub\w*|new|tayo\w*|macaan\w*|'
                    r'caafimaad\w*|dalab\w*|dalbo|sir\w*|siraha|natiij\w*|faa.?iido\w*|lacag\w*|guul\w*|best|secret|sale|offer|'
                    r'price|quality|health\w*|money|result\w*|ramadaan|ciid\w*|hadiyad\w*|gift)$')
# adjective-like words that belong with the noun before them ("subag la'aan", "qiimo dhimis", "dalabyada cusub")
PAIR2 = re.compile(r'^(dhimis|laaan|la.aan|yar|cusub|macaan|jaban|fiican|badan|weyn|dhan|kasta|walba|free|new)$')


def key_word(text, prev=None):
    toks = [re.sub(r"^[^\w$]+|[^\w%$']+$", '', w, flags=re.U) for w in text.split()]
    best, score = None, -99
    for i, w in enumerate(toks):
        if not w:
            continue
        n = norm(w)
        s = 0.0
        if STRONG.match(n):
            s += 4.5
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


# ---------------------------------------------------------------- phrases (what is said together)
SENT_END = re.compile(r'[.!?…]["»”]?$')
SOFT_END = re.compile(r'[,;:]["»”]?$')
JOIN = set('oo laakiin sidoo hase balse but so because which'.split())   # words that start a new clause


def phrases(cues, max_words=7, max_dur=3.0):
    """Groups the spoken words into phrases the way an editor cuts subtitles: at sentence ends, at commas, at pauses,
    never longer than ~7 words / 3 s (a long run is cut at its longest pause or before "iyo / oo / and").
    Works the same for a normal SRT and for a word-by-word SRT."""
    W = sorted((dict(w) for c in cues for w in c['words']), key=lambda w: w['t0'])
    if not W:
        return []
    # a cue end is a soft boundary too (an SRT line usually ends where the speaker breathes)
    ends = set()
    for c in cues:
        if c['words'] and len(c['words']) >= 3:
            ends.add(id(c['words'][-1]))
    flat = [w for c in cues for w in c['words']]
    cue_end = {(round(w['t0'], 3), w['w']) for w in flat if id(w) in ends}
    gaps = sorted(max(0.0, b['t0'] - a['t1']) for a, b in zip(W, W[1:]))
    med = gaps[len(gaps) // 2] if gaps else 0.1
    pause = max(0.26, min(0.5, med * 3.0 + 0.12))

    def score_break(cur, k, nxt):
        """How good it is to end a phrase after cur[k] (higher = better)."""
        a, b = cur[k], (cur[k + 1] if k + 1 < len(cur) else nxt)
        g = b['t0'] - a['t1']
        s = g * 8.0
        if SENT_END.search(a['w']):
            s += 5
        elif SOFT_END.search(a['w']):
            s += 3
        if (round(a['t0'], 3), a['w']) in cue_end:
            s += 2
        if norm(b['w']) in JOIN:
            s += 1.2
        if norm(a['w']) in JOIN:
            s -= 2
        s -= abs((k + 1) - len(cur) / 2.0) * 0.15
        return s

    out, cur = [], []
    for w in W:
        if cur:
            prev = cur[-1]
            gap = w['t0'] - prev['t1']
            n = len(cur)
            dur = w['t1'] - cur[0]['t0']
            if (SENT_END.search(prev['w']) is not None
                    or (SOFT_END.search(prev['w']) is not None and (n >= 3 or dur > 1.2))
                    or (gap > pause and n >= 2)
                    or ((round(prev['t0'], 3), prev['w']) in cue_end and n >= 3)):
                out.append(cur)
                cur = []
            elif n >= max_words or dur > max_dur:
                # cut at the best place inside the run, the tail starts the next phrase
                inside = set()
                for d, i0, i1, c in numbers(' '.join(x['w'] for x in cur)):
                    inside.update(range(i0, i1))          # never between "lix" and "iyo toban doolar"
                ks = [k for k in range(1, n) if k not in inside] or [n - 1]
                k = max(ks, key=lambda k: score_break(cur, k, w)) if n >= 3 else n - 1
                out.append(cur[:k + 1])
                cur = cur[k + 1:]
        cur.append(w)
    if cur:
        out.append(cur)
    # no one-word phrases: join them to a neighbour
    merged = []
    for ph in out:
        if merged and len(ph) == 1 and not SENT_END.search(merged[-1][-1]['w']) and ph[0]['t0'] - merged[-1][-1]['t1'] < 0.7 and len(merged[-1]) <= max_words:
            merged[-1] = merged[-1] + ph
        elif merged and len(merged[-1]) == 1 and ph[0]['t0'] - merged[-1][-1]['t1'] < 0.7 and len(ph) < max_words:
            merged[-1] = merged[-1] + ph
        else:
            merged.append(ph)
    return [{'t0': p[0]['t0'], 't1': p[-1]['t1'], 'text': ' '.join(w['w'] for w in p), 'words': p} for p in merged]


# ---------------------------------------------------------------- numbers said in words ("lix iyo toban doolar" → 16$)
ONES = {'kow': 1, 'koow': 1, 'kowda': 1, 'laba': 2, 'labo': 2, 'labada': 2, 'saddex': 3, 'sadex': 3, 'saddexda': 3, 'afar': 4,
        'afarta': 4, 'shan': 5, 'shanta': 5, 'lix': 6, 'lixda': 6, 'toddoba': 7, 'todoba': 7, 'toddobada': 7, 'siddeed': 8,
        'sideed': 8, 'sagaal': 9, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
TENS = {'toban': 10, 'tobanka': 10, 'labaatan': 20, 'soddon': 30, 'sodon': 30, 'afartan': 40, 'konton': 50, 'lixdan': 60,
        'toddobaatan': 70, 'todobaatan': 70, 'siddeetan': 80, 'sideetan': 80, 'sagaashan': 90, 'ten': 10, 'eleven': 11, 'twelve': 12,
        'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
MULT = {'boqol': 100, 'kun': 1000, 'hundred': 100, 'thousand': 1000, 'malyuun': 1000000, 'million': 1000000}
CUR = {'doolar': '$', 'doolarka': '$', 'doollar': '$', 'dollar': '$', 'dollars': '$', 'usd': '$', 'euro': '€', 'yuuro': '€',
       'boqolkiiba': '%', 'percent': '%'}
DIGITS = re.compile(r'^\$?(\d+(?:[.,]\d+)?)([$%€]?)$')


def _numval(t):
    m = DIGITS.match(t)
    if m:
        return float(m.group(1).replace(',', '.')), m.group(2) or ('$' if t.startswith('$') else '')
    if t in ONES:
        return ONES[t], ''
    if t in TENS:
        return TENS[t], ''
    return None


def numbers(text):
    """[(display, first word index, last word index, currency)] for every number said in `text`."""
    toks = [norm(w) for w in text.split()]
    out, i = [], 0
    while i < len(toks):
        v = _numval(toks[i])
        if v is None and toks[i] not in MULT:
            i += 1
            continue
        total, cur, last, cur_sym, j = 0.0, 0.0, i, '', i
        while j < len(toks):
            t = toks[j]
            nv = _numval(t)
            if nv is not None:
                cur += nv[0]
                cur_sym = cur_sym or nv[1]
                last = j
            elif t in MULT:
                cur = max(cur, 1) * MULT[t]
                total += cur
                cur = 0
                last = j
            elif t in ('iyo', 'and') and j + 1 < len(toks) and (_numval(toks[j + 1]) is not None or toks[j + 1] in MULT):
                pass
            else:
                break
            j += 1
        val = total + cur
        k = last
        if k + 1 < len(toks) and toks[k + 1] in CUR:
            cur_sym = CUR[toks[k + 1]]
            k += 1
        disp = ('%d' % val) if abs(val - round(val)) < 1e-6 else ('%g' % val)
        if cur_sym in ('$', '%', '€'):
            disp += cur_sym
        out.append((disp, i, k, cur_sym))
        i = k + 1
    return out


def has_word(text, *pats):
    n = ' ' + ' '.join(norm(w) for w in text.split()) + ' '
    return any(re.search(p, n) for p in pats)


ICONS = [(r' (dalb|iibs|order|buy|cart)', 'cart'), (r' (jecel|jeclaa|love)', 'heart'), (r' (tayo|quality|ugu fiican|best)', 'star'),
         (r' (degdeg|dhaqso|fast|quick)', 'bolt'), (r' (saac|wakhti|waqti|time|daqiiq)', 'clock'),
         (r' (qiim|price|diskoon|discount|dhimis|jaban|rakhiis|sale|offer)', 'tag'), (r' (fiidiyo|fiidyo|video|muuqaal)', 'play')]
OLD = (r' (hore|ahayd|ahaa|jirtay|jirey|before|was|used) ',)
NEW = (r' (hadda|haatan|haddeer|now|today|maanta) ',)
LIST_SEP = ('iyo', 'ama', 'and', 'or')
KINDS = (r' (nooc|noocy|dhammaan|dhamaan|kala dooro|cayn|all |types|kinds|every)',)
SUPER = (r' ugu (\w+)',)
PUNCHY = (r' (ogow|muhiim|xasuuso|fiiro|important|remember|listen)',)
CTA = re.compile(r'\b(follow|subscribe|comment|faallo|la soco|lasoco|nala soco|like|share|wadaag|wac|whatsapp|la xiriir|nagala soo xiriir|dalbo)\w*', re.I)


def topic_words(phrs, n=6):
    """The things the video is about (most repeated content words, grouped by stem), best first."""
    count, forms = {}, {}
    for p in phrs:
        for w in p['text'].split():
            c = re.sub(r'^[^\w]+|[^\w]+$', '', w, flags=re.U)
            k = norm(c)
            if k in STOP or len(k) < 4 or _numval(k) is not None or k in CUR or k in MULT:
                continue
            stem = k[:6] if len(k) > 6 else k
            count[stem] = count.get(stem, 0) + 1
            forms.setdefault(stem, {}).setdefault(c, 0)
            forms[stem][c] += 1
    best = sorted(count, key=lambda s: -count[s])
    out = []
    for s in best[:n]:
        f = min(forms[s], key=lambda x: (len(x), -forms[s][x]))
        out.append((f.capitalize(), count[s], s))
    return out


VERBS = re.compile(r'^(dooro|doorta|dooran|kala|choose|pick|select|iibso|dalbo|eeg|fiiri|helo|hesho|raadi|look|get|buy|try|isku)$')


def named_things(text, n=5):
    """The things a phrase names (for orbit / chips when it lists kinds): its content words, no verbs or numbers."""
    out = []
    for w in text.split():
        c = re.sub(r"^[^\w]+|[^\w']+$", '', w, flags=re.U)
        k = norm(c)
        if not c or k in STOP or len(k) < 3 or VERBS.match(k) or _numval(k) is not None or k in CUR:
            continue
        if c.capitalize() not in out:
            out.append(c.capitalize())
    return out[:n]


def list_items(text, exclude=()):
    toks = [re.sub(r'^[^\w$]+|[^\w%$]+$', '', w, flags=re.U) for w in text.split()]
    seps = sum(1 for w in text.split() if norm(w) in LIST_SEP) + text.count(',')
    if seps < 2:
        return []
    ex = set(norm(x) for x in exclude)
    items = []
    for w in toks:
        k = norm(w)
        if not w or k in STOP or len(k) < 3 or _numval(k) is not None or k in ex:
            continue
        if w.capitalize() not in items:
            items.append(w.capitalize())
    return items[:4] if len(items) >= 3 else []


UNITS = {'kiilo': 'KIILO', 'kilo': 'KIILO', 'kg': 'KG', 'garaam': 'G', 'gram': 'G', 'litir': 'L', 'xabbo': 'XABBO', 'xabo': 'XABBO',
         'sanadood': 'SANO', 'sano': 'SANO', 'maalmood': 'MAALMOOD', 'saacadood': 'SAAC', 'daqiiqo': 'DAQIIQO', 'bilood': 'BILOOD'}
ORDINAL = re.compile(r'^(koowaad|kowaad|labaad|saddexaad|sadexaad|afraad|shanaad|first|second|third|fourth|fifth)$')


def phrase_bigs(text, prev=None):
    """The word(s) a lockup shows huge for a spoken phrase: a price, "3 SABABOOD", the two things two ordinals name,
    a noun + its adjective ("QIIMO DHIMIS"), else the strongest single word."""
    raw = text.split()
    toks = [re.sub(r"^[^\w$]+|[^\w%$']+$", '', w, flags=re.U) for w in raw]
    ns = [norm(w) for w in toks]
    nums = numbers(text)
    # ordinals: "koowaad waa tayada, labaad waa qiimaha" → TAYADA + QIIMAHA
    outs = []
    for i, n in enumerate(ns):
        if ORDINAL.match(n):
            for j in range(i + 1, min(len(ns), i + 4)):
                if ns[j] not in STOP and len(ns[j]) >= 3 and _numval(ns[j]) is None:
                    outs.append(toks[j].upper())
                    break
    if outs:
        return outs[:2]
    # a number with what it counts: "saddex sababood" → 3 SABABOOD, "labada kiilo" → 2 KIILO
    for d, i0, i1, cur in nums:
        if cur:
            continue
        if i1 + 1 < len(ns):
            nx = ns[i1 + 1]
            said = ' '.join(toks[i0:i1 + 2])
            if nx in UNITS:
                return [{'text': d + ' ' + UNITS[nx], 'on': said}]
            if nx not in STOP and len(nx) >= 4 and _numval(nx) is None:
                return [{'text': d + ' ' + toks[i1 + 1].upper(), 'on': said}]
    k = key_word(text, prev)
    kn = norm(k)
    idx = next((i for i, n in enumerate(ns) if n == kn), None)
    if idx is not None and idx + 1 < len(ns) and PAIR2.match(ns[idx + 1]) and len(toks[idx]) + len(toks[idx + 1]) <= 16:
        return [(toks[idx] + ' ' + toks[idx + 1]).upper()]
    if idx is not None and idx > 0 and PAIR2.match(ns[idx]) and ns[idx - 1] not in STOP and len(toks[idx - 1]) + len(toks[idx]) <= 16:
        return [(toks[idx - 1] + ' ' + toks[idx]).upper()]
    return [k]


def auto(cues, dur, has_mask=True, fmt='9:16', brand=None, captions=True, closeup=False):
    """A complete draft storyboard: every phrase is a lockup (its words built as they are said, the key word huge),
    with a picture or graphic that shows what is said where the words call for it (a product card, the price, a date,
    the list, an icon); a few effects from the library only where the words fit them; the end card.
    Claude then reads every phrase and fixes the key words, the visuals and the effects."""
    phrs = phrases(cues)
    if not phrs:
        return {'beats': []}
    topics = topic_words(phrs)
    product = topics[0] if topics and topics[0][1] >= 3 else None
    beats = []
    last = {'card': -99.0, 'icon': -99.0, 'fx': -99.0, 'bg': -99.0, 'trans': -99.0}
    used = {}
    prev_key = None
    first_t = phrs[0]['t0']
    for i, ph in enumerate(phrs):
        text = ph['text']
        t0 = ph['t0']
        nxt = phrs[i + 1]['t0'] if i + 1 < len(phrs) else min(dur, ph['t1'] + 0.8)
        end = min(nxt - 0.02, ph['t1'] + 0.75)
        if end - t0 < 0.5:
            end = min(nxt - 0.02, t0 + 0.6)
        at = max(0.0, t0 - 0.06, beats[-1]['end'] if beats else 0.0)
        is_last = i >= len(phrs) - 2
        nums = numbers(text)
        price = [n for n in nums if n[3]]
        fx = None
        # ---- effects from the library, only where the words call for them
        if t0 - last['fx'] >= 7.0 and not (beats and beats[-1].get('fx') not in ('say',)):
            if CTA.search(text) and t0 > dur * 0.6 and not used.get('comment-cta'):
                fx = {'fx': 'comment-cta', 'word': (product[0] if product else key_word(text)).upper(), 'name': brand or 'Hal-abuur'}
            elif t0 - first_t < 9 and (brand or product) and not used.get('cover-story') and i <= 2 and has_word(text, r' (' + re.escape(norm(brand or product[0])[:5]) + r')'):
                fx = {'fx': 'cover-story', 'word': (brand or product[0]).upper(), 'items': [t[0] for t in topics[1:4]]}
            elif has_word(text, *KINDS) and not used.get('orbit') and has_mask:
                items = list_items(text) or named_things(text)
                if len(items) < 2:
                    items = [t[0] for t in topics[:4]]
                fx = {'fx': 'orbit', 'items': items[:5]}
            elif has_word(text, *PUNCHY) and not used.get('blackout') and has_mask:
                fx = {'fx': 'blackout', 'word': key_word(text)}
            elif has_word(text, *SUPER) and not used.get('spotlight') and has_mask:
                m = re.search(r' ugu (\w+)', ' ' + ' '.join(norm(w) for w in text.split()) + ' ')
                fx = {'fx': 'spotlight', 'word': (m.group(1) if m else key_word(text)).upper(), 'word2': (product[0] if product else key_word(text)).upper()}
        if fx:
            name = fx['fx']
            used[name] = used.get(name, 0) + 1
            e = min(nxt - 0.02, t0 + MAXDUR.get(name, 2.6), max(end, t0 + 1.2))
            b = dict({'at': round(at, 2), 'end': round(e, 2), 'say': text}, **fx)
            if name in BG_CHANGE:
                if t0 - last['bg'] < 3.0:
                    b = None
                else:
                    last['bg'] = e
            if b:
                last['fx'] = t0
                beats.append(b)
                # the rest of the phrase (if long) goes back to the lockup
                if end - e >= 1.0:
                    beats.append({'at': round(e, 2), 'end': round(end, 2), 'fx': 'say', 'big': key_word(text, prev_key)})
                continue
        # ---- the lockup and what shows what is said
        b = {'at': round(at, 2), 'end': round(end, 2), 'fx': 'say'}
        vis = None
        if price:
            d = price[-1]
            strike = has_word(text, *OLD) and not has_word(text, *NEW)
            b['big'] = [{'text': d[0], 'strike': True}] if strike else [d[0]]
        elif nums and has_word(text, r' (bish|bilood|month|ramadaan|ramadan|maalin|day)'):
            n = nums[0]
            top = 'RAMADAAN' if has_word(text, r' (ramadaan|ramadan)') else ('BISHA' if has_word(text, r' (bish|month)') else 'MAALIN')
            vis = {'type': 'date', 'top': top, 'day': n[0].rstrip('$%'), 'on': text.split()[n[1]]}
            b['big'] = key_word(' '.join(w for k, w in enumerate(text.split()) if not (n[1] <= k <= n[2])), prev_key)
        else:
            items = list_items(text)
            if items:
                rest = ' '.join(w for w in text.split() if w.strip('.,!?').capitalize() not in items)
                b['big'] = key_word(rest or text, prev_key)
                vis = {'type': 'chips', 'items': items}
            else:
                b['big'] = phrase_bigs(text, prev_key)
                if len(b['big']) == 1 and isinstance(b['big'][0], str):
                    b['big'] = b['big'][0]
                if product and not closeup and has_word(text, r' ' + re.escape(product[2])) and t0 - last['card'] >= 9.0:
                    vis = {'type': 'card', 'label': product[0], 'on': product[0]}
                    last['card'] = t0
                elif t0 - last['icon'] >= 5.0:
                    for pat, name in ICONS:
                        if has_word(text, pat):
                            vis = {'type': 'icon', 'name': name}
                            last['icon'] = t0
                            break
        bb = b['big'] if isinstance(b['big'], list) else [b['big']]
        prev_key = str(bb[0].get('text') if isinstance(bb[0], dict) else bb[0])
        if vis:
            b['visual'] = vis
        # a soft transition where a new sentence starts after a pause
        if i > 0 and SENT_END.search(phrs[i - 1]['text']) and t0 - phrs[i - 1]['t1'] > 0.25 and t0 - last['trans'] >= 9.0:
            b['transition'] = 'halftone' if (len(beats) // 2) % 2 == 0 else 'streak'
            last['trans'] = t0
        b['say'] = text
        beats.append(b)
    end_t = max(beats[-1]['at'] + 0.8, dur - 1.6)
    beats.append({'at': round(end_t, 2), 'end': round(dur, 2), 'fx': 'end', 'word': (brand or (product[0] if product else 'Hal-abuur')), 'button': 'Follow', 'done': 'Following'})
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
        if i > 0 and b['at'] < beats[i - 1]['end'] - 0.06:
            out.append('beat %d (%s) starts %.2f s before beat %d ends (one beat at a time)' % (i + 1, fx, beats[i - 1]['end'] - b['at'], i))
        if fx == 'say' and not b.get('big') and not b.get('word'):
            out.append('beat %d (say): no big word' % (i + 1))
        if fx == last and fx not in ('word', 'cards', 'pills', 'say'):
            out.append('beat %d: %s twice in a row' % (i + 1, fx))
        count[fx] = count.get(fx, 0) + 1
        if fx in BG_CHANGE:
            if b['at'] - prev_bg_end < 2.5:
                out.append('beat %d: %s comes %.1f s after the last background effect (keep 3 s of the normal look)' % (i + 1, fx, b['at'] - prev_bg_end))
            prev_bg_end = b['end']
        last = fx
    for fx, n in count.items():
        if fx not in ('word', 'cards', 'pills', 'broll', 'end', 'say') and n > 2:
            out.append('%s is used %d times (at most twice)' % (fx, n))
    return out


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save(plan, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)
