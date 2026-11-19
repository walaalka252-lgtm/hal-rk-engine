# Hal-abuur Red Kinetic engine

Renders the **Red Kinetic Talk** edit of a talking-head video from the footage, its SRT and a small `plan.json`.
Like the reference reels, **every phrase is designed as it is said**: its small italic words appear word by word and its
key word lands huge in the red gradient (letter by letter, never over the face, behind the head when there is room);
what is said gets its picture or graphic at that moment (the product card, the price, the old price struck out, a date,
the list as chips, an icon); a few of the **36 creator-reel effects** come in where the words fit them; transitions,
camera curves, the end card, and the template captions (Inter ExtraBold Italic, a red box that slides onto the spoken
word) for the moments that show no words.

Needs only what a normal Python sandbox has: Python 3.9+, numpy, Pillow, OpenCV (4.8+: it runs the person-mask and
face models itself, no mediapipe/torch) and ffmpeg. Fonts and models are in `assets/`.

## Commands (run from this folder, or `python3 /path/to/run.py …`)

```bash
python3 run.py doctor                              # checks ffmpeg, OpenCV person mask + face, fonts
python3 run.py auto VIDEO.mp4 VIDEO.srt -o plan.json --format 9:16 --brand "Shop name"
python3 run.py fx                                  # every effect, what it is for and its fields
python3 run.py check plan.json                     # problems a human editor would flag
python3 run.py stills plan.json -o stills          # one still per beat + stills/sheet.jpg: LOOK AT IT
python3 run.py render plan.json -o final.mp4 --budget 240   # resumable: if it prints RESUME, run it again
```

If `doctor` says the person mask is not available: `pip install -U opencv-python-headless` (needs OpenCV ≥ 4.8).
Without the mask the engine still renders; effects that need it fall back to similar ones.

`render` works in 6-second parts and keeps them in `final.rkwork/`; with `--budget N` it stops after N seconds
and the same command continues where it stopped. It ends with `DONE: final.mp4` (H.264, AAC, the original voice
normalised to −14 LUFS, a few soft sound effects).

## plan.json

```json
{
  "video": "IMG_1234.MOV", "srt": "IMG_1234.srt", "format": "9:16", "captions": true, "brand": "Buskud Shop",
  "beats": [
    {"at": 0.14, "end": 2.68, "fx": "say", "say": "macaamiisheena discount ayaan idinku wadnaa", "big": "DISCOUNT", "visual": {"type": "icon", "name": "tag"}},
    {"at": 2.68, "end": 5.08, "fx": "cover-story", "word": "BUSKUD SHOP", "items": ["Timir", "Saliid"]},
    {"at": 6.15, "end": 7.68, "fx": "say", "say": "waa lix iyo toban doolar", "big": [{"text": "16$", "on": "lix iyo toban doolar"}]},
    {"at": 7.68, "end": 10.28, "fx": "say", "say": "hore waxay ahayd siddeed iyo toban doolar", "big": [{"text": "18$", "strike": true}]},
    {"at": 10.28, "end": 12.88, "fx": "orbit", "items": ["Timir", "Saliid", "Chocolate"]},
    {"at": 25.88, "end": 28.48, "fx": "say", "say": "koowaad waa tayada labaad waa qiimaha", "big": ["TAYADA", "QIIMAHA"]},
    {"at": 40.4, "end": 42.0, "fx": "end", "word": "Buskud Shop", "button": "Follow", "done": "Following"}
  ]
}
```

- Times are seconds of the footage. One beat at a time (no overlaps); a gap = the plain footage with captions.
- **`say` is the normal beat: one per phrase** (`auto` cuts the SRT into phrases, also a word-by-word SRT). Its words
  come from the SRT between `at` and `end` and appear as they are said. `say` in the plan is only a note for you.
- `big`: the word(s) shown huge: a string, or a list of up to 2 (two words stack like "Marketing / Agency"). An item can
  be `{"text": "16$", "on": "lix iyo toban doolar"}` (shown when those words are said) and `"strike": true` (an old
  price, a myth). Use the strongest word of the phrase: a name, product, number, price, place or action, never a filler.
- `visual` (optional), what shows what is said: `{"type": "card", "img": "frame:12.3@x,y,w,h" | "photo.jpg", "label": "Buskud"}`
  (only a clear picture: a product the speaker holds, a photo the user gave), `{"type": "cards", "imgs": [...], "labels": [...]}`,
  `{"type": "chips", "items": ["Timir", "Saliid", "Chocolate"]}` (a list that is said; each chip on its word),
  `{"type": "date", "top": "BISHA", "day": "1"}`, `{"type": "icon", "name": "cart|heart|star|tag|clock|bolt|check|play|percent"}`.
- `place` (big words): `auto` · `behind` (behind the head, needs headroom) · `above` · `chest` (under the chin).
  The engine checks the face over the beat and never puts text over the eyes, nose or mouth.
- Effects: `fx` = an effect name (`run.py fx`); `word`, `items`, `line`, `img` as listed there. `transition`:
  `streak` · `halftone` · `blurzoom` · `flash` · `rgb` · `none`, played at the start of the beat.
- Captions: `"captions": true` shows them where no lockup shows the words; `"always"` keeps them on under lockups too.
- Optional top-level: `"sfx": false`, `"camera": "none"` (no slow zooms), `"mask": false`, `"fps": 30`.

## Beat types

Normal look: `say` (one per phrase), `end`; also `word`, `cards`, `pills`, `punch`, `broll`, `split`.
Effects library (run `python3 run.py fx` for the fields of each): orbit, couch-wall, ui-world, frame-break,
digital-zoom, phone, checklist, headline-wall, zoom-through, chart, pull-back, green-screen, paper-cut, doodle-boil,
vertigo, cover-story, glass-type, type-wall, stack, depth-scatter, rack-focus, spatial-glass, scene-text,
freeze-sticker, sticky-text, clone, blackout, freeze-label, shatter, chip-row, serif-flow, spotlight, prompt-box,
counter-list, comment-cta, blur-in.

## How to make a good plan (what the template asks)

1. Start from `auto`: one `say` beat per phrase, the price / list / date / icon where the words call for it, a few
   effects where the words fit them, the end card.
2. Read every phrase (`say`) like an editor: what is the person saying **here**? Make `big` the word that carries it,
   and give the phrase the `visual` that shows it (a product → its card, a price → the price as `big`, the old price
   struck, a list → chips, a date → the calendar, an action or a feeling → an icon). Not every phrase needs a visual.
3. Replace a `say` beat with an effect only where the meaning fits the effect (lists of products → orbit / chip-row;
   numbers → chart / counter-list / stack; a hook or a turn → frame-break / zoom-through; a statement → type-wall /
   cover-story / spotlight; one punch word → blackout / punch; an app → phone / ui-world / prompt-box; a joke about
   me → freeze-sticker / doodle-boil; asking for comments or follows → comment-cta). About one effect every 8–12 s,
   never two in a row, each at most twice, at least 3 s between two effects that change the background.
4. Use the footage and the photos the user gave for pictures (`img`: `"frame:12.3@x,y,w,h"` crops, 0–1 of the frame).
5. `check`, then `stills`, and look at `stills/sheet.jpg`: every phrase readable, the big word on its word, the visual
   matching the words, nothing over the face. Fix the plan and run `stills` again until it is right; then `render`.

Fonts: Inter, Instrument Serif, Gloock, Nothing You Could Do (SIL OFL). Models: MediaPipe selfie segmentation and
BlazeFace short range (Apache 2.0). See LICENSES.md.
