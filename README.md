# Hal-abuur Red Kinetic engine

Renders the **Red Kinetic Talk** edit of a talking-head video from the footage, its SRT and a small `plan.json`:
huge red-gradient words built letter by letter (never over the face, behind the head when there is room),
red-bordered photo cards, B-roll beats in a phone frame, red pixel blocks, light streak / halftone / blur-zoom
transitions, the **36 creator-reel effects** of the template, and bottom captions in the template design
(Inter ExtraBold Italic, white with a dark edge, a red box that slides onto the spoken word).

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
    {"at": 0.3, "end": 2.6, "fx": "word", "word": "DISCOUNT", "place": "auto"},
    {"at": 2.7, "end": 5.0, "fx": "cards", "word": "BUSKUD", "cards": [{"img": "frame:6.2@0.0,0.0,0.3,0.3", "label": "Buskud timir"}]},
    {"at": 5.1, "end": 7.4, "fx": "cover-story", "word": "BUSKUD", "items": ["Timir", "Saliid"], "line": "qiimaha waa jaban", "transition": "streak"},
    {"at": 7.5, "end": 9.0, "fx": "blackout", "word": "QIIMAHA"},
    {"at": 38.8, "end": 40.4, "fx": "end", "word": "Buskud Shop", "button": "Follow", "done": "Following"}
  ]
}
```

- Times are seconds of the footage. One beat at a time (beats must not overlap); a gap = the plain footage.
- `word` = the one big key word of the beat (a name, number, product, place or action the speaker says right then),
  in the speaker's language and spelling. It builds on the frame where the SRT says it (found automatically).
- `place` (big words): `auto` · `behind` (behind the head, needs headroom) · `above` · `chest` (under the chin).
  The engine checks the face every frame range and never puts text over the eyes, nose or mouth.
- `items`: short words for lists (chips, checklist, orbit, counter, bars like `"16$"`); each appears when it is said
  (or give `{"text": "Timir", "at": 12.4}`).
- `img`: `"frame:12.3"` (a still of the footage), `"frame:12.3@x,y,w,h"` (a crop, 0–1 of the frame) or a photo file.
- `transition`: `streak` · `halftone` · `blurzoom` · `flash` · `rgb` · `none`, played at the start of the beat.
- `line`: a sentence for the effects that show one (prompt box, UI world, serif flow, cover story…). With captions on,
  a thin line that only repeats the spoken words is dropped (never the same words twice).
- Optional top-level: `"sfx": false`, `"camera": "none"` (no slow zooms), `"mask": false`, `"fps": 30`.

## Beat types

Normal look: `word`, `cards`, `pills`, `punch`, `broll`, `split`, `end`.
Effects library (run `python3 run.py fx` for the fields of each): orbit, couch-wall, ui-world, frame-break,
digital-zoom, phone, checklist, headline-wall, zoom-through, chart, pull-back, green-screen, paper-cut, doodle-boil,
vertigo, cover-story, glass-type, type-wall, stack, depth-scatter, rack-focus, spatial-glass, scene-text,
freeze-sticker, sticky-text, clone, blackout, freeze-label, shatter, chip-row, serif-flow, spotlight, prompt-box,
counter-list, comment-cta, blur-in.

## How to make a good plan (what the template asks)

1. Start from `auto` (it already has one beat per line, an effect every ~5 s, B-roll every ~10 s and an end card).
2. Read every line of the SRT and fix each `word` so it is the strongest word of that line, never a filler.
3. Pick each effect because it fits what is said at that moment (lists → chip-row / checklist / orbit; numbers → chart /
   counter-list / stack; a hook or a turn → frame-break / zoom-through; a statement → type-wall / cover-story /
   spotlight; one punch word → blackout / punch; AI / an app → prompt-box / ui-world / spatial-glass / phone;
   a joke about me → freeze-sticker / doodle-boil; asking for comments / follows → comment-cta).
   About one effect every 4–8 s, never the same effect twice in a row, each at most twice per video, at least 3 s of
   the normal look between two effects that change the background.
4. Use the footage itself for pictures: give `cards`/`img` crops of the frames where the product, place or person is.
5. `stills`, then look at `stills/sheet.jpg`: text readable, nothing over the face, every effect on its word.
   Fix the plan and run `stills` again until it is right; then `render`.

Fonts: Inter, Instrument Serif, Gloock, Nothing You Could Do (SIL OFL). Models: MediaPipe selfie segmentation and
BlazeFace short range (Apache 2.0). See LICENSES.md.
