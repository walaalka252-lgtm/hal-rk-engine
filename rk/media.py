"""ffmpeg / ffprobe helpers: probe, frame reader (rgb24 pipe), chunk writer, audio."""
import json
import os
import shutil
import subprocess
import numpy as np

FFMPEG = os.environ.get('RK_FFMPEG') or shutil.which('ffmpeg') or 'ffmpeg'
FFPROBE = os.environ.get('RK_FFPROBE') or shutil.which('ffprobe') or 'ffprobe'


def have_ffmpeg():
    return bool(shutil.which(FFMPEG) or os.path.exists(FFMPEG))


def probe(path):
    """Display size (rotation applied), fps, duration and whether there is audio."""
    out = subprocess.run([FFPROBE, '-v', 'error', '-print_format', 'json', '-show_streams', '-show_format', path],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError('ffprobe could not read %s: %s' % (path, out.stderr.strip()[:300]))
    info = json.loads(out.stdout or '{}')
    v = next((s for s in info.get('streams', []) if s.get('codec_type') == 'video'), None)
    a = next((s for s in info.get('streams', []) if s.get('codec_type') == 'audio'), None)
    if not v:
        raise RuntimeError('no video stream in ' + path)
    w, h = int(v['width']), int(v['height'])
    rot = 0
    for sd in v.get('side_data_list', []) or []:
        if 'rotation' in sd:
            rot = int(float(sd['rotation']))
    if 'rotate' in (v.get('tags') or {}):
        rot = int(float(v['tags']['rotate']))
    if abs(rot) % 180 == 90:
        w, h = h, w

    def rate(s):
        try:
            n, d = s.split('/')
            return float(n) / float(d) if float(d) else 0.0
        except Exception:
            return 0.0
    fps = rate(v.get('avg_frame_rate', '0/1')) or rate(v.get('r_frame_rate', '0/1')) or 30.0
    dur = float(info.get('format', {}).get('duration') or v.get('duration') or 0)
    return {'w': w, 'h': h, 'fps': fps, 'dur': dur, 'audio': bool(a),
            'hdr': (v.get('color_transfer') in ('arib-std-b67', 'smpte2084')), 'pix_fmt': v.get('pix_fmt')}


class Reader:
    """Decodes frames [t0, t1) at a fixed fps, scaled to (w, h), as uint8 RGB arrays."""

    def __init__(self, path, w, h, fps, t0=0.0, t1=None, hdr=False):
        self.w, self.h = int(w), int(h)
        vf = ['fps=%g' % fps, 'scale=%d:%d:flags=bicubic' % (self.w, self.h)]
        cmd = [FFMPEG, '-v', 'error', '-nostdin']
        if t0 > 0:
            cmd += ['-ss', '%.3f' % t0]
        cmd += ['-i', path]
        if t1 is not None:
            cmd += ['-t', '%.3f' % max(0.04, t1 - t0)]
        cmd += ['-an', '-vf', ','.join(vf), '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.w * self.h * 3 * 2)
        self.last = None
        self.n = 0

    def read(self):
        size = self.w * self.h * 3
        buf = self.p.stdout.read(size)
        if len(buf) < size:
            if self.last is None:
                err = self.p.stderr.read().decode('utf-8', 'replace')[:400]
                raise RuntimeError('could not decode video frames: ' + err)
            return self.last          # past the end: hold the last frame
        self.last = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
        self.n += 1
        return self.last

    def close(self):
        try:
            self.p.stdout.close()
            self.p.kill()
            self.p.wait(timeout=5)
        except Exception:
            pass


class Writer:
    """Encodes uint8 RGB frames to an H.264 video-only file (yuv420p)."""

    def __init__(self, path, w, h, fps, crf=18, preset='veryfast'):
        self.path = path
        cmd = [FFMPEG, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (w, h),
               '-r', '%g' % fps, '-i', '-', '-an', '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
               '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-g', str(int(round(fps * 2))), '-movflags', '+faststart', path]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.p.stdin.close()
        err = self.p.stderr.read().decode('utf-8', 'replace')
        if self.p.wait() != 0:
            raise RuntimeError('ffmpeg encode failed: ' + err[:400])


def extract_audio(src, dst_wav, sr=48000):
    r = subprocess.run([FFMPEG, '-v', 'error', '-y', '-i', src, '-vn', '-ac', '2', '-ar', str(sr), '-c:a', 'pcm_s16le', dst_wav],
                       capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(dst_wav)


def read_wav(path):
    import wave
    with wave.open(path, 'rb') as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        a = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768.0
    return a.reshape(-1, ch), sr


def write_wav(path, a, sr):
    import wave
    a = np.clip(a, -1, 1)
    with wave.open(path, 'wb') as w:
        w.setnchannels(a.shape[1] if a.ndim > 1 else 1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((a * 32767).astype(np.int16).tobytes())


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(' '.join(cmd[:4]) + ' … failed: ' + r.stderr[-600:])
    return r
