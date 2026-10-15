"""Shadowing video: gapiruvchi Zamira personaji + katta matnli kartochka.

Ikki joylashuv (LAYOUT muhit o'zgaruvchisi yoki plan["layout"]):
  corner: personaj chap pastda, katta kartochka o'ngda (132px gacha)
  panel:  personaj tepada markazda, gap pastdagi keng panelda (110px gacha)  [standart]
Personaj (CHARACTER): cartoon (standart, chizilgan qiz) yoki zamira (rasmdan).
"""
import math, os, subprocess, tempfile

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from . import puppet
from .character import head as cartoon_head, body as cartoon_body

W, H, FPS = 1920, 1080, 25
TAIL = 2.5
FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")

CREAM = (247, 241, 232)
RED = (200, 50, 60)
TEAL = (30, 111, 104)
TEAL_L = (214, 234, 229)
DARK = (34, 37, 43)
GRAY = (150, 143, 134)
LIGHT = (200, 193, 184)
DIM = (226, 221, 214)
MARK = (255, 214, 102)
WHITE = (255, 255, 255)

LAYOUTS = {
    "corner": dict(
        card=(620, 110, 1880, 970), char_scale=0.95, char_x=300, char_bottom=1080, circle=(300, 820, 320),
        sent=(132, 120, 108, 96), stem=(140, 128, 116), maxw=1140, maxlines=3, uz=(54, 48), uzw=1160,
        num=(56, 34), num_style="corner", cartoon=dict(cx=300, base=700, sc=0.9)),
    "panel": dict(
        card=(60, 590, 1860, 1050), char_scale=0.86, char_x=960, char_bottom=680, circle=(960, 368, 205),
        cartoon=dict(cx=960, base=330, sc=0.62),
        sent=(110, 100, 90, 80), stem=(116, 106, 96), maxw=1640, maxlines=2, uz=(46, 42), uzw=1660,
        num=(40, 28), num_style="pill"),
}

CHARACTER = os.environ.get("CHARACTER", "cartoon")

_fonts = {}


def F(name, size):
    k = (name, size)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    return _fonts[k]


def ease(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def fade(img, a):
    if a >= 1:
        return img
    img.putalpha(img.getchannel("A").point(lambda v: int(v * max(0, a))))
    return img


# ---------------- assets ----------------
_assets = {}


def assets(name):
    if name in _assets:
        return _assets[name]
    L = LAYOUTS[name]
    card = L["card"]
    bg = Image.new("RGBA", (W, H), CREAM + (255,))
    bl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bl)
    bd.ellipse([1500, -260, 2200, 380], fill=(250, 214, 200, 140))
    bd.ellipse([-200, 760, 380, 1300], fill=(255, 226, 160, 110))
    bg.alpha_composite(bl.filter(ImageFilter.GaussianBlur(60)))
    d = ImageDraw.Draw(bg)
    cx, cy, r = L["circle"]
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=TEAL_L)
    # karta (soya bilan) alohida qatlam: personaj ustidan chiqishi uchun
    cardl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([card[0], card[1] + 18, card[2], card[3] + 18], 44, fill=(90, 60, 40, 55))
    cardl.alpha_composite(sh.filter(ImageFilter.GaussianBlur(26)))
    ImageDraw.Draw(cardl).rounded_rectangle(card, 44, fill=WHITE)
    sc = L["cartoon"]["sc"]
    heads = {(m, b): cartoon_head(m, b).resize((int(468 * sc), int(558 * sc)), Image.LANCZOS)
             for m in range(5) for b in (0, 0.4, 1)}
    body_ = cartoon_body().resize((int(700 * sc), int(479 * sc)), Image.LANCZOS)
    _assets[name] = dict(bg=bg, card=cardl, heads=heads, body=body_,
                         faces=puppet.frames(L["char_scale"]) if CHARACTER == "zamira" else None)
    return _assets[name]


# ---------------- text ----------------
def layout(words, font, maxw, lh):
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    sp = d.textlength(" ", font=font)
    lines, cur, cw = [], [], 0
    for wd in words:
        wl = d.textlength(wd, font=font)
        if cur and cw + sp + wl > maxw:
            lines.append((cur, cw)); cur, cw = [], 0
        cw = cw + (sp if cur else 0) + wl
        cur.append((wd, wl))
    if cur:
        lines.append((cur, cw))
    out = []
    for li, (ws, lw) in enumerate(lines):
        x = -lw / 2
        for wd, wl in ws:
            out.append((wd, x, li * lh, wl))
            x += wl + sp
    return out, len(lines)


def fit(text, maxw, sizes, maxlines):
    words = text.split()
    for s in sizes:
        f = F("InterDisplay-Bold.otf", s)
        lay, nl = layout(words, f, maxw, int(s * 1.18))
        if nl <= maxlines:
            break
    return f, lay, nl, int(f.size * 1.18)


def fit_line(text, maxw, sizes):
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    for s in sizes:
        f = F("Inter-Medium.otf", s)
        if d.textlength(text, font=f) <= maxw:
            return f, [text]
    f = F("Inter-Medium.otf", sizes[-1])
    words = text.split(); half = (len(words) + 1) // 2
    return f, [" ".join(words[:half]), " ".join(words[half:])]


def word_state(t, ev, n):
    """(aytilgan so'zlar, faol so'z, faol so'z ulushi, lo, hi)"""
    lo, hi = ev["span"] if ev.get("span") else (0, n)
    if ev.get("wt"):
        spoken, act, frac = set(), None, 0.0
        for i in range(lo, hi):
            s, e = ev["wt"][i]
            if t >= s - 0.04:
                spoken.add(i); act = i; frac = min(1, (t - s) / max(0.08, e - s))
        return spoken, act, frac, lo, hi
    return None


def word_state_prop(t, ev, lens, n):
    lo, hi = ev["span"] if ev.get("span") else (0, n)
    L = lens[lo:hi]; tot = sum(L)
    p = max(0, min(1, (t - ev["s"]) / max(0.05, ev["e"] - ev["s"] - 0.15)))
    acc = 0; spoken, act, frac = set(), None, 0.0
    for i, l in enumerate(L):
        if p * tot >= acc:
            spoken.add(lo + i); act = lo + i; frac = min(1, (p * tot - acc) / l)
        acc += l
    return spoken, act, frac, lo, hi


def draw_words(layer, lay, font, cx, top, colors, mark=None):
    d = ImageDraw.Draw(layer)
    if mark and mark[0] is not None:
        i, frac = mark
        wd, x, y, wl = lay[i]
        y0 = top + y + font.size * 0.62
        d.rounded_rectangle([cx + x - 10, y0, cx + x - 10 + (wl + 20) * ease(frac * 1.6), y0 + font.size * 0.42], 12, fill=MARK)
    for i, (wd, x, y, wl) in enumerate(lay):
        d.text((cx + x, top + y), wd, font=font, fill=colors[i])


# ---------------- render ----------------
def render(audio_path, plan, out_path, progress=None, only_frames=None, layout_name=None):
    name = layout_name or plan.get("layout") or os.environ.get("LAYOUT", "panel")
    if name not in LAYOUTS:
        name = "panel"
    L = LAYOUTS[name]
    A = assets(name)
    card = L["card"]
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "a.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", audio_path, "-ac", "1", "-ar", "48000", wav], check=True)
        a, sr = sf.read(wav, dtype="float32")
        audio_dur = len(a) / sr
        dur = audio_dur + TAIL
        nfr = int(dur * FPS) + 1
        hop = sr // FPS
        rms = np.array([np.sqrt(np.mean(a[i * hop:(i + 1) * hop] ** 2)) if i * hop < len(a) else 0.0 for i in range(nfr)])
        nz = rms[rms > 0]
        ref = np.percentile(nz, 95) if len(nz) else 1
        env = np.clip(rms / max(ref, 1e-6), 0, 1.2)
        sm = env.copy()
        for i in range(1, len(sm)):
            sm[i] = max(env[i], sm[i - 1] * 0.45)

        scenes = plan["scenes"]
        intro_end = plan["intro_end"]
        end_start = min(plan["end_start"], audio_dur)
        day = plan["day"]
        cx = (card[0] + card[2]) / 2
        cy = (card[1] + card[3]) / 2 + 10
        pre = {}
        for sc in scenes:
            pre[sc["n"]] = dict(
                sent=fit(sc["sent"], L["maxw"], L["sent"], L["maxlines"]),
                stem=fit(sc["stem"], L["maxw"], L["stem"], L["maxlines"]),
                uz=fit_line(sc["uz"], L["uzw"], L["uz"]), stem_uz=fit_line(sc["stem_uz"], L["uzw"], L["uz"]))
        blinks = [b + 0.6 * math.sin(b * 7) for b in np.arange(1.7, dur, 3.9)]
        nf, ns = L["num"]

        def scene_at(t):
            cur = None
            for sc in scenes:
                if t >= sc["start"]:
                    cur = sc
            return cur

        def block(layer, fitted, uzf, colors, mark=None):
            f, lay, nl, lh = fitted
            fu, lines = uzf
            h = nl * lh + 22 + len(lines) * (fu.size + 10)
            top = cy - h / 2
            draw_words(layer, lay, f, cx, top, colors, mark)
            d = ImageDraw.Draw(layer)
            for k, ln in enumerate(lines):
                d.text((cx, top + nl * lh + 22 + k * (fu.size + 10)), ln, font=fu, fill=GRAY, anchor="mt")

        def card_content(t, sc):
            Lr = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(Lr)
            fn = F("InterDisplay-Black.otf", nf)
            if L["num_style"] == "pill":
                txt = f"{sc['n']} / 5"
                tw = d.textlength(txt, font=fn)
                x0, y0 = card[0] + 50, card[1] - nf // 2 - 14
                d.rounded_rectangle([x0, y0, x0 + tw + 44, y0 + nf + 28], (nf + 28) // 2, fill=RED)
                d.text((x0 + 22, y0 + 14 + nf / 2), txt, font=fn, fill=WHITE, anchor="lm")
            else:
                d.text((card[0] + 50, card[1] + 36), str(sc["n"]), font=fn, fill=RED)
                d.text((card[0] + 50 + d.textlength(str(sc["n"]), font=fn) + 10, card[1] + 36 + nf - ns - 4),
                       "/ 5", font=F("Inter-SemiBold.otf", ns), fill=LIGHT)
            evs = sc["ev"]
            exs = [e for e in evs if e["kind"] == "ex"]
            first_ex = exs[0]["s"] if exs else 1e9
            speaking = [e for e in evs if e["s"] <= t <= e["e"]]
            mix = ease((t - (first_ex - 0.6)) / 0.45) if t >= first_ex - 0.6 else 0.0
            p = pre[sc["n"]]
            if mix < 1:
                S = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                lay = p["stem"][1]
                cols = [DARK] * len(lay)
                sp = [e for e in speaking if e["kind"] == "stem"]
                if sp:
                    spoken, *_ = word_state_prop(t, sp[0], [w[3] for w in lay], len(lay))
                    cols = [RED if i in spoken else DARK for i in range(len(lay))]
                block(S, p["stem"], p["stem_uz"], cols)
                Lr.alpha_composite(fade(S, 1 - mix), (0, int(-30 * mix)))
            if mix > 0:
                E = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                lay = p["sent"][1]
                sp = [e for e in speaking if e["kind"] == "ex"]
                mark = None
                if sp:
                    st = word_state(t, sp[0], len(lay)) or word_state_prop(t, sp[0], [w[3] for w in lay], len(lay))
                    spoken, act, frac, lo, hi = st
                    cols = [DARK if i in spoken else (LIGHT if lo <= i < hi else DIM) for i in range(len(lay))]
                    mark = (act, frac)
                else:
                    cols = [DARK] * len(lay)
                block(E, p["sent"], p["uz"], cols, mark)
                Lr.alpha_composite(fade(E, mix), (0, int(30 * (1 - mix))))
            return Lr

        def intro():
            Lr = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(Lr)
            f, lay, nl, lh = fit(plan["topic"], L["maxw"], L["stem"], L["maxlines"])
            top = cy - nl * lh / 2 + 20
            d.text((cx, top - 50), f"{day}-KUN", font=F("InterDisplay-Black.otf", nf), fill=RED, anchor="mm")
            draw_words(Lr, lay, f, cx, top, [DARK] * len(lay))
            return Lr

        def outro():
            Lr = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(Lr)
            d.text((cx, cy - 50), "Barakalla!", font=F("InterDisplay-Black.otf", 130), fill=RED, anchor="mm")
            d.text((cx, cy + 70), f"Ertaga: {day + 1}-kun", font=F("Inter-Medium.otf", 52), fill=GRAY, anchor="mm")
            return Lr

        def character(fr, i, t):
            amp = float(sm[min(i, len(sm) - 1)])
            if A["faces"]:
                lvl = 0 if amp < 0.1 else int(min(5, 1 + amp * 4.5))
                bk = any(b <= t <= b + 0.16 for b in blinks)
                face = A["faces"][(lvl, 1 if bk else 0)]
                bob = math.sin(t * 1.7) * 3 - amp * 3
                x = int(L["char_x"] - face.width / 2 + math.sin(t * 0.9) * 2)
                y = int(L["char_bottom"] - face.height + bob + 4)
                fr.alpha_composite(face, (x, y))
                return
            C = L["cartoon"]; sc = C["sc"]
            m = int(min(4, 1 + amp * 3.6)) if amp > 0.12 else 0
            bk = 0
            for b in blinks:
                if b <= t <= b + 0.18:
                    ph = (t - b) / 0.18
                    bk = 1 if 0.25 < ph < 0.75 else 0.4
                    break
            hd = A["heads"][(m, bk)]
            d = ImageDraw.Draw(fr)
            ccx, ccy, r = L["circle"]
            rr = r + 12 * sc / 0.62 + amp * 14
            d.ellipse([ccx - rr, ccy - rr, ccx + rr, ccy + rr], outline=(178, 214, 206), width=4)
            breath = math.sin(t * 2.1) * 3 * sc
            by = int(C["base"] + breath * 0.5)
            fr.alpha_composite(A["body"], (C["cx"] - A["body"].width // 2, by))
            ang = math.sin(t * 1.3) * 2.2 + amp * 2.5 * math.sin(t * 6)
            hr = hd.rotate(ang, resample=Image.BICUBIC, center=(hd.width / 2, hd.height * 0.8))
            fr.alpha_composite(hr, (C["cx"] - hd.width // 2, int(by - 459 * sc - amp * 6 * sc + breath)))

        def frame(i):
            t = i / FPS
            fr = A["bg"].copy()
            character(fr, i, t)
            fr.alpha_composite(A["card"])
            if t < intro_end:
                k = ease(t / 0.6) * (1 - ease((t - (intro_end - 0.35)) / 0.35))
                fr.alpha_composite(fade(intro(), k), (0, int(30 * (1 - ease(t / 0.6)))))
            elif t >= end_start:
                k = ease((t - end_start) / 0.6)
                fr.alpha_composite(fade(outro(), k), (0, int(30 * (1 - k))))
            else:
                sc = scene_at(t)
                if sc:
                    nxt = [s["start"] for s in scenes if s["start"] > sc["start"]] + [end_start]
                    k = ease((t - sc["start"]) / 0.5)
                    a_ = k * (1 - ease((t - (nxt[0] - 0.3)) / 0.3))
                    fr.alpha_composite(fade(card_content(t, sc), a_), (int(40 * (1 - k)), 0))
            return fr.convert("RGB")

        if only_frames:
            for tt in only_frames:
                frame(int(tt * FPS)).save(f"{out_path}_{name}_{tt:.1f}.png")
            return out_path

        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
               "-i", "-", "-i", wav, "-af", f"apad=pad_dur={TAIL}", "-map", "0:v", "-map", "1:a",
               "-c:v", "libx264", "-preset", os.environ.get("X264_PRESET", "veryfast"), "-crf", "21",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-t", f"{dur:.2f}",
               "-movflags", "+faststart", out_path]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for i in range(nfr):
                p.stdin.write(frame(i).tobytes())
                if progress and i % 125 == 0:
                    progress(i / nfr)
        finally:
            p.stdin.close()
            p.wait()
        if p.returncode != 0:
            raise RuntimeError("ffmpeg xatosi")
    return out_path
