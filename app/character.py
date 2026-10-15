from PIL import Image, ImageDraw, ImageFilter

S = 3  # supersampling
SKIN = (246, 205, 176)
SKIN_SH = (232, 182, 150)
HAIR = (58, 36, 30)
HAIR_HI = (92, 60, 48)
EYE = (40, 28, 26)
LIP = (196, 82, 92)
MOUTH_IN = (110, 30, 45)
TONGUE = (228, 112, 120)
TOP = (30, 111, 104)
TOP_DK = (22, 86, 80)
GOLD = (227, 169, 59)


def _e(d, cx, cy, rx, ry, fill, s=S, **kw):
    d.ellipse([(cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry) * s], fill=fill, **kw)


def head(mouth=0, blink=0.0):
    """mouth: 0..4 openness, blink 0 (open) .. 1 (closed). Returns RGBA 520x620."""
    W, H = 520, 620
    im = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    # hair back + bun
    _e(d, 260, 300, 205, 220, HAIR)
    _e(d, 260, 92, 82, 74, HAIR)
    _e(d, 238, 72, 30, 20, HAIR_HI)
    # bun band
    d.rounded_rectangle([212 * S, 150 * S, 308 * S, 172 * S], radius=10 * S, fill=(200, 50, 60))
    # ears
    _e(d, 95, 350, 26, 36, SKIN_SH)
    _e(d, 425, 350, 26, 36, SKIN_SH)
    _e(d, 95, 392, 8, 8, GOLD)
    _e(d, 425, 392, 8, 8, GOLD)
    # face
    _e(d, 260, 335, 168, 195, SKIN)
    # fringe: swept dome, clipped to head shape
    fm = Image.new("L", im.size, 0)
    fd = ImageDraw.Draw(fm)
    fd.chord([86 * S, 108 * S, 434 * S, 380 * S], 180, 360, fill=255)
    fd.chord([170 * S, 200 * S, 520 * S, 340 * S], 180, 360, fill=0)
    fd.pieslice([86 * S, 108 * S, 434 * S, 380 * S], 180, 215, fill=255)
    hm = Image.new("L", im.size, 0)
    _e(ImageDraw.Draw(hm), 260, 300, 205, 220, 255)
    from PIL import ImageChops
    fm = ImageChops.multiply(fm, hm)
    im.paste(Image.new("RGBA", im.size, HAIR), (0, 0), fm)
    # brows
    for cx, sgn in ((192, -1), (328, 1)):
        d.arc([(cx - 40) * S, 282 * S, (cx + 40) * S, 320 * S], 200, 340, fill=HAIR, width=8 * S)
    # eyes
    for cx in (192, 328):
        if blink < 0.5:
            ry = 27 * (1 - blink * 1.6)
            _e(d, cx, 352, 22, max(ry, 4), EYE)
            if ry > 10:
                _e(d, cx + 7, 342, 7, 7, (255, 255, 255))
                _e(d, cx - 6, 362, 3, 3, (255, 255, 255))
        else:
            d.arc([(cx - 24) * S, 330 * S, (cx + 24) * S, 368 * S], 20, 160, fill=EYE, width=6 * S)
        # lashes
        if blink < 0.5:
            d.line([((cx + 18) * S, 334 * S), ((cx + 30) * S, 324 * S)], fill=EYE, width=5 * S)
    # nose
    d.arc([244 * S, 382 * S, 276 * S, 412 * S], 20, 160, fill=SKIN_SH, width=6 * S)
    # mouth
    mx, my = 260, 458
    if mouth <= 0:
        d.arc([(mx - 42) * S, (my - 30) * S, (mx + 42) * S, (my + 14) * S], 20, 160, fill=LIP, width=8 * S)
    else:
        w = 30 + 5 * mouth
        h = 6 + 9 * mouth
        m = Image.new("L", im.size, 0)
        md = ImageDraw.Draw(m)
        md.chord([(mx - w) * S, (my - h * 0.55) * S, (mx + w) * S, (my + h) * S], 0, 360, fill=255)
        layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        _e(ld, mx, my + h * 0.22, w, h * 0.78, MOUTH_IN)
        _e(ld, mx, my + h * 0.95, w * 0.62, h * 0.5, TONGUE)
        ld.rectangle([(mx - w) * S, (my - h) * S, (mx + w) * S, (my - h * 0.55 + 7 + mouth) * S], fill=(255, 255, 255))
        im.paste(layer, (0, 0), Image.composite(layer, Image.new("RGBA", im.size), m).split()[3])
        d.ellipse([(mx - w) * S, (my - h * 0.56) * S, (mx + w) * S, (my + h) * S], outline=LIP, width=6 * S)
    # blush
    bl = Image.new("RGBA", im.size, (0, 0, 0, 0))
    bd = ImageDraw.Draw(bl)
    _e(bd, 160, 418, 30, 18, (240, 120, 120, 90))
    _e(bd, 360, 418, 30, 18, (240, 120, 120, 90))
    im = Image.alpha_composite(im, bl)
    return im.resize((W, H), Image.LANCZOS)


def body():
    W, H = 760, 520
    im = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    # long hair behind shoulders
    d.rounded_rectangle([200 * S, 0, 560 * S, 230 * S], radius=120 * S, fill=HAIR)
    # neck
    d.rounded_rectangle([318 * S, 0, 442 * S, 170 * S], radius=40 * S, fill=SKIN_SH)
    # torso
    d.rounded_rectangle([60 * S, 120 * S, 700 * S, 900 * S], radius=230 * S, fill=TOP)
    # collar V
    d.polygon([(300 * S, 122 * S), (460 * S, 122 * S), (380 * S, 250 * S)], fill=SKIN_SH)
    d.line([(296 * S, 120 * S), (380 * S, 256 * S), (464 * S, 120 * S)], fill=TOP_DK, width=14 * S, joint="curve")
    # atlas-like stripes accent on shoulders
    for i, c in enumerate([(227, 169, 59), (200, 50, 60), (227, 169, 59)]):
        y = 300 + i * 26
        d.rounded_rectangle([90 * S, y * S, 230 * S, (y + 12) * S], radius=6 * S, fill=c)
        d.rounded_rectangle([530 * S, y * S, 670 * S, (y + 12) * S], radius=6 * S, fill=c)
    return im.resize((W, H), Image.LANCZOS)


if __name__ == "__main__":
    bg = Image.new("RGBA", (1600, 680), (247, 241, 232, 255))
    for i, (m, b) in enumerate([(0, 0), (1, 0), (2, 0), (4, 0), (0, 1)]):
        h = head(m, b).resize((300, 358))
        bg.alpha_composite(h, (i * 320, 0))
    bd = body()
    bg.alpha_composite(bd.resize((380, 260)), (10, 400))
    bg.convert("RGB").save("char_preview.png")
