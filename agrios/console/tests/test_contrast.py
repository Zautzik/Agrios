"""Colour contrast, computed from the stylesheet's tokens in both themes.

app.css claims its text pairs meet WCAG 2.2 AA. This holds it to that:
4.5:1 for text (SC 1.4.3), 3:1 for the focus ring against what it sits on
(SC 1.4.11). Computed from the tokens rather than measured in a browser, so
it covers every page using them, and the dark theme without emulating it.
"""

import re
from pathlib import Path

import pytest

CSS = (Path(__file__).resolve().parents[1] / "static" / "app.css").read_text(encoding="utf-8")

TEXT = 4.5
NON_TEXT = 3.0
# (foreground, background, minimum): every pairing the stylesheet uses.
PAIRS = [
    ("text", "bg", TEXT),
    ("text", "surface", TEXT),
    ("muted", "bg", TEXT),
    ("muted", "surface", TEXT),
    ("accent", "surface", TEXT),  # links
    ("accent", "bg", TEXT),
    ("accent-text", "accent", TEXT),  # primary buttons
    ("ok-text", "ok-bg", TEXT),  # saved messages, your chat bubble
    ("error-text", "error-bg", TEXT),
    ("error-text", "surface", TEXT),  # field errors, a failed tool step
    ("note-text", "note-bg", TEXT),  # demo banner, hints
    ("focus", "surface", NON_TEXT),
    ("focus", "bg", NON_TEXT),
]


def tokens(block: str) -> dict[str, str]:
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})\s*;", block))


def themes(css: str) -> dict[str, dict[str, str]]:
    light = tokens(css[css.index(":root {") : css.index("}", css.index(":root {"))])
    media = css.index("@media (prefers-color-scheme: dark)")
    dark_root = css.index(":root {", media)
    dark = tokens(css[dark_root : css.index("}", dark_root)])
    # The dark theme only overrides; anything it leaves out is the light value.
    return {"light": light, "dark": {**light, **dark}}


def luminance(hex_colour: str) -> float:
    """WCAG 2.2 relative luminance of an sRGB colour."""
    channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    r, g, b = linear
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def failures(css: str) -> list[str]:
    found = []
    for theme, palette in themes(css).items():
        for fg, bg, minimum in PAIRS:
            ratio = contrast(palette[fg], palette[bg])
            if ratio < minimum:
                found.append(f"{theme}: {fg} on {bg} is {ratio:.2f}:1, needs {minimum}:1")
    return found


def test_the_contrast_formula_matches_known_values():
    assert contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast("#777777", "#ffffff") == pytest.approx(4.48, abs=0.01)  # the classic near-miss grey


def test_every_pair_meets_wcag_aa_in_both_themes():
    palettes = themes(CSS)
    assert set(palettes["light"]) >= {fg for fg, _, _ in PAIRS} | {bg for _, bg, _ in PAIRS}
    assert palettes["dark"]["bg"] != palettes["light"]["bg"]  # the dark block was found
    assert failures(CSS) == []


def test_the_check_catches_a_low_contrast_token():
    # Negative control: the same stylesheet with muted text lightened in the
    # dark theme's opposite direction must fail.
    broken = CSS.replace("--muted: #545c50;", "--muted: #9aa096;", 1)
    assert broken != CSS
    assert any(f.startswith("light: muted on") for f in failures(broken))
