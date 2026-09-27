"""Contrast and theme-coherence tests.

These exist because a light ground was introduced while the dark theme's ink
classes were left in place, and the result was ``text-white`` on cream at
**1.08:1** — invisible. It shipped because nothing rendered the page and nothing
checked the numbers.

So the numbers are now checked. Every text token must clear a contrast floor
against the surfaces it is used on, and the dark theme's ink classes must not
appear in the markup at all.
"""

import re
import unittest
from pathlib import Path

CLIENT = Path(__file__).resolve().parents[1] / "client"
CSS = (CLIENT / "app" / "globals.css").read_text()
PAGE = (CLIENT / "app" / "page.tsx").read_text()

# Written out rather than pulled from a library, because the point is that the
# rule is legible to whoever changes the palette next.
CONTEXT_SURFACES = {
    "paper": ("#f7f6f2", "the page ground"),
    "white": ("#ffffff", "a card surface"),
}

# The floor each role has to clear. 4.5:1 is AA for body text; 7:1 is AAA and is
# what small uppercase labels are held to, because they are the worst case.
FLOORS = {
    "--ink": 7.0,
    "--ink-2": 4.5,
    "--ink-3": 4.5,
    "--accent": 4.5,
    "--caution": 4.5,
}

# The dark theme's inks. None of these may appear in the markup: they were
# designed for a near-black ground and are unreadable on paper.
FORBIDDEN = [
    "text-white",
    "text-slate-100",
    "text-slate-200",
    "text-slate-300",
    "text-slate-400",
    "text-slate-500",
    "text-gray-300",
    "text-gray-400",
    "bg-slate-900",
    "bg-slate-950",
    "bg-slate-800",
    "bg-black/20",
    "bg-white/10",
]


def _channel(value: int) -> float:
    value = value / 255
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    red, green, blue = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(red) + 0.7152 * _channel(green) + 0.0722 * _channel(blue)


def contrast(foreground: str, background: str) -> float:
    a, b = luminance(foreground), luminance(background)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


def token(name: str) -> str:
    match = re.search(rf"^\s*{re.escape(name)}:\s*(#[0-9a-fA-F]{{6}})", CSS, re.M)
    if not match:
        raise AssertionError(f"{name} is not declared in globals.css")
    return match.group(1)


class PaletteContrastTests(unittest.TestCase):
    def test_every_text_token_clears_its_floor_on_both_surfaces(self):
        for name, floor in FLOORS.items():
            colour = token(name)
            for surface_name, (surface, why) in CONTEXT_SURFACES.items():
                with self.subTest(token=name, surface=surface_name):
                    ratio = contrast(colour, surface)
                    self.assertGreaterEqual(
                        ratio, floor,
                        f"{name} ({colour}) on {why} ({surface}) is {ratio:.2f}:1, "
                        f"below the {floor}:1 floor",
                    )

    def test_the_paper_ground_is_actually_light(self):
        # A "light theme" token that is dark is the whole bug in one assertion.
        self.assertGreaterEqual(luminance(token("--paper")), 0.85)

    def test_ink_hierarchy_is_ordered(self):
        # --ink is the darkest and --ink-3 the lightest, so luminance must ascend.
        # A palette where the small uppercase label is darker than the body text
        # has inverted the hierarchy and will read as emphasis.
        steps = [luminance(token(n)) for n in ("--ink", "--ink-2", "--ink-3")]
        self.assertEqual(steps, sorted(steps), steps)

    def test_caution_is_distinguishable_from_body_ink(self):
        # Modelled and unconfirmed figures must be visually distinct, or the
        # distinction they carry is invisible.
        self.assertNotEqual(token("--caution"), token("--ink-2"))


class MarkupThemeTests(unittest.TestCase):
    def test_no_dark_theme_ink_classes_survive(self):
        found = {cls: PAGE.count(cls) for cls in FORBIDDEN if cls in PAGE}
        self.assertEqual(found, {}, f"unreadable on a light ground: {found}")

    def test_every_text_utility_is_one_of_our_tokens(self):
        used = set(re.findall(r"\btext-ink(?:-\d)?\b", PAGE))
        allowed = {"text-ink", "text-ink-2", "text-ink-3"}
        self.assertTrue(used <= allowed, f"undeclared text utilities: {used - allowed}")

    def test_figures_use_the_monospaced_class(self):
        # The rule that carries the most weight: every figure is tabular.
        self.assertIn(".fig", CSS)
        self.assertIn("font-variant-numeric: tabular-nums", CSS)

    def test_modelled_and_unconfirmed_get_their_own_treatment(self):
        for status in ("observed", "derived", "modelled", "unconfirmed"):
            with self.subTest(status=status):
                self.assertIn(f".status-{status}", CSS)
        # Modelled data must be marked in a colour of its own, not merely labelled.
        self.assertIn("status-modelled", CSS)
        self.assertIn("var(--caution)", CSS)


if __name__ == "__main__":
    unittest.main()
