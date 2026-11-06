"""What every page is checked for in the browser. Each check returns the
problems it found, so the negative control can show it finds them."""

from playwright.sync_api import Page

# A control has a name a screen reader can announce: a <label>, aria-label,
# aria-labelledby, its own text (buttons and links), or its value (submit and
# button inputs). Text marked aria-hidden isn't announced, so it doesn't count:
# an icon-only button with no label fails. Rendered means it has a box, which
# also covers position: fixed (offsetParent is null for those).
UNNAMED = """() => {
  const spoken = el => [...el.childNodes].map(n =>
    n.nodeType === Node.TEXT_NODE ? n.textContent
    : n.nodeType === Node.ELEMENT_NODE && n.getAttribute('aria-hidden') !== 'true' ? spoken(n) : ''
  ).join('').trim();
  return [...document.querySelectorAll('input, select, textarea, button, a[href]')]
    .filter(el => el.type !== 'hidden' && el.getClientRects().length > 0)
    .filter(el => {
      if (el.getAttribute('aria-label')?.trim()) return false;
      const by = el.getAttribute('aria-labelledby');
      const named = id => document.getElementById(id) && spoken(document.getElementById(id));
      if (by && by.split(' ').some(named)) return false;
      if (el.labels && [...el.labels].some(l => spoken(l))) return false;
      if ((el.tagName === 'BUTTON' || el.tagName === 'A') && spoken(el)) return false;
      const valued = ['submit', 'button', 'reset'].includes(el.type);
      if (el.tagName === 'INPUT' && valued && el.value.trim()) return false;
      return true;
    })
    .map(el => el.outerHTML.slice(0, 120));
}"""

FOCUSED = """() => {
  const el = document.activeElement;
  if (!el || el === document.body) return null;
  const style = getComputedStyle(el);
  return {
    html: el.outerHTML.slice(0, 100),
    visible: style.outlineStyle !== 'none' && parseFloat(style.outlineWidth) > 0,
  };
}"""


def overflow(page: Page) -> list[str]:
    width = page.viewport_size["width"] if page.viewport_size else 0
    scroll = int(page.evaluate("document.documentElement.scrollWidth"))
    return [f"wider than the screen: {scroll}px in {width}px"] if scroll > width else []


def unnamed_controls(page: Page) -> list[str]:
    return [f"no accessible name: {html}" for html in page.evaluate(UNNAMED)]


def keyboard(page: Page, max_stops: int = 60) -> tuple[list[str], list[str]]:
    """Tabs through a freshly loaded page from the top. Returns (problems,
    what was reached): every stop must show where the focus is (SC 2.4.7).
    Nothing is clicked first: a click moves where tabbing starts."""
    problems: list[str] = []
    reached: list[str] = []
    for _ in range(max_stops):
        page.keyboard.press("Tab")
        focused = page.evaluate(FOCUSED)
        if focused is None or (reached and focused["html"] == reached[0]):
            break  # wrapped round to the start
        reached.append(focused["html"])
        if not focused["visible"]:
            problems.append(f"no visible focus: {focused['html']}")
    return problems, reached
