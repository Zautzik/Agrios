"""What every page is checked for in the browser. Each check returns the
problems it found, so the negative control can show it finds them."""

from playwright.sync_api import Page

# A control has a name a screen reader can announce: a <label>, aria-label,
# aria-labelledby, or (buttons and links) its own text. Hidden inputs and
# controls inside <template> don't count.
UNNAMED = """() => [...document.querySelectorAll('input, select, textarea, button, a[href]')]
  .filter(el => el.type !== 'hidden' && el.offsetParent !== null)
  .filter(el => {
    if (el.getAttribute('aria-label')) return false;
    const by = el.getAttribute('aria-labelledby');
    if (by && by.split(' ').some(id => document.getElementById(id)?.textContent.trim())) return false;
    if (el.labels && el.labels.length && [...el.labels].some(l => l.textContent.trim())) return false;
    if ((el.tagName === 'BUTTON' || el.tagName === 'A') && el.textContent.trim()) return false;
    return true;
  })
  .map(el => el.outerHTML.slice(0, 120))"""

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
