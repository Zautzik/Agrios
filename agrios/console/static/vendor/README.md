# Vendored browser libraries

Served from this directory rather than a CDN: the console works with no internet access, and no third party sees when it's used. `agrios/console/tests/test_vendor.py` checks each file against the hash below, so a vendored file can't change without this record changing too.

| File | Upstream | Licence | SHA-256 |
|---|---|---|---|
| `htmx-2.0.11.min.js` | [htmx.org 2.0.11](https://www.npmjs.com/package/htmx.org/v/2.0.11), `dist/htmx.min.js`, fetched from cdn.jsdelivr.net on 2026-11-04 | 0BSD ([htmx-LICENSE.txt](htmx-LICENSE.txt)) | `d6fdc75f204e6bdefa99b69bf1e6d4ac69b8a364f77929f45c13476b4000f717` |

Alpine.js (MIT) joins this table with the first screen that needs it (the calibration canvas).
