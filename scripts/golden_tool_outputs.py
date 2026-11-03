"""Records exactly what the agent's tools present to the model -- each tool's
name, description and argument schema -- and exactly what they return for a
fixed set of calls, with the network faked. test_tools_golden.py compares a
fresh run against testdata/tool_outputs.json.

Why this exists: C1 moves the tools' logic out of main.py into a shared
service layer, so the console and the agent run the same code. The model
must not be able to tell. Byte-identical names, descriptions, schemas and
outputs before and after the move is what "the refactor changed nothing the
model sees" means, measured instead of assumed.

Runs in its own process against its own throwaway database, so the result
doesn't depend on test order or on where the code lives.

    uv run python scripts/golden_tool_outputs.py            # print the snapshot
    uv run python scripts/golden_tool_outputs.py --update   # rewrite testdata/tool_outputs.json
    uv run python scripts/golden_tool_outputs.py --perturb  # the negative control: altered upstream data
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "testdata" / "tool_outputs.json"

FORECAST = {"daily": {"time": ["2026-11-02", "2026-11-03"], "temperature_2m_max": [21.1, 18.2],
                      "temperature_2m_min": [6.0, -0.5], "precipitation_sum": [0.0, 4.2]}}
FAO_CL = [
    {"lastUpdated": "26/02/2025", "crop": {"name": "Potato"}, "aez": {"name": "Mediterranean"},
     "sessions": [{"early_sowing": {"day": 15, "month": 11}, "later_sowing": {"day": 15, "month": 2},
                   "early_harvest": {"day": 15, "month": 3}, "late_harvest": {"day": 15, "month": 6}}]},
    {"lastUpdated": "26/02/2025", "crop": {"name": "Wheat, bread"}, "aez": {"name": "Moist Temperate"},
     "sessions": [{"early_sowing": {"day": "1", "month": 5}, "later_sowing": {"day": "31", "month": 7},
                   "early_harvest": {"day": 1, "month": 10}, "late_harvest": {"day": 31, "month": 1}}]},
]
GEOCODE = {"results": [{"name": "Talca", "admin1": "Maule", "country": "Chile",
                        "latitude": -35.43, "longitude": -71.67}]}

CALLS = [
    ("query_field_status", {"field_id": "field-1"}),
    ("query_field_status", {"field_id": "field-2"}),
    ("query_field_status", {"field_id": "field-99"}),
    ("log_task", {"field_id": "field-2", "description": 'irrigate "north" rows'}),
    ("log_task", {"field_id": "field-1", "description": "inspect for blight"}),
    ("get_pending_tasks", {}),
    ("get_weather_forecast", {"location": "field-1", "days": 2}),
    ("get_weather_forecast", {"location": "Talca", "days": 99}),
    ("get_weather_forecast", {"location": "Atlantis"}),
    ("lookup_crop_calendar", {"crop": "potatoes", "region": "central Chile"}),
    ("lookup_crop_calendar", {"crop": "wheat", "region": "southern Chile"}),
    ("lookup_crop_calendar", {"crop": "kiwi", "region": "central Chile"}),
    ("lookup_crop_calendar", {"crop": "potato", "region": "Midwest USA"}),
]


class _Response:
    def __init__(self, payload: object, status: int = 200) -> None:
        self.payload, self.status = payload, status

    def json(self) -> object:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status >= 400:
            import httpx

            raise httpx.HTTPStatusError("boom", request=None, response=None)  # type: ignore[arg-type]


def snapshot(perturb: bool = False) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="agrios-golden-") as tmp:
        os.environ["DATABASE_URL"] = f"sqlite:///{Path(tmp) / 'agrios.db'}"
        sys.path.insert(0, str(REPO))
        import httpx

        import main

        forecast = json.loads(json.dumps(FORECAST))
        if perturb:  # a 1 °C warmer night: frost_risk must flip, so the comparison must notice
            forecast["daily"]["temperature_2m_min"][1] = 2.5

        def fake_get(url: str, params: dict[str, object] | None = None, timeout: float | None = None) -> _Response:
            if url == main.FAO_CROP_CALENDAR:
                return _Response(FAO_CL)
            if url == main.OPEN_METEO_GEOCODE:
                return _Response({} if (params or {}).get("name") == "Atlantis" else GEOCODE)
            return _Response(forecast)

        httpx.get = fake_get  # type: ignore[assignment]
        schemas = [
            {"name": t.name, "description": t.description, "args": t.args} for t in main.tools
        ]
        outputs = [
            {"tool": name, "args": args, "output": main.tools_by_name[name].invoke(args)} for name, args in CALLS
        ]
        main.engine.dispose()  # release the SQLite file so the temp dir can be removed on Windows
        return {"tools": schemas, "calls": outputs}


def main_() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--perturb", action="store_true")
    args = parser.parse_args()
    data = json.dumps(snapshot(args.perturb), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.update:
        GOLDEN.write_text(data, encoding="utf-8")
        print(f"wrote {GOLDEN.relative_to(REPO)}")
    else:
        sys.stdout.write(data)
    return 0


if __name__ == "__main__":
    sys.exit(main_())
