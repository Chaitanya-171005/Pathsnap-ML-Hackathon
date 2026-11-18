"""Local maps of observed GPS points, directed network arcs and reference geometry."""

from html import escape
from pathlib import Path
import re

import folium
import numpy as np

from src.preprocessing.loaders import Trip
from src.preprocessing.validation import TripValidity


GPS_COLOR = "#1761c0"
ROAD_COLOR = "#59616a"
ROUTE_COLOR = "#008048"


def write_map(trip: Trip, valid: TripValidity, category: str, output: Path,
              road_limit: int) -> None:
    gps = trip.track[valid.gps, :2]
    road_ids = np.flatnonzero(valid.arc_geometry)
    route_ids = trip.route[valid.route_geometry]
    shown_ids = road_ids[np.linspace(0, len(road_ids) - 1,
                                    min(road_limit, len(road_ids)), dtype=int)]
    anchor = gps if len(gps) else trip.nodes[valid.nodes]
    if not len(anchor):
        return
    route = trip.nodes[trip.arcs[route_ids]]
    roads = trip.nodes[trip.arcs[shown_ids]]
    result = folium.Map(location=np.median(anchor, axis=0)[::-1].tolist(), zoom_start=12,
                        control_scale=True)
    road_layer = folium.FeatureGroup(name="Road network", show=True).add_to(result)
    if len(roads):
        folium.PolyLine(roads[:, :, ::-1].tolist(), color=ROAD_COLOR,
                        weight=1, opacity=0.5).add_to(road_layer)
    route_layer = folium.FeatureGroup(name="Reference route", show=True).add_to(result)
    if len(route):
        # A multi-polyline never draws a fabricated connector between route arcs.
        folium.PolyLine(route[:, :, ::-1].tolist(), color=ROUTE_COLOR,
                        weight=4, opacity=0.88).add_to(route_layer)
    gps_layer = folium.FeatureGroup(name="GPS observations", show=True).add_to(result)
    if len(gps):
        # GeoJSON uses longitude/latitude. One feature avoids per-point HTML overhead.
        folium.GeoJson(
            {"type": "Feature", "geometry": {"type": "MultiPoint", "coordinates": gps.tolist()},
             "properties": {}},
            marker=folium.CircleMarker(radius=2, color=GPS_COLOR, fill=True,
                                       weight=1, fill_opacity=0.7),
        ).add_to(gps_layer)
    focus = np.concatenate((anchor, route.reshape(-1, 2)))
    result.fit_bounds([focus.min(axis=0)[::-1].tolist(), focus.max(axis=0)[::-1].tolist()],
                      max_zoom=17)
    folium.LayerControl().add_to(result)

    title = f"{category.title()} training trip {trip.files.trip_id}"
    result.get_root().header.add_child(folium.Element(f"<title>{escape(title)}</title>"))
    legend = f"""
    <aside style="position:fixed;bottom:24px;left:16px;z-index:1000;background:white;
                 color:#182128;padding:12px 16px;border:1px solid #aeb4b8;border-radius:6px;
                 font:14px/1.55 system-ui,sans-serif">
      <strong>{escape(title)}</strong><br>
      <span style="color:{GPS_COLOR}">●</span> GPS observations ({len(gps):,})<br>
      <span style="color:{ROAD_COLOR}">━━</span> Road arcs ({len(roads):,} of {len(road_ids):,} with geometry)<br>
      <span style="color:{ROUTE_COLOR}">━━</span> Reference arcs ({len(route):,})
    </aside>
    """
    result.get_root().html.add_child(folium.Element(legend))

    # Folium generates random element IDs. Renaming consistently makes reruns comparable.
    names: dict[str, str] = {}

    def stable_name(match: re.Match) -> str:
        name = match[0]
        return names.setdefault(name, f"{name.rsplit('_', 1)[0]}_pathsnap_{len(names)}")

    html = re.sub(r"\b[a-z][a-z_]*_[0-9a-f]{32}(?![0-9a-f])",
                  stable_name, result.get_root().render())
    output.write_text(html, encoding="utf-8")
