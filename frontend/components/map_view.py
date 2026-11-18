"""Canvas-rendered map layers. Never connect independent selections as a route."""
from html import escape
import folium
import streamlit as st
from streamlit.components.v1 import html
from frontend.services.prediction_adapter import classifier_selections, has_route

GPS_COLOR = "#39c8ff"
ROAD_COLOR = "#7891ad"
CLASSIFIER_COLOR = "#f6b85b"
ROUTE_COLOR = "#32e6bb"


def route_lines(result):
    """Return separate backend lines. Never flatten disconnected segments."""
    if result and result.phase5:
        return [(f"Segment {feature.properties.segment_id} · {feature.properties.arc_count} directed arcs"
                 f" · break: {feature.properties.break_reason or 'none'}", feature.geometry.coordinates)
                for feature in result.phase5.route_geojson.features]
    if result and result.decoded_route:
        return [("SYNTHETIC route example · not HMM" if result.decoded_route.provenance == "synthetic_demo"
                 else "Backend HMM / Viterbi decoded route", result.decoded_route.coordinates)]
    return []


def latlon(coordinate):
    return coordinate[1], coordinate[0]


def display_indices(count, limit):
    """Evenly spaced inspection points, including both endpoints; input is untouched."""
    if count <= limit:
        return list(range(count))
    return sorted({round(i * (count - 1) / (limit - 1)) for i in range(limit)})


def build_map(geometry, result=None, layers=None, point_limit=150, road_limit=1500):
    layers = set(layers if layers is not None else ["GPS", "Road network", "Classifier selections", "Decoded route"])
    points = [latlon(obs.coordinate) for obs in geometry.observations]
    center = points[0] if points else (12.97, 77.60)
    view = folium.Map(location=center, zoom_start=15, tiles=None, prefer_canvas=True, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap (online)").add_to(view)
    # A no-tile option still displays synthetic vectors when tile servers are offline.
    folium.FeatureGroup(name="Vectors only", overlay=False, show=False).add_to(view)
    if points:
        view.fit_bounds(points)
    if "GPS" in layers and points:
        group = folium.FeatureGroup(name="Raw GPS · observation order").add_to(view)
        # Full trajectory kept; Leaflet smooth_factor is display-only simplification.
        folium.PolyLine(points, color=GPS_COLOR, weight=3, opacity=.8, smooth_factor=1).add_to(group)
        for i in display_indices(len(points), point_limit):
            obs = geometry.observations[i]
            popup = f"Observation {obs.observation_id}<br>Relative time: {obs.relative_timestamp_s if obs.relative_timestamp_s is not None else 'Not available'} s"
            folium.CircleMarker(points[i], radius=3, color=GPS_COLOR, fill=True, fill_opacity=.85, popup=popup).add_to(group)
        folium.Marker(points[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(group)
        folium.Marker(points[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(group)
    if "Road network" in layers:
        group = folium.FeatureGroup(name="Directed road network").add_to(view)
        for arc in geometry.arcs[:road_limit]:
            folium.PolyLine([latlon(c) for c in arc.coordinates], color=ROAD_COLOR, weight=2, opacity=.45,
                tooltip=f"Arc {arc.arc_id}: {arc.from_node} → {arc.to_node}").add_to(group)
    if result and "Classifier selections" in layers:
        group = folium.FeatureGroup(name="Classifier selections · independent arcs").add_to(view)
        arcs = {arc.arc_id: arc for arc in geometry.arcs}
        by_arc = {}
        for selection in classifier_selections(result):
            by_arc.setdefault(selection.arc_id, []).append(selection)
        for arc_id, selections in list(by_arc.items())[:road_limit]:
            arc = arcs.get(arc_id)
            if arc:
                sample = selections[:12]
                detail = "<br>".join(f"Observation {s.observation_id} · candidate rank {s.candidate_rank if s.candidate_rank is not None else 'N/A'} · raw p={s.probability if s.probability is not None else 'N/A'}" for s in sample)
                folium.PolyLine([latlon(c) for c in arc.coordinates], color=CLASSIFIER_COLOR, weight=5,
                    tooltip=f"Directed arc {arc_id} · {len(selections)} selections", popup=folium.Popup(detail, max_width=400)).add_to(group)
    if has_route(result) and "Decoded route" in layers:
        name = ("SYNTHETIC route example · not HMM" if result.synthetic else "Backend HMM / Viterbi decoded route")
        if result.phase5:
            name += f" · {result.phase5.status}"
        group = folium.FeatureGroup(name=name).add_to(view)
        for label, coordinates in route_lines(result):
            folium.PolyLine([latlon(c) for c in coordinates], color=ROUTE_COLOR, weight=7, opacity=.85,
                tooltip=escape(label)).add_to(group)
        if not points:
            bounds = [latlon(c) for _, coordinates in route_lines(result) for c in coordinates]
            view.fit_bounds(bounds)
    folium.LayerControl(collapsed=False).add_to(view)
    return view


def render_map(geometry, result=None, key="map"):
    if geometry.synthetic:
        st.info("SYNTHETIC DEMO · Invented coordinates, road network and predictions. No model inference is performed.")
    layers = st.multiselect("Visible map layers", ["GPS", "Road network", "Classifier selections", "Decoded route"],
        default=["GPS", "Road network", "Classifier selections", "Decoded route"], key=f"{key}-layers")
    with st.expander("Display performance & inspection"):
        point_limit = st.slider("Maximum GPS inspection markers", 25, 500, 150, key=f"{key}-points")
        road_limit = st.select_slider("Maximum displayed road arcs", [500, 1500, 5000, 15000], value=1500, key=f"{key}-roads")
        st.caption("Display limits affect markers and visible road arcs only. The full GPS line and inference inputs remain intact. Leaflet simplifies the line visually at each zoom.")
    if len(geometry.observations) > point_limit:
        st.caption(f"Inspecting {point_limit} evenly spaced markers of {len(geometry.observations):,} observations. Use the table to inspect every point.")
    if len(geometry.arcs) > road_limit:
        st.warning(f"Showing the first {road_limit:,} of {len(geometry.arcs):,} road arcs. Increase the display cap to see more.")
    selections = classifier_selections(result) if result else []
    if result and len({s.arc_id for s in selections}) > road_limit:
        st.warning(f"Classifier layer is capped at {road_limit:,} unique selected arcs for display.")
    if result and {s.arc_id for s in selections} - {a.arc_id for a in geometry.arcs}:
        st.warning("Some selected arcs have no backend geometry; their IDs remain available in the results table.")
    # Embed the complete Folium document: Leaflet handles pan/zoom/layer inspection
    # locally, without a Python rerun on every map interaction.
    html(build_map(geometry, result, layers, point_limit, road_limit).get_root().render(), height=590)
    st.caption("🔵 Raw GPS   ·   Grey: directed roads   ·   Amber: independent classifier selections   ·   Teal: decoded / labelled synthetic route")
    st.caption("Basemap tiles and Leaflet assets need internet access. Synthetic data generation works offline; vector-only maps work once Leaflet assets are loaded.")
    if result and result.phase5 and result.phase5.status == "disconnected":
        st.warning(f"Disconnected route · {len(result.phase5.route_segments)} separate segments. Gaps are preserved; no connecting lines are added.")
    if not has_route(result):
        st.caption("HMM / Viterbi route: Not available. Independent selections are never connected into a route.")
