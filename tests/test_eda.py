import csv

import pytest

from src.eda.run_eda import run


def test_eda_writes_stable_reports_and_real_layers(tmp_path):
    data = tmp_path / "raw"
    reports = tmp_path / "reports"
    plots = tmp_path / "plots"
    for trip_id, count in (("00001", 2), ("00002", 3), ("00003", 5)):
        trip = data / "train" / trip_id
        trip.mkdir(parents=True)
        (trip / f"{trip_id}.track").write_text(
            "".join(f"122 48.0{i} {i}\n" for i in range(count)))
        (trip / f"{trip_id}.nodes").write_text("122 48\n122 48.01\n122 48.02\n")
        (trip / f"{trip_id}.arcs").write_text("0 1\n1 2\n")
        (trip / f"{trip_id}.route").write_text("1\n0\n" if count == 5 else "0\n1\n")
    raw = {path: path.read_bytes() for path in data.rglob("*.*")}

    run(data, reports, plots, (5, 10), 55, 1)
    with (reports / "dataset_inventory.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["trip_id"] for r in rows] == ["00001", "00002", "00003"]
    assert [r["track_rows"] for r in rows] == ["2", "3", "5"]
    assert len(list(reports.glob("*.csv"))) == 5
    assert len(list(plots.glob("*.png"))) == 8
    assert len(list(plots.glob("*.html"))) == 3
    html = (plots / "training_short.html").read_text()
    assert "GPS observations" in html and "Road network" in html and "Reference route" in html
    assert '"coordinates": [[122.0, 48.0], [122.0, 48.01]]' in html
    assert "[[[48.0, 122.0], [48.01, 122.0]], [[48.01, 122.0], [48.02, 122.0]]]" in html
    assert "Road arcs (1 of 2 with geometry)" in html

    outputs = {path: path.read_bytes() for directory in (reports, plots) for path in directory.iterdir()}
    run(data, reports, plots, (5, 10), 55, 1)
    assert all(path.read_bytes() == content for path, content in outputs.items())
    assert all(path.read_bytes() == content for path, content in raw.items())


def test_output_path_cannot_write_inside_raw(tmp_path):
    with pytest.raises(ValueError, match="outside the input"):
        run(tmp_path / "raw", tmp_path / "raw" / "reports", tmp_path, (5,), 55, 10)
