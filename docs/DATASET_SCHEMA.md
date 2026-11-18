# PathSnap dataset schema

Keep the organiser-supplied data in `data/raw/train/<trip_id>/` and `data/raw/student_test/<trip_id>/`. Each directory contains four headerless whitespace-separated files with the directory name as the filename stem. The loaders preserve zero padding and use the partition and string trip ID as the trip key. Row IDs are local to each trip and each file type.

| File | Columns, in order | NumPy shape and type | Identifier |
|---|---|---|---|
| `.track` | Longitude (°), latitude (°), timestamp (s) | `(n, 3)` float64 | Observation ID is the zero-based row index |
| `.nodes` | Longitude (°), latitude (°) | `(n, 2)` float64 | Node ID is the zero-based row index |
| `.arcs` | From-node ID, to-node ID | `(n, 2)` int64 | Arc ID is the zero-based row index |
| `.route` | Reference arc ID | `(n,)` int64 | Sequence position is the zero-based row index |

The four loaders in `src.preprocessing.loaders` return `(array, issues)`. `load_trip` returns the four arrays, the trip metadata and the combined parse issues. Arrays preserve source order. Numeric fields that cannot be parsed are represented by NaN for floating-point data or −1 for integer data. Short rows are padded at the end and reported. Rows with extra fields are represented by a full sentinel row because column identity is uncertain. Blank physical lines inside a file are malformed rows. A normal final newline adds no row. Missing, unreadable and empty files return empty arrays with the appropriate shape and a file-level issue. Present values such as infinity and out-of-range geographic coordinates are checked separately by `validate_trip`.

Coordinates are stored as longitude, latitude in the loader and GeoJSON. Folium's direct polyline API receives latitude, longitude. The source docs describe geographic degrees and seconds. The analysis relies on the timestamp differences and does not assign a wall-clock epoch.

The local data contain 80 training and 10 student-testing trip directories, each with all four required files. No filename, field-count or type discrepancies were observed. The measured share of positive adjacent intervals at exactly one second is 96.50% for training and 98.62% for student testing, consistent with the guideline's approximate 1 Hz description. The training GPS row count ranges from 361 to 19,313, and the student-testing range is 262 to 3,958.

Source: organiser's *Dataset Exploration*, sections 2–7, and the local [dataset inventory](../reports/eda/dataset_inventory.csv).
