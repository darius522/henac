import os
import random
import shutil
from pathlib import Path

def build_audio_comparison_html(
    systems_dict,
    output_dir="comparison_output",
    n_samples=5,
    seed=None
):
    """
    systems_dict: dict
        {"S1": "/path/to/S1", "S2": "/path/to/S2"}
        
    Expects subfolders in each system as:
        input/
        output_<kbps>_<suffix>/
    """
    if seed is not None:
        random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Convert paths
    systems = {name: Path(path) for name, path in systems_dict.items()}

    # ---- Discover all kbps + suffixes ----
    # We collect: kbps -> { system_name : suffix }
    kbps_map = {}

    for sys_name, sys_path in systems.items():
        for folder in sys_path.iterdir():
            if not folder.is_dir():
                continue
            name = folder.name

            if name.lower() == "input":
                continue

            if not name.startswith("output_"):
                continue

            # Expect output_<kbps>_<suffix>
            parts = name.split("_", 2)
            if len(parts) != 3:
                raise ValueError(f"Invalid folder format: {name}")

            _, kbps, suffix = parts

            if kbps not in kbps_map:
                kbps_map[kbps] = {}
            kbps_map[kbps][sys_name] = suffix

    # Sort kbps keys for consistent display
    sorted_kbps = sorted(kbps_map.keys())

    # ---- Select input wavs ----
    first_sys = next(iter(systems.values()))
    input_wavs = list((first_sys / "input").glob("*.wav"))
    if len(input_wavs) < n_samples:
        raise ValueError("Not enough wav files in input folder.")

    selected_files = random.sample(input_wavs, n_samples)
    selected_names = [f.name for f in selected_files]

    # ---- Copy files to output folder ----
    for sys_name, sys_path in systems.items():
        for kbps, suffixes in kbps_map.items():
            folder = f"output_{kbps}_{suffixes[sys_name]}"
            src_folder = sys_path / folder

            if not src_folder.exists():
                raise ValueError(f"Missing folder: {src_folder}")

            for fname in selected_names:
                src = src_folder / fname
                if src.exists():
                    dest = output_dir / sys_name / folder
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest / fname)

    # Copy input reference files
    input_dest = output_dir / "input"
    input_dest.mkdir(exist_ok=True)
    for fname in selected_names:
        shutil.copy2(first_sys / "input" / fname, input_dest / fname)

    # ---- Generate HTML ----
    html = [
        "<html><head><title>Audio Comparison</title>",
        "<style>",
        "table{border-collapse:collapse;margin-bottom:40px;}",
        "td,th{border:1px solid #ccc;padding:6px;text-align:center;}",
        "audio{width:200px;}",
        "</style></head><body>",
        "<h1>Audio Comparison</h1>",
    ]

    for kbps in sorted_kbps:
        html.append(f"<h2>{kbps}</h2>")
        html.append("<table>")

        # Header row
        header = "<tr><th>File</th><th>Input</th>"
        for sys_name in systems.keys():
            suffix = kbps_map[kbps][sys_name]
            column_name = f"{sys_name}_{suffix}"
            header += f"<th>{column_name}</th>"
        header += "</tr>"
        html.append(header)

        # Rows
        for fname in selected_names:
            row = f"<tr><td>{fname}</td>"
            row += f"<td><audio controls src='input/{fname}'></audio></td>"

            for sys_name in systems.keys():
                suffix = kbps_map[kbps][sys_name]
                folder = f"output_{kbps}_{suffix}"
                rel_path = f"{sys_name}/{folder}/{fname}"
                row += f"<td><audio controls src='{rel_path}'></audio></td>"

            row += "</tr>"
            html.append(row)

        html.append("</table>")

    html.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(html), encoding="utf-8")

    print(f"✔ HTML written to: {output_dir/'index.html'}")
    print(f"✔ Copied audio files to: {output_dir}")


# Example usage:
systems = {
    "DAC": "/N/slate/daripete/jstsp-dac/runs_32khz/baseline_31cb_large_fr_80/300k/DAC",
    "HENAC": "/N/slate/daripete/jstsp-dac/runs_32khz/hb_16cb_4_1cb_2_1cb_fr_80_320_500/300k/HENAC",
}
build_audio_comparison_html(systems, output_dir="comparison_output", n_samples=8, seed=3)
