import os
import random
import shutil
from pathlib import Path

def build_audio_comparison_html(systems_dict, output_dir="comparison_output", n_samples=5, seed=None):
    """
    systems_dict: dict
        e.g. {"S1": "/path/to/S1", "S2": "/path/to/S2"}
    output_dir: str or Path
        Folder where the HTML and copied audio files will be placed.
    n_samples: int
        Number of wav files to randomly select for comparison.
    seed: int or None
        Optional random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Convert all paths to Path objects
    systems = {name: Path(path) for name, path in systems_dict.items()}

    # Assume all systems share the same subfolder structure
    first_sys_path = next(iter(systems.values()))
    subfolders = [d.name for d in first_sys_path.iterdir() if d.is_dir()]
    kbps_folders = sorted([f for f in subfolders if f.lower() != "input"])

    # Collect wav files from input folder of the first system
    input_wavs = list((first_sys_path / "input").glob("*.wav"))
    if len(input_wavs) < n_samples:
        raise ValueError(f"Not enough WAV files in input/ (found {len(input_wavs)}, need {n_samples}).")

    selected_files = random.sample(input_wavs, n_samples)
    selected_names = [f.name for f in selected_files]

    # Copy files into comparison_output preserving structure
    for sys_name, sys_path in systems.items():
        for folder in ["input"] + kbps_folders:
            src_folder = sys_path / folder
            if not src_folder.exists():
                print(src_folder, "does not exist, skipping.")
                continue
            for fname in selected_names:
                src = src_folder / fname
                if not src.exists():
                    continue
                dest = output_dir / sys_name / folder
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / fname)

    # Also copy input (from the first system) to output/input
    input_dest = output_dir / "input"
    input_dest.mkdir(exist_ok=True)
    for fname in selected_names:
        shutil.copy2(first_sys_path / "input" / fname, input_dest / fname)

    # Generate HTML
    html = [
        "<html><head><title>Audio Comparison</title>",
        "<style>",
        "table{border-collapse:collapse;margin-bottom:40px;}",
        "td,th{border:1px solid #ccc;padding:6px;text-align:center;}",
        "audio{width:200px;}",
        "body{font-family:Arial, sans-serif;margin:40px;}",
        "h1,h2{font-family:Arial, sans-serif;}",
        "</style></head><body>",
        f"<h1>Audio Comparison for {', '.join(systems.keys())}</h1>"
    ]

    for kbps in kbps_folders:
        html.append(f"<h2>{kbps}</h2>")
        html.append("<table>")
        html.append("<tr><th>File</th><th>Input</th>" +
                    "".join(f"<th>{sys_name}</th>" for sys_name in systems.keys()) +
                    "</tr>")

        for fname in selected_names:
            row = f"<tr><td>{fname}</td>"
            # Input column
            row += f"<td><audio controls src='input/{fname}'></audio></td>"
            # Systems columns
            for sys_name in systems.keys():
                rel_path = f"{sys_name}/{kbps}/{fname}"
                row += f"<td><audio controls src='{rel_path}'></audio></td>"
            row += "</tr>"
            html.append(row)

        html.append("</table>")

    html.append("</body></html>")

    html_path = output_dir / "index.html"
    html_path.write_text("\n".join(html), encoding="utf-8")

    print(f"✅ HTML written to: {html_path}")
    print(f"✅ Copied {n_samples} random audio files into: {output_dir}")


systems = {
    "DAC": "/N/slate/daripete/jstsp-dac/runs_32khz/baseline_32_1cb_large_fr_80/300k/DAC",
    "HENAC": "/N/slate/daripete/jstsp-dac/runs_32khz/hb_32_1cb_2_1cb_1cb_fr_80/300k/HENAC",
}
build_audio_comparison_html(systems, output_dir="comparison_output", n_samples=8, seed=40)
