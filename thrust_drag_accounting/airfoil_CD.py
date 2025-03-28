import pandas as pd
from scipy.interpolate import LinearNDInterpolator
import os
base_dir = os.path.dirname(__file__)

def parse_xfoil_file(filepath, delta_e):
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Find start of data
    for i, line in enumerate(lines):
        if line.strip().startswith("alpha"):
            start_idx = i + 2
            break

    data = []
    for line in lines[start_idx:]:
        if line.strip() == "":
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        # Only extract alpha, CL, CD, CDp, CM
        row = [float(parts[j]) for j in range(5)] + [float(delta_e)]
        data.append(row)

    df = pd.DataFrame(data, columns=["alpha", "CL", "CD", "CDp", "CM", "delta_e"])
    return df

def build_interpolators(df):
    points = df[["alpha", "delta_e"]].values
    cl_values = df["CL"].values
    cd_values = df["CD"].values

    cl_interp = LinearNDInterpolator(points, cl_values)
    cd_interp = LinearNDInterpolator(points, cd_values)

    return cl_interp, cd_interp


file_paths = {
    "+10": os.path.join(base_dir, "aseq_-10_10_+10de.txt"),
    "0":   os.path.join(base_dir, "aseq_-10_10_0de.txt"),
    "-10": os.path.join(base_dir, "aseq_-10_10_-10de.txt"),
}

dfs = [
    parse_xfoil_file(file_paths["+10"], 10),
    parse_xfoil_file(file_paths["0"], 0),
    parse_xfoil_file(file_paths["-10"], -10),
]

full_df = pd.concat(dfs, ignore_index=True)

xfoil_airfoil_CL, xfoil_airfoil_CD = build_interpolators(full_df)


