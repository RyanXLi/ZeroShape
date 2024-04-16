import pandas as pd
import numpy as np
import json
import trimesh
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

symm_gt = {}

# Function to process each file
def process_row(row, candidate_planes, threshold, subset):
    filename = row[0]  # Get the filename from the 0th column
    score = row[1:]  # Get the score from the other columns
    
    # Get the indices of the scores that are below the threshold
    indices = np.where(score < threshold)[0].tolist()

    # Get the center and scale of the object
    pc_fname = f"train_data/{subset}/pointclouds/{filename}"
    pc = np.load(pc_fname)
    trimeshpc = trimesh.PointCloud(pc)
    sphere = trimeshpc.bounding_sphere
    radius = (sphere.bounds[1] - sphere.bounds[0]) * 0.5
    radius = radius.tolist()[0]
    center = sphere.center.tolist()

    return filename, {
        "center": center,
        "radius": radius,
        "normal": candidate_planes[indices].tolist()
    }

# Load the data
with open("symm/candidate_planes.json") as json_file:
    candidate_planes = json.load(json_file)
candidate_planes = np.array(candidate_planes)
df_shapenet = pd.read_csv("symm/output_shapenet_pc_axis_v2.csv", header=None)
df_obja_lvis = pd.read_csv("symm/output_obja_lvis_pc_axis_v2.csv", header=None)

# Setup datasets
datasets = [{"df": df_shapenet, "threshold": 0.02, "subset": "ShapeNet55"}, {"df": df_obja_lvis, "threshold": 0.06, "subset": "objaverse_LVIS"}]

# Use ThreadPoolExecutor to parallelize processing
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for dataset in datasets:
        df = dataset["df"]
        threshold = dataset["threshold"]
        subset = dataset["subset"]
        for index, row in df.iterrows():
            futures.append(executor.submit(process_row, row, candidate_planes, threshold, subset))

    # Collecting results
    for future in tqdm(as_completed(futures), total=len(futures)):
        filename, data = future.result()
        symm_gt[filename] = data

# Save results
json.dump(symm_gt, open("symm/symm_gt.json", "w"))
