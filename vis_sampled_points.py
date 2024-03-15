import trimesh
import numpy as np
import pyvista as pv



# fname = f"{category}/{category}_{object_name}"
# gt_fname = f"{self.path}/{subset}/gt_sdf/{fname}.npy"
gt_fname = "/home/ryan/work/data/processed_final_versions_synthetic/objaverse_LVIS/gt_sdf/aerosol_can/aerosol_can_1c74e24e7ce14cce982e9f2bd7857a8e.npy"
gt_dict = np.load(gt_fname, allow_pickle=True).item()
gt_dict['sample_pt']
gt_sample_points = gt_dict['sample_pt']
gt_sample_sdf = gt_dict['sample_sdf'] - 0.003

cloud = pv.PolyData(gt_sample_points)

# Plot the point cloud
p = pv.Plotter()
# p.open_movie('vis.mp4')
p.add_points(cloud, color='blue', point_size=5)
# p.show()
p.show(auto_close=False)
# p.write_frame()

# # Update scalars on each frame
# for i in range(360):
#     random_points = np.random.random(mesh.points.shape)
#     mesh.points = random_points * 0.01 + mesh.points * 0.99
#     mesh.points -= mesh.points.mean(0)
#     mesh.cell_data["data"] = np.random.random(mesh.n_cells)
#     # p.add_text(f"Iteration: {i}", name='time-label')
#     p.write_frame()  # Write this frame

# # Be sure to close the plotter when finished
# p.close()