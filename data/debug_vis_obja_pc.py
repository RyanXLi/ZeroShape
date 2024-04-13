import sys
import trimesh
import pyvista as pv
import numpy as np

# visualize any model
base_path = "/home/ryan/work/ZeroShape/data/train_data/objaverse_LVIS/pointclouds/"
path = base_path + sys.argv[1]
pc = np.load(path, allow_pickle=True)

trimeshpc = trimesh.PointCloud(pc)
sphere = trimeshpc.bounding_sphere
# Get the center and radius of the bounding sphere
center = sphere.center
radius = (sphere.bounds[1] - sphere.bounds[0]) * 0.5
# Center the mesh at the specified center point, normalize the ball to have radius 1
pc -= center
pc /= radius




plotter = pv.Plotter()
plotter.show_axes()
plotter.view_xy()
pvpc1 = pv.PolyData(pc)
plotter.add_mesh(pvpc1, color='blue')
# plotter.add_mesh(pv.wrap(meshc), color='yellow')
# plotter.add_mesh(pv.wrap(meshsphere), color='pink')
# plotter.add_mesh(pv.wrap(meshcc), color='red')
# plotter.add_mesh(pv.wrap(circle_vis), color='green')
# plotter.add_mesh(pv.wrap(meshcc_b4trans), color='purple')
# plotter.add_mesh(pv.wrap(meshcc_inv_rot), color='orange')
# print(f"{len(planes)} planes found")
planes = eval(sys.argv[2])
for i in range(len(planes)):
    plotter.add_mesh(pv.Plane(center=[0,0,0], direction=planes[i], i_size=10, j_size=10), 
                     color='green')
plotter.show()
