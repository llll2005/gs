import numpy as np
from plyfile import PlyData

ply = PlyData.read(
    "data/matrix_city/aerial/train/block_all/depth_init/block_12.ply"
)
ply["vertex"]["opacity"] = np.log(0.05 / (1.0 - 0.05))
ply.write(
    "data/matrix_city/aerial/train/block_all/depth_init_o05/block_12.ply"
)
