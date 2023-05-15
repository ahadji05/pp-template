import sys
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

snapshot = sys.argv[1]
model = sys.argv[2]
nz = int(sys.argv[3])
nx = int(sys.argv[4])

float_type = np.float64
snapshot_image = np.reshape(np.fromfile(snapshot, dtype=float_type), (nz,nx))
model_image = np.reshape(np.fromfile(model, dtype=float_type), (nz,nx))

vm = np.max(np.abs(snapshot_image))

plt.imshow(model_image, cmap='gray')
plt.imshow(snapshot_image, alpha=0.5, vmin=-vm, vmax=vm, cmap='seismic')
plt.colorbar()
plt.title("Snapshot")
plt.show()