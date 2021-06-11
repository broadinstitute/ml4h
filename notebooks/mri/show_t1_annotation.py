# %%
import imageio
import matplotlib.pyplot as plt
import matplotlib

im = imageio.imread('/home/pdiachil/projects/t1map/annotations/1003150_2_0.png.mask.png')

f, ax = plt.subplots()
ax.imshow(im[:, :, 0], vmin=0, vmax=20, cmap='gray', norm=matplotlib.colors.Normalize())
# %%
