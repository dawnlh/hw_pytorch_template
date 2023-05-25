# libraries
import matplotlib.pyplot as plt
import numpy as np
import math

## data
method = ["Jin et al.", "Shedligeri et al.", "Slow Motion",
           "BIN", "Animation from Blur", "BDNeRV(Ours)"]
psnr = np.array([26.35, 26.94, 27.68, 25.52, 28.1, 30.64])
param = np.array([18.21, 8.60, 18.23, 1.14, 7.4, 3.70])
macs = np.array([211.31, 52.26, 80, 261.80, 250, 242.66])


## data mapping
x, x_label = param, 'Param. # (M)'
y, y_label = psnr, 'PSNR (dB)'
s = macs
label = method
c = range(len(method))
# label_pos = [(213, 25), (52, 26.6), (80, 27.2), (262, 25.4), (252,28), (243,30)]


## figure
# setting
fig, ax = plt.subplots(facecolor='w', figsize=(8, 6))
save_name, save_dpi = 'fig_bubble.svg', 1200
cm = plt.cm.get_cmap('tab10')  # Spectral, Set2, tab10
plt.grid(linestyle='--', linewidth=0.5)
plt.gca().set(xlim=(0, 20), ylim=(24.5, 31.5))
plt.xlabel(x_label, {'size': 14})
plt.ylabel(y_label, {'size': 14})
plt.xticks(fontsize=13)  # ticks=range(40, 300, 40),
plt.yticks(fontsize=13)  # ticks=range(25, 32),
# plt.title("Performance benchmark")

# plot without label: recommented (use PPT to add annotation afterwards)
plt.scatter(x, y, s*5, c=c, cmap=cm, alpha=.8)


# plot with label: not recommented (hard to control the accurate position)
# size_scale = 5
# for i in range(len(label)):
#     ax.scatter(x[i], y[i], s=s[i]*size_scale, alpha=.5)
#     ax.annotate(label[i], xy=(x[i], y[i]))

## show and save
# plt.savefig(save_name, dpi=save_dpi) # `save` should be before `show`
plt.show()

