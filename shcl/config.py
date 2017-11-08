import matplotlib.pyplot as plt

try:
    from mpltools import style
    style.use('ggplot')
except:
    pass

fontsize = 13
params = {'axes.labelsize': fontsize + 2,
          'font.size': fontsize,
          'legend.fontsize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize}
plt.rcParams.update(params)

markers = ['s','d','^','v','<','p','>']
# colors = ['b','r','g','k','c','m','']
# colors = plt.rcParams['axes.color_cycle']
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
