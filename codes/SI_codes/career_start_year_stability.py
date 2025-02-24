import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from plot_config import *

# run python 01-01-data_selection.py --remove_birth -y [YYYY] to get the below data
# year man woman
# 1990 41738 24030
# 1995 38273 22546
# 2000 32519 19506
# 2005 21492 12920

# institutional inequality (gender neutral)
# run python 03-00-gender_preference_assign.py -y [YYYY] --remove_birth to get the below data (ins, neutral)
# 1990 {0: 4136, 1: 2147, 2: 1436})
# 1995 {0: 3909, 1: 1829, 2: 1358}
# 2000 {0: 3362, 1: 1346, 2: 1140}
# 2005 {0: 1692, 1: 642, 2: 546})

# plot SI fig2d
year_list = [1990, 1995, 2000, 2005]
man_list = [41738, 38273, 32519, 21492]
woman_list = [24030, 22546, 19506, 12920]
ratio = np.array(woman_list) / (np.array(man_list) + np.array(woman_list))

plt.clf()
plt.plot(year_list, ratio, 'o-')
plt.ylim(0.3, 0.5)
for this_year, this_ratio in zip(year_list, ratio):
    plt.text(this_year, this_ratio, "{:.2f}".format(this_ratio))

plt.xticks(ticks=[1990, 1995, 2000, 2005])
plt.yticks(ticks=[0.3, 0.35, 0.4, 0.45, 0.5])
plt.xlabel("Career Start Year Threshold")
plt.ylabel("Portion of Women Artists in Selection")
plt.savefig("../../results/SI/fig2/fig2-d.pdf")

# plot SI fig2e
year_list = [1990, 1995, 2000, 2005]
data = {1990: {0: 4566, 1: 1942, 2: 1392},
        1995: {0: 4210, 1: 1807, 2: 1407},
        2000: {0: 3574, 1: 1488, 2: 1280},
        2005: {0: 2196, 2: 1055, 1: 991}}

portion_data = {}
for year in data:
    this_year = data[year]
    num_of_ins = sum(list(this_year.values()))
    this_year_portion = {label: this_year[label] / num_of_ins for label in this_year}
    portion_data[year] = this_year_portion

plot_data = {"Gender Neutral": [portion_data[year][0] for year in year_list],
             "Man Overrepresented": [portion_data[year][1] for year in year_list],
             "Woman Overrepresented": [portion_data[year][2] for year in year_list]}
color_list = ["#989898", "#6E99DD", "#FF5072"]

bottom = np.zeros(len(year_list))
width = 2

fig, ax = plt.subplots(figsize=(8, 6))
for i, (boolean, weight_count) in enumerate(plot_data.items()):
    p = plt.bar(year_list, weight_count, width, label=boolean, bottom=bottom, color=color_list[i])
    bottom += weight_count

plt.xticks(ticks=year_list)
plt.ylabel("Portion")
plt.xlabel("Career Start Year Threshold")
plt.ylim(0, 1.1)

labels_legend = list(plot_data.keys())
handles = [patches.Rectangle((0, 0), 1, 1, color=color_list[i]) for i, label in enumerate(labels_legend)]
for handle in handles:
    ax.add_patch(handle)

ax.legend(handles, labels_legend, loc="upper center", ncol=3, prop={'size': 10})
plt.tight_layout()
plt.savefig("../../results/SI/fig2/fig2-e.pdf")


