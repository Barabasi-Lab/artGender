from adjustText import adjust_text
from scipy.stats import binom
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import pandas as pd
from plot_config import *
import argparse
import matplotlib as mpl
import os

mpl.rcParams['hatch.linewidth'] = 2
parser = argparse.ArgumentParser(description='earliest year for data selection')
parser.add_argument('year', metavar='year', type=int,
                    help='select year')
# parser.add_argument('country', metavar='country', type=int,
# help='inspect country(ies)')

args = parser.parse_args()
select_year = args.year

folder = "../results/threshold_0.6_filter_True/data"
year_folder = "../results/threshold_0.6_filter_True/data/year_%s/" % select_year

artists_select = pd.read_csv(os.path.join(year_folder, "artists_recog_select.csv"))
shows_select = pd.read_csv(os.path.join(year_folder, "shows_select.csv"))
percentile_prestige = json.load(
    open(os.path.join(folder, "percentile_prestige.json")))
ins_name_mapping = {"223": "Guggenheim"}
# binom test attributes
# p_null = 0.5
p_null = Counter(artists_select["gender_recog"])[
             "Female"] / len(artists_select)
print(p_null)
alpha = 0.05

cmap = {1: "#6E99DD", 2: "#FF5072", 0: "#3ac8a4"}


def binom_demo(ins, group_attr, p_null, alpha, ax, legend=False):
    darker_colors = ["#ADC4EA", "#FFA7B8", "#9CE3D1"]  # balance
    lighter_colors = ["#cddbf2", "#ffd3db", "#cdf1e8"]  # neutral
    # demo to show how the test works
    this_ins = shows_select[shows_select[group_attr] == ins]
    n = len(this_ins)
    null_female_count = np.random.binomial(
        n, p_null, size=100000)  # null distribution
    # save for alice
    json.dump(list(map(int, null_female_count)),
              open("../main_paper_plot_data/2B_left_dst_raw.json", "w"))
    female_count = Counter(this_ins["gender_recog"])["Female"]
    neg_critical_neutral = binom.ppf(alpha, n, p_null)
    pos_critical_neutral = binom.ppf(1 - alpha, n, p_null)
    dist = sns.distplot(null_female_count,
                        color="black", hist=False, ax=ax, kde_kws={"gridsize": 1000, "lw": 1})
    ax.axvline(female_count, color=sns.color_palette()
    [1])
    ax.axvline(neg_critical_neutral, color="gray", ls="--", lw=1)
    ax.axvline(pos_critical_neutral, color="gray", ls='--', lw=1)
    print("Neutral:", neg_critical_neutral, pos_critical_neutral)
    print(female_count, n)
    l = dist.lines[0]
    x, y = l.get_xydata()[:, 0], l.get_xydata()[:, 1]
    # print(x[0])
    # x = np.linspace(min(x), max(x), 100000)
    f1 = ax.fill_between(x, y, color=lighter_colors[0], where=x <= neg_critical_neutral, interpolate=True,
                         label="Male-Preferred")
    f2 = ax.fill_between(x, y, color=lighter_colors[
        2], where=(x >= neg_critical_neutral) & (x <= pos_critical_neutral), label="Gender-Neutral")
    f3 = ax.fill_between(x, y, color=lighter_colors[
        1], where=x >= pos_critical_neutral, interpolate=True, label="Female-Preferred")
    # p_null = 0.5
    p_null = 0.5
    n = len(this_ins)
    null_female_count = np.random.binomial(
        n, p_null, size=100000)
    # save for plotting
    json.dump(list(map(int, null_female_count)),
              open("../main_paper_plot_data/2B_right_dst_raw.json", "w"))
    female_count = Counter(this_ins["gender_recog"])["Female"]
    neg_critical_balance = binom.ppf(alpha, n, p_null)
    pos_critical_balance = binom.ppf(1 - alpha, n, p_null)
    dist2 = sns.distplot(null_female_count,
                         color="black", hist=False, ax=ax, kde_kws={"gridsize": 1000, "lw": 1, "ls": "--"})
    ax.axvline(female_count, color=sns.color_palette()
    [1])
    plt.text(female_count + 1, 0, "Female\nExhibition", va="bottom", ha="left", fontsize=10, fontweight="bold")
    ax.axvline(neg_critical_balance, color="#4c4c4c", ls="-.", lw=1)
    ax.axvline(pos_critical_balance, color="#4c4c4c", ls='-.', lw=1)
    print("Balance: ", neg_critical_balance, pos_critical_balance)
    print(female_count, n)
    l = dist.lines[4]
    x, y = l.get_xydata()[:, 0], l.get_xydata()[:, 1]
    # print(x, y)
    # x = np.linspace(min(x), max(x), 100000)
    f4 = ax.fill_between(x, y, color="None", edgecolor=darker_colors[0], where=x <= neg_critical_balance,
                         interpolate=True, label="Male-Preferred", hatch=3 * "-")
    f5 = ax.fill_between(x, y, color="None", edgecolor=darker_colors[2],
                         where=(x >= neg_critical_balance) & (x <= pos_critical_balance), label="Gender-Balanced",
                         hatch=3 * "-")
    f6 = ax.fill_between(x, y, color="None", edgecolor=darker_colors[1], where=x >= pos_critical_balance,
                         interpolate=True, label="Female-Preferred", hatch=3 * "-")
    plt.suptitle(ins_name_mapping[str(ins)], x=0.55, y=0.9, fontsize=20)
    ax.set_title("Female Exhibitions: %s Male Exhibitions: %s" %
                 (female_count, n - female_count), fontsize=10)
    ax.set_ylim(0, )
    ax.set_ylabel("P")
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    t1 = plt.text(xmin + 1, ymax * 0.99,
                  "Gender-Neutral\nExpected: [%.0f, %.0f]" % (neg_critical_neutral, pos_critical_neutral), fontsize=8,
                  va="top", ha="left", fontweight="bold")
    t2 = plt.text(xmax - 1, ymax * 0.99,
                  "Gender-Balance\nExpected: [%.0f, %.0f]" % (neg_critical_balance, pos_critical_balance), fontsize=8,
                  va="top", ha="right", fontweight="bold")
    return ax


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
binom_demo(223, "institution", p_null, alpha, ax, legend=True)
plt.xlabel("Number of Female Exhibitions")
plt.tight_layout()
plt.savefig(
    os.path.join("../results/threshold_0.6_filter_True", "figures/year_1990", "02-gender_preference_assign_demo.pdf"))
