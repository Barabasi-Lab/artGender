# scatter plot to show top institutions and countries with decision boundary calculation
import argparse
import json
import os
from collections import Counter

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.legend_handler import HandlerTuple
from matplotlib.text import TextPath
from scipy.stats import beta, binom
from scipy.integrate import quad

from tqdm import tqdm

from plot_config import *

mpl.rcParams['hatch.linewidth'] = 2


def bayesfactor_binom(k, n, p=0.5, a=1, b=1, oneside=False):
    assert 0 < p < 1, "p must be between 0 and 1."
    assert isinstance(k, int), "k must be int."
    assert isinstance(n, int), "n must be int."
    assert k <= n, "k (successes) cannot be higher than n (trials)."
    assert a > 0, "a must be positive."
    assert b > 0, "b must be positive."
    def fun(g):
        return beta.pdf(g, a, b) * binom.pmf(k, n, g)
    if oneside:
        if k / n <= p:
            bf10 = quad(fun, 0, p)[0] / binom.pmf(k, n, p)
        else:
            bf10 = quad(fun, p, 1)[0] / binom.pmf(k, n, p)
    else:
        bf10 = quad(fun, 0, 1)[0] / binom.pmf(k, n, p)
    return bf10


def get_critical_values(n_total, p_null):
    if p_null == 0.5:
        if n_total < 1000:
            portion_range = [0, 1]
        elif n_total > 15000:
            portion_range = [0.48, 0.52]
        else:
            portion_range = [0.4, 0.6]
    else:
        if n_total < 1000:
            portion_range = [0, 1]
        elif n_total > 15000:
            portion_range = [0.34, 0.38]
        else:
            portion_range = [0.3, 0.5]
    trials = range(int(n_total * portion_range[0])-1, int(n_total * p_null))[::-1]
    bf10_list = []
    after = 0
    for n_trial in trials:
        bf_score = bayesfactor_binom(n_trial, n_total, p=p_null, a=1, b=1, oneside=True)
        bf10_list.append(bf_score)
        if after < 3 and bf_score >= 3:
            critical_value_menover = n_trial
            break
        else:
            after = bf_score
    # determine the lowest n_women to claim p_women > p
    trials = range(int(n_total * p_null), int(n_total * portion_range[1]) + 1)
    bf10_list = []
    previous = 0
    for n_trial in trials:
        bf_score = bayesfactor_binom(n_trial, n_total, p=p_null, a=1, b=1, oneside=True)
        bf10_list.append(bf_score)
        if previous < 3 and bf_score >= 3:
            critical_value_womenover = n_trial
            break
        else:
            previous = bf_score
    # determine the gender neutral range
    trials = range(int(n_total * portion_range[0]), int(n_total * portion_range[1]) + 1)
    bf10_list = []
    for n_trial in trials:
        bf_score = bayesfactor_binom(int(n_trial), n_total, p=p_null, a=1, b=1, oneside=False)
        bf10_list.append(bf_score)
        if bf_score < 1/3:
            critical_null_low = n_trial
            break
    bf10_list = []
    for n_trial in trials[::-1]:
        bf_score = bayesfactor_binom(n_trial, n_total, p=p_null, a=1, b=1, oneside=False)
        bf10_list.append(bf_score)
        if bf_score < 1 / 3:
            critical_null_high = n_trial
            break
    critical_value_womenover = max(critical_value_womenover, critical_null_high)
    critical_value_menover = min(critical_value_menover, critical_null_high)
    return critical_value_menover, critical_value_womenover, critical_null_low, critical_null_high


def get_boundary(total_n, p_null, low, high):
    samples = np.logspace(np.log10(low), np.log10(high), num=50) #np.logspace(np.log10(low), np.log10(high), 2, base=10)
    sample_count = sorted(list(set(list(map(int, samples)))))
    print(samples)
    menover_line, womenover_line = [], []
    null_low_line, null_high_line, null_mid_line = [], [], []
    for each_n in tqdm(sample_count):
        try:
            critical_value_menover, critical_value_womenover, critical_null_low, critical_null_high = get_critical_values(
                each_n, p_null)
        except:
            critical_value_menover, critical_value_womenover, critical_null_low, critical_null_high = 0, 0, 0, 0
        menover_point = ((each_n - critical_value_menover), critical_value_menover)
        womenover_point = ((each_n - critical_value_womenover), critical_value_womenover)
        null_low_point = ((each_n - critical_null_low), critical_null_low)
        null_high_point = ((each_n - critical_null_high), critical_null_high)
        if menover_point in menover_line and womenover_point in womenover_line:
            continue
        menover_line.append(menover_point)
        womenover_line.append(womenover_point)
        null_low_line.append(null_low_point)
        null_high_line.append(null_high_point)
        null_mid_line.append((each_n - each_n * p_null, each_n * p_null))
    menover_x, menover_y = zip(*menover_line)
    womenover_x, womenover_y = zip(*womenover_line)
    null_low_x, null_low_y = zip(*null_low_line)
    null_high_x, null_high_y = zip(*null_high_line)
    null_mid_x, null_mid_y = zip(*null_mid_line)
    return {"menover_x": menover_x, "menover_y": menover_y,
            "womenover_x": womenover_x, "womenover_y": womenover_y,
            "null_low_x": null_low_x, "null_low_y": null_low_y,
            "null_high_x": null_high_x, "null_high_y": null_high_y,
            "null_mid_x": null_mid_x, "null_mid_y": null_mid_y}


def get_gender_portion(df, groupby_attr="institution"):
    female_shows = df[df["gender_recog"] == "Female"]
    male_shows = df[df["gender_recog"] == "Male"]
    ins_female_count = female_shows.groupby([groupby_attr])["show"].count().reset_index()
    ins_male_count = male_shows.groupby([groupby_attr])["show"].count().reset_index()
    ins_female_count["female_portion"] = ins_female_count["show"]
    ins_male_count["male_portion"] = ins_male_count["show"]
    ins_gender_portion = pd.merge(ins_female_count, ins_male_count, how="outer", on=groupby_attr)
    # print(ins_gender_portion)
    ins_gender_portion = ins_gender_portion.fillna(
        {"female_portion": 10 ** (-7), "male_portion": 10 ** (-7)})
    df = ins_gender_portion[[groupby_attr, "male_portion", "female_portion"]].rename({groupby_attr: "name"}, axis=1)
    if groupby_attr == "institution":
        df["ins_name"] = [ins_name_mapping[str(item)] if str(item) in ins_name_mapping else item for item in df["name"]]
    return df


def create_marker_style(darker_cmap, lighter_cmap):
    marker_style = {}
    for i in darker_cmap:
        for j in lighter_cmap:
            this_marker = dict(fillstyle="left", markersize=8,
                               c=darker_cmap[i],
                               markerfacecoloralt=lighter_cmap[j],
                               markeredgecolor="none"
                               )
            marker_style[(i, j)] = this_marker
    return marker_style


def plot_scatter(artists_select, shows_select, neutral_dict, balance_dict, marker_style, labels, figname,
                 groupby_attr="institution"):
    # get_boundary
    p_null = Counter(artists_select["gender_recog"])["Female"] / len(artists_select)
    total_n = len(shows_select)
    total_female = len(artists_select[artists_select["gender_recog"] == "Female"])
    total_male = len(artists_select[artists_select["gender_recog"] == "Male"])
    groupby_attr_counts = shows_select.groupby(groupby_attr)["show"].count()
    low, high = min(groupby_attr_counts), max(groupby_attr_counts) * 1.01
    critical_points_neutral = get_boundary(total_n, p_null, 10, high)
    critical_points_balance = get_boundary(total_n, 0.5, 10, high)
    # save data for plot
    df_neutral = pd.DataFrame()
    for key in critical_points_neutral:
        df_neutral[key] = critical_points_neutral[key]
    if groupby_attr == "country":
        df_neutral.to_csv(
            "../main_paper_plot_data/2D-3B-boundary_neutral_bf10.csv", index=False)
    df_balance = pd.DataFrame()
    for key in critical_points_balance:
        df_balance[key] = critical_points_balance[key]
    if groupby_attr == "country":
        df_balance.to_csv(
            "../main_paper_plot_data/2D-3B-boundary_balance_bf10.csv", index=False)
    # get portion data frame
    df_gender_portion = get_gender_portion(shows_select, groupby_attr=groupby_attr)
    if groupby_attr == "institution":
        df_gender_portion = df_gender_portion[df_gender_portion["name"].isin(list(map(int, list(neutral_dict.keys()))))]
        df_gender_portion = df_gender_portion[df_gender_portion["name"].isin(list(map(int, list(balance_dict.keys()))))]
    else:
        df_gender_portion = df_gender_portion[df_gender_portion["name"].isin(neutral_dict)]
        df_gender_portion = df_gender_portion[df_gender_portion["name"].isin(balance_dict)]
    if groupby_attr == "institution":  # take top 100 prestigious and add type
        gallery_type_ins_name = {i: gallery_dict[i]["type"] for i in gallery_dict}
        df_gender_portion["type"] = [gallery_type_ins_name[str(ins)] for ins in df_gender_portion["name"]]
        prestige_ins = shows_select[shows_select["institution"].isin(df_gender_portion["name"])][[
            "institution", "percentile_prestige"]].drop_duplicates().sort_values("percentile_prestige")
        top_prestige_ins = prestige_ins.tail(100)["institution"]
        # add neutral and balance info to the df
        map_dict = {0: "Neutral", 1: "Man-Preferred", 2: "Woman-Preferred"}
        df_gender_portion["Gender-Neutral Criteria"] = [
            map_dict[neutral_dict[str(item)]] for item in df_gender_portion["name"]]
        map_dict = {0: "Balance", 1: "Man-Preferred", 2: "Woman-Preferred"}
        df_gender_portion["Gender-Balanced Criteria"] = [
            map_dict[balance_dict[str(item)]] for item in df_gender_portion["name"]]
        df_gender_portion = df_gender_portion[df_gender_portion["name"].isin(top_prestige_ins)]
    # add neutral and balance info to the df
    map_dict = {0: "Neutral", 1: "Man-Preferred", 2: "Woman-Preferred"}
    df_gender_portion["Gender-Neutral Criteria"] = [map_dict[neutral_dict[str(item)]] for item in df_gender_portion["name"]]
    map_dict = {0: "Balance", 1: "Man-Preferred", 2: "Woman-Preferred"}
    df_gender_portion["Gender-Balanced Criteria"] = [map_dict[balance_dict[str(item)]] for item in df_gender_portion["name"]]
    # filter df_gender_portion
    df_gender_portion = df_gender_portion[(df_gender_portion['male_portion'] >= 10) & (df_gender_portion['female_portion']>=10)]
    # save data for plotting
    df_gender_portion.to_csv("../main_paper_plot_data/%s_bf10.csv" % figname, index=False)
    # scatter plotMain Paper Data
    fig, ax = plt.subplots(figsize=(6, 6))
    if groupby_attr == "country":
        for item, male_portion, female_portion in zip(df_gender_portion["name"], df_gender_portion['male_portion'],
                                                      df_gender_portion['female_portion']):
            neutral, balance = neutral_dict[str(item)], balance_dict[str(item)]
            plt.plot([male_portion], [female_portion],
                     "o", **marker_style[(balance, neutral)])
    else:  # seperate gallery and museums
        for item, male_portion, female_portion, ins_type in zip(df_gender_portion["name"],
                                                                df_gender_portion['male_portion'],
                                                                df_gender_portion['female_portion'],
                                                                df_gender_portion["type"]):
            neutral, balance = neutral_dict[str(item)], balance_dict[str(item)]
            this_marker_style = marker_style[(balance, neutral)]
            if ins_type == "gallery":
                plt.plot([male_portion], [female_portion],
                         "h", **this_marker_style)
            else:
                plt.plot([male_portion], [female_portion],
                         "o", **this_marker_style)
    # add boundary and fill
    lower_bound = min(list(df_gender_portion["female_portion"]) + list(df_gender_portion["male_portion"]))
    upper_bound = max(list(df_gender_portion["female_portion"]) + list(df_gender_portion["male_portion"]))
    plt.xlim(10, 1.1 * upper_bound)
    plt.ylim(10, 1.1 * upper_bound)
    # add fill
    darker_colors = ["#ADC4EA", "#FFA7B8", "#9CE3D1"]  # balance
    lighter_colors = ["#cddbf2", "#ffd3db", "#cdf1e8"]  # neutral

    # fill neutral
    # neutral1 = plt.fill_between(df_neutral["womenover_x"], y1=df_neutral["womenover_y"], y2=1.1 * upper_bound,
    #                             color="None",
    #                             facecolor=lighter_colors[1], zorder=0, rasterized=True)
    neutral2 = plt.fill_between(df_neutral["null_low_x"], y1=df_neutral["null_low_y"], y2=df_neutral["null_high_y"],
                                color="None",
                                facecolor=lighter_colors[2], zorder=0, rasterized=True)
    plt.fill_between(df_neutral["null_high_x"], y1=df_neutral["null_high_y"], y2=df_neutral["null_low_y"], color="None",
                     facecolor=lighter_colors[2], zorder=0, rasterized=True)
    # neutral0 = plt.fill_between(df_neutral["menover_x"], y1=0.9 * lower_bound, y2=df_neutral["menover_y"], color="None",
    #                             facecolor=lighter_colors[0], zorder=0, rasterized=True)

    # fill_balance
    # balance1 = plt.fill_between(df_balance["womenover_x"], y1=df_balance["womenover_y"], y2=1.1 * upper_bound,
    #                             color="None",
    #                             facecolor=darker_colors[1], zorder=0, rasterized=True, hatch=3 * "-")
    balance2 = plt.fill_between(df_balance["null_low_x"], y1=df_balance["null_low_y"], y2=df_balance["null_high_y"],
                                color="None",
                                facecolor=darker_colors[2], zorder=0, rasterized=True, hatch=3 * "-")
    plt.fill_between(df_balance["null_high_x"], y1=df_balance["null_high_y"], y2=df_balance["null_low_y"], color="None",
                     facecolor=darker_colors[2], zorder=0, rasterized=True, hatch=3 * "-")
    # balance0 = plt.fill_between(df_balance["menover_x"], y1=0.9 * lower_bound, y2=df_balance["menover_y"], color="None",
    #                             facecolor=darker_colors[0], zorder=0, rasterized=True, hatch=3 * "-")
    # add boundary
    # neutral boundary
    # plt.plot(df_neutral["womenover_x"], df_neutral["womenover_y"], ls="--", lw=0.5, color="gray", zorder=0)
    # plt.plot(df_neutral["menover_x"], df_neutral["menover_y"], ls="--", lw=0.5, color="gray", zorder=0)
    neutral_bound, = plt.plot(df_neutral["null_low_x"], df_neutral["null_low_y"], ls="--", lw=1, color="gray", zorder=0)
    neutral_bound, = plt.plot(df_neutral["null_high_x"], df_neutral["null_high_y"], ls="--", lw=1, color="gray",
                              zorder=0)

    # balance boundary
    # plt.plot(df_balance["womenover_x"], df_balance["womenover_y"], ls="-.", lw=0.5, color="#4c4c4c", zorder=0)
    # plt.plot(df_balance["menover_x"], df_balance["menover_y"], ls="-.", lw=0.5, color="#4c4c4c", zorder=0)
    balance_bound, = plt.plot(df_balance["null_low_x"], df_balance["null_low_x"], ls="-", lw=1, color="#4c4c4c", zorder=0)
    neutral_bound, = plt.plot(df_balance["null_high_x"], df_balance["null_high_y"], ls="--", lw=1, color="gray",
                              zorder=0)

    plt.xscale('log')
    plt.yscale('log')
    # create legend
    balance_styles = []
    neutral_styles = []
    for i in darker_cmap:
        scatter_marker_style = dict(fillstyle="left", markersize=8,
                                    c=darker_cmap[i],
                                    markerfacecoloralt="none",
                                    markeredgecolor="none"
                                    )
        balance_styles.append(scatter_marker_style)
    for i in darker_cmap:
        scatter_marker_style = dict(fillstyle="left", markersize=8,
                                    c="none",
                                    markerfacecoloralt=lighter_cmap[i],
                                    markeredgecolor="none"
                                    )
        neutral_styles.append(scatter_marker_style)
    balance_scatters = [mlines.Line2D(
        [], [], marker='o', linestyle='', **balance_styles[i]) for i in range(len(balance_styles))]
    neutral_scatters = [mlines.Line2D(
        [], [], marker='o', linestyle='', **neutral_styles[i]) for i in range(len(neutral_styles))]
    paths = [TextPath((-7, -7), i) for i in ['M', 'F', 'B']]
    category_legend = [mlines.Line2D([], [], marker=paths[i], linestyle='',
                                     markersize=8, color=sns.color_palette()[i]) for i in range(len(paths))]
    if groupby_attr == "institution":
        museum_scatter = mlines.Line2D(
            [], [], marker='o', linestyle='', markersize=8, markerfacecolor="none", markeredgecolor="black")
        gallery_scatter = mlines.Line2D(
            [], [], marker='h', linestyle='', markersize=8, markerfacecolor="none", markeredgecolor="black")
        # l = plt.legend([museum_scatter, gallery_scatter], ["Museum", "Gallery"], ncol=2)
        scatter2 = mlines.Line2D(
            [], [], marker='h', linestyle='', **scatter_marker_style)
        l = plt.legend(
            [tuple(category_legend), tuple(balance_scatters), tuple(neutral_scatters), museum_scatter, gallery_scatter],
            ['', 'Gender-Balance', 'Gender-Neutral', 'Museum', 'Gallery'], numpoints=1,
            handler_map={tuple: HandlerTuple(ndivide=None)}, labelspacing=0.1)
    else:
        l = plt.legend([tuple(category_legend), tuple(balance_scatters), tuple(neutral_scatters)],
                       ['', 'Gender-Balance', 'Gender-Neutral'], numpoints=1,
                       handler_map={tuple: HandlerTuple(ndivide=None)}, labelspacing=0.1)
    # add labels
    texts = []
    if groupby_attr == "institution":
        key = "ins_name"
    else:
        key = "name"
    df_gender_portion.sort_values("male_portion", ascending=False).to_csv(
        "../main_paper_plot_data/df_gender_portion_%s_bf10.csv" % groupby_attr, index=False)
    for i, name in enumerate(df_gender_portion.sort_values("male_portion", ascending=False)[key]):
        x = list(df_gender_portion[df_gender_portion[key] == name]["male_portion"])[0]
        y = list(df_gender_portion[df_gender_portion[key] == name]["female_portion"])[0]
        if name in labels:
            texts.append(plt.text(x, y, name, fontsize=5,
                                  color=sns.color_palette()[-1], ha='center', va='center'))
        else:
            plt.text(x, y, i + 1, fontsize=3,
                     color=sns.color_palette()[-1], ha='center', va='center')
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'))
    plt.xlabel("Number of Male Exhibitions")
    plt.ylabel("Number of Female Exhibitions")
    plt.tight_layout()
    plt.savefig(
        os.path.join("../results/threshold_0.6_filter_True", "figures/year_1990", "%s_scatter_bf10.pdf" % figname),
        dpi=400)
    plt.close()


parser = argparse.ArgumentParser(
    description='earliest year for data selection')
parser.add_argument('--year', metavar='year', type=int,
                    help='select year', default=1990)
args = parser.parse_args()
select_year = args.year

year_folder = "../results/threshold_0.6_filter_True/data/year_%s/" % select_year

artists_select = pd.read_csv(os.path.join(year_folder, "artists_recog_select.csv"))
shows_select = pd.read_csv(os.path.join(year_folder, "shows_select.csv"))
percentile_prestige = json.load(
    open(os.path.join("../results/threshold_0.6_filter_True/data", "percentile_prestige.json")))
ins_name_mapping = json.load(open("../raw_data/ins_name_mapping.json"))
gallery_dict = json.load(open("../raw_data/gallery_dict.json"))

cmap = {1: "#6E99DD", 2: "#FF5072", 0: "#3ac8a4"}
darker_cmap = {1: "#6E99DD", 2: "#FF5072", 0: "#3ac8a4"}
lighter_cmap = {1: "#8BADE4", 2: "#FF738E", 0: "#61D3B6"}

marker_style = create_marker_style(darker_cmap, lighter_cmap)

# country
gender_neutral_country = json.load(open(os.path.join(year_folder, "gender_neutral_country_bf10.json")))
gender_balance_country = json.load(open(os.path.join(year_folder, "gender_balance_country_bf10.json")))
label_country = {"United States", "Sweden", "China", "Norway", "Finland", "Tunisia", "Cameroon", "Venezuela", "Serbia",
                 "Indonesia",
                 # "Algeria", filtered with bayes factor
                 "Dominican Republic", "Bangladesh", "Ireland", "Netherlands", "Slovenia",
                 "Iceland", "Vietnam", "Israel", "Turkey", "United Kingdom", "Switzerland", "Pakistan", "Canada",
                 "Armenia", "Egypt"}
plot_scatter(artists_select, shows_select, gender_neutral_country, gender_balance_country,
             marker_style, label_country, figname="02-gender_preference_country", groupby_attr="country")

# institutions
gender_neutral_ins = json.load(open(os.path.join(year_folder, "gender_neutral_ins_bf10.json")))
gender_balance_ins = json.load(open(os.path.join(year_folder, "gender_balance_ins_bf10.json")))
label_ins = set(ins_name_mapping.values())
plot_scatter(artists_select, shows_select, gender_neutral_ins, gender_balance_ins,
             marker_style, label_ins, figname="02-gender_preference_ins", groupby_attr="institution")
