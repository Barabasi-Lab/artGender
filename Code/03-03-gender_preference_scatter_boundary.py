import matplotlib.lines as mlines
import argparse
from adjustText import adjust_text
from scipy.stats import binom
import numpy as np
import json
import matplotlib.colors as clr
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import pandas as pd
from plot_config import *
import matplotlib as mpl
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib.text import TextPath
mpl.rcParams['hatch.linewidth'] = 2

# scatter plot to show top institutions and countries
# decision boundary calculation


def get_boundary(total_n, p_null, low, high, alpha=0.05):
    sample_portion = np.logspace(
        np.log10(low), np.log10(high), 100000, base=10)
    low_line, mid_line, high_line = [], [], []
    for each_portion in sample_portion:
        each_n = int(total_n * each_portion)
        neg_critical = binom.ppf(alpha, each_n, p_null)
        pos_critical = binom.ppf(1 - alpha, each_n, p_null)
        if str(neg_critical) == "nan" or str(pos_critical) == "nan":
            continue
        low_point = ((each_n - neg_critical),
                     neg_critical)
        high_point = ((each_n - pos_critical),
                      pos_critical)
        if low_point in low_line and high_point in high_line:
            continue
        low_line.append(low_point)
        high_line.append(high_point)
        mid_line.append((each_n - each_n * p_null, each_n * p_null))
    low_x, low_y = zip(*low_line)
    mid_x, mid_y = zip(*mid_line)
    high_x, high_y = zip(*high_line)
    return (low_x, low_y, mid_x, mid_y, high_x, high_y)


def get_gender_portion(df, groupby_attr="institution"):
    female_artists = artists_select[artists_select["gender_recog"] == "Female"]
    male_artists = artists_select[artists_select["gender_recog"] == "Male"]
    female_shows = df[df["gender_recog"] == "Female"]
    male_shows = df[df["gender_recog"] == "Male"]
    ins_female_count = female_shows.groupby(
        [groupby_attr])["show"].count().reset_index()
    ins_male_count = male_shows.groupby(
        [groupby_attr])["show"].count().reset_index()
    ins_female_count["female_portion"] = ins_female_count[
        "show"]
    ins_male_count["male_portion"] = ins_male_count["show"]
    ins_gender_portion = pd.merge(
        ins_female_count, ins_male_count, how="outer", on=groupby_attr)
    # print(ins_gender_portion)
    ins_gender_portion = ins_gender_portion.fillna(
        {"female_portion": 10**(-7), "male_portion": 10**(-7)})
    df = ins_gender_portion[[groupby_attr, "male_portion", "female_portion"]].rename(
        {groupby_attr: "name"}, axis=1)
    if groupby_attr == "institution":
        df["ins_name"] = [ins_name_mapping[str(item)] for item in df["name"]]
    return df


def create_marker_style(darker_cmap, lighter_cmap):
    marker_style = {}
    for i in darker_cmap:
        for j in lighter_cmap:
            this_marker = dict(fillstyle="left", markersize=8,
                               c=darker_cmap[i],
                               markerfacecoloralt=lighter_cmap[j],
                               markeredgecolor="none",
                               # markeredgewidth=0.5
                               )
            marker_style[(i, j)] = this_marker
    return marker_style


def plot_scatter(artists_select, shows_select, neutral_dict, balance_dict, marker_style, labels, figname, groupby_attr="institution"):
    # get_boundary
    p_null = Counter(artists_select["gender_recog"])[
        "Female"] / len(artists_select)
    alpha = 0.05
    total_n = len(shows_select)
    total_female = len(
        artists_select[artists_select["gender_recog"] == "Female"])
    total_male = len(artists_select[artists_select["gender_recog"] == "Male"])
    low, high = 10**(-8), 8
    low_x_neutral, low_y_neutral, mid_x_neutral, mid_y_neutral, high_x_neutral, high_y_neutral = get_boundary(
        total_n, p_null, low, high, alpha=0.05)
    low_x_balance, low_y_balance, mid_x_balance, mid_y_balance, high_x_balance, high_y_balance = get_boundary(
        total_n, 0.5, low, high, alpha=0.05)
    # save data for alice
    df_neutral = pd.DataFrame()
    df_neutral["low_x"] = low_x_neutral
    df_neutral["low_y"] = low_y_neutral
    df_neutral["mid_x"] = mid_x_neutral
    df_neutral["mid_y"] = mid_y_neutral
    df_neutral["high_x"] = high_x_neutral
    df_neutral["high_y"] = high_y_neutral
    df_neutral.to_csv(
        "/Users/xindiwang/Dropbox (CCNR)/Success Team/Wang, Xindi/Gender Inequality in Art/For Alice/Main Paper Data/2D-3B-boundary_neutral.csv", index=False)
    df_balance = pd.DataFrame()
    df_balance["low_x"] = low_x_balance
    df_balance["low_y"] = low_y_balance
    df_balance["mid_x"] = mid_x_balance
    df_balance["mid_y"] = mid_y_balance
    df_balance["high_x"] = high_x_balance
    df_balance["high_y"] = high_y_balance
    df_balance.to_csv(
        "/Users/xindiwang/Dropbox (CCNR)/Success Team/Wang, Xindi/Gender Inequality in Art/For Alice/Main Paper Data/2D-3B-boundary_balance.csv", index=False)
    # boundaries = {"neutral":{"low": (low_x_neutral, low_y_neutral), "mid": (mid_x_neutral,mid_y_neutral), "high":(high_x_neutral, high_y_neutral)},
    # "balance":{"low": (low_x_balance, low_y_balance), "mid": (mid_x_balance,mid_y_balance),"high":(high_x_balance, high_y_balance)}}
    # json.dump(boundaries, open("/Users/xindiwang/Dropbox (CCNR)/Success Team/Wang, Xindi/Gender Inequality in Art/For Alice/Main Paper Data/2D-3B-boundaries.json", "w"))
    # get portion data frame
    df_gender_portion = get_gender_portion(
        shows_select, groupby_attr=groupby_attr)
    if groupby_attr == "institution":
        df_gender_portion = df_gender_portion[df_gender_portion["name"].isin(
            list(map(int, list(neutral_dict.keys()))))]
    else:
        df_gender_portion = df_gender_portion[df_gender_portion["name"].isin(
            neutral_dict)]
    if groupby_attr == "institution":  # take top 100 prestigious and add type
        gallery_type_ins_name = {
            i: gallery_dict[i]["type"] for i in gallery_dict}
        df_gender_portion["type"] = [
            gallery_type_ins_name[str(ins)] for ins in df_gender_portion["name"]]
        shows_select["ins_name"] = [ins_name_mapping[
            str(ins)] for ins in shows_select["institution"]]
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
        # df_gender_portion.to_csv("full_ins_exhibition_info.csv", index=False)
        df_gender_portion = df_gender_portion[
            df_gender_portion["name"].isin(top_prestige_ins)]
        top_prestige_ins_name = [
            ins_name_mapping[str(ins)] for ins in top_prestige_ins]
    # add neutral and balance info to the df
    map_dict = {0: "Neutral", 1: "Man-Preferred", 2: "Woman-Preferred"}
    df_gender_portion["Gender-Neutral Criteria"] = [
        map_dict[neutral_dict[str(item)]] for item in df_gender_portion["name"]]
    map_dict = {0: "Balance", 1: "Man-Preferred", 2: "Woman-Preferred"}
    df_gender_portion["Gender-Balanced Criteria"] = [
        map_dict[balance_dict[str(item)]] for item in df_gender_portion["name"]]
    # save data for alice
    df_gender_portion.to_csv(
        "/Users/xindiwang/Dropbox (CCNR)/Success Team/Wang, Xindi/Gender Inequality in Art/For Alice/Main Paper Data/%s.csv" % figname, index=False)
    # scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    if groupby_attr == "country":
        for item, male_portion, female_portion in zip(df_gender_portion["name"], df_gender_portion['male_portion'], df_gender_portion['female_portion']):
            neutral, balance = neutral_dict[str(item)], balance_dict[str(item)]
            plt.plot([male_portion], [female_portion],
                     "o", **marker_style[(balance, neutral)])
    else:  # seperate gallery and musuems
        for item, male_portion, female_portion, ins_type in zip(df_gender_portion["name"], df_gender_portion['male_portion'], df_gender_portion['female_portion'], df_gender_portion["type"]):
            neutral, balance = neutral_dict[str(item)], balance_dict[str(item)]
            this_marker_style = marker_style[(balance, neutral)]
            if ins_type == "gallery":
                plt.plot([male_portion], [female_portion],
                         "h", **this_marker_style)
            else:
                plt.plot([male_portion], [female_portion],
                         "o", **this_marker_style)
    # add boundary and fill
    lower_bound = min(list(
        df_gender_portion["female_portion"]) + list(df_gender_portion["male_portion"]))
    upper_bound = max(list(df_gender_portion[
        "female_portion"]) + list(df_gender_portion["male_portion"]))
    plt.xlim(0.9 * lower_bound, 1.1 * upper_bound)
    plt.ylim(0.9 * lower_bound, 1.1 * upper_bound)
    # add fill
    darker_colors = ["#91b2e5", "#FF9595", "#6BD5BA"]
    darker_colors = ["#ADC4EA", "#FFA7B8", "#9CE3D1"]  # balance
    lighter_colors = ["#cddbf2", "#ffd3db", "#cdf1e8"]  # neutral
    neutral1 = plt.fill_between(high_x_neutral, y1=high_y_neutral, y2=1.1 * upper_bound, color="None",
                              facecolor=lighter_colors[1], zorder=0, rasterized=True)
    neutral2 = plt.fill_between(low_x_neutral, y1=low_y_neutral, y2=high_y_neutral, color="None",
                              facecolor=lighter_colors[2], zorder=0, rasterized=True)
    plt.fill_between(high_x_neutral, y1=low_y_neutral, y2=high_y_neutral, color="None",
                     facecolor=lighter_colors[2], zorder=0, rasterized=True)
    neutral0 = plt.fill_between(low_x_neutral, y1=0.9 * lower_bound, y2=low_y_neutral, color="None",
                              facecolor=lighter_colors[0], zorder=0, rasterized=True)
    balance1 = plt.fill_between(high_x_balance, y1=high_y_balance, y2=1.1 * upper_bound, color="None",
                                edgecolor=darker_colors[1], zorder=0, hatch=3*"-", rasterized=True)
    balance2 = plt.fill_between(low_x_balance, y1=low_y_balance, y2=low_x_balance, color="None",
                                edgecolor=darker_colors[2], zorder=0, hatch=3*"-", rasterized=True)
    plt.fill_between(high_x_balance, y1=high_x_balance, y2=high_y_balance, color="None",
                     edgecolor=darker_colors[2], zorder=0, hatch=3*"-", rasterized=True)
    balance0 = plt.fill_between(low_x_balance, y1=0.9 * lower_bound, y2=low_y_balance, color="None",
                                edgecolor=darker_colors[0], zorder=0, hatch=3*"-", rasterized=True)
    # add boundary
    plt.plot(low_x_neutral, low_y_neutral, ls="--", lw=0.5, color="gray", zorder=0)
    neutral_bound, = plt.plot(mid_x_neutral, mid_y_neutral,
                            ls="--", lw=1, color="gray", zorder=0)
    plt.plot(high_x_neutral, high_y_neutral, ls="--",
             lw=0.5, color="gray", zorder=0)
    plt.plot(low_x_balance, low_y_balance, ls="-.",
             lw=0.5, color="#4c4c4c", zorder=0)
    balance_bound, = plt.plot(
        mid_x_balance, mid_y_balance, ls="-", lw=1, color="#4c4c4c", zorder=0)
    plt.plot(high_x_balance, high_y_balance, ls="-.",
             lw=0.5, color="#4c4c4c", zorder=0)
    plt.xscale('log')
    plt.yscale('log')
    # create legend
    balance_styles = []
    neutral_styles = []
    for i in darker_cmap:
        scatter_marker_style = dict(fillstyle="left", markersize=8,
                                    c=darker_cmap[i],
                                    markerfacecoloralt="none",
                                    markeredgecolor="none",
                                    # markeredgewidth=0.5
                                    )
        balance_styles.append(scatter_marker_style)
    for i in darker_cmap:
        scatter_marker_style = dict(fillstyle="left", markersize=8,
                                    c="none",
                                    markerfacecoloralt=lighter_cmap[i],
                                    markeredgecolor="none",
                                    # markeredgewidth=0.5
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
        l = plt.legend([tuple(category_legend), tuple(balance_scatters), tuple(neutral_scatters), museum_scatter, gallery_scatter], ['', 'Gender-Balance', 'Gender-Neutral', 'Museum', 'Gallery'], numpoints=1,
                       handler_map={tuple: HandlerTuple(ndivide=None)}, labelspacing=0.1)
    else:
        l = plt.legend([tuple(category_legend), tuple(balance_scatters), tuple(neutral_scatters)], ['', 'Gender-Balance', 'Gender-Neutral'], numpoints=1,
                       handler_map={tuple: HandlerTuple(ndivide=None)}, labelspacing=0.1)
        # l = plt.legend([(balance0, balance1, balance2, balance_bound),(neutral0, neutral1, neutral2, neutral_bound), scatter], ['Gender-Balance','Gender-Neutral','Balance Type|Neutral Type'], numpoints=4,
        #        handler_map={tuple: HandlerTuple(ndivide=None)}, bbox_to_anchor = (1, 1), frameon=False)
    # add labels
    texts = []
    if groupby_attr == "institution":
        labels.update(set(top_prestige_ins_name[-10:]))
        key = "ins_name"
    else:
        key = "name"
    df_gender_portion.sort_values("male_portion", ascending=False).to_csv(
        "../related_data_results/df_gender_portion_%s.csv" % groupby_attr, index=False)
    for i, name in enumerate(df_gender_portion.sort_values("male_portion", ascending=False)[key]):
        x = list(df_gender_portion[df_gender_portion[
            key] == name]["male_portion"])[0]
        y = list(df_gender_portion[df_gender_portion[
            key] == name]["female_portion"])[0]
        if name in labels:
            texts.append(plt.text(x, y, name, fontsize=5,
                                  color=sns.color_palette()[-1], ha='center', va='center'))
        else:
            # print(i, name,sep=",")
            plt.text(x, y, i+1, fontsize=3,
                     color=sns.color_palette()[-1], ha='center', va='center')
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'))
    plt.xlabel("Number of Male Exhibitions")
    plt.ylabel("Number of Female Exhibitions")
    plt.tight_layout()
    plt.savefig("../figures/%s_scatter.pdf" % figname, dpi=400)
    plt.close()


parser = argparse.ArgumentParser(
    description='earliest year for data selection')
parser.add_argument('year', metavar='year', type=int,
                    help='select year')
args = parser.parse_args()
select_year = args.year

artists_select = pd.read_csv(
    "../related_data_results/artists_select_%s.csv" % select_year)
shows_select = pd.read_csv(
    "../related_data_results/shows_select_%s.csv" % select_year)
percentile_prestige = json.load(
    open("../related_data_results/percentile_prestige.json"))
ins_name_mapping = json.load(
    open("../related_data_results/ins_name_mapping.json"))
gallery_dict = json.load(open("../related_data_results/gallery_dict.json"))

cmap = {1: "#6E99DD", 2: "#FF5072", 0: "#3ac8a4"}
darker_cmap = {1: "#6E99DD", 2: "#FF5072", 0: "#3ac8a4"}
lighter_cmap = {1: "#8BADE4", 2: "#FF738E", 0: "#61D3B6"}

marker_style = create_marker_style(darker_cmap, lighter_cmap)

# country
gender_neutral_country = json.load(open(
    "../related_data_results/gender_neutral_country.json"))
gender_balance_country = json.load(open(
    "../related_data_results/gender_balance_country.json"))
label_country = set(["United States", "Sweden", "China", "Norway", "Finland", "Tunisia", "Cameroon", "Venezuela", "Serbia", "Indonesia", "Algeria", "Dominican Republic", "Bangladesh",
                     "Ireland", "Netherlands", "Slovenia", "Iceland", "Vietnam", "Israel",
                     "Turkey", "United Kingdom",
                     "Switzerland", "Pakistan", "Canada", "Armenia", "Egypt"])
plot_scatter(artists_select, shows_select, gender_neutral_country, gender_balance_country,
             marker_style, label_country, figname="02-gender_preference_country", groupby_attr="country")

# institutions
gender_neutral_ins = json.load(open(
    "../related_data_results/gender_neutral_ins.json"))
gender_balance_ins = json.load(open(
    "../related_data_results/gender_balance_ins.json"))
label_ins = set(["MoMA PS1",
                 "ZKM Museum of Contemporary Art",
                 "Museum of Fine Arts Boston",
                 "Albertina",
                 "Fondation Maeght",
                 "Hammer Museum",
                 "Paul Kasmin Gallery",
                 "The Columns Gallery",
                 "National Museum of Modern Art Tokyo",
                 "Paula Cooper Gallery",
                 "Skarstedt Gallery",
                 "Galerie Thomas",
                 "Alan Cristea Gallery",
                 "Galerie Lelong",
                 "Philadelphia Museum of Art",
                 "The Getty Center",
                 # "MUMOK",
                 "MOCA",
                 # "Walker Art Center",
                 "Moderna Museet - Stockholm",
                 # "Matthew Marks Gallery",
                 "Indianapolis Museum of Art",
                 "Fundación Juan March",
                 "Universalmuseum Joanneum",
                 # "Contemporary Arts Museum Houston"
                 ])
plot_scatter(artists_select, shows_select, gender_neutral_ins, gender_balance_ins,
             marker_style, label_ins, figname="02-gender_preference_ins", groupby_attr="institution")
