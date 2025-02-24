import argparse
import json
import os
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats

from plot_config import *

mpl.rcParams['hatch.linewidth'] = 2
darker_colors = ["#ADC4EA", "#FFA7B8", "#9CE3D1"]  # balance
lighter_colors = ["#cddbf2", "#ffd3db", "#cdf1e8"]  # neutral

def make_autopct(sizes):
    def absolute_value(val):
        a = int(np.round(val / 100. * sum(sizes), 0))
        return a

    return absolute_value


class PlotGenderPreference():

    def __init__(self, genderizeio_threshold, remove_birth, career_start_threshold, preference_type, save):
        self.genderizeio_threshold = genderizeio_threshold
        self.remove_birth = remove_birth
        self.career_start_threshold = career_start_threshold
        self.preference_type = preference_type
        self.save = save

        if self.preference_type == "neutral":
            sns.set_palette(sns.color_palette(lighter_colors))
        elif self.preference_type == "balance":
            sns.set_palette(sns.color_palette(darker_colors))

        self.gallery_dict = json.load(open("../../../raw_data/gallery_dict.json"))

        folder_path = os.path.join("..", "..", "..", "results",
                                   f"threshold_{self.genderizeio_threshold}_filter_{self.remove_birth}",
                                   "data")
        self.percentile_prestige = json.load(open(os.path.join(folder_path, "percentile_prestige.json")))

        self.save_data_path = os.path.join("..", "..", "..", "results",
                                           f"threshold_{self.genderizeio_threshold}_filter_{self.remove_birth}",
                                           "data",
                                           f"year_{self.career_start_threshold}")
        self.save_fig_path = os.path.join("..", "..", "..", "results",
                                          "SI",
                                          "fig4")
        try:
            os.makedirs(self.save_data_path, exist_ok=True)
        except:
            pass
        try:
            os.makedirs(self.save_fig_path, exist_ok=True)
        except:
            pass

        self.artists_recog_select = pd.read_csv(os.path.join(self.save_data_path, "artists_recog_select.csv"))
        self.shows_select = pd.read_csv(os.path.join(self.save_data_path, "shows_select.csv"))
        self.gender_preference_dict = json.load(open(
            os.path.join(self.save_fig_path, f"gender_{self.preference_type}_ins_solo_bf10.json")))

        self.ins_info_df = self.get_ins_info_df()
        self.gender_preference_df, self.museum_gender_preference_df, self.gallery_gender_preference_df = self.get_gender_preference_df()

    def get_ins_info_df(self):
        # get ins info
        ins_info_df = pd.DataFrame()
        ins_info_df["gallery_id"] = list(self.gender_preference_dict.keys())
        ins_info_df["gender_preference"] = list(self.gender_preference_dict.values())
        ins_info_df["type"] = [self.gallery_dict[ins]["type"].title() for ins in ins_info_df["gallery_id"]]
        ins_info_df["grade"] = [self.gallery_dict[ins]["grade"] for ins in ins_info_df["gallery_id"]]
        ins_info_df["prestige"] = [self.percentile_prestige[str(ins)] for ins in ins_info_df["gallery_id"]]
        pmap = {1: "Low", 2: "Mid", 3: "High"}
        ins_info_df["prestige_40_70_bin"] = [pmap[np.digitize(round(item, 1),
                                                              [0,
                                                               np.percentile(
                                                                   ins_info_df[ins_info_df["gender_preference"] != -1][
                                                                       "prestige"], 40),
                                                               np.percentile(
                                                                   ins_info_df[ins_info_df["gender_preference"] != -1][
                                                                       "prestige"], 70),
                                                               1.1])]
                                             for item in ins_info_df["prestige"]]
        ins_info_df["prestige_30_60_bin"] = [pmap[np.digitize(
            round(item, 1), [0, np.percentile(ins_info_df[ins_info_df["gender_preference"] != -1]["prestige"], 30),
                             np.percentile(ins_info_df[ins_info_df["gender_preference"] != -1]["prestige"], 60), 1.1])]
                                             for item in
                                             ins_info_df["prestige"]]
        tenth_bins = np.array(
            [np.percentile(ins_info_df[ins_info_df["gender_preference"] != -1]["prestige"], percentile)
             for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]] + [1.1])
        ins_info_df["prestige_tenth_bin"] = [np.digitize(item, tenth_bins) for item in ins_info_df["prestige"]]
        ins_info_df.to_csv(os.path.join(self.save_data_path, f"ins_information_{self.preference_type}.csv"),
                           index=False)
        return ins_info_df

    def get_gender_preference_df(self):
        # all
        count = Counter(self.gender_preference_dict.values())
        gender_preference_df = pd.DataFrame({"Equality": ["Male-Preferred", "Female-Preferred", "Balanced"],
                                             "Count": [count[1], count[2], count[0]]})
        # museum
        museum_count = Counter(self.ins_info_df[self.ins_info_df["type"] == "Museum"]["gender_preference"])
        sizes = museum_count[1], museum_count[2], museum_count[0]
        museum_gender_preference_df = pd.DataFrame(
            {"Equality": ["Male-Preferred", "Female-Preferred", "Balanced"], "Count": sizes})
        # gallery
        gallery_count = Counter(self.ins_info_df[self.ins_info_df["type"] == "Gallery"]["gender_preference"])
        sizes = gallery_count[1], gallery_count[2], gallery_count[0]
        gallery_gender_preference_df = pd.DataFrame(
            {"Equality": ["Male-Preferred", "Female-Preferred", "Balanced"], "Count": sizes})

        return gender_preference_df, museum_gender_preference_df, gallery_gender_preference_df

    def plot_pie_chart(self, tag="all"):
        if tag == "all":
            sizes = list(self.gender_preference_df["Count"])
        elif tag == "museum":
            sizes = list(self.museum_gender_preference_df["Count"])
        elif tag == "gallery":
            sizes = list(self.gallery_gender_preference_df["Count"])
        else:
            raise Exception("tag not supported.")
        fig1, ax1 = plt.subplots(figsize=(3, 3))
        piechart = ax1.pie(sizes, autopct=make_autopct(sizes), textprops=dict(color="black", fontsize=20))
        if self.preference_type == "balance":
            for i in range(len(piechart[0])):
                piechart[0][i].set_hatch(2 * "-")
                piechart[0][i].set_color("none")
                piechart[0][i].set_edgecolor(sns.color_palette()[i])
        plt.tight_layout()
        pie_plot_path = os.path.join(self.save_fig_path, f"{tag}_pie")
        try:
            os.makedirs(pie_plot_path, exist_ok=True)
        except:
            pass
        plt.savefig(os.path.join(pie_plot_path, f"{self.preference_type}_bf10.pdf"))
        plt.close()

    def plot_gender_preference_portion_vs_prestige(self, prestige_bin_col_name):
        plt.clf()
        colors = ["#6E99DD", "#FF5072", "#3ac8a4", "#E66100", "#5D3A9B", "#000000"]
        sns.set_palette(sns.color_palette(colors))
        plt.figure(figsize=(4, 6))
        if prestige_bin_col_name == "prestige_tenth_bin":
            prestige_list = sorted(set(self.ins_info_df["prestige_tenth_bin"]))
        else:
            prestige_list = ["Low", "Mid", "High"]
        male_portion_list, female_portion_list, balanced_portion_list = [], [], []
        for prestige in prestige_list:
            this_prestige_level = self.ins_info_df[(self.ins_info_df[prestige_bin_col_name] == prestige)]
            male_count = Counter(this_prestige_level["gender_preference"])[1]
            female_count = Counter(this_prestige_level["gender_preference"])[2]
            balance_count = Counter(this_prestige_level["gender_preference"])[0]
            total_count = len(this_prestige_level)
            male_portion_list.append(male_count / total_count)
            female_portion_list.append(female_count / total_count)
            balanced_portion_list.append(balance_count / total_count)

        # print(prestige_list, male_portion_list, female_portion_list, balanced_portion_list)
        plt.plot(prestige_list, male_portion_list, "o-", color=sns.color_palette()[0], markersize=15)
        plt.plot(prestige_list, female_portion_list, "o-", color=sns.color_palette()[1], markersize=15)
        plt.plot(prestige_list, balanced_portion_list, "o-", color=sns.color_palette()[2], markersize=15)
        plt.xlabel("Institution Prestige", fontsize=25)
        plt.ylabel("Fraction of Gender-Preferenced Institutions")
        if prestige_bin_col_name == "prestige_30_60_bin":
            plt.xticks(ticks=[0, 1, 2], labels=["<30", "<60", ">60"])
        if prestige_bin_col_name == "prestige_tenth_bin":
            plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                       labels=["<%s" % percentile for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]] + [">90"],
                       fontsize=10)
        else:
            plt.xticks(["Low", "Mid", "High"])

        total_ins_num = len([i for i in self.gender_preference_dict if self.gender_preference_dict[i] != -1])
        male_baseline = Counter(self.gender_preference_dict.values())[1] / total_ins_num
        female_baseline = Counter(self.gender_preference_dict.values())[2] / total_ins_num
        balanced_baseline = Counter(self.gender_preference_dict.values())[0] / total_ins_num

        plt.axhline(male_baseline, color=sns.color_palette()[0], ls="--", linewidth=1)
        plt.axhline(female_baseline, color=sns.color_palette()[1], ls="--", linewidth=1)
        plt.axhline(balanced_baseline, color=sns.color_palette()[2], ls="--", linewidth=1)
        plt.ylim(0, )
        plt.tight_layout()

        preference_prestige_plot_meta_path = os.path.join(self.save_fig_path, f"preference_prestige")
        try:
            os.makedirs(preference_prestige_plot_meta_path, exist_ok=True)
        except:
            pass

        preference_prestige_plot_path = os.path.join(preference_prestige_plot_meta_path, prestige_bin_col_name)
        try:
            os.makedirs(preference_prestige_plot_path, exist_ok=True)
        except:
            pass
        plt.savefig(os.path.join(preference_prestige_plot_path, f"{self.preference_type}_bf10.pdf"))

    def cramer_v(self):
        # plot heatmap of the result and solo exhibition result
        reference_gender_preference_dict = json.load(open(
            os.path.join("..", "..", "..", "results",
                         f"threshold_{self.genderizeio_threshold}_filter_{self.remove_birth}",
                         "data",
                         f"year_{self.career_start_threshold}", f"gender_{self.preference_type}_ins_bf10.json")))

        pairs = []
        print(len(reference_gender_preference_dict), len(self.gender_preference_dict))
        for key in self.gender_preference_dict:
            if key not in reference_gender_preference_dict:
                continue
            solo_result = self.gender_preference_dict[key]
            result = reference_gender_preference_dict[key]
            pairs.append((result, solo_result))
        pair_count = Counter(pairs)
        matrix = np.zeros((3, 3))
        for each_pair in pair_count:
            matrix[each_pair[0]][each_pair[1]] = pair_count[each_pair]

        print(pair_count)
        print(matrix)

        X2 = stats.chi2_contingency(matrix, correction=False)[0]
        n = np.sum(matrix)
        minDim = min(matrix.shape) - 1

        # calculate Cramer's V
        V = np.sqrt((X2 / n) / minDim)
        print(V)

        plt.clf()
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt=".0f", cmap="YlGnBu")
        plt.ylabel("Exhibition Gender Inequality")
        plt.xlabel("Solo Exhibition Gender Inequality")
        plt.title(f"Cramer's V Coefficient: {V:.2f}")
        plt.tight_layout()
        if self.preference_type == "neutral":
            plt.savefig(os.path.join(self.save_fig_path, f"c-gender_{self.preference_type}_heatmap.pdf"))
        else:
            plt.savefig(os.path.join(self.save_fig_path, f"d-gender_{self.preference_type}_heatmap.pdf"))

def main():
    parser = argparse.ArgumentParser(
        description='select artists with career start year > [year_threshold]')
    parser.add_argument('-t', '--genderizeio_threshold', type=float, help='genderize.io threshold', default=0.6)
    parser.add_argument('-f', '--remove_birth', action=argparse.BooleanOptionalAction)
    parser.add_argument('-y', '--career_start_threshold', type=int,
                        help='earliest career start year of selected artists', default=1990)
    parser.add_argument('-p', '--preference_type', type=str, help='neutral or balance')
    args = parser.parse_args()

    plotter = PlotGenderPreference(args.genderizeio_threshold, args.remove_birth,
                                   args.career_start_threshold, args.preference_type, False)
    plotter.plot_pie_chart(tag="all")
    plotter.plot_pie_chart(tag="museum")
    plotter.plot_pie_chart(tag="gallery")
    plotter.plot_gender_preference_portion_vs_prestige(prestige_bin_col_name="prestige_tenth_bin")
    plotter.plot_gender_preference_portion_vs_prestige(prestige_bin_col_name="prestige_40_70_bin")
    plotter.plot_gender_preference_portion_vs_prestige(prestige_bin_col_name="prestige_30_60_bin")
    plotter.cramer_v()


if __name__ == '__main__':
    main()
