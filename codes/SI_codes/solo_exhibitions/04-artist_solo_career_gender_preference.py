import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

from plot_config import *


def most_common(s):
    return pd.Series(s).mode().values[0]


class CareerGenderPreference:
    def __init__(self, genderizeio_threshold, remove_birth, career_start_threshold, preference_type,
                 num_exhibition_threshold, save):
        self.genderizeio_threshold = genderizeio_threshold
        self.remove_birth = remove_birth
        self.career_start_threshold = career_start_threshold
        self.preference_type = preference_type
        self.num_exhibition_threshold = num_exhibition_threshold
        self.save = save

        self.gallery_dict = json.load(open("../../../raw_data/gallery_dict.json"))

        self.save_data_path = os.path.join("..", "..", "..", "results",
                                           f"threshold_{self.genderizeio_threshold}_filter_{self.remove_birth}",
                                           "data",
                                           f"year_{self.career_start_threshold}")
        self.save_fig_path = os.path.join("..", "..", "..", "results",
                                          "SI",
                                          "fig4")

        # read data
        self.artists_recog_select = pd.read_csv(os.path.join(self.save_data_path, "artists_recog_select.csv"))
        self.shows_select = pd.read_csv(os.path.join(self.save_data_path, "solo_shows_select.csv"))
        self.gender_preference_dict = json.load(open(
            os.path.join(self.save_fig_path, f"gender_{self.preference_type}_ins_solo.json")))

        self.shows_select["gender_preference"] = [int(self.gender_preference_dict[str(ins)])
                                                  if str(ins) in self.gender_preference_dict else np.nan
                                                  for ins in self.shows_select["institution"]]

        self.null_portion = self.get_null_portion()

        self.agg_fname = os.path.join(self.save_fig_path, f"artist_solo_exh_info_agg_{self.preference_type}.csv")
        if not os.path.exists(self.agg_fname):
            self.agg = self.get_agg_artist_df()
        else:
            self.agg = pd.read_csv(self.agg_fname)

        # filter agg based on num_exhibition_threshold
        self.agg_legit = self.agg[(self.agg["exhibition_count"] >= self.num_exhibition_threshold) & (
                self.agg["portion_overall"] != (0, 0, 0))]

    def get_null_portion(self):
        count_null_model = Counter(self.gender_preference_dict.values())
        total_num_ins = len(self.gender_preference_dict)
        null_portion = np.array([count_null_model[1] / total_num_ins,
                                 count_null_model[2] / total_num_ins,
                                 count_null_model[0] / total_num_ins])
        return null_portion

    def get_artist_info_dicts(self):
        portion_overall_dict, portion_init_dict, portion_late_dict = {}, {}, {}
        prestige_init_dict, prestige_late_dict = {}, {}
        museum_gallery_portion_overall_dict, museum_gallery_portion_init_dict, museum_gallery_portion_late_dict = {}, {}, {}
        museum_gallery_sliding_window = {}
        for artist in set(self.shows_select["artist"]):
            this_person_shows = self.shows_select[(self.shows_select["artist"] == artist)].dropna(
                subset=["gender_preference"])
            num_shows = len(this_person_shows)
            if len(this_person_shows) <= 5:
                portion_overall_dict[artist] = (0, 0, 0)
                portion_init_dict[artist], portion_late_dict[artist] = (0, 0, 0), (0, 0, 0)
                prestige_init_dict[artist], prestige_late_dict[artist] = 0, 0
                museum_gallery_portion_overall_dict[artist] = (0, 0)
                museum_gallery_portion_init_dict[artist], museum_gallery_portion_late_dict[artist] = (0, 0), (0, 0)
                museum_gallery_sliding_window[artist] = []
            else:
                # total portion
                gender_count = Counter(this_person_shows["gender_preference"])
                portion_overall_dict[artist] = (gender_count[1] / num_shows,
                                                gender_count[2] / num_shows,
                                                gender_count[0] / num_shows)
                # init portion
                gender_count_init = Counter(list(this_person_shows["gender_preference"])[:5])
                portion_init_dict[artist] = (gender_count_init[1] / 5,
                                             gender_count_init[2] / 5,
                                             gender_count_init[0] / 5)
                # late portion
                gender_count_late = Counter(
                    list(this_person_shows["gender_preference"])[-5:])
                portion_late_dict[artist] = (gender_count_late[1] / 5,
                                             gender_count_late[2] / 5,
                                             gender_count_late[0] / 5)
                # init prestige
                prestige_init = np.mean(this_person_shows["percentile_prestige"].iloc[:5])
                prestige_init_dict[artist] = prestige_init
                # late parestige
                prestige_late = np.mean(this_person_shows["percentile_prestige"].iloc[-5:])
                prestige_late_dict[artist] = prestige_late
                # museum gallery portion
                ins_type_map = {"museum": 1, "gallery": 2}
                ins_type = [ins_type_map[self.gallery_dict[str(ins)]["type"]] for ins in
                            this_person_shows["institution"]]
                museum_gallery_sliding_window[artist] = list(
                    pd.Series(ins_type).rolling(window=5).apply(most_common).iloc[5 - 1:].values)
                # museum gallery count
                museum_gallery_count_overall = Counter(ins_type)
                museum_gallery_portion_overall_dict[artist] = (
                    museum_gallery_count_overall[1], museum_gallery_count_overall[2])
                # init museum gallery count
                museum_gallery_count_init = Counter(ins_type[:5])
                museum_gallery_portion_init_dict[artist] = (
                    museum_gallery_count_init[1] / 5, museum_gallery_count_init[2] / 5)
                # late museum gallery count
                museum_gallery_count_late = Counter(ins_type[-5:])
                museum_gallery_portion_late_dict[artist] = (
                    museum_gallery_count_late["museum"] / 5, museum_gallery_count_late["gallery"] / 5)
        artists_info_dicts = {"portion_overall_dict": portion_overall_dict,
                              "portion_init_dict": portion_init_dict,
                              "portion_late_dict": portion_late_dict,
                              "prestige_init_dict": prestige_init_dict,
                              "prestige_late_dict": prestige_late_dict,
                              "museum_gallery_portion_overall_dict": museum_gallery_portion_overall_dict,
                              "museum_gallery_portion_init_dict": museum_gallery_portion_init_dict,
                              "museum_gallery_portion_late_dict": museum_gallery_portion_late_dict,
                              "museum_gallery_sliding_window": museum_gallery_sliding_window}
        return artists_info_dicts

    def get_agg_artist_df(self):
        artists_info_dicts = self.get_artist_info_dicts()
        number_of_exhibitions = self.shows_select.groupby("artist")["show"].count().reset_index().rename(
            columns={"show": "exhibition_count"})
        different_ins_count = self.shows_select.groupby("artist")["institution"].nunique().reset_index().rename(
            columns={"institution": "different_ins_count"})
        career_prestige = self.shows_select.groupby("artist")["percentile_prestige"].mean().reset_index().rename(
            columns={"percentile_prestige": "career_prestige"})  # mean
        gender = self.shows_select[["artist", "gender_recog"]].drop_duplicates().reset_index()
        agg = number_of_exhibitions.join(career_prestige.set_index("artist"), on="artist")
        agg = agg.join(different_ins_count.set_index("artist"), on="artist")
        agg = agg.join(gender.set_index("artist"), on="artist")

        agg["round_prestige"] = [round(item, 1) for item in agg["career_prestige"]]

        agg["portion_overall"] = [artists_info_dicts["portion_overall_dict"][artist] for artist in agg["artist"]]
        agg["portion_init"] = [artists_info_dicts["portion_init_dict"][artist] for artist in agg["artist"]]
        agg["portion_late"] = [artists_info_dicts["portion_late_dict"][artist] for artist in agg["artist"]]

        agg["prestige_init"] = [artists_info_dicts["prestige_init_dict"][artist] for artist in agg["artist"]]
        agg["prestige_late"] = [artists_info_dicts["prestige_late_dict"][artist] for artist in agg["artist"]]

        agg["museum_gallery_portion_overall"] = [artists_info_dicts["museum_gallery_portion_overall_dict"][artist]
                                                 for artist in agg["artist"]]
        agg["museum_gallery_portion_init"] = [artists_info_dicts["museum_gallery_portion_init_dict"][artist]
                                              for artist in agg["artist"]]
        agg["museum_gallery_portion_late"] = [artists_info_dicts["museum_gallery_portion_late_dict"][artist]
                                              for artist in agg["artist"]]
        agg["museum_gallery_sliding_window"] = [artists_info_dicts["museum_gallery_sliding_window"][artist]
                                                for artist in agg["artist"]]

        # create co-exhibition gender
        gender_map = {0: 1, 1: 2, 2: 0}
        agg["ins_gender"] = [gender_map[np.argmax((np.array(portion_overall) - self.null_portion) / self.null_portion)]
                             for portion_overall in agg["portion_overall"]]
        agg["ins_gender_init"] = [
            gender_map[np.argmax((np.array(portion_overall) - self.null_portion) / self.null_portion)]
            for portion_overall in agg["portion_init"]]
        agg["ins_gender_late"] = [
            gender_map[np.argmax((np.array(portion_overall) - self.null_portion) / self.null_portion)]
            for portion_overall in agg["portion_late"]]

        # create prestige bin
        pmap = {1: "Low", 2: "Mid", 3: "High"}
        bin_edge_40_70 = [0, np.percentile(agg["career_prestige"], 40), np.percentile(agg["career_prestige"], 70), 1.1]
        bin_edge_30_60 = [0, np.percentile(agg["career_prestige"], 30), np.percentile(agg["career_prestige"], 60), 1.1]
        bin_edge_tenth = np.array([np.percentile(agg["career_prestige"], percentile)
                                   for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]] + [1.1])

        agg["prestige_40_70_bin"] = [pmap[np.digitize(item, bin_edge_40_70)] for item in agg["career_prestige"]]
        agg["prestige_30_60_bin"] = [pmap[np.digitize(item, bin_edge_30_60)] for item in agg["career_prestige"]]
        agg["prestige_tenth_bin"] = [np.digitize(item, bin_edge_tenth) for item in agg["career_prestige"]]

        agg["early_prestige_40_70_bin"] = [pmap[np.digitize(item, bin_edge_40_70)] for item in agg["prestige_init"]]
        agg["late_prestige_40_70_bin"] = [pmap[np.digitize(item, bin_edge_40_70)] for item in agg["prestige_late"]]

        agg.to_csv(self.agg_fname)
        return agg

    def plot_coexhibit_gender_vs_gender_count(self, coexhibit_gender_col):
        agg_percent = self.agg_legit.groupby("gender_recog")[coexhibit_gender_col].value_counts(normalize=True)
        agg_percent = agg_percent.mul(100)
        agg_percent = agg_percent.rename('Percent').reset_index()
        plt.figure(figsize=(8, 6))
        g = sns.barplot(x="gender_recog", y='Percent', hue=coexhibit_gender_col, hue_order=[
            1, 2, 0], order=["Male", "Female"], data=agg_percent)
        g.set_xticklabels(["Man", "Woman"])
        g.set_ylim(0, )
        for p in g.patches:
            txt = str(p.get_height().round(1)) + '%'
            txt_x = p.get_x()
            txt_y = p.get_height()
            g.text(txt_x + p.get_width() / 2., txt_y, txt,
                   ha="center", va="top", fontsize=20, color="white")
        g.set_xlabel("")
        g.set_ylabel("Portion of Artist of Co-exhibition Gender")
        g.tick_params(axis='x', labelsize=25)
        g.legend_.remove()
        plt.tight_layout()
        coexh_gender_vs_gender_fig_path = os.path.join(self.min_exh_fig_path, "coexh_gender_vs_gender")
        os.makedirs(coexh_gender_vs_gender_fig_path, exist_ok=True)
        if coexhibit_gender_col == "ins_gender_init":
            fig_path = os.path.join(coexh_gender_vs_gender_fig_path, "early_career")
            os.makedirs(fig_path, exist_ok=True)
            plt.savefig(os.path.join(fig_path, f"{self.preference_type}.pdf"))
        else:
            fig_path = os.path.join(coexh_gender_vs_gender_fig_path, "full_career")
            os.makedirs(fig_path, exist_ok=True)
            plt.savefig(os.path.join(fig_path, f"{self.preference_type}.pdf"))
        if self.save and self.preference_type == "neutral":
            agg_percent.to_csv('../main_paper_plot_data/ins_gender.csv', index=False)

    def plot_coexhibit_gender_vs_gender_vs_prestige_count(self, prestige_bin_col_name):
        agg_percent = self.agg_legit.groupby(["gender_recog", prestige_bin_col_name])["ins_gender"].value_counts(
            normalize=True)
        agg_percent = agg_percent.mul(100)
        agg_percent = agg_percent.rename('Percent').reset_index()
        with sns.axes_style("whitegrid"):
            if prestige_bin_col_name == "prestige_tenth_bin":
                order = sorted(list(set(agg_percent[prestige_bin_col_name])))
            else:
                order = ["Low", "Mid", "High"]
            ax = sns.catplot(x=prestige_bin_col_name, y='Percent', hue="ins_gender", col="gender_recog", hue_order=[
                1, 2, 0], kind="point", col_order=["Male", "Female"], order=order, data=agg_percent,
                             legend=False, sharex=False)
        for i in range(2):
            texts = []
            for j, p in enumerate(ax.axes[0][i].lines):
                xlist, ylist = p.get_data()
                if len(ylist) < 3:
                    continue
                for x, y in zip(xlist, ylist):
                    texts.append(ax.axes[0][i].text(
                        x, y, "%.1f" % y + "%", ha="center", va="top", color="black", fontsize=20))
            adjust_text(texts, ax=ax.axes[0][i])

        ax.axes[0][0].set_xlabel("")
        ax.axes[0][1].set_xlabel("")
        ax.axes[0][0].set_ylabel("Portion of Artist of Co-exhibit Gender")
        ax.axes[0][1].set_ylabel("")
        ax.axes[0][0].set_title("Man Artist", fontsize=25)
        ax.axes[0][1].set_title("Woman Artist", fontsize=25)
        ax.axes[0][0].tick_params(axis='x', labelsize=25)
        ax.axes[0][1].tick_params(axis='x', labelsize=25)
        ax.axes[0][0].set_ylim(0, )
        ax.axes[0][1].set_ylim(0, )
        if prestige_bin_col_name == "prestige_40_70_bin":
            ax.axes[0][0].set_xticklabels(["Low", "Mid", "High"])
            ax.axes[0][1].set_xticklabels(["Low", "Mid", "High"])
        if prestige_bin_col_name == "prestige_30_60_bin":
            ax.axes[0][0].set_xticklabels(["<30", "<60", ">60"])
            ax.axes[0][1].set_xticklabels(["<30", "<60", ">60"])
        if prestige_bin_col_name == "prestige_tenth_bin":
            ax.axes[0][0].set_xticklabels(
                ["<%s" % percentile for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]] + [">90"],
                fontsize=10)
            ax.axes[0][1].set_xticklabels(
                ["<%s" % percentile for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]] + [">90"],
                fontsize=10)
        plt.tight_layout()

        coexh_gender_vs_gender_vs_prestige_fig_path = os.path.join(self.save_fig_path,
                                                                   "coexh_gender_vs_gender_vs_prestige")
        os.makedirs(coexh_gender_vs_gender_vs_prestige_fig_path, exist_ok=True)

        fig_path = os.path.join(coexh_gender_vs_gender_vs_prestige_fig_path, prestige_bin_col_name)
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(os.path.join(fig_path, f"{self.preference_type}.pdf"))

    @staticmethod
    def cal_confusion_matrix(count):
        confusion_matrix = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                i_index = i + 1
                j_index = j + 1
                if i_index == 3:
                    i_index = 0
                if j_index == 3:
                    j_index = 0
                confusion_matrix[i][j] = count[(i_index, j_index)]
        normalize = confusion_matrix / np.sum(confusion_matrix, axis=1).reshape(3, 1)
        return normalize

    def plot_coexhibit_gender_lockin(self):
        count = Counter(tuple(zip(self.agg_legit["ins_gender_init"], self.agg_legit["ins_gender_late"])))
        normalize_cm = self.cal_confusion_matrix(count)
        plt.figure(figsize=(6, 6))
        ax = sns.heatmap(normalize_cm, annot=True, fmt=".2%",
                         cmap="Blues", annot_kws={"size": 25}, cbar=False)
        plt.xlabel("Late Co-exhibition Gender", fontsize=25)
        plt.ylabel("Early Co-exhibition Gender", fontsize=25)
        ax.set_xticklabels(["Co-exhibit\nMan", "Co-exhibit\nWoman", "Co-exhibit\n%s" % self.preference_type.title()])
        plt.yticks([0.5, 1.5, 2.5],
                   ["Co-exhibit\nMan", "Co-exhibit\nWoman", "Co-exhibit\n%s" % self.preference_type.title()],
                   ma="center", va="center")
        plt.tight_layout()
        lockin_fig_path = os.path.join(self.save_fig_path, "lockin")
        os.makedirs(lockin_fig_path, exist_ok=True)

        fig_path = os.path.join(lockin_fig_path, "all")
        os.makedirs(fig_path, exist_ok=True)

        plt.savefig(os.path.join(fig_path, f"{self.preference_type}.pdf"))

    def plot_coexhibit_gender_lockin_prestige(self):
        for prestige in ["Low", "Mid", "High"]:
            agg_legit_select = self.agg_legit[self.agg_legit["prestige_40_70_bin"] == prestige]
            count = Counter(tuple(zip(agg_legit_select["ins_gender_init"], agg_legit_select["ins_gender_late"])))
            normalize_cm = self.cal_confusion_matrix(count)
            plt.figure(figsize=(6, 6))
            ax = sns.heatmap(normalize_cm, annot=True, fmt=".2%",
                             cmap="Blues", annot_kws={"size": 25}, cbar=False)
            plt.xlabel("Late Co-exhibition Gender", fontsize=25)
            plt.ylabel("Early Co-exhibition Gender", fontsize=25)
            ax.set_xticklabels(
                ["Co-exhibit\nMan", "Co-exhibit\nWoman", "Co-exhibit\n%s" % self.preference_type.title()])
            plt.yticks([0.5, 1.5, 2.5],
                       ["Co-exhibit\nMan", "Co-exhibit\nWoman", "Co-exhibit\n%s" % self.preference_type.title()],
                       ma="center", va="center")
            plt.tight_layout()

            lockin_fig_path = os.path.join(self.save_fig_path, "lockin")
            os.makedirs(lockin_fig_path, exist_ok=True)

            fig_path = os.path.join(lockin_fig_path, f"{prestige.lower()}_prestige")
            os.makedirs(fig_path, exist_ok=True)

            plt.savefig(os.path.join(fig_path, f"{self.preference_type}.pdf"))

    def plot_coexhibit_gender_transition_prestige(self):
        ins_gender_dict = {1: "Co-exhibit Man",
                           2: "Co-exhibit Woman", 0: "Co-exhibit Neutral"}
        transition_prestige_condition = self.agg_legit.groupby(["ins_gender_init", "ins_gender_late"])[
            "prestige_40_70_bin"].value_counts(normalize=True).reset_index(name="portion")
        transition_prestige_condition["transition_type"] = ["%s-%s" % (ins_gender_dict[i], ins_gender_dict[j])
                                                            for (i, j) in
                                                            zip(transition_prestige_condition["ins_gender_init"],
                                                                transition_prestige_condition["ins_gender_late"])]

        fig, ax = plt.subplots(3, 1, figsize=(8, 12))
        for i in [1, 2, 0]:
            df_select = transition_prestige_condition[transition_prestige_condition["ins_gender_init"] == i]
            sns.barplot(x="ins_gender_late", y="portion", hue="prestige_40_70_bin",
                        order=[1, 2, 0], hue_order=["Low", "Mid", "High"], data=df_select, palette="Blues_d", ax=ax[i])
            ax[i].set_xlabel("")
            ax[i].set_ylabel("Portion of Artists")
            ax[i].set_xticklabels(["Co-exhibit\nMan", "Co-exhibit\nWoman", "Co-exhibit\nNeutral"])
            ax[i].get_legend().remove()
            ax[i].set_title("Early %s" % ins_gender_dict[i])

        plt.tight_layout()
        coexh_transition_fig_path = os.path.join(self.save_fig_path, "coexh_transition")
        os.makedirs(coexh_transition_fig_path, exist_ok=True)

        plt.savefig(os.path.join(coexh_transition_fig_path, f"{self.preference_type}.pdf"))


def main():
    parser = argparse.ArgumentParser(
        description='select artists with career start year > [year_threshold]')
    parser.add_argument('-t', '--genderizeio_threshold', type=float, help='genderize.io threshold', default=0.6)
    parser.add_argument('-f', '--remove_birth', action=argparse.BooleanOptionalAction)
    parser.add_argument('-y', '--career_start_threshold', type=int,
                        help='earliest career start year of selected artists', default=1990)
    parser.add_argument('-p', '--preference_type', type=str, help='neutral or balance')
    parser.add_argument('-n', '--num_exhibition_threshold', type=int, help='number of exhibition threshold', default=10)
    parser.add_argument('-a', '--save', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    career_gender_preference = CareerGenderPreference(args.genderizeio_threshold,
                                                      args.remove_birth,
                                                      args.career_start_threshold,
                                                      args.preference_type,
                                                      args.num_exhibition_threshold,
                                                      args.save)

    # career_gender_preference.plot_coexhibit_gender_vs_gender_count(coexhibit_gender_col="ins_gender")
    # career_gender_preference.plot_coexhibit_gender_vs_gender_count(coexhibit_gender_col="ins_gender_init")
    # career_gender_preference.plot_coexhibit_gender_vs_gender_vs_prestige_count(
    #     prestige_bin_col_name="prestige_40_70_bin")
    # career_gender_preference.plot_coexhibit_gender_vs_gender_vs_prestige_count(
    #     prestige_bin_col_name="prestige_30_60_bin")
    # career_gender_preference.plot_coexhibit_gender_vs_gender_vs_prestige_count(
    #     prestige_bin_col_name="prestige_tenth_bin")
    # career_gender_preference.plot_coexhibit_gender_lockin()
    # career_gender_preference.plot_coexhibit_gender_lockin_prestige()
    # career_gender_preference.plot_coexhibit_gender_transition_prestige()


if __name__ == '__main__':
    main()





#
# import pandas as pd
# import numpy as np
# import json
# from collections import Counter
# import matplotlib.pyplot as plt
# from adjustText import adjust_text
# from plot_config import *
# import argparse
#
# parser = argparse.ArgumentParser(
#     description='earliest year for data selection')
# parser.add_argument('year', metavar='year', type=int,
#                     help='select year')
# parser.add_argument('type', metavar='type', type=str,
#                     help='balance or neutral')
#
# args = parser.parse_args()
# select_year = args.year
# preference_type = args.type
#
# artists_select = pd.read_csv(
#     "../../../related_data_results/artists_select_%s.csv" % select_year)
# solo_shows_select = pd.read_csv(
#     "../../../related_data_results/solo_shows_select_%s.csv" % select_year)
# solo_gender_preference_dict = json.load(
#     open("../../../related_data_results/solo_gender_%s_ins.json" % preference_type))
# agg_ori = pd.read_csv(
#     "../../../related_data_results/artist_exh_info_agg_%s.csv" % preference_type)
#
# solo_shows_select["gender_preference"] = [int(solo_gender_preference_dict[
#     str(ins)]) if str(ins) in solo_gender_preference_dict else np.nan for ins in solo_shows_select["institution"]]
#
# count_null_model = Counter(solo_gender_preference_dict.values())
# p_null = np.array([count_null_model[1] / len(solo_gender_preference_dict), count_null_model[2] /
#                    len(solo_gender_preference_dict), count_null_model[0] / len(solo_gender_preference_dict)])
#
# portion_overall_dict = {}
# portion_init_dict = {}
# portion_late_dict = {}
# prestige_init_dict = {}
# prestige_late_dict = {}
# for artist in set(solo_shows_select["artist"]):
#     this_person_shows = solo_shows_select[
#         (solo_shows_select["artist"] == artist)].dropna(subset=["gender_preference"])
#     if len(this_person_shows) <= 5:
#         portion_overall_dict[artist] = (0, 0, 0)
#         portion_init_dict[artist] = (0, 0, 0)
#         portion_late_dict[artist] = (0, 0, 0)
#         prestige_init_dict[artist] = 0
#         prestige_late_dict[artist] = 0
#     else:
#         # total portion
#         gender_count = Counter(this_person_shows["gender_preference"])
#         portion_overall_dict[artist] = (gender_count[1] / len(this_person_shows), gender_count[
#             2] / len(this_person_shows), gender_count[0] / len(this_person_shows))
#         # init portion
#         gender_count_init = Counter(
#             list(this_person_shows["gender_preference"])[:5])
#         portion_init_dict[artist] = (
#             gender_count_init[1] / 5, gender_count_init[2] / 5, gender_count_init[0] / 5)
#         # late portion
#         gender_count_late = Counter(
#             list(this_person_shows["gender_preference"])[-5:])
#         portion_late_dict[artist] = (
#             gender_count_late[1] / 5, gender_count_late[2] /
#             5, gender_count_late[0] / 5)
#         # init prestige
#         prestige_init = np.mean(this_person_shows["percentile_prestige"][:5])
#         prestige_init_dict[artist] = prestige_init
#         # late parestige
#         prestige_late = np.mean(this_person_shows["percentile_prestige"][-5:])
#         prestige_late_dict[artist] = prestige_late


# number_of_exhibitions = solo_shows_select.groupby("artist")["show"].count(
# ).reset_index().rename(columns={"show": "exhibition_count"})
# different_ins_count = solo_shows_select.groupby("artist")["institution"].nunique(
# ).reset_index().rename(columns={"institution": "different_ins_count"})
# career_prestige = solo_shows_select.groupby("artist")["percentile_prestige"].mean(
# ).reset_index().rename(columns={"percentile_prestige": "career_prestige"})  # mean
# gender = solo_shows_select[["artist", "gender_recog"]
#                       ].drop_duplicates().reset_index()
# agg = number_of_exhibitions.join(
#     career_prestige.set_index("artist"), on="artist")
# agg = agg.join(different_ins_count.set_index("artist"), on="artist")
# agg = agg.join(
#     gender.set_index("artist"), on="artist")
#
# agg["round_prestige"] = [
#     round(item, 1) for item in agg["career_prestige"]]
# gender_map = {0: 1, 1: 2, 2: 0}
# agg["portion_overall"] = [portion_overall_dict[artist]
#                           for artist in agg["artist"]]
# agg["portion_init"] = [portion_init_dict[artist] for artist in agg["artist"]]
# agg["portion_late"] = [portion_late_dict[artist] for artist in agg["artist"]]
#
# agg["prestige_init"] = [prestige_init_dict[artist] for artist in agg["artist"]]
# agg["prestige_late"] = [prestige_late_dict[artist] for artist in agg["artist"]]
# agg["ins_gender"] = [gender_map[np.argmax((np.array(
#     portion_overall) - p_null) / p_null)] for portion_overall in agg["portion_overall"]]
# agg["ins_gender_init"] = [gender_map[np.argmax((np.array(
#     portion_overall) - p_null) / p_null)] for portion_overall in agg["portion_init"]]
# agg["ins_gender_late"] = [gender_map[np.argmax((np.array(
#     portion_overall) - p_null) / p_null)] for portion_overall in agg["portion_late"]]
# pmap = {1: "Low", 2: "Mid", 3: "High"}
# bin_edge = [0, np.percentile(agg_ori["career_prestige"], 40), np.percentile(
#     agg_ori["career_prestige"], 70), 1.1]
# print("bin_edge:", bin_edge)
# agg["prestige_bin"] = [pmap[np.digitize(
#     item, bin_edge)] for item in agg["career_prestige"]]
# bin_edge = [0, np.percentile(agg_ori[agg_ori["prestige_init"] > 0]["prestige_init"], 40), np.percentile(
#     agg_ori[agg_ori["prestige_init"] > 0]["prestige_init"], 70), 1.1]
# agg["early_prestige_bin"] = [pmap[np.digitize(
#     item, bin_edge)] for item in agg["prestige_init"]]
# bin_edge = [0, np.percentile(agg_ori[agg_ori["prestige_late"] > 0]["prestige_late"], 40), np.percentile(
#     agg_ori[agg_ori["prestige_late"] > 0]["prestige_late"], 70), 1.1]
# agg["late_prestige_bin"] = [pmap[np.digitize(
#     item, bin_edge)] for item in agg["prestige_late"]]
# print(len(agg))
#
# agg.to_csv("../../../related_data_results/artist_solo_exh_info_agg_%s.csv" %
#            preference_type)
#
# agg = pd.read_csv(
#     "../../../related_data_results/artist_solo_exh_info_agg_%s.csv" % preference_type)
# # select artist with more than 10 exhibitions
# agg_legit = agg[(agg["exhibition_count"] >= 10) &
#                 (agg["portion_overall"] != (0, 0, 0))]
# # agg_legit["prestige_bin"] = [cmap[np.digitize(
# #     item, [0, np.percentile(agg_legit["career_prestige"], 40), np.percentile(agg_legit["career_prestige"], 70), max(agg_legit["career_prestige"]) + 1])] for item in agg_legit["career_prestige"]]
# # agg_legit["prestige_bin"] = [pmap[np.digitize(
# # item, [0, np.percentile(agg_legit["career_prestige"], 40),
# # np.percentile(agg_legit["career_prestige"], 70), 1.1])] for item in
# # agg_legit["career_prestige"]]
#
# male_low_count = len(agg_legit[(agg_legit["gender_recog"] == "Male") & (
#     agg_legit["prestige_bin"] == "Low")])
# male_mid_count = len(agg_legit[(agg_legit["gender_recog"] == "Male") & (
#     agg_legit["prestige_bin"] == "Mid")])
# male_high_count = len(agg_legit[(agg_legit["gender_recog"] == "Male") & (
#     agg_legit["prestige_bin"] == "High")])
#
# female_low_count = len(agg_legit[(agg_legit["gender_recog"] == "Female") & (
#     agg_legit["prestige_bin"] == "Low")])
# female_mid_count = len(agg_legit[(agg_legit["gender_recog"] == "Female") & (
#     agg_legit["prestige_bin"] == "Mid")])
# female_high_count = len(agg_legit[(agg_legit["gender_recog"] == "Female") & (
#     agg_legit["prestige_bin"] == "High")])
#
# agg_percent = agg_legit.groupby("gender_recog")[
#     "ins_gender"].value_counts(normalize=True)
# agg_percent = agg_percent.mul(100)
# agg_percent = agg_percent.rename('Percent').reset_index()
#
# plt.figure(figsize=(8, 6))
# g = sns.barplot(x="gender_recog", y='Percent', hue="ins_gender", hue_order=[
#                 1, 2, 0], order=["Male", "Female"], data=agg_percent)
# g.set_xticklabels(["Man", "Woman"])
# g.set_ylim(0, )
#
# for p in g.patches:
#     txt = str(p.get_height().round(1)) + '%'
#     txt_x = p.get_x()
#     txt_y = p.get_height()
#     g.text(txt_x + p.get_width() / 2., txt_y - 5, txt,
#            ha="center", va="top", fontsize=20, color="white")
#
# # g.set_xticklabels(
# #     ["Male-Preferred\nDominant", "Female-Preferred\nDominant", "Balance\nDominant"])
# g.set_xlabel("")
# g.set_ylabel("Portion of Artist of Institutional Gender")
# g.tick_params(axis='x', labelsize=25)
# g.legend_.remove()
# plt.tight_layout()
# plt.savefig("../../../figures/SI_figures/solo_exhibitions/03-solo_ins_gender_count_diff_gender_%s.pdf" %
#             preference_type)
#
# agg_percent = agg_legit.groupby(["gender_recog", "prestige_bin"])[
#     "ins_gender"].value_counts(normalize=True)
# agg_percent = agg_percent.mul(100)
# agg_percent = agg_percent.rename('Percent').reset_index()
#
# texts = []
# with sns.axes_style("whitegrid"):
#     ax = sns.catplot(x="prestige_bin", y='Percent', hue="ins_gender", col="gender_recog", hue_order=[
#         1, 2, 0], kind="point", col_order=["Male", "Female"], order=["Low", "Mid", "High"], data=agg_percent, legend=False, sharex=False)
# for i in range(2):
#     texts = []
#     for j, p in enumerate(ax.axes[0][i].lines):
#         xlist, ylist = p.get_data()
#         if len(ylist) < 3:
#             continue
#         print(j, ylist)
#         for x, y in zip(xlist, ylist):
#             texts.append(ax.axes[0][i].text(
#                 x, y, "%.1f" % y + "%", ha="center", va="top", color="black", fontsize=20))
#     adjust_text(texts, ax=ax.axes[0][i])
#
# ax.axes[0][0].set_xlabel("")
# ax.axes[0][1].set_xlabel("")
# # ax.axes[0][2].set_xlabel("")
# ax.axes[0][0].set_ylabel("Portion of Artist of Institutional Gender")
# ax.axes[0][1].set_ylabel("")
# # ax.axes[0][2].set_ylabel("")
# ax.axes[0][0].set_title("Man Artist", fontsize=25)
# ax.axes[0][1].set_title("Woman Artist", fontsize=25)
# ax.axes[0][0].tick_params(axis='x', labelsize=25)
# ax.axes[0][1].tick_params(axis='x', labelsize=25)
# # ax.axes[0][2].set_title("High")
# ax.axes[0][0].set_ylim(0, )
# ax.axes[0][1].set_ylim(0, )
# ax.axes[0][0].set_xticklabels(
#     ["Low", "Mid", "High"])
# ax.axes[0][1].set_xticklabels(
#     ["Low", "Mid", "High"])
# # ax.axes[0][0].set_xticklabels(
# #     ["Low\n(%s)" % male_low_count, "Mid\n(%s)" % male_mid_count, "High\n(%s)" % male_high_count])
# # ax.axes[0][1].set_xticklabels(
# #     ["Low\n(%s)" % female_low_count, "Mid\n(%s)" % female_mid_count, "High\n(%s)" % female_high_count])
# plt.tight_layout()
# plt.savefig(
#     "../../../figures/SI_figures/solo_exhibitions/03-solo_ins_gender_count_diff_gender_prestige_%s.pdf" % preference_type)
# # if preference_type == "neutral":
# #     agg_percent.to_csv(
# #         '/Users/xindiwang/Dropbox (CCNR)/Success Team/Wang, Xindi/Gender Inequality in Art/For Alice/Main Paper Data/ins_gender_prestige.csv', index=False)
#
#
# agg_percent = agg_legit.groupby("gender_recog")[
#     "ins_gender_init"].value_counts(normalize=True)
# agg_percent = agg_percent.mul(100)
# agg_percent = agg_percent.rename('Percent').reset_index()
#
# plt.figure(figsize=(8, 6))
# g = sns.barplot(x="gender_recog", y='Percent', hue="ins_gender_init", hue_order=[
#                 1, 2, 0], order=["Male", "Female"], data=agg_percent)
# g.set_ylim(0, )
# g.set_xticklabels(["Man", "Woman"])
#
# for p in g.patches:
#     txt = str(p.get_height().round(1)) + '%'
#     txt_x = p.get_x()
#     txt_y = p.get_height()
#     g.text(txt_x + p.get_width() / 2., txt_y - 5, txt,
#            ha="center", va="top", fontsize=20, color="white")
#
# g.set_xlabel("")
# g.set_ylabel("Portion of Artist of Early Institutional Gender")
# g.tick_params(axis='x', labelsize=25)
# g.legend_.remove()
# plt.tight_layout()
# plt.savefig("../../../figures/SI_figures/solo_exhibitions/03-solo_ins_gender_early_count_diff_gender_%s.pdf" %
#             preference_type)
#
# # equality lock-in
# count = Counter(
#     tuple(zip(agg_legit["ins_gender_init"], agg_legit["ins_gender_late"])))
# print(count)
# confusion_matrix = np.zeros((3, 3))
# for i in range(3):
#     for j in range(3):
#         i_index = i + 1
#         j_index = j + 1
#         if i_index == 3:
#             i_index = 0
#         if j_index == 3:
#             j_index = 0
#         confusion_matrix[i][j] = count[(i_index, j_index)]
# print(confusion_matrix, np.sum(confusion_matrix, axis=0))
# normalize = confusion_matrix / np.sum(confusion_matrix, axis=1).reshape(3, 1)
# plt.figure(figsize=(6, 6))
# ax = sns.heatmap(normalize, annot=True, fmt=".2%",
#                  cmap="Blues", annot_kws={"size": 25}, cbar=False)
# plt.xlabel("Late Institional Gender", fontsize=25)
# plt.ylabel("Early Institutional Gender", fontsize=25)
# print(ax.get_yticks())
# ax.set_xticklabels(["Institutional\nMan",
#                     "Institutional\nWoman", "Institutional\n%s" % preference_type.title()])
# plt.yticks([0.5, 1.5, 2.5], ["Institutional\nMan",
#                              "Institutional\nWoman", "Institutional\n%s" % preference_type.title()], ma="center", va="center")
# plt.tight_layout()
# plt.savefig("../../../figures/SI_figures/solo_exhibitions/03-solo_gender_%s_lockin.pdf" % preference_type)
#
#
# # equality lock-in
# for prestige in ["Low", "Mid", "High"]:
#     agg_legit_select = agg_legit[agg_legit["prestige_bin"] == prestige]
#     count = Counter(
#         tuple(zip(agg_legit_select["ins_gender_init"], agg_legit_select["ins_gender_late"])))
#     print(count)
#     confusion_matrix = np.zeros((3, 3))
#     for i in range(3):
#         for j in range(3):
#             i_index = i + 1
#             j_index = j + 1
#             if i_index == 3:
#                 i_index = 0
#             if j_index == 3:
#                 j_index = 0
#             confusion_matrix[i][j] = count[(i_index, j_index)]
#     print(confusion_matrix, np.sum(confusion_matrix, axis=0))
#     normalize = confusion_matrix / \
#         np.sum(confusion_matrix, axis=1).reshape(3, 1)
#     plt.figure(figsize=(6, 6))
#     ax = sns.heatmap(normalize, annot=True, fmt=".2%",
#                      cmap="Blues", annot_kws={"size": 25}, cbar=False)
#     plt.xlabel("Late Institional Gender", fontsize=25)
#     plt.ylabel("Early Institutional Gender", fontsize=25)
#     print(ax.get_yticks())
#     ax.set_xticklabels(["Institutional\nMan",
#                         "Institutional\nWoman", "Institutional\n%s" % preference_type.title()])
#     plt.yticks([0.5, 1.5, 2.5], ["Institutional\nMan",
#                                  "Institutional\nWoman", "Institutional\n%s" % preference_type.title()], ma="center", va="center")
#     plt.tight_layout()
#     plt.savefig("../../../figures/SI_figures/solo_exhibitions/03-gender_%s_%s_lockin.pdf" %
#                 (preference_type, prestige))
#
# ins_gender_dict = {1: "Institutional Man",
#                    2: "Institutional Woman", 0: "Institutional Neutral"}
# transition_prestige_condition = agg_legit.groupby(["ins_gender_init", "ins_gender_late"])[
#     "prestige_bin"].value_counts(normalize=True).reset_index(name="portion")
# transition_prestige_condition["transition_type"] = ["%s-%s" % (ins_gender_dict[i], ins_gender_dict[j]) for (
#     i, j) in zip(transition_prestige_condition["ins_gender_init"], transition_prestige_condition["ins_gender_late"])]
#
# fig, ax = plt.subplots(3, 1, figsize=(8, 12))
# for i in [1, 2, 0]:
#     df_select = transition_prestige_condition[transition_prestige_condition["ins_gender_init"] == i]
#     sns.barplot(x="ins_gender_late", y="portion", hue="prestige_bin",
#                 order=[1, 2, 0], hue_order=["Low", "Mid", "High"], data=df_select, palette="Blues_d", ax=ax[i])
#     ax[i].set_xlabel("")
#     ax[i].set_ylabel("Portion of Artists")
#     ax[i].set_xticklabels(["Institutional\nMan",
#                            "Institutional\nWoman", "Institutional\nNeutral"])
#     ax[i].get_legend().remove()
#     ax[i].set_title("Early %s" % ins_gender_dict[i])
#
# plt.tight_layout()
# plt.savefig("../../../figures/SI_figures/solo_exhibitions/03-solo_gender_%s_transition_prestige.pdf" %
#             preference_type)
