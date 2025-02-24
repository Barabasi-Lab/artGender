import argparse
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import numpy as np
import pandas as pd
import json
import seaborn as sns
import os
import matplotlib.patches as mpatches
from adjustText import adjust_text

sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
colors = ["#6E99DD", "#FF5072", "#3ac8a4", "#720d0f", "#000000"]
sns.set_palette(sns.color_palette(colors))


class NeighborhoodAnalysis:

    def __init__(self, genderizeio_threshold, remove_birth, career_start_threshold, preference_type):
        self.genderizeio_threshold = genderizeio_threshold
        self.remove_birth = remove_birth
        self.career_start_threshold = career_start_threshold
        self.preference_type = preference_type

        self.attribute = "genderNeutral" if self.preference_type == "neutral" else "genderBalance"

        self.save_data_path = os.path.join("..", "..", "results",
                                           f"threshold_{self.genderizeio_threshold}_filter_{self.remove_birth}",
                                           "data")
        self.save_fig_path = os.path.join("..", "..", "results",
                                          "SI",
                                          "fig1")

        # get or generate dataframe
        dataframe_path = os.path.join(self.save_fig_path, f"institution_neighborhood_info_{self.preference_type}_bf10.csv")
        if os.path.exists(dataframe_path):
            self.df = pd.read_csv(dataframe_path)
        else:
            # read data
            self.G = nx.read_gml(os.path.join(self.save_data_path, "year_1990", "full_asso_bf10.gml"))
            self.percentile_prestige = json.load(open(os.path.join(self.save_data_path, "percentile_prestige.json")))
            self.gender_preference_dict = nx.get_node_attributes(self.G, self.attribute)
            self.df = self.get_neighborhood_info()

        # some constants for plot
        self.group_name_list = ["Man-Overrepresented",
                                "Woman-Overrepresented", f"Gender {self.preference_type.title()}"]

    def get_neighborhood_info(self):
        degree = dict(self.G.degree())
        print(len(degree), len(self.gender_preference_dict))
        weighted_degree = dict(self.G.degree(weight="weight"))
        node_list = []
        gender_preference_list = []
        node_prestige_list = []
        degree_list = []
        weighted_degree_list, out_weight_list, in_weight_list = [], [], []
        balance_neighbor_portion_list, male_neighbor_portion_list, female_neighbor_portion_list = [], [], []
        balance_suc_portion_list, male_suc_portion_list, female_suc_portion_list = [], [], []
        balance_pre_portion_list, male_pre_portion_list, female_pre_portion_list = [], [], []
        out_male_strength_portion_list, out_female_strength_portion_list, out_balanced_strength_portion_list = [], [], []
        in_male_strength_portion_list, in_female_strength_portion_list, in_balanced_strength_portion_list = [], [], []
        male_strength_portion_list, female_strength_portion_list, balanced_strength_portion_list = [], [], []
        for node in degree:
            if node not in self.gender_preference_dict:
                continue
            suc_neighbors = set(self.G.successors(node))
            pre_neighbors = set(self.G.predecessors(node))
            # all neighbors
            neighbors = set(suc_neighbors).union(pre_neighbors)
            neighbors_bin = [self.gender_preference_dict[n] for n in neighbors if n != node and n in self.gender_preference_dict]
            bin_count = Counter(neighbors_bin)
            if sum(bin_count.values()) == 0:
                continue
            portion_dict = {key: bin_count[key] / sum(bin_count.values()) for key in [0, 1, 2]}
            node_list.append(node)
            degree_list.append(degree[node])
            weighted_degree_list.append(weighted_degree[node])
            gender_preference_list.append(self.gender_preference_dict[node])
            node_prestige_list.append(self.percentile_prestige[node])
            balance_neighbor_portion_list.append(portion_dict[0])
            male_neighbor_portion_list.append(portion_dict[1])
            female_neighbor_portion_list.append(portion_dict[2])
            # out_neighbors
            neighbors_bin = [self.gender_preference_dict[n] for n in suc_neighbors if n != node and n in self.gender_preference_dict]
            bin_count = Counter(neighbors_bin)
            if sum(bin_count.values()) == 0:
                balance_suc_portion_list.append(np.nan)
                male_suc_portion_list.append(np.nan)
                female_suc_portion_list.append(np.nan)
            else:
                portion_dict = {key: bin_count[
                                         key] / sum(bin_count.values()) for key in [0, 1, 2]}
                balance_suc_portion_list.append(portion_dict[0])
                male_suc_portion_list.append(portion_dict[1])
                female_suc_portion_list.append(portion_dict[2])
            # in_neighbors
            neighbors_bin = [self.gender_preference_dict[n] for n in pre_neighbors if n != node and n in self.gender_preference_dict]
            bin_count = Counter(neighbors_bin)
            if sum(bin_count.values()) == 0:
                balance_pre_portion_list.append(np.nan)
                male_pre_portion_list.append(np.nan)
                female_pre_portion_list.append(np.nan)
            else:
                portion_dict = {key: bin_count[
                                         key] / sum(bin_count.values()) for key in [0, 1, 2]}
                balance_pre_portion_list.append(portion_dict[0])
                male_pre_portion_list.append(portion_dict[1])
                female_pre_portion_list.append(portion_dict[2])
            # strength
            out_edges = self.G.out_edges(node, data="weight")
            in_edges = self.G.in_edges(node, data="weight")
            # out_edges
            temp_0, temp_1, temp_2, out_weight_sum = 0, 0, 0, 0
            for (_, target, weight) in out_edges:
                if target == node or target not in self.gender_preference_dict:
                    continue
                target_bin = self.gender_preference_dict[target]
                out_weight_sum += weight
                if target_bin == 0:
                    temp_0 += weight
                elif target_bin == 1:
                    temp_1 += weight
                else:
                    temp_2 += weight
            if out_weight_sum == 0:
                out_male_strength_portion_list.append(np.nan)
                out_female_strength_portion_list.append(np.nan)
                out_balanced_strength_portion_list.append(np.nan)
            else:
                out_balanced_strength_portion_list.append(temp_0 / out_weight_sum)
                out_male_strength_portion_list.append(temp_1 / out_weight_sum)
                out_female_strength_portion_list.append(temp_2 / out_weight_sum)
            out_weight_list.append(out_weight_sum)
            # in_edges
            temp_0, temp_1, temp_2, in_weight_sum = 0, 0, 0, 0
            for (target, _, weight) in in_edges:
                if target == node or target not in self.gender_preference_dict:
                    continue
                target_bin = self.gender_preference_dict[target]
                in_weight_sum += weight
                if target_bin == 0:
                    temp_0 += weight
                elif target_bin == 1:
                    temp_1 += weight
                else:
                    temp_2 += weight
            if in_weight_sum == 0:
                in_male_strength_portion_list.append(np.nan)
                in_female_strength_portion_list.append(np.nan)
                in_balanced_strength_portion_list.append(np.nan)
            else:
                in_balanced_strength_portion_list.append(temp_0 / in_weight_sum)
                in_male_strength_portion_list.append(temp_1 / in_weight_sum)
                in_female_strength_portion_list.append(temp_2 / in_weight_sum)
            in_weight_list.append(in_weight_sum)
            # all edges
            temp_0, temp_1, temp_2, weight_sum = 0, 0, 0, 0
            for (_, target, weight) in out_edges:
                if target == node or target not in self.gender_preference_dict:
                    continue
                target_bin = self.gender_preference_dict[target]
                weight_sum += weight
                if target_bin == 0:
                    temp_0 += weight
                elif target_bin == 1:
                    temp_1 += weight
                else:
                    temp_2 += weight
            for (target, _, weight) in in_edges:
                if target == node or target not in self.gender_preference_dict:
                    continue
                target_bin = self.gender_preference_dict[target]
                weight_sum += weight
                if target_bin == 0:
                    temp_0 += weight
                elif target_bin == 1:
                    temp_1 += weight
                else:
                    temp_2 += weight
            if weight_sum == 0:
                male_strength_portion_list.append(np.nan)
                female_strength_portion_list.append(np.nan)
                balanced_strength_portion_list.append(np.nan)
            else:
                male_strength_portion_list.append(temp_1 / weight_sum)
                female_strength_portion_list.append(temp_2 / weight_sum)
                balanced_strength_portion_list.append(temp_0 / weight_sum)

        df = pd.DataFrame()
        df["node"] = node_list
        df["gender_preference"] = gender_preference_list
        df["prestige"] = node_prestige_list
        df["prestige_bin"] = [np.digitize(
            x, [0, np.percentile(df["prestige"], 40), np.percentile(df["prestige"], 70), 1.1]) for x in df["prestige"]]
        df["degree"] = degree_list
        df["weighted_degree"] = weighted_degree_list
        df["male_portion"] = male_neighbor_portion_list
        df["female_portion"] = female_neighbor_portion_list
        df["balance_portion"] = balance_neighbor_portion_list
        df["male_strength_portion"] = male_strength_portion_list
        df["female_strength_portion"] = female_strength_portion_list
        df["balance_strength_portion"] = balanced_strength_portion_list

        df["out_male_portion"] = male_suc_portion_list
        df["out_female_portion"] = female_suc_portion_list
        df["out_balance_portion"] = balance_suc_portion_list
        df["in_male_portion"] = male_pre_portion_list
        df["in_female_portion"] = female_pre_portion_list
        df["in_balance_portion"] = balance_pre_portion_list
        df["out_male_strength_portion"] = out_male_strength_portion_list
        df["out_female_strength_portion"] = out_female_strength_portion_list
        df["out_balance_strength_portion"] = out_balanced_strength_portion_list
        df["in_male_strength_portion"] = in_male_strength_portion_list
        df["in_female_strength_portion"] = in_female_strength_portion_list
        df["in_balance_strength_portion"] = in_balanced_strength_portion_list
        print(len(df), Counter(df["prestige_bin"]))
        df.to_csv(os.path.join(self.save_fig_path, f"institution_neighborhood_info_{self.preference_type}.csv"),
                  index=False)
        return df

    def stacked_bar(self, group_name_list, group_size_list, figname):
        portion_list = [[each_list[i] for each_list in group_size_list]
                        for i in range(len(group_size_list[0]))]
        fig, ax = plt.subplots(figsize=(7, 4))
        plt.box(False)
        if "neutral" in figname:
            colors = ["#FF5072", "#6E99DD", "#989898"]
        else:
            colors = ["#FF5072", "#6E99DD", "#3ac8a4"]
        for i, color in enumerate(colors):
            widths = portion_list[i]
            starts = sum(np.array(portion_list[:i]))
            if i == 0:
                starts = np.array([0] * len(widths))
            ax.barh(group_name_list, widths, left=starts, height=0.5, color=color)
            xcenters = starts + np.array(widths) / 2
            for y, (x, c) in enumerate(zip(xcenters, widths)):
                if "neutral" in figname:
                    ha = 'center'
                else:
                    if c > 0:
                        if i < 2:
                            ha = "left"
                        else:
                            ha = "center"
                ax.text(x, y, "%.1f" % (c * 100) + "%", ha=ha, va='center',
                            color="white", fontsize=12)
        plt.xticks([])
        plt.gca().set_yticklabels(["Man-Overrepresented", "Woman-Overrepresented",
                                   "Gender-Neutral"][::-1])
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_fig_path, "%s.pdf" % figname))

    @staticmethod
    def group_size(df, keys=["male_portion", "female_portion", "balance_portion"]):
        # male
        df_select = df[df["gender_preference"] == 1]
        male_portion = np.mean(df_select[keys[0]])
        female_portion = np.mean(df_select[keys[1]])
        balance_portion = np.mean(df_select[keys[2]])
        sum_portion = male_portion + female_portion + balance_portion
        male_group_size = [female_portion / sum_portion,
                           male_portion / sum_portion, balance_portion / sum_portion]
        # female
        df_select = df[df["gender_preference"] == 2]
        male_portion = np.mean(df_select[keys[0]])
        female_portion = np.mean(df_select[keys[1]])
        balance_portion = np.mean(df_select[keys[2]])
        sum_portion = male_portion + female_portion + balance_portion
        female_group_size = [female_portion / sum_portion,
                             male_portion / sum_portion, balance_portion / sum_portion]
        # balance/neutral
        df_select = df[df["gender_preference"] == 0]
        male_portion = np.mean(df_select[keys[0]])
        female_portion = np.mean(df_select[keys[1]])
        balance_portion = np.mean(df_select[keys[2]])
        sum_portion = male_portion + female_portion + balance_portion
        balance_group_size = [female_portion / sum_portion,
                              male_portion / sum_portion, balance_portion / sum_portion]
        return male_group_size, female_group_size, balance_group_size

    def plot_neighborhood_composition(self):
        # stacked plot for out strength composition
        male_group_size, female_group_size, balanced_group_size = self.group_size(
            self.df, keys=["out_male_strength_portion", "out_female_strength_portion", "out_balance_strength_portion"])
        group_size_list = [male_group_size, female_group_size,
                           balanced_group_size]
        self.stacked_bar(self.group_name_list[::-1], group_size_list[::-1],
                         f"out_strength_neighborhood_composition_{self.preference_type}_stacked")

    def plot_neighborhood_prestige(self):
        # if category == "man":
        #     index = 1
        # if category == "woman":
        #     index = 0
        # if category == "neutral" or category=="balance":
        #     index = 2
        pmap = {1: "Low", 2: "Mid", 3: "High"}
        man_preferred_color = sns.light_palette(
            "#6E99DD", reverse=False, as_cmap=False, n_colors=3)
        woman_preferred_color = sns.light_palette(
            "#FF5072", reverse=False, as_cmap=False, n_colors=3)
        neutral_preferred_color = sns.light_palette(
            "#989898", reverse=False, as_cmap=False, n_colors=3)
        color_order = {0: man_preferred_color[0],
                       1: woman_preferred_color[0],
                       2: neutral_preferred_color[0],
                       3: man_preferred_color[1],
                       4: woman_preferred_color[1],
                       5: neutral_preferred_color[1],
                       6: man_preferred_color[2],
                       7: woman_preferred_color[2],
                       8: neutral_preferred_color[2],
                       }
        text_order = ["Low"] * 3 + ["Mid"] * 3 + ["High"] * 3
        prestige_list = []
        man_preferred_portion_list = []
        woman_preferred_portion_list = []
        balance_preferred_portion_list = []
        gender_bias_list = []
        for prestige in [1, 2, 3]:
            male_group_size, female_group_size, balance_group_size = self.group_size(
                self.df[self.df["prestige_bin"] == prestige],
                keys=["out_male_strength_portion", "out_female_strength_portion", "out_balance_strength_portion"])
            man_preferred_portion_list += [male_group_size[1], female_group_size[1], balance_group_size[1]]
            woman_preferred_portion_list += [male_group_size[0], female_group_size[0], balance_group_size[0]]
            balance_preferred_portion_list += [male_group_size[2], female_group_size[2], balance_group_size[2]]
            prestige_list += [pmap[prestige]] * 3
            gender_bias_list += self.group_name_list

        df_prestige_portion = pd.DataFrame()
        df_prestige_portion["prestige"] = prestige_list
        df_prestige_portion["gender_bias"] = gender_bias_list
        df_prestige_portion["man_preferred_portion"] = man_preferred_portion_list
        df_prestige_portion["woman_preferred_portion"] = woman_preferred_portion_list
        df_prestige_portion["balance_preferred_portion"] = balance_preferred_portion_list

        plt.clf()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6), sharey=True)
        sns.barplot(x="gender_bias", y="man_preferred_portion", hue="prestige",
                    hue_order=["Low", "Mid", "High"], order=self.group_name_list,
                    data=df_prestige_portion,
                    palette=sns.light_palette("#6E99DD", reverse=False, as_cmap=False, n_colors=3),
                    ax=ax1)
        for i, bar in enumerate(ax1.patches):
            bar.set_color(color_order[i])
            ax1.text(x=bar.get_x() + bar.get_width() / 2., y=bar.get_height(),
                     s=text_order[i], va="top", ha="center")
        ax1.set_xlabel("")
        ax1.set_ylabel("Weighted Out Neighborhood Portion")
        ax1.set_title("Man-Overrepresented Institutions")
        ax1.legend_.remove()

        sns.barplot(x="gender_bias", y="woman_preferred_portion", hue="prestige",
                    hue_order=["Low", "Mid", "High"], order=self.group_name_list,
                    data=df_prestige_portion,
                    palette=sns.light_palette("#FF5072", reverse=False, as_cmap=False, n_colors=3), ax=ax2)
        for i, bar in enumerate(ax2.patches):
            bar.set_color(color_order[i])
            ax2.text(x=bar.get_x() + bar.get_width() / 2., y=bar.get_height(),
                     s=text_order[i], va="top", ha="center")
        ax2.set_xlabel("")
        ax2.set_ylabel("Weighted Out Neighborhood Portion")
        ax2.set_title("Woman-Overrepresented Institutions")
        ax2.legend_.remove()

        sns.barplot(x="gender_bias", y="balance_preferred_portion", hue="prestige",
                    hue_order=["Low", "Mid", "High"], order=self.group_name_list,
                    data=df_prestige_portion,
                    palette=sns.light_palette("#989898", reverse=False, as_cmap=False, n_colors=3), ax=ax3)
        for i, bar in enumerate(ax3.patches):
            bar.set_color(color_order[i])
            ax3.text(x=bar.get_x() + bar.get_width() / 2., y=bar.get_height(),
                     s=text_order[i], va="top", ha="center")
        ax3.set_xlabel("")
        ax3.set_ylabel("Weighted Out Neighborhood Portion")
        ax3.set_title("Gender Neutral Institutions")
        ax3.legend_.remove()

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_fig_path,
                         f"{self.preference_type}_out_strength_portion_prestige_bar.pdf"))
        #
        # pmap = {1: "Low", 2: "Mid", 3: "High"}
        # prestige_list = []
        # woman_preferred_portion_list = []
        # gender_bias_list = []
        # for prestige in [1, 2, 3]:
        #     male_group_size, female_group_size, balance_group_size = group_size(
        #         df[df["prestige_bin"] == prestige],
        #         keys=["out_male_strength_portion", "out_female_strength_portion", "out_balance_strength_portion"])
        #     # male_group_size, female_group_size, balance_group_size = group_size(
        #     # df[df["prestige_bin"] == prestige], keys=["male_strength_portion",
        #     # "female_strength_portion", "balance_strength_portion"])
        #     woman_preferred_portion_list += [male_group_size[0],
        #                                      female_group_size[0], balance_group_size[0]]
        #     prestige_list += [pmap[prestige]] * 3
        #     gender_bias_list += ["Man-Preferred", "Woman-Preferred", "Gender Neutral"]
        #
        # df_woman_preferred = pd.DataFrame()
        # df_woman_preferred["prestige"] = prestige_list
        # df_woman_preferred["gender_bias"] = gender_bias_list
        # df_woman_preferred["woman_preferred_portion"] = woman_preferred_portion_list
        # # print(df_woman_preferred)
        # plt.clf()
        # plt.figure(figsize=(8, 6))
        # ax = sns.barplot(x="gender_bias", y="woman_preferred_portion", hue="prestige",
        #                  hue_order=["Low", "Mid", "High"], order=["Man-Preferred", "Woman-Preferred", "Gender Neutral"],
        #                  data=df_woman_preferred,
        #                  palette=sns.light_palette("#FF5072", reverse=False, as_cmap=False, n_colors=3))
        # for i, bar in enumerate(ax.patches):
        #     bar.set_color(color_order[i])
        #     plt.text(x=bar.get_x() + bar.get_width() / 2., y=bar.get_height(),
        #              s=text_order[i], va="top", ha="center")
        # plt.xlabel("")
        # plt.ylabel("Weighted Out Neighborhood Portion")
        # plt.title("Woman-Preferred Institutions")
        # ax.legend_.remove()
        # plt.savefig(
        #     "../../figures/SI_figures/01-woman_preferred_%s_out_strength_portion_prestige_bar.pdf" % attribute)
        #
        # pmap = {1: "Low", 2: "Mid", 3: "High"}
        # prestige_list = []
        # woman_preferred_portion_list = []
        # gender_bias_list = []
        # for prestige in [1, 2, 3]:
        #     male_group_size, female_group_size, balance_group_size = group_size(
        #         df[df["prestige_bin"] == prestige],
        #         keys=["out_male_strength_portion", "out_female_strength_portion", "out_balance_strength_portion"])
        #     # male_group_size, female_group_size, balance_group_size = group_size(
        #     # df[df["prestige_bin"] == prestige], keys=["male_strength_portion",
        #     # "female_strength_portion", "balance_strength_portion"])
        #     woman_preferred_portion_list += [male_group_size[2],
        #                                      female_group_size[2], balance_group_size[2]]
        #     prestige_list += [pmap[prestige]] * 3
        #     gender_bias_list += ["Man-Preferred", "Woman-Preferred", "Gender Neutral"]
        #
        # df_woman_preferred = pd.DataFrame()
        # df_woman_preferred["prestige"] = prestige_list
        # df_woman_preferred["gender_bias"] = gender_bias_list
        # df_woman_preferred["woman_preferred_portion"] = woman_preferred_portion_list
        # # print(df_woman_preferred)
        # plt.clf()
        # plt.figure(figsize=(8, 6))
        # ax = sns.barplot(x="gender_bias", y="woman_preferred_portion", hue="prestige",
        #                  hue_order=["Low", "Mid", "High"], order=["Man-Preferred", "Woman-Preferred", "Gender Neutral"],
        #                  data=df_woman_preferred,
        #                  palette=sns.light_palette("#989898", reverse=False, as_cmap=False, n_colors=3))
        # for i, bar in enumerate(ax.patches):
        #     bar.set_color(color_order[i])
        #     plt.text(x=bar.get_x() + bar.get_width() / 2., y=bar.get_height(),
        #              s=text_order[i], va="top", ha="center")
        # plt.xlabel("")
        # plt.ylabel("Weighted Out Neighborhood Portion")
        # plt.title("Gender Neutral Institutions")
        # ax.legend_.remove()
        # plt.savefig(
        #     "../../figures/SI_figures/01-%s_out_strength_portion_prestige_bar.pdf" % attribute)


def main():
    parser = argparse.ArgumentParser(description='')
    parser = argparse.ArgumentParser(
        description='select artists with career start year > [year_threshold]')
    parser.add_argument('-t', '--genderizeio_threshold', type=float, help='genderize.io threshold', default=0.6)
    parser.add_argument('-f', '--remove_birth', action=argparse.BooleanOptionalAction)
    parser.add_argument('-y', '--career_start_threshold', type=int,
                        help='earliest career start year of selected artists', default=1990)
    parser.add_argument('-p', '--preference_type', type=str, help='neutral or balance')
    args = parser.parse_args()

    analyzer = NeighborhoodAnalysis(args.genderizeio_threshold, args.remove_birth,
                                    args.career_start_threshold, args.preference_type)

    analyzer.plot_neighborhood_composition()
    analyzer.plot_neighborhood_prestige()


if __name__ == '__main__':
    main()
