import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from plot_config import *
import argparse
import os




class TimeTrendAnalysis:
    def __init__(self, genderizeio_threshold, remove_birth, career_start_threshold):
        self.genderizeio_threshold = genderizeio_threshold
        self.remove_birth = remove_birth
        self.career_start_threshold = career_start_threshold
        self.save_data_path = os.path.join("..", "..", "results",
                                           f"threshold_{self.genderizeio_threshold}_filter_{self.remove_birth}",
                                           "data",
                                           f"year_{self.career_start_threshold}")

        self.save_fig_path = os.path.join("..", "..", "results", "SI",
                                          "fig2")

        os.makedirs(self.save_fig_path, exist_ok=True)

        self.artists_recog_select = pd.read_csv(os.path.join(self.save_data_path, "artists_recog_select.csv"))
        self.shows_select = pd.read_csv(os.path.join(self.save_data_path, "shows_select.csv"))
        self.career_span = pd.read_csv(os.path.join(self.save_data_path, "career_span_select.csv"))

        self.population_baseline_select = len(self.artists_recog_select[self.artists_recog_select[
                                                                            "gender_recog"] == "Male"]) / len(
            self.artists_recog_select[self.artists_recog_select["gender_recog"] == "Female"])
        self.exhibition_baseline_select = Counter(self.shows_select["gender_recog"])['Male'] / \
                                          Counter(self.shows_select["gender_recog"])['Female']

    def plot_panel_a_b(self):
        career_start_year_artist_count = self.career_span.groupby(["career_start_year", "gender_recog"])[
            "artist"].count(
        ).reset_index().pivot(index="career_start_year", columns="gender_recog", values="artist").reset_index()
        career_start_year_artist_count["gender_portion"] = career_start_year_artist_count["Male"] / \
                                                           career_start_year_artist_count["Female"]
        career_start_year_artist_count["sum"] = career_start_year_artist_count["Male"] + career_start_year_artist_count[
            "Female"]

        plt.plot(career_start_year_artist_count["career_start_year"],
                 career_start_year_artist_count["gender_portion"], "o-", markersize=8,
                 color=sns.color_palette()[-3])
        plt.xlim(1990, )
        plt.ylim(1, 3)
        plt.axhline(self.population_baseline_select, label="Population Gender Ratio", ls=':', lw=1,
                    color=sns.color_palette()[-3])
        plt.legend()
        plt.xlabel("Career Start Year")
        plt.ylabel("Men/Women Artists")
        plt.savefig(os.path.join(self.save_fig_path,
                                 f"a-population_ratio_career_start_{self.career_start_threshold}.pdf"))
        print(career_start_year_artist_count["career_start_year"],
              np.cumsum(career_start_year_artist_count["sum"]) / career_start_year_artist_count["sum"].sum())

        plt.clf()
        plt.plot(career_start_year_artist_count[
                     "career_start_year"],
                 np.cumsum(career_start_year_artist_count["sum"]) / career_start_year_artist_count["sum"].sum(), "o-",
                 markersize=8, color=sns.color_palette()[-3])
        plt.xlim(1990, )
        plt.xlabel("Career Start Year")
        plt.ylabel("Cumulative Portion of Artists")
        plt.savefig(os.path.join(self.save_fig_path,
                                 f"b-population_cumu_portion_career_start_year_{self.career_start_threshold}.pdf"))

    def plot_panel_c(self):
        plt.figure()
        male_artist_count_dict = {}
        female_artist_count_dict = {}
        male_exhibit_count = {}
        female_exhibit_count = {}
        total_artist_count_dict = {}
        total_exhibit_count_dict = {}
        for year in set(self.shows_select["show_year"]):
            this_year = self.shows_select[self.shows_select["show_year"] == year]
            male = this_year[this_year["gender_recog"] == "Male"]
            female = this_year[this_year["gender_recog"] == "Female"]
            male_exhibit_count[year] = len(male)
            female_exhibit_count[year] = len(female)
            male_artist_count_dict[year] = len(set(male["artist"]))
            female_artist_count_dict[year] = len(set(female["artist"]))
            total_exhibit_count_dict[year] = len(male) + len(female)
            total_artist_count_dict[year] = len(
                set(male["artist"])) + len(set(female["artist"]))

        female_artist_count_list = []
        male_artist_count_list = []
        for year in sorted(male_artist_count_dict):
            try:
                if female_artist_count_dict[year] != 0:
                    female_artist_count_list.append(female_artist_count_dict[year])
                else:
                    female_artist_count_list.append(0.1)
            except:
                female_artist_count_list.append(0.1)
            male_artist_count_list.append(male_artist_count_dict[year])

        male_exhibit_count_list = []
        female_exhibit_count_list = []
        for year in sorted(male_exhibit_count):
            male_exhibit_count_list.append(male_exhibit_count[year])
            if female_exhibit_count[year] != 0:
                female_exhibit_count_list.append(female_exhibit_count[year])
            else:
                female_exhibit_count_list.append(0.1)

        plt.clf()
        plt.figure(figsize=(8, 6))
        print(male_artist_count_list)
        print(female_artist_count_list)
        print(np.array(male_artist_count_list) /
              np.array(female_artist_count_list))
        plt.plot(sorted(male_artist_count_dict), np.array(male_artist_count_list) /
                 np.array(female_artist_count_list), 'o-', markersize=8, color=sns.color_palette()[-3],
                 label="All Artists")
        plt.plot(sorted(male_exhibit_count), np.array(male_exhibit_count_list) /
                 np.array(female_exhibit_count_list), 'o-', markersize=8, color=sns.color_palette()[-2],
                 label="Number of Exhibitions")
        plt.axhline(self.population_baseline_select,
                    label="Population Gender Ratio", ls=':', lw=1, color=sns.color_palette()[-3])
        plt.axhline(self.exhibition_baseline_select,
                    label="Exhibition Gender Ratio", ls=':', lw=1, color=sns.color_palette()[-2])
        plt.text(1990, self.population_baseline_select, "%.2f" %
                 self.population_baseline_select, color=sns.color_palette()[-3], va="center", fontsize=25)
        plt.text(1990, self.exhibition_baseline_select, "%.2f" %
                 self.exhibition_baseline_select, color=sns.color_palette()[-2], va="center", fontsize=25)
        plt.legend()
        plt.xlabel('Exhibition Year', fontsize=25)
        plt.ylabel('Men/Women Ratio', fontsize=25)
        plt.xlim(min(sorted(male_artist_count_dict)), 2014)
        # plt.yticks([1, 1.5, 2, 2.5])
        # plt.xticks([1990, 1995, 2000, 2005, 2010])
        plt.ylim(0.99, )
        ylim = plt.gca().get_ylim()
        if ylim[-1] > 10:
            plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_fig_path,
                                 f"c-population_exhibition_ratio_year_{self.career_start_threshold}.pdf"))

def main():
    parser = argparse.ArgumentParser(
        description='select artists with career start year > [year_threshold]')
    parser.add_argument('-t', '--genderizeio_threshold', type=float, help='genderize.io threshold', default=0.6)
    parser.add_argument('-f', '--remove_birth', action=argparse.BooleanOptionalAction)
    parser.add_argument('-y', '--career_start_threshold', type=int,
                        help='earliest career start year of selected artists', default=1990)
    args = parser.parse_args()
    print(args)

    assign_gender_preference = TimeTrendAnalysis(args.genderizeio_threshold, args.remove_birth,
                                                      args.career_start_threshold)
    assign_gender_preference.plot_panel_a_b()
    assign_gender_preference.plot_panel_c()


if __name__ == '__main__':
    main()