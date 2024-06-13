"""Select Data."""
import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

from plot_config import *


class SelectData:

    def __init__(self, genderizeio_threshold, remove_birth, career_start_threshold, save):
        # parse input
        self.genderizeio_threshold = genderizeio_threshold
        self.remove_birth = remove_birth
        self.career_start_threshold = career_start_threshold
        self.save = save

        folder_name = f"threshold_{self.genderizeio_threshold}_filter_{self.remove_birth}"

        data_path = os.path.join("..", "results", folder_name, "data")
        self.shows = pd.read_csv(os.path.join(data_path, "shows.csv"))
        self.solo_shows = pd.read_csv(os.path.join(data_path, "solo_shows.csv"))
        self.percentile_prestige = json.load(open(os.path.join(data_path, "percentile_prestige.json")))
        self.career_span = pd.read_csv(os.path.join(data_path, "career_span.csv"))
        self.sales = pd.read_csv(os.path.join(data_path, "sales.csv"))

        self.artists_recog = pd.read_csv("../processed_data/artists_gender_recog_%s.csv" % self.genderizeio_threshold)

        self.select_artists = None
        self.artists_recog_select = None
        self.shows_select = None
        self.solo_shows_select = None
        self.sales_select = None
        self.career_span_select = None

        self.save_data_path = os.path.join("..", "results", folder_name, "data", f"year_{self.career_start_threshold}")
        self.save_fig_path = os.path.join("..", "results", folder_name, "figures",
                                          f"year_{self.career_start_threshold}")
        try:
            os.makedirs(self.save_data_path, exist_ok=True)
            os.makedirs(self.save_fig_path, exist_ok=True)
        except:
            pass

    def get_select_artists(self):
        self.select_artists = self.career_span[
            (self.career_span["career_start_year"] >= self.career_start_threshold)]["artist"]

    def filter_data(self):
        self.artists_recog_select = self.artists_recog[self.artists_recog["artist"].isin(self.select_artists)]
        self.shows_select = self.shows[self.shows["artist"].isin(self.select_artists)]
        self.solo_shows_select = self.solo_shows[self.solo_shows["artist"].isin(self.select_artists)]
        self.sales_select = self.sales[self.sales["artist"].isin(self.select_artists)]
        self.career_span_select = self.career_span[self.career_span["artist"].isin(self.select_artists)]

    def save_data(self):
        self.artists_recog_select.to_csv(os.path.join(self.save_data_path, "artists_recog_select.csv"), index=False)
        self.shows_select.to_csv(os.path.join(self.save_data_path, "shows_select.csv"), index=False)
        self.solo_shows_select.to_csv(os.path.join(self.save_data_path, "solo_shows_select.csv"), index=False)
        self.career_span_select.to_csv(os.path.join(self.save_data_path, "career_span_select.csv"), index=False)
        self.sales_select.to_csv(os.path.join(self.save_data_path, "sales_select.csv"), index=False)

    def report_statistics(self):
        print("====================")
        print(f"SELECTED DATA STATISTICS (Genderize.io threshold: {self.genderizeio_threshold}, "
              f"Filter birth: {self.remove_birth}, Career start threshold: {self.career_start_threshold}")
        print("Number of Selected Artists", len(self.select_artists))
        print("Selected Artists Gender Info:", Counter(self.artists_recog_select["gender_recog"]))
        print("Involved Number of Institutions", len(set(self.shows_select["institution"])))
        print("Total Number of Exhibitions involved with Selected Artists:", len(set(self.shows_select["show"])))
        print("Total Number of Exhibition Records of Selected Artists (with multiple counting for group shows)",
              len(self.shows_select), Counter(self.shows_select["gender_recog"]))
        print("Selected Artists With Solo Shows Gender Info:",
              Counter(self.solo_shows_select[["artist", "gender_recog"]].drop_duplicates()["gender_recog"]))
        print("Total Number of Solo Exhibition Records of Selected Artists",
              len(self.solo_shows_select), Counter(self.solo_shows_select["gender_recog"]))
        print("Number of Selected Artists with Auction:", len(set(self.sales_select["artist"])))
        print("Total Number of Auctions of Selected Artists", len(self.sales_select))
        print("====================")
        print("POPULATION IMBALANCE")
        print("Male:", len(self.artists_recog_select[self.artists_recog_select["gender_recog"] == "Male"]))
        print("Female:", len(self.artists_recog_select[self.artists_recog_select["gender_recog"] == "Female"]))
        population_baseline_select = len(
            self.artists_recog_select[self.artists_recog_select["gender_recog"] == "Male"]) / len(
            self.artists_recog_select[self.artists_recog_select["gender_recog"] == "Female"])
        print("Population Male/Female:", population_baseline_select)
        print("====================")
        print("EXHIBITION IMBALANCE")
        print("Male:", len(self.shows_select[self.shows_select["gender_recog"] == "Male"]))
        print("Female:", len(self.shows_select[self.shows_select["gender_recog"] == "Female"]))
        print("Exhibition Male/Female:",
              Counter(self.shows_select["gender_recog"])['Male'] / Counter(self.shows_select["gender_recog"])['Female'])

    def plot_full_vs_select_data(self):
        fig, ax = plt.subplots(2, 1, sharex=False, figsize=(8, 6))

        # group by exhibition start year, artist count
        career_start_year_artist_count = self.career_span.groupby(["career_start_year", "gender_recog"])[
            "artist"].count(
        ).reset_index().pivot(index="career_start_year", columns="gender_recog", values="artist").reset_index()

        career_start_year_artist_count_select = self.career_span_select.groupby(["career_start_year", "gender_recog"])[
            "artist"].count().reset_index().pivot(index="career_start_year", columns="gender_recog",
                                                  values="artist").reset_index()

        # shows over year
        shows_count = self.shows.groupby(["show_year", "gender_recog"])["show"].count().reset_index().pivot(
            index="show_year", columns='gender_recog', values="show").reset_index()
        shows_select_count = self.shows_select.groupby(["show_year", "gender_recog"])[
            "show"].count().reset_index().pivot(index="show_year",
                                                columns='gender_recog',
                                                values="show").reset_index()

        ax[0].plot(career_start_year_artist_count["career_start_year"], career_start_year_artist_count["Male"],
                   label="Male Total")
        ax[0].plot(career_start_year_artist_count["career_start_year"], career_start_year_artist_count["Female"],
                   label="Female Total")
        ax[0].stackplot(career_start_year_artist_count_select["career_start_year"],
                        career_start_year_artist_count_select["Female"],
                        career_start_year_artist_count_select["Male"] - career_start_year_artist_count_select["Female"],
                        alpha=0.3,
                        labels=["Female Selected", "Male Selected"],
                        colors=[colors[1], colors[0]])
        ax[0].set_xlabel("Career Start Year")
        ax[0].set_ylabel("Number of\nArtists")
        ax[0].legend(loc="upper left", fontsize=8)
        legend_handles, legend_labels = ax[0].get_legend_handles_labels()
        order = [0, 1, 3, 2]
        ax[0].legend([legend_handles[idx] for idx in order],
                     [legend_labels[idx]
                      for idx in order], ncol=2, fontsize=10)

        ax[1].plot(shows_count["show_year"], shows_count["Male"], label="Male Total")
        ax[1].plot(shows_count["show_year"], shows_count["Female"], label="Female Total")
        ax[1].stackplot(shows_select_count["show_year"], shows_select_count["Female"],
                        shows_select_count["Male"] - shows_select_count["Female"],
                        alpha=0.3, labels=["Female Selected", "Male Selected"],
                        colors=[colors[1], colors[0]])
        ax[1].set_xlabel("Year")
        ax[1].set_ylabel("Number of\nExhibitions")

        ax[0].set_title("Population Imbalance", fontsize=20)
        ax[1].set_title("Exhibition Imbalance", fontsize=20)
        if self.remove_birth:
            ax[0].set_yticks([0, 2500, 5000])
            ax[0].set_yticklabels([0, r'$2.5 \times 10^3$', r'$5 \times 10^3$'])
        else:
            ax[0].set_yticks([0, 10000, 20000])
            ax[0].set_yticklabels([0, r'$1 \times 10^4$', r'$2 \times 10^4$'])
        ax[1].set_yticks([0, 5 * 10 ** 4, 10 ** 5])
        ax[1].set_yticklabels([0, r'$0.5 \times 10^5$', r'$1 \times 10^5$'])
        ax[1].set_xlim(1900, )
        ax[0].set_xlim(1900, )
        plt.tight_layout()
        # save figure
        suffix = f"threshold_{self.genderizeio_threshold}_filter_{self.remove_birth}_year_{self.career_start_threshold}"
        plt.savefig(os.path.join(self.save_fig_path, "00-full_vs_select_data.pdf"))

        if self.save:
            # save data for alice
            career_start_year_artist_count.to_csv("../for_plotting//Main Paper Data/1B_top_line.csv", index=False)
            career_start_year_artist_count_select.to_csv("../main_paper_plot_data/1B_top_shade.csv", index=False)
            shows_count.to_csv("../main_paper_plot_data/1B_bottom_line.csv", index=False)
            shows_select_count.to_csv("../main_paper_plot_data/1B_bottom_shade.csv", index=False)

    def plot_ins_num_over_year(self):
        # exhibited institutions over year
        ins_count = self.shows[["show_year", "institution"]].drop_duplicates().groupby(["show_year"])[
            "institution"].count().reset_index()
        ins_select_count = self.shows_select[["show_year", "institution"]].drop_duplicates().groupby(["show_year"])[
            "institution"].count().reset_index()
        plt.figure(figsize=(8, 3))
        plt.plot(ins_count["show_year"], ins_count[
            "institution"], label="Total Institutions", color="black")
        plt.plot(ins_select_count["show_year"], ins_select_count["institution"], label="Selected Institutions",
                 color="grey")
        plt.legend()
        # plt.yscale("log")
        plt.gca().set_yticks([0, 5 * 10 ** 3, 10 ** 4])
        plt.gca().set_yticklabels([0, r'$5 \times 10^3$', r'$1 \times 10^4$'])
        plt.xlabel("Year")
        plt.ylabel("Number of\nInstitutions")
        plt.xlim(1900, )
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_fig_path, "00-number_of_institutions.pdf"))

    def plot_population(self):
        # Number of male and female artists
        fig, ax1 = plt.subplots(figsize=(4.5, 6))
        graph = sns.countplot(x="gender_recog", data=self.artists_recog_select, ax=ax1)
        for p in graph.patches:
            height = p.get_height()
            graph.text(p.get_x() + p.get_width() / 2., height * 0.8,
                       "{:,}".format(height), ha="center", va="top", color="white", fontsize=18)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(
            0, 0), useOffset=False, useMathText=True)
        ax1.get_yaxis().get_major_formatter().set_useOffset(False)
        if self.remove_birth:
            ax1.set_yticks([0, 0.5 * 10 ** 4, 10 ** 4, 1.5 * 10 ** 4,
                            2 * 10 ** 4, 2.5 * 10 ** 4, 3 * 10 ** 4, 3.5 * 10 ** 4])
            ax1.set_yticklabels([0, r'$0.5 \times 10^4$', r'$1 \times 10^4$', r'$1.5 \times 10^4$',
                                 r'$2 \times 10^4$', r'$2.5 \times 10^4$', r'$3 \times 10^4$', r'$3.5 \times 10^4$'])
        else:
            ax1.set_yticks([0, 0.5 * 10 ** 5, 10 ** 5, 1.5 * 10 ** 5,
                            2 * 10 ** 5, 2.5 * 10 ** 5])
            ax1.set_yticklabels([0, r'$0.5 \times 10^5$', r'$1 \times 10^5$', r'$1.5 \times 10^5$',
                                 r'$2 \times 10^5$', r'$2.5 \times 10^5$'])
        plt.title("Population\nImbalance", fontsize=25)
        plt.xlabel('')
        plt.gca().tick_params(axis='x', labelsize=25)
        plt.ylabel('Number of Artists', fontsize=25)
        plt.tight_layout()

        plt.savefig(os.path.join(self.save_fig_path, "01-population_count.pdf"))

        # save for alice
        if self.save:
            lines = ["Male,Female\n",
                     str(len(self.artists_recog_select[self.artists_recog_select["gender_recog"] == "Male"])) + "," +
                     str(len(self.artists_recog_select[self.artists_recog_select["gender_recog"] == "Female"]))]
            f = open("../main_paper_plot_data/1C.csv", "w")
            f.writelines(lines)
            f.close()

    def plot_exhibition(self):
        fig, ax1 = plt.subplots(figsize=(4.5, 6))
        graph = sns.countplot(x="gender_recog", data=self.shows_select, ax=ax1)
        max_height = 0
        for p in graph.patches:
            height = p.get_height()
            graph.text(p.get_x() + p.get_width() / 2., height * 0.8,
                       "{:,}".format(height), ha="center", va="top", color="white", fontsize=18)
            max_height = max(max_height, height)
        if self.remove_birth:
            ax1.set_yticks([0, 1 * 10 ** 5, 2 * 10 ** 5, 3 * 10 ** 5,
                            4 * 10 ** 5, 5 * 10 ** 5, 6 * 10 ** 5])
            ax1.set_yticklabels([0, r'$1 \times 10^5$', r'$2 \times 10^5$', r'$3 \times 10^5$',
                                 r'$4 \times 10^5$', r'$5 \times 10^5$', r'$6 \times 10^5$'])
        else:
            ax1.set_yticks(
                [0, 0.2 * 10 ** 6, 0.4 * 10 ** 6, 0.6 * 10 ** 6, 0.8 * 10 ** 6, 1.0 * 10 ** 6, 1.2 * 10 ** 6])
            ax1.set_yticklabels(
                [0, r'$0.2 \times 10^6$', r'$0.4 \times 10^6$', r'$0.6 \times 10^6$', r'$0.8 \times 10^6$',
                 r'$1.0 \times 10^6$', r'$1.2 \times 10^6$'])
        plt.xlabel("")
        plt.gca().tick_params(axis='x', labelsize=25)
        plt.ylabel("Number of Exhibitions", fontsize=25)
        plt.title("Exhibition\nImbalance", fontsize=25)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_fig_path, "01-exhibition_count.pdf"))

        if self.save:
            lines = ["Male,Female\n",
                     str(len(self.shows_select[self.shows_select["gender_recog"] == "Male"])) + "," +
                     str(len(self.shows_select[self.shows_select["gender_recog"] == "Female"]))]
            f = open("../main_paper_plot_data/1D.csv", "w")
            f.writelines(lines)
            f.close()

    def plot_solo_population(self):
        fig, ax1 = plt.subplots(figsize=(4.5, 6))
        graph = sns.countplot(x="gender_recog", data=self.solo_shows_select[[
            "artist", "gender_recog"]].drop_duplicates(), ax=ax1)
        for p in graph.patches:
            height = p.get_height()
            graph.text(p.get_x() + p.get_width() / 2., height * 0.8,
                       "{:,}".format(height), ha="center", va="top", color="white", fontsize=18)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(
            0, 0), useOffset=False, useMathText=True)
        ax1.get_yaxis().get_major_formatter().set_useOffset(False)
        ax1.set_xticklabels(["Man", "Woman"])
        if self.remove_birth:
            ax1.set_yticks([0, 0.5 * 10 ** 4, 10 ** 4, 1.5 * 10 ** 4,
                            2 * 10 ** 4, 2.5 * 10 ** 4, 3 * 10 ** 4, 3.5 * 10 ** 4])
            ax1.set_yticklabels([0, r'$0.5 \times 10^4$', r'$1 \times 10^4$', r'$1.5 \times 10^4$',
                                 r'$2 \times 10^4$', r'$2.5 \times 10^4$', r'$3 \times 10^4$', r'$3.5 \times 10^4$'])
        plt.title("Population\nImbalance", fontsize=25)
        plt.xlabel('')
        plt.gca().tick_params(axis='x', labelsize=25)
        plt.ylabel('Number of Solo-Exhibited Artists', fontsize=25)
        plt.tight_layout()

        plt.savefig(os.path.join(self.save_fig_path, "01-solo_population_count.pdf"))

    def plot_solo_exhibition(self):
        fig, ax1 = plt.subplots(figsize=(4.5, 6))
        graph = sns.countplot(x="gender_recog", data=self.solo_shows_select, ax=ax1)
        for p in graph.patches:
            height = p.get_height()
            graph.text(p.get_x() + p.get_width() / 2., height * 0.8,
                       "{:,}".format(height), ha="center", va="top", color="white", fontsize=18)
        ax1.set_xticklabels(["Man", "Woman"])
        if self.remove_birth:
            ax1.set_yticks([0, 1 * 10 ** 5, 2 * 10 ** 5])
            ax1.set_yticklabels([0, r'$1 \times 10^5$', r'$2 \times 10^5$'])
        plt.xlabel("")
        plt.gca().tick_params(axis='x', labelsize=25)
        plt.ylabel("Number of\nSolo Exhibitions", fontsize=25)
        plt.title("Exhibition\nImbalance", fontsize=25)
        plt.tight_layout()

        plt.savefig(os.path.join(self.save_fig_path, "01-solo_exhibition_count.pdf"))


def main():
    parser = argparse.ArgumentParser(
        description='select artists with career start year > [year_threshold]')
    parser.add_argument('-t', '--genderizeio_threshold', type=float, help='genderize.io threshold', default=0.6)
    parser.add_argument('-f', '--remove_birth', action=argparse.BooleanOptionalAction)
    parser.add_argument('-y', '--career_start_threshold', type=int,
                        help='earliest career start year of selected artists', default=1990)
    parser.add_argument('-a', '--save', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    print(args)

    # init
    select_data = SelectData(args.genderizeio_threshold,
                             args.remove_birth,
                             args.career_start_threshold,
                             args.save)
    # select
    select_data.get_select_artists()
    select_data.filter_data()
    # save
    select_data.save_data()
    # report
    select_data.report_statistics()
    # plot
    select_data.plot_full_vs_select_data()
    select_data.plot_ins_num_over_year()
    select_data.plot_population()
    select_data.plot_exhibition()
    select_data.plot_solo_population()
    select_data.plot_solo_exhibition()


if __name__ == '__main__':
    main()
