import pandas as pd
import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from plot_config import *
import argparse
import os


class PrepareSale:
    def __init__(self, genderizeio_threshold, remove_birth, career_start_threshold, preference_type,
                 num_exhibition_threshold, save_alice):
        self.genderizeio_threshold = genderizeio_threshold
        self.remove_birth = remove_birth
        self.career_start_threshold = career_start_threshold
        self.preference_type = preference_type
        self.num_exhibition_threshold = num_exhibition_threshold
        self.save_alice = save_alice

        base_data_path = os.path.join("..", "results",
                                      f"threshold_{self.genderizeio_threshold}_filter_{self.remove_birth}",
                                      "data")
        base_figure_path = os.path.join("..", "results",
                                        f"threshold_{self.genderizeio_threshold}_filter_{self.remove_birth}",
                                        "figures")
        self.year_data_path = os.path.join(base_data_path, f"year_{self.career_start_threshold}")
        self.auction_fig_path = os.path.join(base_figure_path, f"year_{self.career_start_threshold}",
                                             "auction")
        os.mkdir(self.auction_fig_path) if not os.path.exists(self.auction_fig_path) else None
        self.sales_select = pd.read_csv(os.path.join(self.year_data_path, "sales_select.csv"))

        self.min_exh_data_path = os.path.join(self.year_data_path,
                                              f"minimum_exh_count_{self.num_exhibition_threshold}")

        self.agg = pd.read_csv(os.path.join(self.year_data_path, f"artist_exh_info_agg_{self.preference_type}.csv"))

        # create agg_sales
        self.agg_sales = self.create_agg_sales()

    def create_agg_sales(self):
        # create an agg_sales dataframe
        pieces_sold = self.sales_select.groupby("artist")["auction"].count().reset_index().rename(
            columns={"auction": "pieces_sold"})
        total_sales = self.sales_select.groupby("artist")["price real"].sum(
        ).reset_index().rename(columns={"price real": "total_sales"})
        avg_sales = self.sales_select.groupby("artist")["price real"].mean(
        ).reset_index().rename(columns={"price real": "avg_sales"})
        medium = self.sales_select.groupby("artist")["medium"].apply(list).reset_index()
        agg_sales = pieces_sold.join(total_sales.set_index("artist"), on="artist")
        agg_sales = agg_sales.join(avg_sales.set_index("artist"), on="artist")
        agg_sales = agg_sales.join(medium.set_index("artist"), on="artist")
        agg_sales = agg_sales.join(self.agg.set_index("artist"), on="artist")
        agg_sales.to_csv(os.path.join(self.year_data_path, f"artist_sale_info_agg_{self.preference_type}.csv"),
                         index=False)
        return agg_sales

    def plot_population_imbalance(self):
        count_df = self.sales_select[["artist", "gender_recog"]].drop_duplicates().groupby("gender_recog")[
            "artist"].count()
        count_df = count_df.to_dict()
        print("Auction population gender ratio: Male/Female = %.2f" %
              (count_df["Male"] / count_df["Female"]))

        fig, ax1 = plt.subplots(figsize=(4.5, 6))
        graph = sns.countplot(x="gender_recog", data=self.sales_select[["artist", "gender_recog"]].drop_duplicates())
        for p in graph.patches:
            height = p.get_height()
            graph.text(p.get_x() + p.get_width() / 2., height * 0.8,
                       "{:,}".format(height), ha="center", va="top", color="white", fontsize=20)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(
            0, 0), useOffset=False, useMathText=True)

        if self.remove_birth:
            ax1.set_yticks([0, 2 * 10 ** 3, 4 * 10 ** 3, 6 * 10 ** 3, 8 * 10 ** 3, 10 * 10 ** 3])
            ax1.set_yticklabels([0, r'$2 \times 10^3$', r'$4 \times 10^3$',
                                 r'$6 \times 10^3$', r'$8 \times 10^3$', r'$10 \times 10^3$'])

        plt.xlabel("")
        plt.ylabel("Number of Auctioned Artists", fontsize=25)
        ax1.tick_params(axis='x', labelsize=25)
        plt.title("Auction Population\nImbalance", fontsize=25)
        plt.tight_layout()
        plt.savefig(os.path.join(self.auction_fig_path, "04-number_auction_artist.pdf"))

    def plot_records_balance(self):
        count_df = self.sales_select.groupby("gender_recog")["artist"].count()
        # save for alice
        if self.save_alice:
            count_df.reset_index().to_csv("../for_alice/Main Paper Data/4A-2.csv", index=False)
        count_df = count_df.to_dict()
        print("Auction record gender ratio: Male/Female = %.2f" %
              (count_df["Male"] / count_df["Female"]))

        fig, ax1 = plt.subplots(figsize=(4.5, 6))
        graph = sns.countplot(x="gender_recog", data=self.sales_select)
        for p in graph.patches:
            height = p.get_height()
            graph.text(p.get_x() + p.get_width() / 2., height * 0.8,
                       "{:,}".format(height), ha="center", va="top", color="white", fontsize=20)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(
            0, 0), useOffset=False, useMathText=True)
        plt.xlabel("")
        plt.ylabel("Number of Auction Records", fontsize=25)
        ax1.tick_params(axis='x', labelsize=25)
        plt.title("Auction Record\nImbalance", fontsize=25)
        ax1.get_yaxis().get_major_formatter().set_useOffset(False)
        if self.remove_birth:
            ax1.set_yticks([0,
                            2 * 10 ** 4, 4 * 10 ** 4, 6 * 10 ** 4, 8 * 10 ** 4, 10 * 10 ** 4])
            ax1.set_yticklabels([0, r'$2 \times 10^4$', r'$4 \times 10^4$',
                                 r'$6 \times 10^4$', r'$8 \times 10^4$', r'$10 \times 10^4$'])
        plt.tight_layout()
        plt.savefig(os.path.join(self.auction_fig_path, "04-number_auction.pdf"))

    def plot_sales_imbalance(self):
        total_sales_df = self.sales_select.groupby(
            "gender_recog")["price relative"].sum().to_dict()
        print("Auction Total Sales gender ratio: Male/Female =%.2f" %
              (total_sales_df["Male"] / total_sales_df["Female"]))

        total_sales_df = self.sales_select.groupby("gender_recog")["price relative"].sum().reset_index()
        if self.save_alice:
            total_sales_df.reset_index().to_csv(
                "../for_alice/Main Paper Data/4A-3.csv",
                index=False)

        fig, ax1 = plt.subplots(figsize=(4.5, 6))
        graph = sns.barplot(x="gender_recog", y="price relative",
                            data=total_sales_df, order=["Male", "Female"])
        for p in graph.patches:
            height = p.get_height()
            string = "{:.1E}".format(height)
            num, power = string.split("E+0")
            graph.text(p.get_x() + p.get_width() / 2., height * 0.8,
                       r'$%s \times 10^%s$' % (num, power), ha="center", va="top", color="white", fontsize=15)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(
            0, 0), useOffset=False, useMathText=True)
        ax1.get_yaxis().get_major_formatter().set_useOffset(False)
        if self.remove_birth:
            ax1.set_yticks([0,
                            2 * 10 ** 4, 4 * 10 ** 4, 6 * 10 ** 4, 8 * 10 ** 4, 10 * 10 ** 4])
            ax1.set_yticklabels([0, r'$2 \times 10^4$', r'$4 \times 10^4$',
                                 r'$6 \times 10^4$', r'$8 \times 10^4$', r'$10 \times 10^4$'])
        plt.xlabel("")
        plt.ylabel("Total Auction Sales\n(Relative Price)", fontsize=25)
        ax1.tick_params(axis='x', labelsize=25)
        plt.title("Auction Sales\nImbalance", fontsize=25)
        plt.tight_layout()
        plt.savefig(os.path.join(self.auction_fig_path, "04-total_sales_auction.pdf"))

    def plot_sold_pieces_imbalance(self):
        auction_count_male = self.agg_sales[self.agg_sales["gender_recog"] == "Male"]["pieces_sold"]
        auction_count_female = self.agg_sales[self.agg_sales["gender_recog"] == "Female"]["pieces_sold"]

        male_10th, female_10th = np.percentile(
            auction_count_male, 10), np.percentile(auction_count_female, 10)
        male_median, female_median = np.median(
            auction_count_male), np.median(auction_count_female)
        male_mean, female_mean = np.mean(
            auction_count_male), np.mean(auction_count_female)
        male_90th, female_90th = np.percentile(
            auction_count_male, 90), np.percentile(auction_count_female, 90)

        temp_df = pd.DataFrame()
        temp_df["gender"] = ["Male", "Female", "Male",
                             "Female", "Male", "Female", "Male", "Female"]
        temp_df["statistic"] = [male_mean, female_mean, male_10th,
                                female_10th, male_median, female_median, male_90th, female_90th]
        temp_df["name"] = ["Mean", "Mean", "10%tile", "10%tile",
                           "Median", "Median", "90%tile", "90%tile"]
        temp_df["pos"] = [0, 0, 1.2, 1.2, 2.2, 2.2, 3.2, 3.2]

        plt.clf()
        plt.figure(figsize=(8, 6))
        g = sns.barplot(x="pos", y="statistic", hue="gender", data=temp_df)
        plt.xticks(temp_df["pos"], temp_df["name"])
        plt.xlim(-0.5, 3.8)
        for i, p in enumerate(g.patches):
            if i != 0 and i != 4:
                p.set_x(p.get_x() + 0.2)
            txt = str(p.get_height().round(1))
            height = p.get_height()
            if height > 2:
                g.text(p.get_x() + p.get_width() / 2., height * 0.8, txt,
                       ha="center", va="top", color="white", fontsize=20)
            else:
                g.text(p.get_x() + p.get_width() / 2., height, txt,
                       ha="center", va="bottom", color="black", fontsize=20)
        plt.xlabel("")
        plt.gca().tick_params(axis='x', labelsize=25)
        plt.ylabel("Number of Auctions", fontsize=25)
        g.legend_.remove()
        plt.tight_layout()
        plt.savefig(os.path.join(self.auction_fig_path, "04-auction_count_bias_barplot.pdf"))

    def plot_piece_price_imbalance(self):
        auction_price_male = self.sales_select[self.sales_select["gender_recog"] == "Male"]["price/avg"]
        auction_price_female = self.sales_select[self.sales_select["gender_recog"] == "Female"]["price/avg"]

        male_10th, female_10th = np.percentile(
            auction_price_male, 10), np.percentile(auction_price_female, 10)
        male_median, female_median = np.median(
            auction_price_male), np.median(auction_price_female)
        male_mean, female_mean = np.mean(
            auction_price_male), np.mean(auction_price_female)
        male_90th, female_90th = np.percentile(
            auction_price_male, 90), np.percentile(auction_price_female, 90)

        temp_df = pd.DataFrame()
        temp_df["gender"] = ["Male", "Female", "Male",
                             "Female", "Male", "Female", "Male", "Female"]
        temp_df["statistic"] = [male_mean, female_mean, male_10th,
                                female_10th, male_median, female_median, male_90th, female_90th]
        temp_df["name"] = ["Mean", "Mean", "10%tile", "10%tile",
                           "Median", "Median", "90%tile", "90%tile"]
        temp_df["pos"] = [0, 0, 1.2, 1.2, 2.2, 2.2, 3.2, 3.2]

        plt.clf()
        plt.figure(figsize=(8, 6))
        g = sns.barplot(x="pos", y="statistic", hue="gender", data=temp_df)
        plt.xticks(temp_df["pos"], temp_df["name"])
        plt.xlim(-0.5, 3.8)
        plt.ylim(0, 1.5)
        for i, p in enumerate(g.patches):
            if i != 0 and i != 4:
                p.set_x(p.get_x() + 0.2)
            txt = str(p.get_height().round(1))
            if txt == "0.0" or txt == "0.1":
                txt = str(p.get_height().round(2))
            height = p.get_height()
            if height > 2:
                g.text(p.get_x() + p.get_width() / 2., height * 0.8, txt,
                       ha="center", va="top", color="white", fontsize=20)
            else:
                g.text(p.get_x() + p.get_width() / 2., height, txt,
                       ha="center", va="bottom", color="black", fontsize=20)
        plt.xlabel("")
        plt.gca().tick_params(axis='x', labelsize=25)
        plt.ylabel("Auction Price", fontsize=25)
        g.legend_.remove()
        plt.tight_layout()
        plt.savefig(os.path.join(self.auction_fig_path, '04-auction_price_bias_barplot.pdf'))

    def plot_access_rate(self):
        select_artists = self.agg

        select_artists_auction = self.agg_sales[self.agg_sales["artist"].isin(select_artists["artist"])]
        male_transition_rate = 100 * len(select_artists_auction[select_artists_auction[
                                                                    "gender_recog"] == "Male"]) / len(
            select_artists[select_artists["gender_recog"] == "Male"])
        female_transition_rate = 100 * len(select_artists_auction[select_artists_auction[
                                                                      "gender_recog"] == "Female"]) / len(
            select_artists[select_artists["gender_recog"] == "Female"])
        plt.figure(figsize=(8, 6))
        temp = pd.DataFrame({"gender": ["Male", "Female"], "transition Rate": [
            male_transition_rate, female_transition_rate]})
        ax = sns.barplot(x="gender", y="transition Rate", data=temp)
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height * 0.9,
                    "%.2f" % height, ha="center", va="top", color="white", fontsize=25)
        plt.xlabel("")
        plt.ylabel("Auction Access Rate (%)", fontsize=25)
        plt.gca().tick_params(axis='x', labelsize=25)
        plt.tight_layout()
        plt.savefig(os.path.join(self.auction_fig_path, "04-access_rate.pdf"))


def main():
    parser = argparse.ArgumentParser(
        description='select artists with career start year > [year_threshold]')
    parser.add_argument('-t', '--genderizeio_threshold', type=float, help='genderize.io threshold', default=0.6)
    parser.add_argument('-f', '--remove_birth', action=argparse.BooleanOptionalAction)
    parser.add_argument('-y', '--career_start_threshold', type=int,
                        help='earliest career start year of selected artists', default=1990)
    parser.add_argument('-p', '--preference_type', type=str, help='neutral or balance')
    parser.add_argument('-n', '--num_exhibition_threshold', type=int, help='number of exhibition threshold', default=10)
    parser.add_argument('-a', '--save_alice', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    prepare_sale = PrepareSale(args.genderizeio_threshold,
                               args.remove_birth,
                               args.career_start_threshold,
                               args.preference_type,
                               args.num_exhibition_threshold,
                               args.save_alice)

    prepare_sale.plot_population_imbalance()
    prepare_sale.plot_records_balance()
    prepare_sale.plot_sales_imbalance()
    prepare_sale.plot_sold_pieces_imbalance()
    prepare_sale.plot_piece_price_imbalance()
    prepare_sale.plot_access_rate()


if __name__ == '__main__':
    main()