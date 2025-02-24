"""Process Data"""
import argparse
import json
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats


class ProcessData:

    def __init__(self, genderizeio_threshold, remove_birth):
        # parse input
        self.genderizeio_threshold = genderizeio_threshold
        self.remove_birth = remove_birth

        # read data
        # self.shows = pd.read_csv("../raw_data/shows.tsv")

        self.ins_attrs = pd.read_csv("../raw_data/attributes.csv")
        self.sales = pd.read_csv("../raw_data/sales.tsv")
        self.artists_recog = pd.read_csv("../processed_data/artists_gender_recog_%s.csv" % self.genderizeio_threshold)

        self.prestige_dict = dict(zip(self.ins_attrs["Id"], self.ins_attrs["Centrality"]))
        self.artists_info = self.artists_recog[["artist", "birth", "gender_recog"]]

        self.shows_with_person = None
        self.shows_with_person_with_gender = None
        self.career_start_end = None
        self.select_artists = None
        self.solo_shows_with_person_with_gender = None
        self.sales_person = None
        self.percentile_prestige = None

        # set up save data path
        folder_name = f"threshold_{self.genderizeio_threshold}_filter_{self.remove_birth}"
        folder_path = os.path.join("..", "results", folder_name)
        try:
            os.makedirs(folder_path, exist_ok=True)
        except:
            pass
        # save data
        self.save_data_path = os.path.join(folder_path, "data")
        save_figure_path = os.path.join(folder_path, "figures")
        try:
            os.makedirs(self.save_data_path, exist_ok=True)
            os.makedirs(save_figure_path, exist_ok=True)
        except:
            pass

    def create_shows_with_person(self):
        # create dataframe shows with person
        artist_id_list = []
        show_id_list = []
        for artist in self.paths:
            show_ids = self.paths[artist]
            artist_id_list += [artist] * len(show_ids)
            show_id_list += show_ids
        # this dataframe records every (artist, show) pair
        artist_shows = pd.DataFrame()
        artist_shows["artist"] = artist_id_list
        artist_shows["show"] = show_id_list
        self.shows_with_person = artist_shows.join(self.shows.set_index("show"), on="show")
        self.shows_with_person.to_csv("../raw_data/shows.csv", index=False)

    def add_prestige(self):
        # add prestige info
        self.shows_with_person["prestige"] = [self.prestige_dict[ins] if ins in self.prestige_dict else np.nan for ins
                                              in
                                              self.shows_with_person["institution"]]
        # transform raw prestige to percentile prestige
        self.shows_with_person = self.shows_with_person.dropna(subset=["prestige"])
        prestige_dict = dict(zip(self.shows_with_person["institution"], self.shows_with_person["prestige"]))
        prestige_values = list(prestige_dict.values())
        self.percentile_prestige = {ins: stats.percentileofscore(prestige_values, prestige_dict[ins]) / 100
                                    for ins in prestige_dict}
        self.shows_with_person["percentile_prestige"] = [self.percentile_prestige[ins]
                                                         for ins in self.shows_with_person["institution"]]

    def add_artist_info(self):
        # add artist's gender, name and birth in shows_with_person
        self.shows_with_person = self.shows_with_person.join(self.artists_info.set_index("artist"), on="artist")
        self.shows_with_person_with_gender = self.shows_with_person[
            self.shows_with_person["gender_recog"].isin(["Male", "Female"])]
        # add show_year
        self.shows_with_person_with_gender.loc[:, "show_year"] = pd.to_datetime(
            self.shows_with_person_with_gender["start"]).dt.year

    def get_career_start_end(self):
        # get artist career start and career end year
        career_start = self.shows_with_person_with_gender.groupby("artist")["show_year"].min().reset_index().rename(
            {"show_year": "career_start_year"}, axis=1)
        career_end = self.shows_with_person_with_gender.groupby("artist")["show_year"].max().reset_index().rename(
            {"show_year": "career_end_year"}, axis=1)
        self.career_start_end = career_start.join(career_end.set_index("artist"), on="artist").join(
            self.artists_info.set_index("artist"), on="artist")

    def filter_shows_nonadult(self):
        # filter out record before artist's adulthood
        if self.remove_birth:
            self.shows_with_person_with_gender = self.shows_with_person_with_gender[
                (self.shows_with_person_with_gender["show_year"] > self.shows_with_person_with_gender["birth"] + 18)]
        else:
            self.shows_with_person_with_gender = self.shows_with_person_with_gender[
                (self.shows_with_person_with_gender["show_year"] > self.shows_with_person_with_gender["birth"] + 18) |
                (self.shows_with_person_with_gender["birth"].isna())]

    def select_artist_birthyear(self):
        # only select artist whose first exhibition is within 50 year after birth
        if self.remove_birth:
            self.select_artists = self.career_start_end[
                (self.career_start_end["career_start_year"] - self.career_start_end["birth"] <= 50)]["artist"]
        else:
            self.select_artists = self.career_start_end[
                (self.career_start_end["career_start_year"] - self.career_start_end["birth"] <= 50) | (
                    self.career_start_end["birth"].isna())]["artist"]

    def get_solo_shows(self):
        # get solo shows
        show_artist_count = self.shows_with_person.groupby("show")["artist"].count().reset_index()
        solo_show_id = show_artist_count[show_artist_count["artist"] == 1]["show"]
        self.solo_shows_with_person_with_gender = self.shows_with_person_with_gender[
            self.shows_with_person_with_gender["show"].isin(solo_show_id)]

    def filter_shows_data_select_artists(self):
        # filter out data based on the above criteria
        self.shows_with_person_with_gender = self.shows_with_person_with_gender[
            self.shows_with_person_with_gender["artist"].isin(self.select_artists)]
        self.career_start_end = self.career_start_end[self.career_start_end["artist"].isin(self.select_artists)]

    def process_shows_data(self):
        if not os.path.exists("../raw_data/shows.csv"):
            self.shows = pd.read_csv("../raw_data/shows.tsv")
            self.paths = pickle.load(open("../raw_data/paths.pkl", "rb"))
            self.create_shows_with_person()
        else:
            self.shows_with_person = pd.read_csv("../raw_data/shows.csv")
        self.add_prestige()
        self.add_artist_info()
        self.filter_shows_nonadult()
        self.get_career_start_end()
        self.select_artist_birthyear()
        self.filter_shows_data_select_artists()
        self.get_solo_shows()

    def process_sales_data(self):
        # sales data
        sales_person = pd.merge(self.sales, self.artists_recog, on="artist")
        sales_person = sales_person.dropna(subset=["price real"], axis=0)
        sales_person.loc[:, "year"] = pd.to_datetime(sales_person["date"]).dt.year
        sales_person = sales_person[sales_person["year"] > sales_person["birth"] + 18]
        # create relative price price/avg
        year_auction_avg = sales_person.groupby("year")["price real"].mean().reset_index().sort_values("year")
        year_auction_avg_dict = dict(zip(year_auction_avg["year"], year_auction_avg["price real"]))
        sales_person["price/avg"] = [price / year_auction_avg_dict[year]
                                     for price, year in zip(sales_person["price real"], sales_person["year"])]
        self.sales_person = sales_person[sales_person["artist"].isin(self.select_artists)]

    def save_data(self):
        self.shows_with_person_with_gender[["show", "artist", "gender_recog", "institution", "percentile_prestige", "show_year", "country"]].to_csv(os.path.join(self.save_data_path, "shows.csv"), index=False)
        self.solo_shows_with_person_with_gender[["show", "artist", "gender_recog", "institution", "percentile_prestige", "show_year", "country"]].to_csv(os.path.join(self.save_data_path, "solo_shows.csv"), index=False)
        json.dump(self.percentile_prestige, open(os.path.join(self.save_data_path, "percentile_prestige.json"), "w"))
        self.career_start_end.to_csv(os.path.join(self.save_data_path, "career_span.csv"), index=False)
        self.sales_person[["auction", "artist", "gender_recog", "medium", "price real", "price relative", "price/avg"]].to_csv(os.path.join(self.save_data_path, "sales.csv"), index=False)

    def report_raw_data_statistics(self):
        print("=========================")
        print("RAW DATA STATISTICS")
        print("Total Number of Artists and Gender Composition:", len(self.artists_recog))
        print("Curated gender composition:", Counter(self.artists_recog["gender"]))
        print("With genderize.io gender composition: ", Counter(self.artists_recog["gender_recog"]))
        print("Total Number of Exhibitions:", len(self.shows_with_person))
        print("Total Number of Auctions:", len(self.sales))

    def report_processed_data_statistics(self):
        # report statistics on processed data records
        print("=========================")
        print(f"PROCESSED DATA STATISTICS (Genderize.io threshold: {self.genderizeio_threshold}, "
              f"Filter birth {self.remove_birth}")
        print("Total Number of Artists:", len(set(self.shows_with_person_with_gender["artist"])))
        print("Artists Gender Composition:",
              Counter(self.shows_with_person_with_gender[["artist", "gender_recog"]].drop_duplicates()["gender_recog"]))
        print("Total Number of Institutions:", len(set(self.shows_with_person_with_gender["institution"])))
        print("Number of Exhibitions breakdown by gender:", Counter(
            self.shows_with_person_with_gender["gender_recog"]))
        print("Number of Solo Exhibitions breakdown by gender:", Counter(
            self.solo_shows_with_person_with_gender["gender_recog"]))
        print("Total Number of Auctions:", len(self.sales_person))


def main():
    parser = argparse.ArgumentParser(
        description='genderize.io threshold and whether to filter using artist birth year')
    parser.add_argument('-t', '--threshold', type=float, help='genderize.io threshold', default=0.6)
    parser.add_argument('-f', '--remove_birth', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    print(args)
    process_data = ProcessData(genderizeio_threshold=args.threshold, remove_birth=args.remove_birth)

    process_data.process_shows_data()
    process_data.report_raw_data_statistics()
    process_data.process_sales_data()
    process_data.save_data()
    process_data.report_processed_data_statistics()


if __name__ == '__main__':
    main()
