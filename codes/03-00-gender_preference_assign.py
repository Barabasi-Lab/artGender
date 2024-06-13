import argparse
import json
import os
from collections import Counter

import pandas as pd
from scipy.stats import binomtest, binom
import pingouin as pg


class AssignGenderPreference:

    def __init__(self, genderizeio_threshold, remove_birth, career_start_threshold, alpha=0.1):
        self.genderizeio_threshold = genderizeio_threshold
        self.remove_birth = remove_birth
        self.career_start_threshold = career_start_threshold
        self.alpha = alpha

        self.save_data_path = os.path.join("..", "results",
                                           f"threshold_{self.genderizeio_threshold}_filter_{self.remove_birth}",
                                           "data",
                                           f"year_{self.career_start_threshold}")
        self.save_fig_path = os.path.join("..", "results",
                                          f"threshold_{self.genderizeio_threshold}_filter_{self.remove_birth}",
                                          "figures",
                                          f"year_{self.career_start_threshold}")
        try:
            os.makedirs(self.save_data_path, exist_ok=True)
            os.makedirs(self.save_fig_path, exist_ok=True)
        except:
            pass

        self.artists_recog_select = pd.read_csv(os.path.join(self.save_data_path, "artists_recog_select.csv"))
        self.shows_select = pd.read_csv(os.path.join(self.save_data_path, "shows_select.csv"))

        self.p_null_neutral = Counter(self.artists_recog_select["gender_recog"])["Female"] / len(
            self.artists_recog_select)
        self.p_null_balance = 0.5

    @staticmethod
    def binom_power(n, p_baseline, p_obs, alpha, mode):
        # adapted from https://stackoverflow.com/questions/13059011/is-there-any-python-function-library-for-calculate-binomial-confidence-intervals
        # and https://sites.calvin.edu/scofield/courses/m343/F15/handouts/binomialTestPower.pdf
        if mode == "two-sided":
            critical = binom.ppf(alpha / 2, n, p_baseline)
            beta = binom.cdf(n - critical, n, p_obs) - binom.cdf(critical - 1, n, p_obs)
            power = 1 - beta
        elif mode == "greater":
            critical = binom.ppf(1 - alpha, n, p_baseline)
            power = 1 - binom.cdf(critical, n, p_obs)
        elif mode == "less":
            critical = binom.ppf(alpha, n, p_baseline)
            power = binom.cdf(critical, n, p_obs)
        return power

    @staticmethod
    def get_gender_preference(df, group_attr, p_null, alpha):
        gender_preference = {}
        power = {}
        filtered_female_count, filtered_total_count = 0, 0
        size = {}
        positive = {}
        for ins in set(df[group_attr]):
            this_ins = df[df[group_attr] == ins]
            size[ins] = len(this_ins)
            if len(this_ins) < 30:
                total_count = len(this_ins)
                female_count = Counter(this_ins["gender_recog"])["Female"]
                filtered_female_count += female_count
                filtered_total_count += total_count
                continue
            n = len(this_ins)
            female_count = Counter(this_ins["gender_recog"])["Female"]
            positive[ins] = female_count
            two_side_test = binomtest(
                female_count, n=n, p=p_null, alternative='two-sided')
            bf = float(pg.bayesfactor_binom(k=female_count, n=n, p=p_null))
            if two_side_test.pvalue > alpha:
                gender_preference[ins] = 0
                power[ins] = {"label": 0,
                              "n": n,
                              "p_null": p_null,
                              "female_count": female_count,
                              "female_portion": female_count / n,
                              "binom_pvalue": two_side_test.pvalue,
                              "bayes_factor": bf,
                              "power": AssignGenderPreference.binom_power(
                                  n, p_null, female_count / n, alpha, "two-sided")}
            else:
                if female_count < n * p_null:
                    one_sided_test = binomtest(
                        female_count, n=n, p=p_null, alternative='less')
                    gender_preference[ins] = 1
                    power[ins] = {"label": 1,
                                  "n": n,
                                  "p_null": p_null,
                                  "female_count": female_count,
                                  "female_portion": female_count / n,
                                  "binom_pvalue": one_sided_test.pvalue,
                                  "bayes_factor": bf,
                                  "power": AssignGenderPreference.binom_power(
                                      n, p_null, female_count / n, alpha, "less")}
                else:
                    one_sided_test = binomtest(
                        female_count, n=n, p=p_null, alternative='greater')
                    gender_preference[ins] = 2
                    power[ins] = {"label": 2,
                                  "n": n,
                                  "p_null": p_null,
                                  "female_count": female_count,
                                  "female_portion": female_count / n,
                                  "binom_pvalue": one_sided_test.pvalue,
                                  "bayes_factor": bf,
                                  "power": AssignGenderPreference.binom_power(
                                      n, p_null, female_count / n, alpha, "greater")}
        return gender_preference, power, filtered_female_count, filtered_total_count

    def get_ins_gender(self):
        print("==============")
        print("GENDER PREFERENCE INSTITUTIONS")
        for preference_type, p_null in zip(["neutral", "balance"], [self.p_null_neutral, self.p_null_balance]):
            gender_preference, power, filtered_female_count, filtered_total_count = self.get_gender_preference(
                self.shows_select,
                "institution",
                p_null,
                self.alpha)
            json.dump(gender_preference,
                      open(os.path.join(self.save_data_path, f"gender_{preference_type}_ins.json"), "w"),
                      indent=4)
            json.dump(power,
                      open(os.path.join(self.save_data_path, f"gender_{preference_type}_details_ins.json"), "w"),
                      indent=4)
            print(f"Institution with Gender-{preference_type} Info/All Institutions: "
                  f"{len(gender_preference)}/{len(set(self.shows_select['institution']))}")
            print(f"Institution Composition Under Gender {preference_type}: {Counter(gender_preference.values())}")
            print("---------------------")
        print(f"Small Institution female exhibition/total exhibition:", filtered_female_count, filtered_total_count)

    def get_country_gender(self):
        print("==============")
        print("GENDER PREFERENCE COUNTRY")
        for preference_type, p_null in zip(["neutral", "balance"], [self.p_null_neutral, self.p_null_balance]):
            gender_preference, power, filtered_female_count, filtered_total_count = self.get_gender_preference(
                self.shows_select,
                "country",
                p_null,
                self.alpha)
            json.dump(gender_preference,
                      open(os.path.join(self.save_data_path, f"gender_{preference_type}_country.json"), "w"),
                      indent=4)
            print(f"Country with Gender-{preference_type} Info/All Countries: "
                  f"{len(gender_preference)}/{len(set(self.shows_select['institution']))}")
            print(f"Country Composition Under Gender {preference_type}: {Counter(gender_preference.values())}")
            print("---------------------")
        print(f"Small Country female exhibition/total exhibition:", filtered_female_count, filtered_total_count)


def main():
    parser = argparse.ArgumentParser(
        description='select artists with career start year > [year_threshold]')
    parser.add_argument('-t', '--genderizeio_threshold', type=float, help='genderize.io threshold', default=0.6)
    parser.add_argument('-f', '--remove_birth', action=argparse.BooleanOptionalAction)
    parser.add_argument('-y', '--career_start_threshold', type=int,
                        help='earliest career start year of selected artists', default=1990)
    args = parser.parse_args()
    print(args)

    assign_gender_preference = AssignGenderPreference(args.genderizeio_threshold, args.remove_birth,
                                                      args.career_start_threshold)
    assign_gender_preference.get_ins_gender()
    assign_gender_preference.get_country_gender()


if __name__ == '__main__':
    main()
