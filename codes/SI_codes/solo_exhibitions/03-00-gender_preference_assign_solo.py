import argparse
import json
import os
from collections import Counter

import pandas as pd
from scipy.stats import beta, binom
from scipy.integrate import quad


class AssignGenderPreference:

    def __init__(self, genderizeio_threshold, remove_birth, career_start_threshold):
        self.genderizeio_threshold = genderizeio_threshold
        self.remove_birth = remove_birth
        self.career_start_threshold = career_start_threshold

        self.save_data_path = os.path.join("..", "..", "..", "results",
                                           f"threshold_{self.genderizeio_threshold}_filter_{self.remove_birth}",
                                           "data",
                                           f"year_{self.career_start_threshold}")
        self.save_fig_path = os.path.join("..", "..", "..", "results",
                                          f"SI",
                                          "fig4")
        try:
            os.makedirs(self.save_data_path, exist_ok=True)
            os.makedirs(self.save_fig_path, exist_ok=True)
        except:
            pass

        self.artists_recog_select = pd.read_csv(os.path.join(self.save_data_path, "artists_recog_select.csv"))
        self.shows_select = pd.read_csv(os.path.join(self.save_data_path, "solo_shows_select.csv"))

        self.p_null_neutral = Counter(self.artists_recog_select["gender_recog"])["Female"] / len(
            self.artists_recog_select)
        self.p_null_balance = 0.5

    @staticmethod
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

    @staticmethod
    def assign_ins_gender(n, k, p0):
        p1 = k / n
        bf10_two_sided, bf10_one_sided = None, None
        bf10_two_sided = AssignGenderPreference.bayesfactor_binom(k, n, p=p0, a=1, b=1, oneside=False)
        if bf10_two_sided < 1 / 3:
            return 0, bf10_two_sided, bf10_one_sided
        elif bf10_two_sided > 3:
            bf10_one_sided = AssignGenderPreference.bayesfactor_binom(k, n, p=p0, a=1, b=1, oneside=True)
            if bf10_one_sided > 3:
                if p1 < p0:
                    return 1, bf10_two_sided, bf10_one_sided
                else:
                    return 2, bf10_two_sided, bf10_one_sided
        return -1, bf10_two_sided, bf10_one_sided

    @staticmethod
    def get_gender_preference(df, group_attr, p_null):
        gender_preference = {}
        info = {}
        for ins in set(df[group_attr]):
            this_ins = df[df[group_attr] == ins]
            n = len(this_ins)
            k = Counter(this_ins["gender_recog"])["Female"]
            ins_gender, bf10_two_sided, bf10_one_sided = AssignGenderPreference.assign_ins_gender(n, k, p_null)
            if ins_gender != -1:
                gender_preference[ins] = ins_gender
            info[ins] = {"label": ins_gender,
                         "n": n,
                         "female_count": k,
                         "bf10_two_sided": bf10_two_sided,
                         "bf10_one_sided": bf10_one_sided}
        return gender_preference, info

    def get_ins_gender(self):
        print("==============")
        print("GENDER PREFERENCE INSTITUTIONS")
        for preference_type, p_null in zip(["neutral", "balance"], [self.p_null_neutral, self.p_null_balance]):
            gender_preference, info = self.get_gender_preference(
                self.shows_select,
                "institution",
                p_null)
            json.dump(gender_preference,
                      open(os.path.join(self.save_fig_path, f"gender_{preference_type}_ins_solo_bf10.json"), "w"),
                      indent=4)
            json.dump(info,
                      open(os.path.join(self.save_fig_path, f"gender_{preference_type}_details_ins_solo_bf10.json"), "w"),
                      indent=4)
            print(f"Institution with Gender-{preference_type} Info/All Institutions: "
                  f"{len(gender_preference)}/{len(set(self.shows_select['institution']))}")
            print(f"Institution Composition Under Gender {preference_type}: {Counter(gender_preference.values())}")
            print("---------------------")

    def get_country_gender(self):
        print("==============")
        print("GENDER PREFERENCE COUNTRY")
        for preference_type, p_null in zip(["neutral", "balance"], [self.p_null_neutral, self.p_null_balance]):
            gender_preference, info = self.get_gender_preference(
                self.shows_select,
                "country",
                p_null)
            json.dump(gender_preference,
                      open(os.path.join(self.save_fig_path, f"gender_{preference_type}_country_solo_bf10.json"), "w"),
                      indent=4)
            print(f"Country with Gender-{preference_type} Info/All Countries: "
                  f"{len(gender_preference)}/{len(set(self.shows_select['country']))}")
            print(f"Country Composition Under Gender {preference_type}: {Counter(gender_preference.values())}")
            print("---------------------")


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
