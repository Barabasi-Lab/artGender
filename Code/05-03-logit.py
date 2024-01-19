import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry_convert as pc
import statsmodels.formula.api as smf
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

from plot_config import *


def country_to_continent(country_name):
    try:
        country_alpha2 = pc.country_name_to_country_alpha2(country_name)
        country_continent_code = pc.country_alpha2_to_continent_code(
            country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(
            country_continent_code)
        return country_continent_name
    except:
        return np.nan


class Logit:
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

        self.artist_info = pd.read_csv("../raw_data/artists.tsv")
        self.shows_select = pd.read_csv(os.path.join(self.year_data_path, "shows_select.csv"))
        self.sales_select = pd.read_csv(os.path.join(self.year_data_path, "sales_select.csv"))
        self.career_span_select = pd.read_csv(os.path.join(self.year_data_path, "career_span_select.csv"))

        self.min_exh_data_path = os.path.join(self.year_data_path,
                                              f"minimum_exh_count_{self.num_exhibition_threshold}")

        self.agg = pd.read_csv(os.path.join(self.year_data_path, f"artist_exh_info_agg_{self.preference_type}.csv"))

        self.agg = self.agg[(self.agg["exhibition_count"] >= self.num_exhibition_threshold) & (
                self.agg["portion_overall"] != (0, 0, 0))]

        self.agg_sales = pd.read_csv(
            os.path.join(self.year_data_path, f"artist_sale_info_agg_{self.preference_type}.csv"))

        self.artist_nationality_dict, self.artist_most_common_country_dict = self.get_nationality_and_most_country()
        self.logit_data = self.create_logit_data()
        self.scaler, self.logit_data_processed = self.preprocess_logit_data()
        self.logit_data_downsampled = self.downsample_logit_data()

    def get_nationality_and_most_country(self):
        artist_nationality_dict = dict(zip(self.artist_info["artist"], self.artist_info["country"]))

        artist_most_common_country = self.shows_select.groupby("artist")["country"].agg(
            lambda x: pd.Series.mode(x)[0]).reset_index()
        artist_most_common_country_dict = dict(
            zip(artist_most_common_country["artist"], artist_most_common_country["country"]))

        return artist_nationality_dict, artist_most_common_country_dict

    def create_logit_data(self):
        logit_data = self.agg[
            ["artist", "exhibition_count", "portion_overall", "career_prestige", "gender_recog", "ins_gender"]]
        # add cumulative prestige
        artist_cumu_prestige = self.shows_select.groupby("artist")["percentile_prestige"].sum().reset_index()
        artist_cumu_prestige_dict = dict(
            zip(artist_cumu_prestige["artist"], artist_cumu_prestige["percentile_prestige"]))
        logit_data["cumu_prestige"] = [artist_cumu_prestige_dict[artist] for artist in logit_data["artist"]]
        # bin prestige
        pmap = {1: "Low", 2: "Mid", 3: "High"}
        bars = [0, np.percentile(logit_data["career_prestige"], 40), np.percentile(logit_data["career_prestige"], 70),
                1.1]
        logit_data["career_prestige_bin"] = [pmap[np.digitize(prestige, bars)] for prestige in
                                             logit_data["career_prestige"]]
        # add career length
        self.career_span_select["career_length"] = self.career_span_select["career_end_year"] - self.career_span_select[
            "career_start_year"]
        logit_data = logit_data.join(self.career_span_select[["artist", "career_length"]].set_index("artist"),
                                     on="artist", how="left")
        logit_data["career_length"] = logit_data["career_length"] + 1
        # label in_auction
        saled_artist = set(self.agg_sales["artist"])
        logit_data["in_auction"] = [1 if artist in saled_artist else 0 for artist in logit_data["artist"]]
        # add medium info
        medium_dict = dict(zip(self.artist_info["artist"], self.artist_info["category"]))
        logit_data["medium"] = [medium_dict[artist] for artist in logit_data["artist"]]
        medium_set = {'Installation', 'Print', 'Design & Decorative Art', 'Mixed Media', 'Collage', 'Ceramic',
                      'Video & Film', 'Work on Paper', 'Photography', 'Painting', 'Drawing & Watercolor', 'Sculpture'}
        medium_cleaned_list = []
        for each_medium in logit_data["medium"]:
            try:
                each_medium_eval = eval(each_medium)
                each_medium_eval = [each_medium for each_medium in each_medium_eval if each_medium in medium_set]
                if len(each_medium_eval) > 1:
                    medium_cleaned_list.append(each_medium_eval[0].replace(" ", "_").replace("&", ""))
                else:
                    medium_cleaned_list.append(np.nan)
            except:
                medium_cleaned_list.append(each_medium)
        logit_data["medium_cleaned"] = medium_cleaned_list
        # add avg exhibition count per year
        logit_data["avg_exhibition_count"] = (logit_data["exhibition_count"] + 0.1) / logit_data["career_length"]
        # add average cumulative prestige
        logit_data["avg_cumu_prestige"] = (logit_data["cumu_prestige"] + 0.1) / logit_data["career_length"]
        # add man_preferred and woman_preferred portion
        logit_data["man_preferred_portion"] = [eval(item)[0] for item in logit_data["portion_overall"]]
        logit_data["woman_preferred_portion"] = [eval(item)[1] for item in logit_data["portion_overall"]]
        # add nationality and most common country/continent
        logit_data["nationality"] = [self.artist_nationality_dict[artist] for artist in logit_data["artist"]]
        logit_data["most_common_country"] = [self.artist_most_common_country_dict[artist] for artist in
                                             logit_data["artist"]]
        logit_data["nationality_continent"] = [country_to_continent(self.artist_nationality_dict[artist])
                                               for artist in logit_data["artist"]]
        logit_data["most_common_continent"] = [country_to_continent(self.artist_most_common_country_dict[artist])
                                               for artist in logit_data["artist"]]

        # get dummies
        gender_encode = pd.get_dummies(logit_data['gender_recog'], prefix='gender')
        ins_gender_encode = pd.get_dummies(logit_data['ins_gender'], prefix='ins_gender')
        prestige_bin_encode = pd.get_dummies(
            logit_data["career_prestige_bin"], prefix="prestige")
        logit_data["prestige_ins_gender"] = ["%s_%s" % (prestige, ins_gender)
                                             for (prestige, ins_gender) in
                                             zip(logit_data["career_prestige_bin"], logit_data["ins_gender"])]
        prestige_ins_gender_encode = pd.get_dummies(
            logit_data["prestige_ins_gender"], prefix="")
        logit_data["gender_ins_gender"] = ["%s_%s" % (gender, ins_gender) for (
            gender, ins_gender) in zip(logit_data["gender_recog"], logit_data["ins_gender"])]
        gender_ins_gender_encode = pd.get_dummies(
            logit_data["gender_ins_gender"], prefix="")
        nationality_continent_encode = pd.get_dummies(
            logit_data["nationality_continent"], prefix="nationality_continent")
        most_common_continent_encode = pd.get_dummies(
            logit_data["most_common_continent"], prefix="continent")
        medium_encode = pd.get_dummies(logit_data["medium_cleaned"], prefix="medium")

        logit_data = pd.concat([logit_data, gender_encode, ins_gender_encode,
                                prestige_ins_gender_encode, prestige_bin_encode,
                                gender_ins_gender_encode,
                                nationality_continent_encode, most_common_continent_encode,
                                medium_encode], axis=1)

        logit_data.to_csv(os.path.join(self.min_exh_data_path, "logit_data.csv"), index=False)
        return logit_data

    def plot_access_rate_vs_career_length_exhibit_count(self):
        # career length vs access rate
        bins = [1, 5, 10, 15, 20, 30]
        in_auction_count_career = self.logit_data.groupby(pd.cut(self.logit_data.career_length, bins))[
            "in_auction"].value_counts().reset_index(
            name='count').pivot(index="career_length", columns="in_auction", values="count").reset_index()
        in_auction_count_career["total_count"] = in_auction_count_career[
                                                     0] + in_auction_count_career[1]
        in_auction_count_career["access_rate"] = in_auction_count_career[
                                                     1] / in_auction_count_career["total_count"]
        # exhibition count vs access rate
        bins = [0, 1, 3, 5, 10, 40]
        in_auction_exhibition_count = self.logit_data.groupby(pd.cut(self.logit_data.avg_exhibition_count, bins))[
            "in_auction"].value_counts().reset_index(
            name='count').pivot(index="avg_exhibition_count", columns="in_auction", values="count").reset_index()
        in_auction_exhibition_count["total_count"] = in_auction_exhibition_count[
                                                         0] + in_auction_exhibition_count[1]
        in_auction_exhibition_count["access_rate"] = in_auction_exhibition_count[
                                                         1] / in_auction_exhibition_count["total_count"]

        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(2, 1, figsize=(4.5, 6))
            ax[0].plot(in_auction_count_career["career_length"].astype(str), in_auction_count_career[
                "access_rate"] * 100, "o-", color=colors[3])
            ax[0].set_xlabel("Career Length (Year)", fontsize=20)
            ax[0].set_ylabel("Auction Access Rate (%)")
            ax[1].plot(in_auction_exhibition_count["avg_exhibition_count"].astype(str), in_auction_exhibition_count[
                "access_rate"] * 100, "o-", color=colors[3])
            ax[1].set_xlabel("Exhibition Count Per Year", fontsize=20)
            ax[1].set_ylabel("Auction Access Rate (%)")
            ax[0].set_ylim(0, )
            ax[1].set_ylim(0, )
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.auction_fig_path, "04-avg_exhibition_count_career_year_vs_access_rate.pdf"))

            if self.save_alice:
                in_auction_count_career[["career_length", "access_rate"]].to_csv(
                    "../for_alice/Main Paper Data/4B_top.csv", index=False)
            in_auction_exhibition_count[["avg_exhibition_count", "access_rate"]].to_csv(
                "../for_alice/Main Paper Data/4B_bottom.csv", index=False)

    def preprocess_logit_data(self):
        logit_data_processed = self.logit_data.dropna(subset=["most_common_continent"])
        print(len(logit_data_processed), logit_data_processed[logit_data_processed["artist"] == 1171]["most_common_continent"])
        logit_data_processed = logit_data_processed.rename(columns=
                                                           {"continent_South America": "continent_South_America",
                                                            "continent_North America": "continent_North_America"})

        # logarithm
        logit_data_processed["avg_exhibition_count_log"] = np.log(logit_data_processed["avg_exhibition_count"])
        logit_data_processed["career_length_log"] = np.log(logit_data_processed["career_length"])

        # normalization
        norm_features = ['exhibition_count', "avg_exhibition_count", "avg_exhibition_count_log",
                         "career_prestige", "cumu_prestige", "avg_cumu_prestige",
                         "career_length", "career_length_log"]
        scaler = MinMaxScaler()
        logit_data_processed[norm_features] = scaler.fit_transform(logit_data_processed[norm_features].to_numpy())
        logit_data_processed.to_csv(os.path.join(self.min_exh_data_path, "logit_data_processed.csv"), index=False)
        pickle.dump(scaler, open(os.path.join(self.min_exh_data_path, "logit_data_scaler.sav"), "wb"))
        return scaler, logit_data_processed

    def create_test_cases(self):
        avg_exhibition_count_log = np.log(5)
        avg_exhibition_count_log_max, avg_exhibition_count_log_min = self.scaler.data_max_[2], self.scaler.data_min_[2]
        avg_exhibition_count_log_normed = (avg_exhibition_count_log - avg_exhibition_count_log_min) / (
                avg_exhibition_count_log_max - avg_exhibition_count_log_min)

        career_length_log = np.log(11)
        career_length_log_max, career_length_log_min = self.scaler.data_max_[-1], self.scaler.data_min_[-1]
        career_length_log_normed = (career_length_log - career_length_log_min) / \
                                   (career_length_log_max - career_length_log_min)

        test_data = {"avg_exhibition_count_log": [avg_exhibition_count_log_normed, avg_exhibition_count_log_normed,
                                                  avg_exhibition_count_log_normed],
                     "career_length_log": [career_length_log_normed, career_length_log_normed,
                                           career_length_log_normed],
                     "gender_Female": [0, 1, 0],
                     "ins_gender_0": [0, 0, 0],
                     "ins_gender_2": [0, 0, 1]}
        test_data_df = pd.DataFrame.from_dict(test_data)
        return test_data_df

    def downsample_logit_data(self):
        df_majority = self.logit_data_processed[self.logit_data_processed.in_auction == 0]
        df_minority = self.logit_data_processed[self.logit_data_processed.in_auction == 1]
        random_state = 0
        # Downsample majority class
        df_majority_downsampled = resample(df_majority,
                                           replace=False,
                                           n_samples=len(df_minority),
                                           random_state=random_state)
        # Combine minority class with downsampled majority class
        logit_data_downsampled = pd.concat([df_majority_downsampled, df_minority])
        logit_data_downsampled.to_csv(os.path.join(self.min_exh_data_path, "logit_data_downsampled.csv"), index=False)
        return logit_data_downsampled

    def logit_statsmodel(self, formula, data):
        log_reg = smf.logit(formula=formula, data=data).fit()
        return log_reg

    def get_cv_score(self, data, formula, score):
        features = formula.replace("in_auction ~ ", "").split(" + ")
        X = data[features].to_numpy()
        y = data["in_auction"].to_numpy()
        scores = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X, y):
            data_train = data.iloc[train_index]
            data_test = data.iloc[test_index]
            logit_reg1 = self.logit_statsmodel(formula, data_train)
            train_pred = logit_reg1.predict(data_test)
            pred_labels = list(map(round, train_pred))
            scores.append(score(data_test.in_auction, pred_labels))
        return scores

    def get_result_df(self, log_reg):
        conf_int_array = log_reg.conf_int()
        rounded_conf_int_array = np.round(conf_int_array, 2)
        print(conf_int_array, rounded_conf_int_array)
        conf_int_result = map(str, list(zip(rounded_conf_int_array[0], rounded_conf_int_array[1])))
        print(conf_int_result)
        statistics_df = pd.DataFrame({
            'name': log_reg.params.index,
            'odds ratio': np.exp(log_reg.params.values),
            'coef': log_reg.params.values,
            'standard error': log_reg.bse,
            'pvalues': log_reg.pvalues,
            'conf_int': conf_int_result
        })
        return statistics_df

    def create_result_summary_df(self, fitted_model_names, fitted_models):
        fitted_models_statistics = [self.get_result_df(model) for model in fitted_models]

        feature_names = ["Intercept", "avg_exhibition_count_log",
                         "man_preferred_portion", "woman_preferred_portion",
                         "career_length_log",
                         "gender_Female",
                         "ins_gender_0", "ins_gender_2",
                         "_Female_0", "_Female_1", "_Female_2", "_Male_0", "_Male_2",
                         "prestige_Low", "prestige_Mid",
                         "continent_Africa", "continent_Asia", "continent_Europe", "continent_Oceania",
                         "continent_South_America",
                         "medium_Drawing__Watercolor", "medium_Installation", "medium_Mixed_Media",
                         "medium_Photography",
                         "medium_Print", "medium_Sculpture", "medium_Video__Film"]
        feature_names_pretty = ["Intercept", "Exhibition Count Per Year",
                                "Portion of Man-preferred Institution Exhibition",
                                "Portion of Woman-preferred Institution Exhibition",
                                "Career Length",
                                "Woman",
                                "Co-exhibit. Neutral", "Co-exhibit. Woman",
                                "Woman, Co-exhibit. Neutral", "Woman, Co-exhibit. Man", "Woman, Co-exhibit. Woman",
                                "Man, Co-exhibit. Neutral", "Man, Co-exhibit. Woman",
                                "Low Prestige", "Mid Prestige",
                                "Africa", "Asia", "Europe", "Oceania", "South America",
                                "Drawing & Watercolor", "Installation", "Mixed Media", "Photography", "Print",
                                "Sculpture",
                                "Video & Film"]
        metric_names = ["Coef.", "O.R.", "S.E.", "P Val.", "Conf. Int."]
        cols_list = []
        for model in fitted_model_names:
            for metric in metric_names:
                cols_list.append((model, metric))
        cols = pd.MultiIndex.from_tuples(cols_list)

        data = []
        for feature in feature_names:
            this_feature_data = []
            for statistics_df in fitted_models_statistics:
                try:
                    values = statistics_df[statistics_df["name"] == feature][
                        ["coef", "odds ratio", "standard error", "pvalues", "conf_int"]].values.tolist()[0]
                    this_feature_data += values
                except:
                    this_feature_data += ["-", "-", "-", "-","-"]
            data.append(this_feature_data)
        df = pd.DataFrame(data, columns=cols, index=feature_names_pretty)

        bic = [model.bic for model in fitted_models]
        df_bic = pd.DataFrame(np.array([bic]), columns=fitted_model_names, index=["bic"])
        print(df_bic)

        dof = [model.df_model for model in fitted_models] # degree of freedom
        df_dof = pd.DataFrame(np.array([dof]), columns=fitted_model_names, index=["dof"])
        print(df_dof)

        nobs = [model.nobs for model in fitted_models]
        df_nobs = pd.DataFrame(np.array([nobs]), columns=fitted_model_names, index=["nobs"])
        print(df_nobs)

        conf_int = [model.conf_int() for model in fitted_models]


        df.to_excel(os.path.join(self.auction_fig_path, "logit_results.xlsx"))
        df_bic.to_excel(os.path.join(self.auction_fig_path, "logit_bic.xlsx"))
        df_dof.to_excel(os.path.join(self.auction_fig_path, "logit_dof.xlsx"))
        df_nobs.to_excel(os.path.join(self.auction_fig_path, "logit_nobs.xlsx"))


def main():
    parser = argparse.ArgumentParser(
        description='select artists with career start year > [year_threshold]')
    parser.add_argument('-t', '--genderizeio_threshold', type=float, help='genderize.io threshold', default=0.6)
    parser.add_argument('-f', '--remove_birth', action=argparse.BooleanOptionalAction)
    parser.add_argument('-y', '--career_start_threshold', type=int,
                        help='earliest career start year of selected artists', default=1990)
    parser.add_argument('-p', '--preference_type', type=str, help='neutral or balance', default="neutral")
    parser.add_argument('-n', '--num_exhibition_threshold', type=int, help='number of exhibition threshold', default=10)
    parser.add_argument('-a', '--save_alice', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    logit = Logit(args.genderizeio_threshold,
                  args.remove_birth,
                  args.career_start_threshold,
                  args.preference_type,
                  args.num_exhibition_threshold,
                  args.save_alice)

    logit.plot_access_rate_vs_career_length_exhibit_count()

    # model 1
    base_formula = "in_auction ~ avg_exhibition_count_log + career_length_log"
    logit_base = logit.logit_statsmodel(base_formula, logit.logit_data_downsampled)
    print(logit_base.summary())
    # model 2
    add_gender_formula = "in_auction ~ avg_exhibition_count_log + career_length_log + gender_Female"
    logit_add_gender = logit.logit_statsmodel(add_gender_formula, logit.logit_data_downsampled)
    # model 3
    add_coexh_gender_formula = "in_auction ~ avg_exhibition_count_log + career_length_log + gender_Female + ins_gender_0 + ins_gender_2"
    logit_add_coexh_gender = logit.logit_statsmodel(add_coexh_gender_formula, logit.logit_data_downsampled)
    # model 4
    add_preferred_portion_formula = "in_auction ~ avg_exhibition_count_log + career_length_log + gender_Female + ins_gender_0 + ins_gender_2 + man_preferred_portion + woman_preferred_portion"
    logit_add_preferred_portion = logit.logit_statsmodel(add_preferred_portion_formula, logit.logit_data_downsampled)
    # model 5
    gender_coexh_gender_mix_formula = "in_auction ~ avg_exhibition_count_log + career_length_log + _Female_0 + _Female_1 + _Female_2 + _Male_0 + _Male_2"
    logit_gender_coexh_gender_mix = logit.logit_statsmodel(gender_coexh_gender_mix_formula,
                                                           logit.logit_data_downsampled)
    # model 6
    add_prestige_formula = "in_auction ~ avg_exhibition_count_log + career_length_log + _Female_0 + _Female_1 + _Female_2 + _Male_0 + _Male_2 + prestige_Low + prestige_Mid"
    logit_add_prestige = logit.logit_statsmodel(add_prestige_formula, logit.logit_data_downsampled)
    # model 7
    add_continent_formula = "in_auction ~ avg_exhibition_count_log + career_length_log + _Female_0 + _Female_1 + _Female_2 + _Male_0 + _Male_2 + prestige_Low + prestige_Mid + continent_Africa + continent_Asia + continent_Europe + continent_Oceania + continent_South_America"
    logit_add_continent = logit.logit_statsmodel(add_continent_formula, logit.logit_data_downsampled)
    # model 8
    add_medium_formula = "in_auction ~ avg_exhibition_count_log + career_length_log + gender_Female + ins_gender_0 + ins_gender_2 + prestige_Low + prestige_Mid + medium_Drawing__Watercolor + medium_Installation + medium_Mixed_Media + medium_Photography + medium_Print + medium_Sculpture + medium_Video__Film"
    df_downsampled_filtered = logit.logit_data_downsampled[
        logit.logit_data_downsampled["medium_cleaned"] != "Design__Decorative_Art"].dropna(subset=["medium_cleaned"])
    logit_add_medium = logit.logit_statsmodel(add_medium_formula, df_downsampled_filtered)

    # cross validation AUC score
    formula_auc_scores_downsampled = []
    for formula in [base_formula, add_gender_formula, add_coexh_gender_formula, add_preferred_portion_formula,
                    gender_coexh_gender_mix_formula, add_prestige_formula, add_continent_formula]:
        formula_auc_scores_downsampled.append(logit.get_cv_score(logit.logit_data_downsampled, formula, f1_score))

    # formula_auc_scores_downsampled.append(logit.get_cv_score(df_downsampled_filtered, add_medium_formula, f1_score))
    print("downsampled f1 scores", [np.mean(scores) for scores in formula_auc_scores_downsampled])

    # case study df
    test_data_df = logit.create_test_cases()
    print(test_data_df)
    train_pred = logit_add_coexh_gender.predict(test_data_df)
    print(train_pred)

    # get summary statistics excel
    model_names = ["base", "add_gender", "add_coexh_gender", "add_preferred_portion", "gender_coexh_gender_mix",
                   "add_prestige", "add_continent", "add_medium"]
    fitted_models = [logit_base, logit_add_gender, logit_add_coexh_gender, logit_add_preferred_portion,
                     logit_gender_coexh_gender_mix, logit_add_prestige, logit_add_continent, logit_add_medium]
    logit.create_result_summary_df(model_names, fitted_models)


if __name__ == '__main__':
    main()
