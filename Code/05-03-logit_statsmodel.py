from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import pickle
import sys
from patsy import dmatrices
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("../data/logit_data.csv")

# preprocessing
df = df.dropna(subset=["most_common_continent"])

df = df.rename(columns={"continent_South America": "continent_South_America",
                        "continent_North America": "continent_North_America"})


# logarithm
df["avg_exhibition_count_log"] = np.log(df["avg_exhibition_count"])
df["career_length_log"] = np.log(df["career_length"])

# normalization
norm_features = ['exhibition_count', "avg_exhibition_count", "avg_exhibition_count_log",
                 "career_prestige", "cumu_prestige", "avg_cumu_prestige",
                 "career_length", "career_length_log"]
scaler = MinMaxScaler()
df[norm_features] = scaler.fit_transform(
    df[norm_features].to_numpy())
pickle.dump(scaler, open("../logit_models/scaler.sav", "wb"))

# create test examples
avg_exhibition_count_log = np.log(5)
avg_exhibition_count_log_max, avg_exhibition_count_log_min = scaler.data_max_[
    2], scaler.data_min_[2]
avg_exhibition_count_log_normed = (avg_exhibition_count_log - avg_exhibition_count_log_min)/(
    avg_exhibition_count_log_max-avg_exhibition_count_log_min)

career_length_log = np.log(11)
career_length_log_max, career_length_log_min = scaler.data_max_[
    -1], scaler.data_min_[-1]
career_length_log_normed = (career_length_log - career_length_log_min) / \
    (career_length_log_max-career_length_log_min)


test_data = {"avg_exhibition_count_log": [avg_exhibition_count_log_normed, avg_exhibition_count_log_normed, avg_exhibition_count_log_normed],
             "career_length_log": [career_length_log_normed, career_length_log_normed, career_length_log_normed],
             "gender_Female": [0, 1, 0],
             "ins_gender_0": [0, 0, 0],
             "ins_gender_2": [0, 0, 1]}
test_data_df = pd.DataFrame.from_dict(test_data)


def logit_model(formula, data):
    log_reg = smf.logit(formula=formula, data=data).fit()
    summary = log_reg.summary()
    odds_ratios_df = pd.DataFrame({
        'name': log_reg.params.index,
        'odds ratio': np.exp(log_reg.params.values),
        'coef': log_reg.params.values,
        'pvalues': log_reg.pvalues
    })
    print(summary)
    odds_ratios_df.to_csv(sys.stdout, index=False)
    print(log_reg.bic)
    return log_reg, odds_ratios_df


formula1 = "in_auction ~ avg_exhibition_count_log + career_length_log"
logit_reg1 = logit_model(formula1, df)

formula2 = "in_auction ~ avg_exhibition_count_log + career_length_log + gender_Female"
logit_reg2 = logit_model(formula2, df)

formula3 = "in_auction ~ avg_exhibition_count_log + career_length_log + gender_Female + ins_gender_0 + ins_gender_2"
logit_reg3 = logit_model(formula3, df)


formula4 = "in_auction ~ avg_exhibition_count_log + career_length_log + _Female_0 + _Female_1 + _Female_2 + _Male_0 + _Male_2"
logit_reg4 = logit_model(formula4, df)


formula5 = "in_auction ~ avg_exhibition_count_log + career_length_log + _Female_0 + _Female_1 + _Female_2 + _Male_0 + _Male_2 + prestige_Low + prestige_Mid"
logit_reg5 = logit_model(formula5, df)

formula6 = "in_auction ~ avg_exhibition_count_log + career_length_log + _Female_0 + _Female_1 + _Female_2 + _Male_0 + _Male_2 + prestige_Low + prestige_Mid + continent_Africa + continent_Asia + continent_Europe + continent_Oceania + continent_South_America"
logit_reg6 = logit_model(formula6, df)


# cross validation with statsmodel
df_majority = df[df.in_auction == 0]
df_minority = df[df.in_auction == 1]
random_state = 0
# Downsample majority class
df_majority_downsampled = resample(df_majority,
                                   replace=False,
                                   n_samples=len(df_minority),
                                   random_state=random_state)
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])


def get_cv_score(df, formula, score):
    features = formula.replace("in_auction ~ ", "").split(" + ")
    X = df[features].to_numpy()
    y = df["in_auction"].to_numpy()
    scores = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X, y):
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        logit_reg1 = logit_model(formula, df_train)
        train_pred = logit_reg1[0].predict(df_test)
        pred_labels = list(map(round, train_pred))
        conf_matrix = confusion_matrix(df_test.in_auction, pred_labels)
        scores.append(score(df_test.in_auction, pred_labels))
    return scores


formula_f1_scores = []
for formula in [formula1, formula2, formula3, formula4, formula5, formula6]:
    formula_scores.append(get_cv_score(df, formula, f1_score))

print([np.mean(scores) for scores in formula_scores])

formula_auc_scores_downsampled = []
for formula in [formula1, formula2, formula3, formula4, formula5, formula6]:
    formula_scores.append(get_cv_score(df_downsampled, formula, roc_auc_score))

print([np.mean(scores) for scores in formula_scores])


logit_scores = cross_val_score(LogisticRegression(
    solver="lbfgs", random_state=42), X, y, cv=kf, scoring='roc_auc')


formula7 = "in_auction ~ avg_exhibition_count_log + career_length_log + prestige_Low + prestige_Mid +  +  _Female_0 + _Female_1 + _Female_2 + _Male_0 + _Male_2"
logit_reg7 = logit_model(formula7, df)


# break by prestige
low = df[df["career_prestige_bin"] == "Low"]
mid = df[df["career_prestige_bin"] == "Mid"]
high = df[df["career_prestige_bin"] == "High"]
formula1 = "in_auction ~ avg_exhibition_count_log + career_length_log + gender_Female + ins_gender_0 + ins_gender_2"
formula2 = "in_auction ~ avg_exhibition_count_log + career_length_log +_Female_0 + _Female_1 + _Female_2 + _Male_0 + _Male_2"

logit_reg_low1 = logit_model(formula1, low)
logit_reg_mid1 = logit_model(formula1, mid)
logit_reg_high1 = logit_model(formula1, high)

logit_reg_low2 = logit_model(formula2, low)
logit_reg_mid2 = logit_model(formula2, mid)
logit_reg_high2 = logit_model(formula2, high)

log_reg1_low = log_model(formula1, low)
log_reg1_mid = log_model(formula1, mid)
log_reg1_high = log_model(formula1, high)


# add prestige
formula2 = "in_auction ~ avg_exhibition_count_log + career_length_log + C(gender_recog, Treatment('Male')) + C(ins_gender, Treatment(1)) + C(career_prestige_bin, Treatment('High'))"
log_reg2 = log_model(formula2, df)

# interaction with ins_gender and career prestige
formula3 = "in_auction ~ C(ins_gender, Treatment(1)): C(career_prestige_bin, Treatment('Low')) + avg_exhibition_count_log + career_length_log + C(gender_recog, Treatment('Male'))"
log_reg3 = log_model(formula3, df)


formula3 = "in_auction ~ avg_exhibition_count_log + career_length_log + C(gender_recog, Treatment('Male')) + C(ins_gender, Treatment(0))"
formula4 = "in_auction ~ avg_exhibition_count_log + career_length_log + C(gender_recog, Treatment('Male')) + C(ins_gender, Treatment(0))"
formula5 = "in_auction ~ avg_exhibition_count_log + career_length_log + C(gender_recog, Treatment('Male')) + C(ins_gender, Treatment(0))"


log_reg_1 = smf.logit(
    formula=, data=df).fit()
print(log_reg_1.summary())

coef_odds_ratios = pd.DataFrame({
    'coef': log_reg_1.params.values,
    'odds ratio': np.exp(log_reg_1.params.values),
    'name': log_reg_1.params.index
})


log_reg = smf.logit(formula="in_auction ~ avg_exhibition_count_log + career_length_log + C(career_prestige_bin, Treatment('Low')) + C(gender_recog, Treatment('Male')) + C(ins_gender, Treatment(0))", data=df).fit()


print(log_reg.summary())

coef_odds_ratios = pd.DataFrame({
    'coef': log_reg.params.values,
    'odds ratio': np.exp(log_reg.params.values),
    'name': log_reg.params.index
})
