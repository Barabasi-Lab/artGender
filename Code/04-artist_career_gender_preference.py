import pandas as pd
import numpy as np
import json
from collections import Counter
import matplotlib.pyplot as plt
from adjustText import adjust_text
from plot_config import *
import argparse

parser = argparse.ArgumentParser(
    description='earliest year for data selection')
parser.add_argument('year', metavar='year', type=int,
                    help='select year')
parser.add_argument('type', metavar='type', type=str,
                    help='balance or neutral')

args = parser.parse_args()
select_year = args.year
preference_type = args.type

artists_select = pd.read_csv(
    "../related_data_results/artists_select_%s.csv" % select_year)
shows_select = pd.read_csv(
    "../related_data_results/shows_select_%s.csv" % select_year)
gender_preference_dict = json.load(
    open("../related_data_results/gender_%s_ins.json" % preference_type))

shows_select["gender_preference"] = [int(gender_preference_dict[
    str(ins)]) if str(ins) in gender_preference_dict else np.nan for ins in shows_select["institution"]]

count_null_model = Counter(gender_preference_dict.values())
p_null = np.array([count_null_model[1] / len(gender_preference_dict), count_null_model[2] /
                   len(gender_preference_dict), count_null_model[0] / len(gender_preference_dict)])

portion_overall_dict = {}
portion_init_dict = {}
portion_late_dict = {}
prestige_init_dict = {}
prestige_late_dict = {}
for artist in set(shows_select["artist"]):
    this_person_shows = shows_select[
        (shows_select["artist"] == artist)].dropna(subset=["gender_preference"])
    if len(this_person_shows) <= 5:
        portion_overall_dict[artist] = (0, 0, 0)
        portion_init_dict[artist] = (0, 0, 0)
        portion_late_dict[artist] = (0, 0, 0)
        prestige_init_dict[artist] = 0
        prestige_late_dict[artist] = 0
    else:
        # total portion
        gender_count = Counter(this_person_shows["gender_preference"])
        portion_overall_dict[artist] = (gender_count[1] / len(this_person_shows), gender_count[
            2] / len(this_person_shows), gender_count[0] / len(this_person_shows))
        # init portion
        gender_count_init = Counter(
            list(this_person_shows["gender_preference"])[:5])
        portion_init_dict[artist] = (
            gender_count_init[1] / 5, gender_count_init[2] / 5, gender_count_init[0] / 5)
        # late portion
        gender_count_late = Counter(
            list(this_person_shows["gender_preference"])[-5:])
        portion_late_dict[artist] = (
            gender_count_late[1] / 5, gender_count_late[2] /
            5, gender_count_late[0] / 5)
        # init prestige
        prestige_init = np.mean(this_person_shows["percentile_prestige"][:5])
        prestige_init_dict[artist] = prestige_init
        # late parestige
        prestige_late = np.mean(this_person_shows["percentile_prestige"][-5:])
        prestige_late_dict[artist] = prestige_late


number_of_exhibitions = shows_select.groupby("artist")["show"].count(
).reset_index().rename(columns={"show": "exhibition_count"})
different_ins_count = shows_select.groupby("artist")["institution"].nunique(
).reset_index().rename(columns={"institution": "different_ins_count"})
career_prestige = shows_select.groupby("artist")["percentile_prestige"].mean(
).reset_index().rename(columns={"percentile_prestige": "career_prestige"})  # mean
print(career_prestige)
gender = shows_select[["artist", "gender_recog"]
                      ].drop_duplicates().reset_index()
agg = number_of_exhibitions.join(
    career_prestige.set_index("artist"), on="artist")
agg = agg.join(different_ins_count.set_index("artist"), on="artist")
agg = agg.join(
    gender.set_index("artist"), on="artist")

agg["round_prestige"] = [
    round(item, 1) for item in agg["career_prestige"]]
gender_map = {0: 1, 1: 2, 2: 0}
agg["portion_overall"] = [portion_overall_dict[artist]
                          for artist in agg["artist"]]
agg["portion_init"] = [portion_init_dict[artist] for artist in agg["artist"]]
agg["portion_late"] = [portion_late_dict[artist] for artist in agg["artist"]]

agg["prestige_init"] = [prestige_init_dict[artist] for artist in agg["artist"]]
agg["prestige_late"] = [prestige_late_dict[artist] for artist in agg["artist"]]
agg["ins_gender"] = [gender_map[np.argmax((np.array(
    portion_overall) - p_null) / p_null)] for portion_overall in agg["portion_overall"]]
agg["ins_gender_init"] = [gender_map[np.argmax((np.array(
    portion_overall) - p_null) / p_null)] for portion_overall in agg["portion_init"]]
agg["ins_gender_late"] = [gender_map[np.argmax((np.array(
    portion_overall) - p_null) / p_null)] for portion_overall in agg["portion_late"]]
pmap = {1: "Low", 2: "Mid", 3: "High"}
bin_edge = [0, np.percentile(agg["career_prestige"], 40), np.percentile(
    agg["career_prestige"], 70), 1.1]
print("bin_edge:", bin_edge)
agg["prestige_bin"] = [pmap[np.digitize(
    item, bin_edge)] for item in agg["career_prestige"]]
bin_edge = [0, np.percentile(agg[agg["prestige_init"] > 0]["prestige_init"], 40), np.percentile(
    agg[agg["prestige_init"] > 0]["prestige_init"], 70), 1.1]
agg["early_prestige_bin"] = [pmap[np.digitize(
    item, bin_edge)] for item in agg["prestige_init"]]
bin_edge = [0, np.percentile(agg[agg["prestige_late"] > 0]["prestige_late"], 40), np.percentile(
    agg[agg["prestige_late"] > 0]["prestige_late"], 70), 1.1]
agg["late_prestige_bin"] = [pmap[np.digitize(
    item, bin_edge)] for item in agg["prestige_late"]]
print(len(agg))

agg.to_csv("../related_data_results/artist_exh_info_agg_%s.csv" %
           preference_type)

agg = pd.read_csv(
    "../related_data_results/artist_exh_info_agg_%s.csv" % preference_type)
# select artist with more than 10 exhibitions
agg_legit = agg[(agg["exhibition_count"] >= 10) &
                (agg["portion_overall"] != (0, 0, 0))]
# agg_legit["prestige_bin"] = [cmap[np.digitize(
#     item, [0, np.percentile(agg_legit["career_prestige"], 40), np.percentile(agg_legit["career_prestige"], 70), max(agg_legit["career_prestige"]) + 1])] for item in agg_legit["career_prestige"]]
# agg_legit["prestige_bin"] = [pmap[np.digitize(
# item, [0, np.percentile(agg_legit["career_prestige"], 40),
# np.percentile(agg_legit["career_prestige"], 70), 1.1])] for item in
# agg_legit["career_prestige"]]

male_low_count = len(agg_legit[(agg_legit["gender_recog"] == "Male") & (
    agg_legit["prestige_bin"] == "Low")])
male_mid_count = len(agg_legit[(agg_legit["gender_recog"] == "Male") & (
    agg_legit["prestige_bin"] == "Mid")])
male_high_count = len(agg_legit[(agg_legit["gender_recog"] == "Male") & (
    agg_legit["prestige_bin"] == "High")])

female_low_count = len(agg_legit[(agg_legit["gender_recog"] == "Female") & (
    agg_legit["prestige_bin"] == "Low")])
female_mid_count = len(agg_legit[(agg_legit["gender_recog"] == "Female") & (
    agg_legit["prestige_bin"] == "Mid")])
female_high_count = len(agg_legit[(agg_legit["gender_recog"] == "Female") & (
    agg_legit["prestige_bin"] == "High")])

agg_percent = agg_legit.groupby("gender_recog")[
    "ins_gender"].value_counts(normalize=True)
agg_percent = agg_percent.mul(100)
agg_percent = agg_percent.rename('Percent').reset_index()

plt.figure(figsize=(8, 6))
g = sns.barplot(x="gender_recog", y='Percent', hue="ins_gender", hue_order=[
                1, 2, 0], order=["Male", "Female"], data=agg_percent)
g.set_xticklabels(["Man", "Woman"])
g.set_ylim(0, )

for p in g.patches:
    txt = str(p.get_height().round(1)) + '%'
    txt_x = p.get_x()
    txt_y = p.get_height()
    g.text(txt_x + p.get_width() / 2., txt_y, txt,
           ha="center", va="top", fontsize=20, color="white")

# g.set_xticklabels(
#     ["Male-Preferred\nDominant", "Female-Preferred\nDominant", "Balance\nDominant"])
g.set_xlabel("")
g.set_ylabel("Portion of Artist of Co-exhibition Gender")
g.tick_params(axis='x', labelsize=25)
g.legend_.remove()
plt.tight_layout()
plt.savefig("../figures/03-ins_gender_count_diff_gender_%s.pdf" %
            preference_type)
if preference_type == "neutral":
    agg_percent.to_csv(
        '/Users/xindiwang/Dropbox (CCNR)/Success Team/Wang, Xindi/Gender Inequality in Art/For Alice/Main Paper New Figures/ins_gender.csv', index=False)

agg_percent = agg_legit.groupby(["gender_recog", "prestige_bin"])[
    "ins_gender"].value_counts(normalize=True)
agg_percent = agg_percent.mul(100)
agg_percent = agg_percent.rename('Percent').reset_index()

texts = []
with sns.axes_style("whitegrid"):
    ax = sns.catplot(x="prestige_bin", y='Percent', hue="ins_gender", col="gender_recog", hue_order=[
        1, 2, 0], kind="point", col_order=["Male", "Female"], order=["Low", "Mid", "High"], data=agg_percent, legend=False, sharex=False)
for i in range(2):
    texts = []
    for j, p in enumerate(ax.axes[0][i].lines):
        xlist, ylist = p.get_data()
        if len(ylist) < 3:
            continue
        print(j, ylist)
        for x, y in zip(xlist, ylist):
            texts.append(ax.axes[0][i].text(
                x, y, "%.1f" % y + "%", ha="center", va="top", color="black", fontsize=20))
    adjust_text(texts, ax=ax.axes[0][i])

ax.axes[0][0].set_xlabel("")
ax.axes[0][1].set_xlabel("")
# ax.axes[0][2].set_xlabel("")
ax.axes[0][0].set_ylabel("Portion of Artist of Co-exhibit Gender")
ax.axes[0][1].set_ylabel("")
# ax.axes[0][2].set_ylabel("")
ax.axes[0][0].set_title("Man Artist", fontsize=25)
ax.axes[0][1].set_title("Woman Artist", fontsize=25)
ax.axes[0][0].tick_params(axis='x', labelsize=25)
ax.axes[0][1].tick_params(axis='x', labelsize=25)
# ax.axes[0][2].set_title("High")
ax.axes[0][0].set_ylim(0, )
ax.axes[0][1].set_ylim(0, )
ax.axes[0][0].set_xticklabels(
    ["Low", "Mid", "High"])
ax.axes[0][1].set_xticklabels(
    ["Low", "Mid", "High"])
# ax.axes[0][0].set_xticklabels(
#     ["Low\n(%s)" % male_low_count, "Mid\n(%s)" % male_mid_count, "High\n(%s)" % male_high_count])
# ax.axes[0][1].set_xticklabels(
#     ["Low\n(%s)" % female_low_count, "Mid\n(%s)" % female_mid_count, "High\n(%s)" % female_high_count])
plt.tight_layout()
plt.savefig(
    "../otherFigures/03-ins_gender_count_diff_gender_prestige_%s.pdf" % preference_type)
if preference_type == "neutral":
    agg_percent.to_csv(
        '/Users/xindiwang/Dropbox (CCNR)/Success Team/Wang, Xindi/Gender Inequality in Art/For Alice/Main Paper New Figures/ins_gender_prestige.csv', index=False)


agg_percent = agg_legit.groupby("gender_recog")[
    "ins_gender_init"].value_counts(normalize=True)
agg_percent = agg_percent.mul(100)
agg_percent = agg_percent.rename('Percent').reset_index()

plt.figure(figsize=(8, 6))
g = sns.barplot(x="gender_recog", y='Percent', hue="ins_gender_init", hue_order=[
                1, 2, 0], order=["Male", "Female"], data=agg_percent)
g.set_ylim(0, )
g.set_xticklabels(["Man", "Woman"])

for p in g.patches:
    txt = str(p.get_height().round(1)) + '%'
    txt_x = p.get_x()
    txt_y = p.get_height()
    g.text(txt_x + p.get_width() / 2., txt_y, txt,
           ha="center", va="top", fontsize=20, color="white")

g.set_xlabel("")
g.set_ylabel("Portion of Artist of Early Co-exhibit Gender")
g.tick_params(axis='x', labelsize=25)
g.legend_.remove()
plt.tight_layout()
plt.savefig("../otherFigures/03-ins_gender_early_count_diff_gender_%s.pdf" %
            preference_type)

# equality lock-in
count = Counter(
    tuple(zip(agg_legit["ins_gender_init"], agg_legit["ins_gender_late"])))
print(count)
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
print(confusion_matrix, np.sum(confusion_matrix, axis=0))
normalize = confusion_matrix / np.sum(confusion_matrix, axis=1).reshape(3, 1)
plt.figure(figsize=(6, 6))
ax = sns.heatmap(normalize, annot=True, fmt=".2%",
                 cmap="Blues", annot_kws={"size": 25}, cbar=False)
plt.xlabel("Late Co-exhibition Gender", fontsize=25)
plt.ylabel("Early Co-exhibition Gender", fontsize=25)
print(ax.get_yticks())
ax.set_xticklabels(["Co-exhibit\nMan",
                    "Co-exhibit\nWoman", "Co-exhibit\n%s" % preference_type.title()])
plt.yticks([0.5, 1.5, 2.5], ["Co-exhibit\nMan",
                             "Co-exhibit\nWoman", "Co-exhibit\n%s" % preference_type.title()], ma="center", va="center")
plt.tight_layout()
plt.savefig("../otherFigures/03-gender_%s_lockin.pdf" % preference_type)
if preference_type == "neutral":
    np.savetxt('/Users/xindiwang/Dropbox (CCNR)/Success Team/Wang, Xindi/Gender Inequality in Art/For Alice/Main Paper New Figures/lockin_full.out', normalize, delimiter=',')


# equality lock-in
for prestige in ["Low", "Mid", "High"]:
    agg_legit_select = agg_legit[agg_legit["prestige_bin"] == prestige]
    count = Counter(
        tuple(zip(agg_legit_select["ins_gender_init"], agg_legit_select["ins_gender_late"])))
    print(count)
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
    print(confusion_matrix, np.sum(confusion_matrix, axis=0))
    normalize = confusion_matrix / \
        np.sum(confusion_matrix, axis=1).reshape(3, 1)
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(normalize, annot=True, fmt=".2%",
                     cmap="Blues", annot_kws={"size": 25}, cbar=False)
    plt.xlabel("Late Co-exhibition Gender", fontsize=25)
    plt.ylabel("Early Co-exhibition Gender", fontsize=25)
    print(ax.get_yticks())
    ax.set_xticklabels(["Co-exhibit\nMan",
                        "Co-exhibit\nWoman", "Co-exhibit\n%s" % preference_type.title()])
    plt.yticks([0.5, 1.5, 2.5], ["Co-exhibit\nMan",
                                 "Co-exhibit\nWoman", "Co-exhibit\n%s" % preference_type.title()], ma="center", va="center")
    plt.tight_layout()
    plt.savefig("../otherFigures/03-gender_%s_%s_lockin.pdf" %
                (preference_type, prestige))
    if preference_type == "neutral":
        np.savetxt('/Users/xindiwang/Dropbox (CCNR)/Success Team/Wang, Xindi/Gender Inequality in Art/For Alice/Main Paper New Figures/lockin_%s.out' %
                   prestige, normalize, delimiter=',')


ins_gender_dict = {1: "Co-exhibit Man",
                   2: "Co-exhibit Woman", 0: "Co-exhibit Neutral"}
transition_prestige_condition = agg_legit.groupby(["ins_gender_init", "ins_gender_late"])[
    "prestige_bin"].value_counts(normalize=True).reset_index(name="portion")
transition_prestige_condition["transition_type"] = ["%s-%s" % (ins_gender_dict[i], ins_gender_dict[j]) for (
    i, j) in zip(transition_prestige_condition["ins_gender_init"], transition_prestige_condition["ins_gender_late"])]

fig, ax = plt.subplots(3, 1, figsize=(8, 12))
for i in [1, 2, 0]:
    df_select = transition_prestige_condition[transition_prestige_condition["ins_gender_init"] == i]
    sns.barplot(x="ins_gender_late", y="portion", hue="prestige_bin",
                order=[1, 2, 0], hue_order=["Low", "Mid", "High"], data=df_select, palette="Blues_d", ax=ax[i])
    ax[i].set_xlabel("")
    ax[i].set_ylabel("Portion of Artists")
    ax[i].set_xticklabels(["Co-exhibit\nMan",
                           "Co-exhibit\nWoman", "Co-exhibit\nNeutral"])
    ax[i].get_legend().remove()
    ax[i].set_title("Early %s" % ins_gender_dict[i])

plt.tight_layout()
plt.savefig("../otherFigures/03-gender_%s_transition_prestige.pdf" %
            preference_type)
