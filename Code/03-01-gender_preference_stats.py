import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import pandas as pd
from plot_config import *
import matplotlib as mpl
from scipy import stats
mpl.rcParams['hatch.linewidth'] = 2

parser = argparse.ArgumentParser(
    description='earliest year for data selection')
parser.add_argument('year', metavar='year', type=int,
                    help='select year')
parser.add_argument('type', metavar='type', type=str,
                    help='balance or neutral')

darker_colors = ["#ADC4EA", "#FFA7B8", "#9CE3D1"]  # balance
lighter_colors = ["#cddbf2", "#ffd3db", "#cdf1e8"]  # neutral

args = parser.parse_args()
select_year = args.year
preference_type = args.type
if preference_type == "neutral":
    sns.set_palette(sns.color_palette(lighter_colors))
else:
    sns.set_palette(sns.color_palette(darker_colors))

artists_select = pd.read_csv(
    "../related_data_results/artists_select_%s.csv" % select_year)
shows_select = pd.read_csv(
    "../related_data_results/shows_select_%s.csv" % select_year)
percentile_prestige = json.load(
    open("../related_data_results/percentile_prestige.json"))
gender_preference_dict = json.load(open(
    "../related_data_results/gender_%s_ins.json" % preference_type))
gallery_dict = json.load(open("../related_data_results/gallery_dict.json"))

# count
count = Counter(gender_preference_dict.values())
df = pd.DataFrame({"Equality": ["Male-Preferred", "Female-Preferred",
                                "Balanced"], "Count": [count[1], count[2], count[0]]})
# plt.figure(figsize=(8, 6))
# graph = sns.barplot(x="Equality", y="Count", data=df)
# for p in graph.patches:
#     height = p.get_height()
#     graph.text(p.get_x() + p.get_width() / 2., height - 100,
#                "{:,}".format(int(height)), ha="center", va="top", color="white", fontsize=20)
# plt.xlabel("")
# plt.ylabel("Number of Institutions", fontsize=25)
# plt.gca().tick_params(axis='x', labelsize=18)
# plt.tight_layout()
# plt.savefig("../figures/02-gender_%s_count.pdf"%preference_type)

# piechart for the above plot
labels = list(df["Equality"])
sizes = list(df["Count"])
# save for alice
if preference_type == "balance":
    alice_label = "1"
else:
    alice_label = "4"
df.to_csv("/Users/xindiwang/Dropbox (CCNR)/Success Team/Wang, Xindi/Gender Inequality in Art/For Alice/Main Paper Data/2C-%s.csv" %
          alice_label, index=False)


def absolute_value(val):
    a = int(np.round(val / 100. * sum(sizes), 0))
    return a


fig1, ax1 = plt.subplots(figsize=(3, 3))
piechart = ax1.pie(
    sizes, autopct=absolute_value, textprops=dict(color="black", fontsize=20))
if preference_type == "balance":
    for i in range(len(piechart[0])):
        piechart[0][i].set_hatch(2 * "-")
        piechart[0][i].set_color("none")
        piechart[0][i].set_edgecolor(sns.color_palette()[i])
# ax1.legend(wedges, labels,
#            title="",
#            loc="upper center",
#            ncol=3,
#            fontsize=12,
#            )
# plt.title("All")
plt.tight_layout()
plt.savefig("../figures/02-gender_%s_count_pie.pdf" % preference_type)
plt.close()

# gender_preference bin count for museum and gallery separately
df = pd.DataFrame()
df["gallery_id"] = list(gender_preference_dict.keys())
df["gender_preference"] = list(gender_preference_dict.values())
df["type"] = [gallery_dict[ins]["type"].title() for ins in df["gallery_id"]]
# plt.clf()
# plt.figure(figsize=(8, 6))
# graph = sns.countplot(x="type", hue="gender_preference",
#                       data=df, hue_order=[1, 2, 0])
# for p in graph.patches:
#     height = p.get_height()
#     graph.text(p.get_x() + p.get_width() / 2., height - 100,
#                "{:,}".format(int(height)), ha="center", va="top", color="white", fontsize=20)
# plt.xlabel("")
# plt.ylabel("Number of Institutions", fontsize=25)
# plt.gca().set_xticklabels(["Museums", "Galleries"])
# plt.gca().tick_params(axis='x', labelsize=25)
# graph.legend_.remove()
# plt.tight_layout()
# plt.savefig("../figures/02-gender_%s_count_museum_vs_gallery.pdf"%preference_type)

# two pie charts for the above plot
# museum
museum_df = df[df["type"] == "Museum"]
museum_count = Counter(museum_df["gender_preference"])
sizes = museum_count[1], museum_count[2], museum_count[0]

df_museum = pd.DataFrame({"Equality": ["Male-Preferred", "Female-Preferred",
                                       "Balanced"], "Count": sizes})
# save for alice
if preference_type == "balance":
    alice_label = "2"
else:
    alice_label = "5"
df_museum.to_csv("/Users/xindiwang/Dropbox (CCNR)/Success Team/Wang, Xindi/Gender Inequality in Art/For Alice/Main Paper Data/2C-%s.csv" %
                 alice_label, index=False)


fig1, ax1 = plt.subplots(figsize=(3, 3))
piechart = ax1.pie(
    sizes, autopct=absolute_value, textprops=dict(color="black", fontsize=20))
if preference_type == "balance":
    for i in range(len(piechart[0])):
        piechart[0][i].set_hatch(2 * "-")
        piechart[0][i].set_color("none")
        piechart[0][i].set_edgecolor(sns.color_palette()[i])
# plt.title("Museums", fontsize=20)
plt.tight_layout()
plt.savefig("../figures/02-gender_%s_count_museum_pie.pdf" % preference_type)
plt.close()

galley_df = df[df["type"] == "Gallery"]
gallery_count = Counter(galley_df["gender_preference"])
sizes = gallery_count[1], gallery_count[2], gallery_count[0]
df_gallery = pd.DataFrame({"Equality": ["Male-Preferred", "Female-Preferred",
                                        "Balanced"], "Count": sizes})
# save for alice
if preference_type == "balance":
    alice_label = "3"
else:
    alice_label = "6"
df_gallery.to_csv(
    "/Users/xindiwang/Dropbox (CCNR)/Success Team/Wang, Xindi/Gender Inequality in Art/For Alice/Main Paper Data/2C-%s.csv" % alice_label, index=False)

fig1, ax1 = plt.subplots(figsize=(3, 3))
piechart = ax1.pie(
    sizes, autopct=absolute_value, textprops=dict(color="black", fontsize=20))
if preference_type == "balance":
    for i in range(len(piechart[0])):
        piechart[0][i].set_hatch(2 * "-")
        piechart[0][i].set_color("none")
        piechart[0][i].set_edgecolor(sns.color_palette()[i])
# plt.title("Galleries", fontsize=20)
plt.tight_layout()
plt.savefig("../figures/02-gender_%s_count_gallery_pie.pdf" % preference_type)
plt.close()

# portion of male-preferred, female-preferred and balanced institutions at
# different prestige
prestige_score_list = []
gender_preference_list = []
for item in gender_preference_dict:
    prestige_score_list.append(percentile_prestige[str(item)])
    gender_preference_list.append(gender_preference_dict[item])

df = pd.DataFrame()
df["gender_preference"] = gender_preference_list
df["prestige"] = prestige_score_list

pmap = {1: "Low", 2: "Mid", 3: "High"}
df["prestige_bin"] = [pmap[np.digitize(
    round(item, 1), [0, np.percentile(df[df["gender_preference"] != -1]["prestige"], 40), np.percentile(df[df["gender_preference"] != -1]["prestige"], 70), 1.1])] for item in df["prestige"]]

plt.clf()
colors = ["#6E99DD", "#FF5072", "#3ac8a4", "#E66100", "#5D3A9B", "#000000"]
sns.set_palette(sns.color_palette(colors))
plt.figure(figsize=(4, 6))
prestige_list = []
male_portion_list = []
female_portion_list = []
balanced_portion_list = []
for prestige in ["Low", "Mid", "High"]:
    this_prestige = df[(df["prestige_bin"] == prestige)]
    male_count = Counter(this_prestige["gender_preference"])[1]
    female_count = Counter(this_prestige["gender_preference"])[2]
    balance_count = Counter(this_prestige["gender_preference"])[0]
    total_count = len(this_prestige)
    male_portion_list.append(male_count / total_count)
    female_portion_list.append(female_count / total_count)
    balanced_portion_list.append(balance_count / total_count)
    prestige_list.append(prestige)

print(prestige_list, male_portion_list,
      female_portion_list, balanced_portion_list)
plt.plot(prestige_list, male_portion_list, "o-",
         color=sns.color_palette()[0], markersize=15)
plt.plot(prestige_list, female_portion_list,
         "o-", color=sns.color_palette()[1], markersize=15)
plt.plot(prestige_list, balanced_portion_list,
         "o-", color=sns.color_palette()[2], markersize=15)
plt.xlabel("Institution Prestige", fontsize=25)
plt.ylabel("Fraction of Gender-Preferrenced Institutions")
plt.xticks(["Low", "Mid", "High"])
male_baseline = Counter(gender_preference_dict.values())[
    1] / len([i for i in gender_preference_dict if gender_preference_dict[i] != -1])
female_baseline = Counter(gender_preference_dict.values())[
    2] / len([i for i in gender_preference_dict if gender_preference_dict[i] != -1])
balanced_baseline = Counter(gender_preference_dict.values())[
    0] / len([i for i in gender_preference_dict if gender_preference_dict[i] != -1])
plt.axhline(male_baseline, color=sns.color_palette()[0], ls="--", linewidth=1)
plt.axhline(female_baseline, color=sns.color_palette()
            [1], ls="--", linewidth=1)
plt.axhline(balanced_baseline, color=sns.color_palette()
            [2], ls="--", linewidth=1)
plt.ylim(0, )
plt.tight_layout()
plt.savefig("../figures/02-%s_portion_vs_prestige_agg.pdf" % preference_type)

df = pd.DataFrame()
df["prestige_list"] = prestige_list
df["Man-Preferred Portion"] = male_portion_list
df["Woman-Preferred Portion"] = female_portion_list
df["%s Portion" % preference_type.title()] = balanced_portion_list
df.to_csv("/Users/xindiwang/Dropbox (CCNR)/Success Team/Wang, Xindi/Gender Inequality in Art/For Alice/Main Paper New Figures/%s_portion_prestige.csv" %
          preference_type, index=False)

baseline = {"male": male_baseline, "female": female_baseline, preference_type: balanced_baseline}
json.dump(baseline, open("/Users/xindiwang/Dropbox (CCNR)/Success Team/Wang, Xindi/Gender Inequality in Art/For Alice/Main Paper New Figures/%s_portion_prestige_baseline.json" %
          preference_type, "w"))
