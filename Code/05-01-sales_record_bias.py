from adjustText import adjust_text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plot_config import *
import argparse

parser = argparse.ArgumentParser(
    description='earliest year for data selection')
parser.add_argument('year', metavar='year', type=int,
                    help='select year')

args = parser.parse_args()
select_year = args.year
preference_type = "neutral"

artists_select = pd.read_csv(
    "../related_data_results/artists_select_%s.csv" % select_year)
shows_select = pd.read_csv(
    "../related_data_results/shows_select_%s.csv" % select_year)
sales_select = pd.read_csv(
    "../related_data_results/sales_select_%s.csv" % select_year)

agg_sales = pd.read_csv(
    "../related_data_results/artist_sale_info_agg_%s.csv" % preference_type)
agg = pd.read_csv(
    "../related_data_results/artist_exh_info_agg_%s.csv" % preference_type)

sales_select = pd.merge(sales_select, agg[
                        ["artist", "prestige_bin", "ins_gender", "exhibition_count"]], on="artist")
career_start_end_select = pd.read_csv(
    "../related_data_results/career_start_end_select_%s.csv" % select_year)
sales_select = pd.merge(sales_select, career_start_end_select[
                        ["artist", "career_start_year"]], left_on="artist", right_on="artist")


# artists with auction
count_df = sales_select[["artist", "gender_recog"]].drop_duplicates(
).groupby("gender_recog")["artist"].count()
count_df = count_df.to_dict()
print("Auction population gender ratio: Male/Female = %.2f" %
      (count_df["Male"] / count_df["Female"]))

fig, ax1 = plt.subplots(figsize=(4.5, 6))
graph = sns.countplot(x="gender_recog", data=sales_select[
    ["artist", "gender_recog"]].drop_duplicates())
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x() + p.get_width() / 2., height * 0.8,
               "{:,}".format(height), ha="center", va="top", color="white", fontsize=20)
plt.ticklabel_format(axis="y", style="sci", scilimits=(
    0, 0), useOffset=False, useMathText=True)

ax1.set_yticks([0,
                2 * 10**3, 4 * 10**3, 6 * 10**3, 8 * 10**3, 10 * 10**3])
ax1.set_yticklabels([0, r'$2 \times 10^3$', r'$4 \times 10^3$',
                     r'$6 \times 10^3$', r'$8 \times 10^3$', r'$10 \times 10^3$'])

plt.xlabel("")
plt.ylabel("Number of Auctioned Artists", fontsize=25)
ax1.tick_params(axis='x', labelsize=25)
plt.title("Auction Population\nImbalance", fontsize=25)
plt.tight_layout()
plt.savefig("../figures/04-number_auction_artist.pdf")

# auction records count
count_df = sales_select.groupby("gender_recog")["artist"].count()
# save for alice
count_df.reset_index().to_csv(
    "/Users/xindiwang/Dropbox (CCNR)/Success Team/Wang, Xindi/Gender Inequality in Art/For Alice/Main Paper Data/4A-2.csv", index=False)
count_df = count_df.to_dict()
print("Auction record gender ratio: Male/Female = %.2f" %
      (count_df["Male"] / count_df["Female"]))

fig, ax1 = plt.subplots(figsize=(4.5, 6))
graph = sns.countplot(x="gender_recog", data=sales_select)
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
ax1.set_yticks([0,
                2 * 10**4, 4 * 10**4, 6 * 10**4, 8 * 10**4, 10 * 10**4])
ax1.set_yticklabels([0, r'$2 \times 10^4$', r'$4 \times 10^4$',
                     r'$6 \times 10^4$', r'$8 \times 10^4$', r'$10 \times 10^4$'])
plt.tight_layout()
plt.savefig("../figures/04-number_auction.pdf")

# auction total sales
total_sales_df = sales_select.groupby(
    "gender_recog")["price relative"].sum().to_dict()
print("Auction Total Sales gender ratio: Male/Female =%.2f" %
      (total_sales_df["Male"]/total_sales_df["Female"]))

total_sales_df = sales_select.groupby(
    "gender_recog")["price relative"].sum().reset_index()
# save for alice
total_sales_df.reset_index().to_csv(
    "/Users/xindiwang/Dropbox (CCNR)/Success Team/Wang, Xindi/Gender Inequality in Art/For Alice/Main Paper Data/4A-3.csv", index=False)

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
ax1.set_yticks([0,
                2 * 10**4, 4 * 10**4, 6 * 10**4, 8 * 10**4, 10 * 10**4])
ax1.set_yticklabels([0, r'$2 \times 10^4$', r'$4 \times 10^4$',
                     r'$6 \times 10^4$', r'$8 \times 10^4$', r'$10 \times 10^4$'])
plt.xlabel("")
plt.ylabel("Total Auction Sales\n(Relative Price)", fontsize=25)
ax1.tick_params(axis='x', labelsize=25)
plt.title("Auction Sales\nImbalance", fontsize=25)
plt.tight_layout()
plt.savefig("../figures/04-total_sales_auction.pdf")


# pieces sold
auction_count_male = agg_sales[
    agg_sales["gender_recog"] == "Male"]["pieces_sold"]
auction_count_female = agg_sales[
    agg_sales["gender_recog"] == "Female"]["pieces_sold"]

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
plt.savefig('../figures/04-auction_count_bias_barplot.pdf')

record_list = []
raw_count_record_list = []
cmap = {1: "Male-Preferred Dominant",
        2: "Female-Preferred Dominant", 0: "Balanced Dominant"}
for i, prestige in enumerate(["Low", "Mid", "High"]):
    for j, ins_gender in enumerate([1, 2, 0]):
        this_prestige = agg_sales[(agg_sales["prestige_bin"] == prestige) &
                                  (agg_sales["ins_gender"] == ins_gender) &
                                  (agg_sales["exhibition_count"] >= 10)]
        male_data = this_prestige[this_prestige[
            "gender_recog"] == "Male"]["pieces_sold"]
        female_data = this_prestige[this_prestige[
            "gender_recog"] == "Female"]["pieces_sold"]
        record_list.append([prestige, cmap[ins_gender],
                            "Male", np.mean(male_data), len(male_data)])
        record_list.append([prestige, cmap[ins_gender],
                            "Female", np.mean(female_data), len(female_data)])

# piece price
auction_price_male = sales_select[
    sales_select["gender_recog"] == "Male"]["price/avg"]
auction_price_female = sales_select[
    sales_select["gender_recog"] == "Female"]["price/avg"]

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
plt.savefig('../figures/04-auction_price_bias_barplot.pdf')

# access rate
select_artists = agg
print(len(select_artists[select_artists["gender_recog"] == "Male"]), len(
    select_artists[select_artists["gender_recog"] == "Female"]))

select_artists_auction = agg_sales[
    agg_sales["artist"].isin(select_artists["artist"])]
male_transition_rate = 100 * len(select_artists_auction[select_artists_auction[
                                 "gender_recog"] == "Male"]) / len(select_artists[select_artists["gender_recog"] == "Male"])
female_transition_rate = 100 * len(select_artists_auction[select_artists_auction[
                                   "gender_recog"] == "Female"]) / len(select_artists[select_artists["gender_recog"] == "Female"])
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
plt.savefig("../figures/04-access_rate.pdf")
# record_list = []
# raw_count_record_list = []
# cmap = {1: "Male-Preferred Dominant",
#         2: "Female-Preferred Dominant", 0: "Balanced Dominant"}
# for i, prestige in enumerate(["Low", "Mid", "High"]):
#     for j, ins_gender in enumerate([1, 2, 0]):
#         this_prestige = sales_select[(
#             sales_select["prestige_bin"] == prestige) &
#             (sales_select["ins_gender"] == ins_gender) &
#             ((sales_select["exhibition_count"] >= 10))]
#         male_data = this_prestige[this_prestige[
#             "gender_recog"] == "Male"]["price/avg"]
#         female_data = this_prestige[this_prestige[
#             "gender_recog"] == "Female"]["price/avg"]
#         record_list.append([prestige, cmap[ins_gender],
#                             "Male", np.mean(male_data), len(male_data)])
#         record_list.append([prestige, cmap[ins_gender],
#                             "Female", np.mean(female_data), len(female_data)])
