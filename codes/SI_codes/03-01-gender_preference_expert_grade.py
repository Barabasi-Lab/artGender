import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import json
import argparse
import os
from plot_config import *

parser = argparse.ArgumentParser(
    description='preference_type')
parser.add_argument('-p', '--preference_type', type=str, help='neutral or balance')

args = parser.parse_args()
preference_type = args.preference_type

df = pd.read_csv(f"../../results/threshold_0.6_filter_True/data/year_1990/ins_information_{preference_type}.csv")
gender_preference_dict = json.load(open(
    f"../../results/threshold_0.6_filter_True/data/year_1990/gender_{preference_type}_ins.json"))

prestige_list = []
male_portion_list = []
female_portion_list = []
balanced_portion_list = []
for grade in ["D", "C", "B", "A"]:
    this_grade = df[(df["grade"] == grade)]
    male_count = Counter(this_grade["gender_preference"])[1]
    female_count = Counter(this_grade["gender_preference"])[2]
    balance_count = Counter(this_grade["gender_preference"])[0]
    total_count = len(this_grade)
    male_portion_list.append(male_count / total_count)
    female_portion_list.append(female_count / total_count)
    balanced_portion_list.append(balance_count / total_count)
    prestige_list.append(grade)

plt.figure(figsize=(4, 6))
print(prestige_list, male_portion_list,
      female_portion_list, balanced_portion_list)
plt.plot(prestige_list, male_portion_list, "o-",
         color=sns.color_palette()[0], markersize=15)
plt.plot(prestige_list, female_portion_list,
         "o-", color=sns.color_palette()[1], markersize=15)
plt.plot(prestige_list, balanced_portion_list,
         "o-", color=sns.color_palette()[2], markersize=15)
plt.xlabel("Institution Grade", fontsize=25)
plt.ylabel("Fraction of Gender-Preferrenced Institutions")
plt.xticks(["D", "C", "B", "A"])
male_baseline = Counter(gender_preference_dict.values())[1] / len(
    [i for i in gender_preference_dict if gender_preference_dict[i] != -1])
female_baseline = Counter(gender_preference_dict.values())[2] / len(
    [i for i in gender_preference_dict if gender_preference_dict[i] != -1])
balanced_baseline = Counter(gender_preference_dict.values())[0] / len(
    [i for i in gender_preference_dict if gender_preference_dict[i] != -1])
plt.axhline(male_baseline, color=sns.color_palette()[0], ls="--", linewidth=1)
plt.axhline(female_baseline, color=sns.color_palette()[1], ls="--", linewidth=1)
plt.axhline(balanced_baseline, color=sns.color_palette()[2], ls="--", linewidth=1)
plt.ylim(0, )
plt.tight_layout()
os.makedirs("../../results/SI/fig5/", exist_ok=True)
plt.savefig(f"../../results/SI/fig5/expert_grade_{preference_type}.pdf")
