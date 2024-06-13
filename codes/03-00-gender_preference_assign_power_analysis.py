import json
import numpy as np
import argparse
import os


def get_power_statistics(gender_preference_dict):
    bf_list = []
    binom_pvalue_list, power_list = [], []
    for ins in gender_preference_dict:
        label = gender_preference_dict[ins]["label"]
        if label == 0:
            bayes_factor = gender_preference_dict[ins]["bayes_factor"]
            bf_list.append(bayes_factor)
        else:
            binom_pvalue = gender_preference_dict[ins]["binom_pvalue"]
            power = gender_preference_dict[ins]["power"]
            binom_pvalue_list.append(binom_pvalue)
            power_list.append(power)

    return bf_list, binom_pvalue_list, power_list


def power_analysis_results(gender_preference_dict):
    bf_list, binom_pvalue_list, power_list = get_power_statistics(gender_preference_dict)

    bf_array = np.array(bf_list)
    bf_smaller_1 = bf_array[bf_array < 1]
    print("Bayes Factor:", len(bf_smaller_1), len(bf_array), len(bf_smaller_1) / len(bf_array))

    binom_pvalue_array = np.array(binom_pvalue_list)
    binom_pvalue_smaller01 = binom_pvalue_array[binom_pvalue_array < 0.1]
    print("Binom pvalue:", len(binom_pvalue_smaller01), len(binom_pvalue_array), len(binom_pvalue_smaller01) / len(binom_pvalue_array))

    power_array = np.array(power_list)
    power_higher40 = power_array[power_array > 0.4]
    power_higher50 = power_array[power_array > 0.5]
    power_higher60 = power_array[power_array > 0.6]
    power_higher70 = power_array[power_array > 0.7]
    power_higher80 = power_array[power_array > 0.8]
    power_higher90 = power_array[power_array > 0.9]
    print("Power Analysis")
    print("Power > 0.4:", len(power_higher40), len(power_array), len(power_higher40) / len(power_array))
    print("Power > 0.5:", len(power_higher50), len(power_array), len(power_higher50) / len(power_array))
    print("Power > 0.6:", len(power_higher60), len(power_array), len(power_higher60) / len(power_array))
    print("Power > 0.7:", len(power_higher70), len(power_array), len(power_higher70) / len(power_array))
    print("Power > 0.8:", len(power_higher80), len(power_array), len(power_higher80) / len(power_array))
    print("Power > 0.9:", len(power_higher90), len(power_array), len(power_higher90) / len(power_array))


def main():
    parser = argparse.ArgumentParser(
        description='select artists with career start year > [year_threshold]')
    parser.add_argument('-t', '--genderizeio_threshold', type=float, help='genderize.io threshold', default=0.6)
    parser.add_argument('-f', '--remove_birth', action=argparse.BooleanOptionalAction)
    parser.add_argument('-y', '--career_start_threshold', type=int,
                        help='earliest career start year of selected artists', default=1990)
    parser.add_argument('-p', '--preference_type', type=str, help='neutral or balance')
    args = parser.parse_args()

    gender_preference_dict_path = os.path.join("..", "results",
                                               f"threshold_{args.genderizeio_threshold}_filter_{args.remove_birth}",
                                               "data",
                                               f"year_{args.career_start_threshold}",
                                               f"gender_{args.preference_type}_details_ins.json")

    gender_preference_dict = json.load(open(gender_preference_dict_path))

    print(f"Gender {args.preference_type}, Remove birth {args.remove_birth}")
    power_analysis_results(gender_preference_dict)

if __name__ == '__main__':
    main()
