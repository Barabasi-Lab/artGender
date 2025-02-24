"""
This function calculates association rules by:
1) converting each artists' trajectories to transitions and combine them together (i.e., form a graph)
2) calculate support, confidence and lift for each transition (i.e., each edge)
"""
import argparse
import json
import os
from collections import Counter

import networkx as nx
import pandas as pd


def create_trajectories(shows):
    trajectory_df = shows.sort_values("show_year").groupby(
        ["artist"])["institution"].apply(list).reset_index(name='trajectory')
    trajectories = trajectory_df["trajectory"]
    return trajectories


def create_transitions(trajectories):
    transitions = []
    for trajectory in trajectories:
        for i in range(len(trajectory) - 1):
            for j in range(i + 1, len(trajectory)):
                transitions.append((trajectory[i], trajectory[j]))
        # transitions += list(temp_transitions)
    return transitions


east_asia = ["China", "Japan", "South Korea", "Taiwan", "Hong Kong", "Macau"]
east_eu = ["Bulgaria", "Czech Republic", "Hungary", "Poland",
           "Romania", "Russia", "Slovakia", "Belarus", "Moldova", "Ukraine"]
north_eu = ["Iceland", "Norway", "Denmark", "Finland", "Sweden"]
south_america = ["Brazil", "Argentina", "Chile", "Peru",
                 "Mexico", "Columbia", "Venezuela", "Bolivia", "Ecuador", "Uruguay", "El Salvador", "Costa Rica",
                 "Panama", "Guatemala"]
oceania = ["Australia", "New Zealand"]
germany = ["Germany"]
us = ["United States"]


def G_add_info(G):
    ins_country_dict = json.load(
        open("../raw_data/ins_country_mapping.json"))
    percentile_prestige_dict = json.load(
        open(os.path.join("../results/threshold_0.6_filter_True/data", "percentile_prestige.json")))
    ins_gender_neutral_dict = json.load(open(os.path.join(year_folder, "gender_neutral_ins_bf10.json")))
    ins_gender_balance_dict = json.load(open(os.path.join(year_folder, "gender_balance_ins_bf10.json")))
    percentile_prestige_dict = {
        int(ins): percentile_prestige_dict[ins] for ins in percentile_prestige_dict}
    ins_country_dict = {int(ins): ins_country_dict[ins] for ins in ins_country_dict}
    ins_gender_neutral_dict = {int(ins): ins_gender_neutral_dict[ins] for ins in ins_gender_neutral_dict}
    ins_gender_balance_dict = {int(ins): ins_gender_balance_dict[ins] for ins in ins_gender_balance_dict}

    in_east_asia_neutral, in_east_eu_neutral, in_north_eu_neutral, in_south_america_neutral, in_oceania_neutral, in_germany_neutral, in_us_neutral = {
    }, {}, {}, {}, {}, {}, {}
    for ins in ins_gender_neutral_dict:
        country = ins_country_dict[ins]
        gender_preference = ins_gender_neutral_dict[ins]
        for (compare_list, record_dict) in zip([east_asia, east_eu, north_eu, south_america, oceania, germany, us],
                                               [in_east_asia_neutral, in_east_eu_neutral, in_north_eu_neutral,
                                                in_south_america_neutral, in_oceania_neutral, in_germany_neutral,
                                                in_us_neutral]):
            if country in compare_list:
                record_dict[ins] = gender_preference
            else:
                record_dict[ins] = 3

    in_east_asia_balance, in_east_eu_balance, in_north_eu_balance, in_south_america_balance, in_oceania_balance, in_germany_balance, in_us_balance = {
    }, {}, {}, {}, {}, {}, {}
    for ins in ins_gender_balance_dict:
        country = ins_country_dict[ins]
        gender_preference = ins_gender_balance_dict[ins]
        for (compare_list, record_dict) in zip([east_asia, east_eu, north_eu, south_america, oceania, germany, us],
                                               [in_east_asia_balance, in_east_eu_balance, in_north_eu_balance,
                                                in_south_america_balance, in_oceania_balance, in_germany_balance,
                                                in_us_balance]):
            if country in compare_list:
                record_dict[ins] = gender_preference
            else:
                record_dict[ins] = 3

    edge_color = {}
    for (i, j) in G.edges():
        source_type = ins_gender_neutral_dict[i]
        target_type = ins_gender_neutral_dict[j]
        if source_type == target_type:
            edge_color[(i, j)] = source_type
        else:
            edge_color[(i, j)] = 3

    nx.set_node_attributes(G, name="country", values=ins_country_dict)
    nx.set_node_attributes(G, name="percentilePrestige",
                           values=percentile_prestige_dict)
    nx.set_node_attributes(G, name="genderNeutral",
                           values=ins_gender_neutral_dict)
    nx.set_node_attributes(G, name="genderBalance",
                           values=ins_gender_balance_dict)
    nx.set_node_attributes(G, name="inUsNeutral", values=in_us_neutral)
    nx.set_node_attributes(G, name="inGermanyNeutral",
                           values=in_germany_neutral)
    nx.set_node_attributes(G, name="inOceaniaNeutral",
                           values=in_oceania_neutral)
    nx.set_node_attributes(G, name="inSouthAmericaNeutral",
                           values=in_south_america_neutral)
    nx.set_node_attributes(G, name="inNorthEuNeutral",
                           values=in_north_eu_neutral)
    nx.set_node_attributes(G, name="inEastEuNeutral",
                           values=in_east_eu_neutral)
    nx.set_node_attributes(G, name="inEastAsiaNeutral",
                           values=in_east_asia_neutral)

    nx.set_node_attributes(G, name="inUsBalance", values=in_us_balance)
    nx.set_node_attributes(G, name="inGermanyBalance",
                           values=in_germany_balance)
    nx.set_node_attributes(G, name="inOceaniaBalance",
                           values=in_oceania_balance)
    nx.set_node_attributes(G, name="inSouthAmericaBalance",
                           values=in_south_america_balance)
    nx.set_node_attributes(G, name="inNorthEuBalance",
                           values=in_north_eu_balance)
    nx.set_node_attributes(G, name="inEastEuBalance",
                           values=in_east_eu_balance)
    nx.set_node_attributes(G, name="inEastAsiaBalance",
                           values=in_east_asia_balance)

    nx.set_edge_attributes(G, name="color", values=edge_color)
    return G


def cal_asso_measures(transitions, selected_nodes):
    edge_weight = Counter(transitions)
    G = nx.DiGraph()
    G.add_edges_from(transitions)
    G.remove_edges_from(nx.selfloop_edges(G))
    G = G.subgraph(selected_nodes)
    nx.set_edge_attributes(G, name="weight", values=edge_weight)
    edge_weight = nx.get_edge_attributes(G, "weight")
    print(len(edge_weight), sum(edge_weight.values()))
    out_degree = dict(G.out_degree(weight="weight"))
    in_degree = dict(G.in_degree(weight="weight"))
    edge_weight_sum = sum(list(edge_weight.values()))
    # cal support
    support = {edge: edge_weight[edge] /
                     edge_weight_sum for edge in edge_weight}
    # cal confidence
    confidence = {edge: edge_weight[edge] /
                        out_degree[edge[0]] for edge in edge_weight}
    # cal contribution
    contribution = {edge: edge_weight[edge] /
                          in_degree[edge[1]] for edge in edge_weight}
    # cal lift
    lift = {edge: (edge_weight[edge] / out_degree[edge[0]]) /
                  (in_degree[edge[1]] / edge_weight_sum) for edge in edge_weight}
    nx.set_edge_attributes(G, name="support", values=support)
    nx.set_edge_attributes(G, name="confidence", values=confidence)
    nx.set_edge_attributes(G, name="contribution", values=contribution)
    nx.set_edge_attributes(G, name="lift", values=lift)
    edge_weight = nx.get_edge_attributes(G, name="weight")
    return support, confidence, contribution, lift, edge_weight, G


def write_rules_and_graph(G, fname, support_threhold=2.6 * 10 ** (-7), confidence_threshold=0.005,
                          contribution_threshold=0.005, lift_threshold=1):
    support = nx.get_edge_attributes(G, "support")
    confidence = nx.get_edge_attributes(G, "confidence")
    contribution = nx.get_edge_attributes(G, "contribution")
    lift = nx.get_edge_attributes(G, "lift")
    edge_weight = nx.get_edge_attributes(G, "weight")
    G = nx.DiGraph()
    edges_keep = []
    for edge in support:
        if ((support[edge] > support_threhold) and
                (confidence[edge] > confidence_threshold) and
                (contribution[edge] > contribution_threshold) and
                (lift[edge] > lift_threshold)):
            edges_keep.append(edge)

    G.add_edges_from(edges_keep)
    nx.set_edge_attributes(G, name="support", values=support)
    nx.set_edge_attributes(G, name="confidence", values=confidence)
    nx.set_edge_attributes(G, name="contribution", values=contribution)
    nx.set_edge_attributes(G, name="lift", values=lift)
    nx.set_edge_attributes(G, name="weight", values=edge_weight)

    G = G_add_info(G)
    G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
    print(len(G), G.number_of_edges())
    nx.write_gml(G, os.path.join(year_folder, "%s.gml" % fname))


def get_rules(shows, fname, selected_nodes, support_threhold=10 ** (-7), confidence_threshold=0.005,
              contribution_threshold=0, lift_threshold=1):
    trajectories = create_trajectories(shows)
    transitions = create_transitions(trajectories)
    support, confidence, contribution, lift, edge_weight, G = cal_asso_measures(
        transitions, selected_nodes)
    write_rules_and_graph(G, fname,
                          support_threhold, confidence_threshold, contribution_threshold, lift_threshold)
    return support, confidence, contribution, lift


parser = argparse.ArgumentParser(
    description='earliest year for data selection')
parser.add_argument('--year', metavar='year', type=int,
                    help='select year', default=1990)

args = parser.parse_args()
select_year = args.year

year_folder = "../results/threshold_0.6_filter_True/data/year_%s/" % select_year

shows_select = pd.read_csv(os.path.join(year_folder, "shows_select.csv"))
gallery_dict = json.load(open("../raw_data/gallery_dict.json"))
shows_select["ins_type"] = [gallery_dict[str(ins)]["type"] for ins in shows_select["institution"]]

ins_gender_neutral_dict = json.load(open(os.path.join(year_folder, "gender_neutral_ins_bf10.json")))
ins_gender_balance_dict = json.load(open(os.path.join(year_folder, "gender_balance_ins_bf10.json")))
selected_nodes = list(map(int, list(ins_gender_neutral_dict.keys())))

support, confidence, contribution, lift = get_rules(
    shows_select, fname="full_asso_default_filter_bf10", selected_nodes=selected_nodes)

support, confidence, contribution, lift = get_rules(
    shows_select, fname="full_asso_bf10", selected_nodes=selected_nodes, support_threhold=0, confidence_threshold=0,
    contribution_threshold=0, lift_threshold=0)

# support, confidence, contribution, lift = get_rules(
#     shows_select[shows_select["type"] == "gallery"], fname="gallery", selected_nodes=selected_nodes, support_threhold=0,
#     confidence_threshold=0, contribution_threshold=0, lift_threshold=0)
#
# support, confidence, contribution, lift = get_rules(
#     shows_select[shows_select["type"] == "museum"], fname="museum", selected_nodes=selected_nodes, support_threhold=0,
#     confidence_threshold=0, contribution_threshold=0, lift_threshold=0)
