import json
ins_preference_neutral = json.load(open("gender_neutral_details_ins_bf10.json"))
ins_preference_balance = json.load(open("gender_balance_details_ins_bf10.json"))

not_assigned_exh_neutral = [ins_preference_neutral[ins]["n"] for ins in ins_preference_neutral if ins_preference_neutral[ins]["label"]==-1]
all_exh_neutral = [ins_preference_neutral[ins]["n"] for ins in ins_preference_neutral]

not_assigned_exh_balance = [ins_preference_balance[ins]["n"] for ins in ins_preference_balance if ins_preference_balance[ins]["label"]==-1]
all_exh_balance = [ins_preference_balance[ins]["n"] for ins in ins_preference_balance]

print("neutral", len(not_assigned_exh_neutral), sum(not_assigned_exh_neutral)/(sum(all_exh_neutral)))
print("balance", len(not_assigned_exh_balance), sum(not_assigned_exh_balance)/(sum(all_exh_balance)))
