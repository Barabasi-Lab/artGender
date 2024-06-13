python 01-00-data_processing.py -t 0.6 --no-remove_birth
python 01-01-data_selection.py -t 0.6 --no-remove_birth -y 1990 --no-save
python 03-00-gender_preference_assign.py -t 0.6 --no-remove_birth -y 1990
python 03-01-gender_preference_stats.py -t 0.6 --no-remove_birth -y 1990 -p neutral --no-save
python 03-01-gender_preference_stats.py -t 0.6 --no-remove_birth -y 1990 -p balance --no-save
python 04-artist_career_gender_preference.py -t 0.6 --no-remove_birth -y 1990 -p neutral -n 10 --no-save
python 04-artist_career_gender_preference.py -t 0.6 --no-remove_birth -y 1990 -p balance -n 10 --no-save
python 05-00-sales_data_prep.py -t 0.6 --no-remove_birth -y 1990 -p neutral -n 10 --no-save
python 05-03-logit.py -t 0.6 --no-remove_birth -y 1990 -p neutral -n 10 --no-save