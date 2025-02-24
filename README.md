# Codes to generate analysis and figures for _Quantifying Institutional Gender Inequality in Contemporary Visual Art_

## Main paper analysis and figures
To generate all results and figures of the main paper, run `./remove_birth.sh`. 

In this shell script, it contains the following steps:

### Data processing
`python 01-00-data_processing.py -t 0.6 -f`: create merged dataframes (shows_with_person_with_gender, career_start_end, sales_with_person), filter data and print out basic information.

### Select data
`python 01-01-data_selection.py -t 0.6 -f -y 1990 -a`: select data based on the career start year we select, create corresponding selected dataframe. Generate Figure 1.

### Assign institutional gender inequality
`python 03-00-gender_preference_assign.py -t 0.6 -f -y 1990`: assign institution to gender inequality category

### Analysis institutional gender inequality
`python 03-01-gender_preference_stats.py -t 0.6 -f -y 1990 -p neutral -a` and
`python 03-01-gender_preference_stats.py -t 0.6 -f -y 1990 -p balance -a`:
Basic statistics (count) of gender inequality category under gender-neutral and gender-balanced criteria. Generate Figure 2a, 2b 2c, Figure 3a, 3b.

For Figure 3a, 3b, further run
```markdown
cd plot_country_preference
python plot_map_color.py
```

### Artist co-ehixibion gender analysis
`python 04-artist_career_gender_preference.py -t 0.6 -f -y 1990 -p neutral -n 10 -a`: assign co-exhibition gender to artist, and further analysis. Generate Figure 5

### Sales data preparation
`python 05-00-sales_data_prep.py -t 0.6 -f -y 1990 -p neutral -n 10 -a`: prepare sales data and output auction bias. Generate Figure 6a.

### Logistic regression model
`python 05-03-logit.py -t 0.6 -f -y 1990 -p neutral -n 10 -a`: logistic regression model. Generate Table 2, Table 3 and Figure 6b, 6c.

### Remaining figures in main paper
#### Figure 2d and Figure 3c
`python 03-03-gender_preference_scatter_boundary.py 1990`
#### Figure 4
`python 06-network_viz.py 1990` generates the `gml` file of the network and we further process the visualization file.

## Supplementary Material analysis and figures
### Table 1
```shell
# process and get Men Artists/Women Artists/Ratio
python 01-00-data_processing.py --threshold 0.6 --remove_birth
python 01-00-data_processing.py --threshold 0.8 --remove_birth
python 01-00-data_processing.py --threshold 0.9 --remove_birth

# select, and get Selected Men Artists/Selected Women Artists/Ratio
python 01-01-data_selection.py -t 0.6 --remove_birth -y 1990
python 01-01-data_selection.py -t 0.8 --remove_birth -y 1990
python 01-01-data_selection.py -t 0.9 --remove_birth -y 1990
```
Results are printed in the command line.

### Table 2
For Men Exhibitions/Women Exhibitions/Exhibition Ratio
```
python 01-01-data_selection.py -t 0.6 --remove_birth -y 1990
python 01-01-data_selection.py -t 0.8 --remove_birth -y 1990
python 01-01-data_selection.py -t 0.9 --remove_birth -y 1990
```

For Man-overrepresented Institutions/Woman-overrepresented Institutions/Gender-neutral Institutions
```
python 03-00-gender_preference_assign.py -t 0.9 --remove_birth
```
Results are printed in the command line.

### Figure 1
#### Panel a, b, c
```
python generate_neighborhood_csv.py -f -p neutral
python generate_neighborhood_csv.py -f -p balance
```

#### Panel d
The codes for panel d is in `SI_codes/MultiscaleMixing/Art Gender Network Multiscale.ipynb`

### Figure 2
#### Panel a, b, c
```
cd SI_codes
python 02-01-time_trend.py -f
```

#### Panel d, e
```
cd SI_codes
python career_start_year_stability.py
```

### Figure 3
Run `python 04-artist_career_gender_preference.py -t 0.6 --no-remove_birth -y 1990 -p balance -n 10 --no-save`

### Figure 4: solo exhibition
Run all scripts under `SI_codes/solo_exhibitions`

### Table 3
`python 05-03-logit.py -t 0.6 --no-remove_birth -y 1990 -p neutral -n 10 --no-save`

### Figure 5
#### Panel a, b
`python 03-01-gender_preference_stats.py -t 0.6 -f -y 1990 -p neutral`

#### Panel d, e
`python 03-01-gender_preference_stats.py -t 0.6 -f -y 1990 -p balance`

#### Panel c, f
```
cd SI_codes
python 03-01-gender_preference_expert_grade.py -p neutral
python 03-01-gender_preference_expert_grade.py -p balance
```

### Figure 6
Panel a, b: `python 04-artist_career_gender_preference.py --remove_birth -p neutral -n 15`
Panel c, d: `python 04-artist_career_gender_preference.py --remove_birth -p neutral -n 20`

### Figure 7
Run `no_remove_birth.sh`

# License
Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg