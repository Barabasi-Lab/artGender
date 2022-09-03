# Quantifying Systemic Gender Inequality in Visual Art
## Data description
Our dataset can be obtained from the raw dataset deposited to [Harvard dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PGICDM). Below shows anonymized dataset that can be used to generate figures in this project.

1. `df_gender_portion_institution.csv`: information about institutions and their institutional gender. Related to Figure 2, 3, 4.
2. `artist_exh_info_agg_neutral.csv` and `artist_exh_info_agg_balance.csv`: information about artists exhibition and their co-exhibition gender under both gender neutral and gender balance criteria. Related to Figure 5.
3. `artist_sale_info_agg_solo_neutral.csv`: information about artists auctions and co-exhibition gender under gender neutral criteria. Related to Figure 6.
4. `logit_data.csv`: used to construct logit model predicting access to auction. Related to Figure 6.

## Code description
1. `03-01-gender_preference_stats.py` Generate institutional gener plot (Figure 2)
2. `03-03-gender_preference_scatter_boundary.py`: Generate institutional gender scatter plot (Figure 3)
3. `04-artist_career_gender_preference.py`: Artist co-exhibition gender analysis (Figure 5)
4. `05-01-sales_record_bias.py`: Auction biases plot (Figure 6)
5. `05-03-logit_statsmodel.py`: Logit model (Figure 6)
