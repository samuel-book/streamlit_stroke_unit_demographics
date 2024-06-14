import streamlit as st


def page_setup():
    # ----- Page setup -----
    # The following options set up the display in the tab in your browser. 
    # Set page to widescreen must be first call to st.
    st.set_page_config(
        page_title='Stroke unit demographics',
        page_icon=':rainbow:',
        layout='wide'
        )
    # n.b. this can be set separately for each separate page if you like.

# Names of columns that contain absolute values instead
# of proportions:
cols_abs = [
    'ethnic_group_other_than_white_british',
    'ethnic_group_all_categories_ethnic_group',
    'bad_or_very_bad_health',
    'all_categories_general_health',
    'long_term_health_count',
    'all_categories_long_term_health_problem_or_disability',
    'age_65_plus_count',
    'population_all',
    'rural_False',
    'rural_True',
    'over_65_within_30_False',
    'over_65_within_30_True',
    'closest_is_mt_False',
    'closest_is_mt_True'
]

# Make column names more pretty:
cols_prettier_dict = {
    'population_density': 'Population density',
    'income_domain_weighted_mean': 'Income deprivation domain',
    'imd_weighted_mean': 'IMD',
    'weighted_ivt_time': 'Time to IVT unit',
    'mt_time_weighted_mean': 'Time to MT unit',
    'ivt_time_weighted_mean': 'Time to IVT unit (??)',
    'mt_transfer_time_weighted_mean': 'Time to transfer unit',
    'ethnic_minority_proportion': 'Proportion ethnic minority',
    'bad_health_proportion': 'Proportion with bad health',
    'long_term_health_proportion': 'Proportion with long-term health issues',
    'population_all': 'Population',
    'age_65_plus_proportion': 'Proportion aged 65+',
    'proportion_rural': 'Proportion rural',
    'proportion_over_65_within_30': 'Proportion aged 65+ and within 30mins of stroke unit',
    'proportion_closest_is_mt': 'Proportion with MT as closest unit',
    'ivt_rate': 'IVT rate',
    'admissions_2122': 'Admissions (2021/22)'
}
cols_prettier_reverse_dict = dict(
    zip(list(cols_prettier_dict.values()), list(cols_prettier_dict.keys())))