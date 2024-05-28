"""
Streamlit demo of two England maps with LSOA-level colour bands.

Because a long app quickly gets out of hand,
try to keep this document to mostly direct calls to streamlit to write
or display stuff. Use functions in other files to create and
organise the stuff to be shown. In this example, most of the work is
done in functions stored in files named container_(something).py
"""
# ----- Imports -----
import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from utilities.maps import convert_shapely_polys_into_xy

from stroke_maps.utils import find_multiindex_column_names
from stroke_maps.geo import _load_geometry_stroke_units, check_scenario_level
from stroke_maps.catchment import Catchment

# Custom functions:
import utilities.maps as maps
import utilities.plot_maps as plot_maps
import utilities.container_inputs as inputs


# ###########################
# ##### START OF SCRIPT #####
# ###########################
# page_setup()
st.set_page_config(
    page_title='Stroke unit demographics',
    page_icon=':rainbow:',
    layout='wide'
    )

container_maps = st.empty()


# Import the full travel time matrix:
df_demog = pd.read_csv(
    './data/collated_data_amb.csv', index_col='LSOA')
# Rename index to 'lsoa':
df_demog.index.name = 'lsoa'


# TO DO - split off numerical data and region lookup data.
# Make separate region files and allow option to plot them as outlines.

# TO DO - make separate legend type for categorical data (e.g. rural-urban type)

# TO DO - put Wales back into the LSOA and MSOA shapefiles.


# #################################
# ########## USER INPUTS ##########
# #################################

cols_selectable = list(df_demog.columns)
cols_to_remove = [
    'income_domain_rank',
    'idaci_rank',
    'idaopi_rank',
    'closest_ivt_unit',
    'closest_mt_unit',
    'closest_mt_transfer',
    'la_district_name_2019',
    'rural_urban_2011',
    'ambulance_service',
    'local_authority_district_22',
    'LAD22NM',
    'country',
    'ethnic_group_all_categories_ethnic_group',
    'all_categories_general_health'
]
for c in cols_to_remove:
    cols_selectable.remove(c)

cols = st.columns(2)
with cols[0]:
    col1 = st.selectbox('Column 1', options=cols_selectable, index=1)
with cols[1]:
    col2 = st.selectbox('Column 2', options=cols_selectable, index=2)


# User inputs for which hospitals to pick:
catchment = Catchment()
df_units = catchment.get_unit_services()


# Colourmap selection
cmap_names = [
    'cosmic_r', 'viridis_r', 'inferno_r', 'neutral_r'
    ]
cmap_displays = [
    inputs.make_colourbar_display_string(cmap_name, char_line='█', n_lines=15)
    for cmap_name in cmap_names
    ]

# v_min = np.min(df_data[col1])
# v_max = np.max(df_data[col1])
# step_size = (v_max - v_min) * (1.0/5.0)

# v_min_diff = np.min(df_data[col2])
# v_max_diff = np.max(df_data[col2])
# step_size_diff = (v_max_diff - v_min_diff) * (1.0/5.0)

col1_colour_scale_dict = inputs.lookup_colour_scale(col1)
col2_colour_scale_dict = inputs.lookup_colour_scale(col2)

# Inputs for overwriting default colourbars:
with st.expander('Colour setup'):
    cols = st.columns(2)
    with cols[0]:
        container_map1_cbar_setup = st.container()
    with cols[1]:
        container_map2_cbar_setup = st.container()

with container_map1_cbar_setup:
    with st.form('Left map'):
        cols = st.columns(3)
        with cols[0]:
            vmin_map1 = st.number_input('Minimum',
                                        value=col1_colour_scale_dict['vmin'])
        with cols[2]:
            vmax_map1 = st.number_input('Maximum',
                                        value=col1_colour_scale_dict['vmax'])
        with cols[1]:
            step_size_map1 = st.number_input(
                'Step size', value=col1_colour_scale_dict['step_size'])

        cmap_name_map1 = st.radio(
            'Colour display for map',
            cmap_names,
            captions=cmap_displays,
            key='cmap_name_map1'
        )
        flip_cmap1 = st.checkbox('Reverse colour scale', key='flip_map1')
        if flip_cmap1:
            cmap_name_map1 += '_r'
        if cmap_name_map1.endswith('_r_r'):
            # Remove the double reverse reverse.
            cmap_name_map1 = cmap_name_map1[:-4]

        submit_left = st.form_submit_button('Redraw left map')

with container_map2_cbar_setup:
    with st.form('Right map'):
        cols = st.columns(3)
        with cols[0]:
            vmin_map2 = st.number_input('Minimum',
                                        value=col2_colour_scale_dict['vmin'])
        with cols[2]:
            vmax_map2 = st.number_input('Maximum',
                                        value=col2_colour_scale_dict['vmax'])
        with cols[1]:
            step_size_map2 = st.number_input(
                'Step size', value=col2_colour_scale_dict['step_size'])

        cmap_name_map2 = st.radio(
            'Colour display for map',
            cmap_names,
            captions=cmap_displays,
            key='cmap_name_map2'
        )
        flip_cmap2 = st.checkbox('Reverse colour scale', key='flip_map2')
        if flip_cmap2:
            cmap_name_map2 += '_r'
        if cmap_name_map2.endswith('_r_r'):
            # Remove the double reverse reverse.
            cmap_name_map2 = cmap_name_map2[:-4]

        submit_right = st.form_submit_button('Redraw right map')

# Display names:
subplot_titles = [col1, col2]
cmap_titles = subplot_titles


# #######################################
# ########## MAIN CALCULATIONS ##########
# #######################################
# While the main calculations are happening, display a blank map.
# Later, when the calculations are finished, replace with the actual map.
with container_maps:
    plot_maps.plotly_blank_maps(['', ''], n_blank=2)


# # Remove some data that doesn't look useful here:
# cols_to_remove = [
#     'income_domain_rank',
#     'idaci_rank',
#     'idaopi_rank'
# ]
# df_demog = df_demog.drop(cols_to_remove, axis='columns')


# DO THIS JUST ONCE IN ADVANCE AND SAVE RESULTS? TO DO -----------------------------------------------

cols_to_scale = [c for c in df_demog.columns if (
    (
        # (c.startswith('ethnic_group')) |
        # (c.endswith('health')) |
        # (c.startswith('day_to_day_activities')) |
        (c.startswith('age_band_all'))
    )
    )]
# # Stuff that's hard to pick out with conditions:
# cols_to_scale += [
#     'all_categories_long_term_health_problem_or_disability',
#     ]
df_demog[cols_to_scale] = df_demog[cols_to_scale].astype(float)
df_demog.loc[:, cols_to_scale] = (
    df_demog.loc[:, cols_to_scale].div(
        df_demog['population_all'], axis=0))

cols_ethnic = [c for c in df_demog.columns if c.startswith('ethnic_group')]
cols_ethnic.remove('ethnic_group_all_categories_ethnic_group')
df_demog[cols_ethnic] = df_demog[cols_ethnic].astype(float)
df_demog.loc[:, cols_ethnic] = (
    df_demog.loc[:, cols_ethnic].div(
        df_demog['ethnic_group_all_categories_ethnic_group'], axis=0))

cols_activities = [c for c in df_demog.columns if c.startswith('day_to_day')]
# cols_activities.remove('all_categories_long_term_health_problem_or_disability')
df_demog[cols_activities] = df_demog[cols_activities].astype(float)
df_demog.loc[:, cols_activities] = (
    df_demog.loc[:, cols_activities].div(
        df_demog['all_categories_long_term_health_problem_or_disability'],
        axis=0))


cols_health = [c for c in df_demog.columns if c.endswith('health')]
cols_health.remove('all_categories_general_health')
df_demog[cols_health] = df_demog[cols_health].astype(float)
df_demog.loc[:, cols_health] = (
    df_demog.loc[:, cols_health].div(
        df_demog['all_categories_general_health'], axis=0))

cols_female = [c for c in df_demog.columns if (
    ('females' in c) & (c.startswith('population') is False)
    )]
df_demog[cols_female] = df_demog[cols_female].astype(float)
df_demog.loc[:, cols_female] = (
    df_demog.loc[:, cols_female].div(
        df_demog['population_females'], axis=0))

cols_male = [c for c in df_demog.columns if (
    ('_males' in c) & (c.startswith('population') is False)
    )]
df_demog[cols_male] = df_demog[cols_male].astype(float)
df_demog.loc[:, cols_male] = (
    df_demog.loc[:, cols_male].div(
        df_demog['population_males'], axis=0))

st.write(df_demog)

# for col in df_demog.columns:
#     try:
#         st.write(col, np.min(df_demog[col]), np.max(df_demog[col]))
#     except TypeError:
#         pass
# # st.stop()



# ####################################
# ########## SETUP FOR MAPS ##########
# ####################################
# Keep this below the results above because the map creation is slow.

gdf_lhs, colour_dict = maps.create_colour_gdf(
    df_demog[col1],
    col1,
    vmin_map1,
    vmax_map1,
    step_size_map1,
    cmap_name=cmap_name_map1,
    cbar_title=cmap_titles[0],
    )
gdf_rhs, colour_diff_dict = maps.create_colour_gdf(
    df_demog[col2],
    col2,
    vmin_map2,
    vmax_map2,
    step_size_map2,
    cmap_name=cmap_name_map2,
    cbar_title=cmap_titles[1],
    )


# ----- Process geography for plotting -----
# Convert gdf polygons to xy cartesian coordinates:
gdfs_to_convert = [gdf_lhs, gdf_rhs]
for gdf in gdfs_to_convert:
    if gdf is None:
        pass
    else:
        x_list, y_list = maps.convert_shapely_polys_into_xy(gdf)
        gdf['x'] = x_list
        gdf['y'] = y_list


# ----- Plot -----
with container_maps:
    plot_maps.plotly_many_maps(
        gdf_lhs,
        gdf_rhs,
        subplot_titles=subplot_titles,
        colour_dict=colour_dict,
        colour_diff_dict=colour_diff_dict
        )
