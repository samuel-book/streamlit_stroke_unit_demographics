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
    page_title='Colour band map demo',
    page_icon=':rainbow:',
    layout='wide'
    )

container_maps = st.empty()


# Import the full travel time matrix:
df_travel_times = pd.read_csv(
    './data/lsoa_travel_time_matrix_calibrated.csv', index_col='LSOA')
# Rename index to 'lsoa':
df_travel_times.index.name = 'lsoa'


# #################################
# ########## USER INPUTS ##########
# #################################

# User inputs for which hospitals to pick:
catchment = Catchment()
df_units = catchment.get_unit_services()

# Limit to England:
df_units = df_units.loc[~df_units['icb'].isna()].copy()
# Sort by ISDN (approximates sort by region):
df_units = df_units.sort_values('isdn')

# Find where in the list the default options are.
# Ugly conversion to int from int64 so selectbox() can take it.
ind1 = int(np.where(df_units.index == 'LE15WW')[0][0])
ind2 = int(np.where(df_units.index == 'TA15DA')[0][0])

# Select hospitals by name...
unit1_name = st.selectbox(
    'Hospital 1',
    options=df_units['stroke_team'],
    index=ind1
)
unit2_name = st.selectbox(
    'Hospital 2',
    options=df_units['stroke_team'],
    index=ind2
)

# ... then convert names to postcodes for easier lookup.
unit1 = df_units.loc[df_units['stroke_team'] == unit1_name].index.values[0]
unit2 = df_units.loc[df_units['stroke_team'] == unit2_name].index.values[0]


# Colourmap selection
cmap_names = [
    'cosmic_r', 'viridis_r', 'inferno_r', 'neutral_r'
    ]
cmap_diff_names = [
    'iceburn_r', 'seaweed', 'fusion', 'waterlily'
    ]
with st.sidebar:
    st.markdown('### Colour schemes')
    cmap_name, cmap_diff_name = inputs.select_colour_maps(
        cmap_names, cmap_diff_names)

    with st.form('Colour band setup'):
        v_min = st.number_input(
            'LHS vmin',
            min_value=0,
            max_value=480,
            step=5,
            value=0,
            )
        v_max = st.number_input(
            'LHS vmax',
            min_value=0,
            max_value=480,
            step=5,
            value=120,
            )
        step_size = st.number_input(
            'LHS step',
            min_value=5,
            max_value=60,
            step=5,
            value=30,
            )

        v_min_diff = st.number_input(
            'RHS vmin',
            min_value=-480,
            max_value=0,
            step=5,
            value=-120,
            )
        v_max_diff = st.number_input(
            'RHS vmax',
            min_value=0,
            max_value=480,
            step=5,
            value=120,
            )
        step_size_diff = st.number_input(
            'RHS step',
            min_value=5,
            max_value=120,
            step=5,
            value=30,
            )
        submitted = st.form_submit_button('Submit')


# Display names:
subplot_titles = [
    'Time to hospital 1',
    'Time benefit of hospital 2 over hospital 1'
]
cmap_titles = [f'{s} (minutes)' for s in subplot_titles]


# #######################################
# ########## MAIN CALCULATIONS ##########
# #######################################
# While the main calculations are happening, display a blank map.
# Later, when the calculations are finished, replace with the actual map.
with container_maps:
    plot_maps.plotly_blank_maps(['', ''], n_blank=2)

# Pick out the data for hospitals 1 and 2 only:
df_data = df_travel_times[[unit1, unit2]]

# Calculate the time difference between hospitals 1 and 2:
if unit1 != unit2:
    df_data['diff'] = df_data[unit1] - df_data[unit2]
else:
    # Difference is zero when the same unit is selected twice.
    df_data['diff'] = 0.0


# ####################################
# ########## SETUP FOR MAPS ##########
# ####################################
# Keep this below the results above because the map creation is slow.

gdf_lhs, colour_dict = maps.create_colour_gdf(
    df_data,
    unit1,
    v_min,
    v_max,
    step_size,
    cmap_name=cmap_name,
    cbar_title=cmap_titles[0],
    )
gdf_rhs, colour_diff_dict = maps.create_colour_gdf(
    df_data,
    'diff',
    v_min_diff,
    v_max_diff,
    step_size_diff,
    use_diverging=True,
    cmap_name=cmap_diff_name,
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
