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
from utilities.inputs import load_region_outlines


# ###########################
# ##### START OF SCRIPT #####
# ###########################
# page_setup()
st.set_page_config(
    page_title='Stroke unit demographics',
    page_icon=':rainbow:',
    layout='wide'
    )

container_column_select = st.container()
container_maps = st.empty()
container_hist = st.container()
# Inputs for overwriting default colourbars:
with st.expander('Colour setup'):
    cols = st.columns(2)
    with cols[0]:
        container_map1_cbar_setup = st.container()
    with cols[1]:
        container_map2_cbar_setup = st.container()
with st.sidebar:
    container_region_select = st.container()
    container_region_outline_setup = st.container()
container_stats = st.container()
with container_stats:
    container_stats_map1, container_stats_map2 = st.columns(2)
with st.expander('Full data table'):
    container_data_table = st.container()


# --- Import demographic data ---
# Pick the region type to plot:
region_type_dict = {
    'Ambulance service': 'ambulance_service',
    'Closest IVT unit': 'closest_ivt_unit',
    'Closest MT unit': 'closest_mt_unit',
    'Closest transfer unit for MT': 'closest_mt_transfer',
    'ICB (England) & LHB (Wales)': 'icb_lhb',
    # 'Integrated Care Board (England only)': 'icb_code',
    # 'Local Health Board (Wales only)': 'lhb',
    'Integrated Stroke Delivery Network (England only)': 'isdn',
    # 'Local Authority District': 'LAD22NM',
    'LSOA': 'LSOA',
}
with container_region_select:
    region_type_str = st.selectbox(
        'Group data at this level',
        region_type_dict.keys(),
        index=1  # index of default option
        )
region_type = region_type_dict[region_type_str]

# Import the full travel time matrix:
df_demog = pd.read_csv(
    f'./data/collated_data_regional_{region_type}.csv', index_col=0)

with container_data_table:
    st.dataframe(df_demog)

if region_type == 'LSOA':
    # Rename index to 'lsoa':
    df_demog.index.name = 'lsoa'

# TO DO - remove repeat population_all column
# TO DO - fix column names with missing characters

# TO DO - add pretty text explanation of column names, not just the obscure names

# TO DO - violins to show the scatter of all stroke units' data and then highlight some selected units.
# Separate to main map app.
# Could show a reference map with the regions coloured by broader region (south west vs north east e.g.)
# and then colour all points by those reference colours. So still get some geographical info
# but won't take the huge load time.

# TO DO - print means, stds etc across all regions.

# TO DO - show stroke unit locations

# TO DO - does it make sense to attempt an admissions-weighted IVT rate? I think not.


# TO DO - stick the histograms in the same set of subplots as the maps
# to see if the sizing lines up nicely (colourbar bands match histogram bins).


# #################################
# ########## USER INPUTS ##########
# #################################

cols_selectable = list(df_demog.columns)
# Remove columns that are absolute numbers instead of ratios:
cols_abs = [
    'ethnic_group_other_than_white_british',
    'ethnic_group_all_categories_ethnic_group',
    'bad_or_very_bad_health',
    'all_categories_general_health',
    'long_term_health_count',
    'all_categories_long_term_health_problem_or_disability',
    'age_65_plus_count',
    # 'population_all',
    'rural_False',
    'rural_True',
    'over_65_within_30_False',
    'over_65_within_30_True',
    'closest_is_mt_False',
    'closest_is_mt_True'
]
cols_to_remove = ['polygon_area_km2'] + cols_abs
cols_selectable = [c for c in cols_selectable if c not in cols_to_remove]

with container_column_select:
    cols = st.columns(2)
with cols[0]:
    col1 = st.selectbox('Data for left map', options=cols_selectable, index=0)
with cols[1]:
    col2 = st.selectbox('Data for right map', options=cols_selectable, index=1)


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

col1_colour_scale_dict = inputs.lookup_colour_scale(col1, df_demog[col1])
col2_colour_scale_dict = inputs.lookup_colour_scale(col2, df_demog[col2])


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


# Name of the column in the geojson that labels the shapes:
with container_region_outline_setup:
    outline_name = st.radio(
        'Region type to draw on maps',
        [
            'None',
            'Ambulance service',
            'Closest IVT unit',
            'Closest MT unit',
            'Closest MT transfer',
            'ICB (England) & LHB (Wales)',
            'ISDN',
            # 'LAD',
        ]
    )

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

# Find means, std etc.:
def calculate_stats(vals):
    s = {}
    s['mean'] = vals.mean()
    s['std'] = vals.std()
    s['q1'] = vals.quantile(0.25)
    s['median'] = vals.median()
    s['q3'] = vals.quantile(0.75)

    if int(s['median']) == s['median']:
        s['mean'] = int(round(s['mean'], 0))
        s['std'] = int(round(s['std'], 0))
    return s

stats_dict_map1 = calculate_stats(df_demog[col1])
stats_dict_map2 = calculate_stats(df_demog[col2])

stats_series_map1 = pd.Series(stats_dict_map1, name=col1)
stats_series_map2 = pd.Series(stats_dict_map2, name=col2)

with container_stats_map1:
    # for key, val in stats_dict_map1.items():
    #     st.write(f'{key}: {val}')
    st.write(stats_series_map1)

with container_stats_map2:
    # for key, val in stats_dict_map2.items():
    #     st.write(f'{key}: {val}')
    st.write(stats_series_map2)


# ####################################
# ########## SETUP FOR MAPS ##########
# ####################################
# Keep this below the results above because the map creation is slow.

merge_polygons_bool = True if region_type == 'LSOA' else False
use_discrete_cmap = True if region_type == 'LSOA' else False

gdf_lhs, colour_dict_map1 = maps.create_colour_gdf(
    df_demog[col1],
    col1,
    vmin_map1,
    vmax_map1,
    step_size_map1,
    cmap_name=cmap_name_map1,
    cbar_title=cmap_titles[0],
    merge_polygons_bool=merge_polygons_bool,
    region_type=region_type,
    use_discrete_cmap=use_discrete_cmap
    )
gdf_rhs, colour_dict_map2 = maps.create_colour_gdf(
    df_demog[col2],
    col2,
    vmin_map2,
    vmax_map2,
    step_size_map2,
    cmap_name=cmap_name_map2,
    cbar_title=cmap_titles[1],
    merge_polygons_bool=merge_polygons_bool,
    region_type=region_type,
    use_discrete_cmap=use_discrete_cmap
    )

# ----- Region outlines -----
if outline_name == 'None':
    outline_names_col = None
    gdf_catchment_lhs = None
    gdf_catchment_rhs = None
else:
    outline_names_col, gdf_catchment_lhs, gdf_catchment_rhs = (
        load_region_outlines(outline_name))


# ----- Process geography for plotting -----
# Convert gdf polygons to xy cartesian coordinates:
gdfs_to_convert = [gdf_lhs, gdf_rhs, gdf_catchment_lhs, gdf_catchment_rhs]
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
        gdf_catchment_lhs=gdf_catchment_lhs,
        gdf_catchment_rhs=gdf_catchment_rhs,
        outline_names_col=outline_names_col,
        outline_name=outline_name,
        subplot_titles=subplot_titles,
        colour_dict=colour_dict_map1,
        colour_diff_dict=colour_dict_map2,
        use_discrete_cmap=use_discrete_cmap
        )

# ----- Histogram -----
with container_hist:
    plot_maps.plot_hists(
        df_demog, col1, col2,
        colour_dict_map1, colour_dict_map2,
        subplot_titles,
        use_discrete_cmap=use_discrete_cmap
        )
