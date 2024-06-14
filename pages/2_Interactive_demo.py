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
from scipy.stats import linregress

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
from utilities.fixed_params import page_setup
from utilities.inputs import load_region_outlines


# ###########################
# ##### START OF SCRIPT #####
# ###########################
page_setup()
st.markdown('# Stroke unit demographics')

container_column_select = st.container()
container_maps = st.empty()
container_hist = st.container()
# Inputs for overwriting default colourbars:
with st.expander('Colour and histogram setup'):
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
container_scatter = st.container()
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
    st.markdown('### Data selection')
    region_type_str = st.selectbox(
        'Group data at this level',
        region_type_dict.keys(),
        index=1  # index of default option
        )
region_type = region_type_dict[region_type_str]

# Import the full travel time matrix:
df_demog = pd.read_csv(
    f'./data/collated_data_regional_{region_type}.csv', index_col=0)

# Stroke unit info:
catchment = Catchment()
df_units = catchment.get_unit_services()

if (('_ivt_' in region_type) | ('_mt_' in region_type)):
    # The data is saved with stroke unit postcodes as the only
    # identifier. Load in a postcode-name lookup for display.
    cols_to_merge = ['stroke_team', 'ssnap_name']
    df_demog = pd.merge(
        df_units[cols_to_merge], df_demog,
        left_index=True, right_index=True, how='right'
        )

with container_data_table:
    st.dataframe(df_demog)

if region_type == 'LSOA':
    # Rename index to 'lsoa':
    df_demog.index.name = 'lsoa'


# TO DO - add text explanation of column names and how weighted means are done.

# TO DO - violins to show the scatter of all stroke units' data and then highlight some selected units.
# Separate to main map app.
# Could show a reference map with the regions coloured by broader region (south west vs north east e.g.)
# and then colour all points by those reference colours. So still get some geographical info
# but won't take the huge load time.

# TO DO - print means, stds etc across all regions.

# TO DO - show stroke unit locations

# TO DO - stick the histograms in the same set of subplots as the maps
# to see if the sizing lines up nicely (colourbar bands match histogram bins).


# #################################
# ########## USER INPUTS ##########
# #################################


def select_columns(cols_selectable):
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
    cols_unit_names = ['stroke_team', 'ssnap_name']
    cols_to_remove = ['polygon_area_km2'] + cols_abs + cols_unit_names
    cols_selectable = [c for c in cols_selectable if c not in cols_to_remove]

    # Make column names more pretty:
    cols_prettier = {
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
    cols_prettier_reverse = dict(
        zip(list(cols_prettier.values()), list(cols_prettier.keys())))

    cols_selectable = sorted([
        cols_prettier[c] if c in list(cols_prettier.keys()) else c
        for c in cols_selectable
        ])

    with container_column_select:
        cols = st.columns(2)
    with cols[0]:
        col1_pretty = st.selectbox(
            'Data for left map', options=cols_selectable, index=0)
    with cols[1]:
        col2_pretty = st.selectbox(
            'Data for right map', options=cols_selectable, index=1)

    try:
        col1 = cols_prettier_reverse[col1_pretty]
    except KeyError:
        col1 = col1_pretty
    try:
        col2 = cols_prettier_reverse[col2_pretty]
    except KeyError:
        col2 = col2_pretty
    return col1, col2, col1_pretty, col2_pretty


cols_selectable = list(df_demog.columns)
col1, col2, col1_pretty, col2_pretty = select_columns(cols_selectable)

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
    st.markdown('### Outlines')
    outline_name = st.selectbox(
        'Region outlines to draw on maps',
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
subplot_titles = [col1_pretty, col2_pretty]
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
    st.write(stats_series_map1)

with container_stats_map2:
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

if (('_ivt_' in region_type) | ('_mt_' in region_type)):
    # Put the stroke team names back into these gdfs.
    cols_to_merge = ['stroke_team', 'ssnap_name']
    gdf_lhs = pd.merge(
        df_demog[cols_to_merge], gdf_lhs,
        left_index=True, right_on=region_type, how='right'
        )
    gdf_rhs = pd.merge(
        df_demog[cols_to_merge], gdf_rhs,
        left_index=True, right_on=region_type, how='right'
        )

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
        colour_dict=colour_dict_map1 | {'title': subplot_titles[0]},
        colour_diff_dict=colour_dict_map2  | {'title': subplot_titles[1]},
        use_discrete_cmap=use_discrete_cmap
        )

# ----- Histogram -----
with container_hist:
    plot_maps.plot_hists(
        df_demog, col1, col2,
        colour_dict_map1  | {'title': subplot_titles[0]},
        colour_dict_map2  | {'title': subplot_titles[1]},
        subplot_titles,
        use_discrete_cmap=use_discrete_cmap
        )

# ----- Scatter -----

def scatter_fields(
        x_feature_name,
        y_feature_name,
        c_feature_name,
        df,
        stroke_teams_selected,
        team_colours_dict,
        x_feature_display_name,
        y_feature_display_name,
        c_feature_display_name,
        ):
    """
    Scatter selected descriptive stats data for all teams.

    Inputs:
    -------
    x_feature_name         - str. df column for x-axis data.
    y_feature_name         - str. df column for y-axis data.
    c_feature_name         - str. df column for colour data.
    year_restriction       - str. Which years of data to use.
    df                     - pd.DataFrame. Descriptive stats data.
    stroke_teams_selected  - list. Names of stroke teams to highlight.
    team_colours_dict      - dict. Colours for highlighted teams.
    x_feature_display_name - str. x-axis label.
    y_feature_display_name - str. y-axis label.
    c_feature_display_name - str. Colour axis label.
    """
    # df = df.T

    # Create the line of best fit:
    lobf = linregress(
        list(df[x_feature_name].astype(float).values),
        list(df[y_feature_name].astype(float).values)
        )

    fig = go.Figure()

    if c_feature_display_name != 'None':
        # Fig with colourbar
        fig_width = 700 + 5 * len(x_feature_display_name)
    else:
        # Fig without colourbar
        fig_width = 600 + 5 * len(x_feature_display_name)
    # Quick attempt to get near-square axis:
    fig.update_layout(
        width=fig_width,
        height=500,
        margin_l=0, margin_r=0, margin_t=0, margin_b=0
        )

    # Create the legend label for the line of best fit.
    # Round numbers to 3 significant figures and then convert
    # large (>=1000) numbers back from general string format to
    # float to avoid printing scientific notation (e.g. 4.01e+3).
    lobf_int = (
        f'{lobf.intercept:.3g}' if abs(lobf.intercept) < 1000
        else int(float(f'{lobf.intercept:.3g}'))
    )
    lobf_slope = (
        f'{lobf.slope:.3g}' if abs(lobf.slope) < 1000
        else int(float(f'{lobf.slope:.3g}'))
    )
    lobf_name = (
        f'{lobf_int} + ' +
        f'({x_feature_display_name}) × ({lobf_slope})'
        )
    # Plot the line of best fit:
    fig.add_trace(go.Scatter(
        x=df[x_feature_name],
        y=lobf.intercept + lobf.slope * df[x_feature_name].astype(float),
        name=lobf_name,
        hoverinfo='skip',
        marker_color='silver'
    ))

    # Plot highlighted teams:
    # Remove any of the "all teams" or "all region" data:
    # (silly setup to remove teams appearing in the list multiple times
    # while retaining the order in the list)
    a = []
    for t in stroke_teams_selected:
        if (t[:4] != 'All ' and t not in a):
            a.append(t)
    stroke_teams_selected = a
    for stroke_team in stroke_teams_selected:
        mask_team = df.index == stroke_team
        fig.add_trace(go.Scatter(
            x=df[x_feature_name][mask_team],
            y=df[y_feature_name][mask_team],
            mode='markers',
            name=df.index[mask_team].squeeze(),
            text=[df.index[mask_team].squeeze()],
            marker_color='rgba(0, 0, 0, 0)',
            marker_line_color=team_colours_dict[stroke_team],
            marker_size=10,
            marker_line_width=2.5,
            marker_symbol='square',
            # hovertemplate='(%{x}, %{y})<extra>%{text}</extra>'
            hoverinfo='skip'
        ))

    # Plot all teams that are not highlighted:
    if c_feature_display_name != 'None':
        # Colour teams by third value
        fig.add_trace(go.Scatter(
            x=df[x_feature_name],
            y=df[y_feature_name],
            marker_color=df[c_feature_name].astype(float),
            marker_showscale=True,
            marker_colorbar_title_text=c_feature_display_name,
            marker_colorbar_title_side='right',
            mode='markers',
            text=df.index,
            name='Stroke teams',
            # marker_color='grey',
            marker_line_color='black',
            marker_line_width=1.0,
            customdata=np.stack([df[c_feature_name].astype(float)], axis=-1),
            hovertemplate=(
                '(%{x}, %{y})<br>' + c_feature_display_name +
                ': %{customdata[0]}<extra>%{text}</extra>'
                )
        ))
        fig.update_coloraxes(colorbar_title_text=c_feature_display_name)
    else:
        # Show all teams in grey.
        fig.add_trace(go.Scatter(
            x=df[x_feature_name],
            y=df[y_feature_name],
            mode='markers',
            text=df.index,
            name='Stroke teams',
            marker_color='grey',
            marker_line_color='black',
            marker_line_width=1.0,
            hovertemplate='(%{x}, %{y})<extra>%{text}</extra>'
        ))

    # Figure format:
    fig.update_layout(
        xaxis_title=x_feature_display_name,
        yaxis_title=y_feature_display_name
    )
    if c_feature_display_name != 'None':
        # Move the legend to make space for the colour bar.
        fig.update_layout(legend=dict(
            orientation='v',
            yanchor='top',
            y=1.0,
            xanchor='left',
            x=1.3,
            # itemwidth=50
        ))
    plotly_config = {
        'displayModeBar': False
        # 'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    }
    st.plotly_chart(fig, config=plotly_config)


with container_scatter:
    st.header('Relation between two features')
    st.markdown('Compare the variation of two features across hospitals.')
    cols_scatter_inputs = st.columns(3)

    cols_selectable = list(df_demog.columns)
    cols_to_remove = ['stroke_team', 'ssnap_name']
    cols_selectable = [c for c in cols_selectable if c not in cols_to_remove]

    # Pick two features to scatter:
    with cols_scatter_inputs[0]:
        x_feature_display_name = st.selectbox(
            'Feature for x-axis',
            options=cols_selectable
        )
        x_feature_name = x_feature_display_name  # inverse_index_names[x_feature_display_name]

    with cols_scatter_inputs[1]:
        y_feature_display_name = st.selectbox(
            'Feature for y-axis',
            options=cols_selectable
        )
        y_feature_name = y_feature_display_name  # inverse_index_names[y_feature_display_name]

    with cols_scatter_inputs[2]:
        c_feature_display_name = st.selectbox(
            'Feature for colour',
            options=['None'] + list(cols_selectable)
        )
        # c_feature_name = (inverse_index_names[c_feature_display_name]
        #                     if c_feature_display_name != 'None'
        #                     else c_feature_display_name)
        c_feature_name = c_feature_display_name

    # with cols_scatter_inputs[3]:
    #     year_restriction = st.selectbox(
    #         'Years to show',
    #         options=year_options
    #     )

    scatter_fields(
        x_feature_name,
        y_feature_name,
        c_feature_name,
        df_demog,
        [],  # stroke_teams_selected_without_year,
        {},  # team_colours_dict,
        x_feature_display_name,
        y_feature_display_name,
        c_feature_display_name
        )
