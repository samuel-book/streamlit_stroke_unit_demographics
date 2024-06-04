"""
Functions for making plotly maps for Streamlit.
"""
import streamlit as st
import numpy as np
import os
import geopandas

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from utilities.maps import convert_shapely_polys_into_xy

from stroke_maps.utils import find_multiindex_column_names
from stroke_maps.geo import _load_geometry_stroke_units, check_scenario_level
from stroke_maps.catchment import Catchment


def create_stroke_team_markers(df_units=None):
    """
    Create plotly traces for the stroke team scatter markers.

    Create the traces first and then later actually draw them
    on the figures because some traces need to appear identically
    across multiple subplots.

    Inputs
    ------
    df_units - pd.DataFrame. Information about which units provide
               which services, and this affects which markers are
               drawn for each unit.

    Returns
    -------
    traces - dict. An entry for IVT, MT, and MSU units. Contains
             plotly traces for the markers for the stroke units.
    """

    # Add stroke team markers.
    if df_units is None:
        catchment = Catchment()
        df_units = catchment.get_unit_services()
    else:
        pass
    # Build geometry:
    df_units = check_scenario_level(df_units)
    gdf_points_units = _load_geometry_stroke_units(df_units)

    # # Convert to British National Grid.
    # The geometry column should be BNG on import, so just overwrite
    # the longitude and latitude columns that are by default long/lat.
    col_geo = find_multiindex_column_names(
        gdf_points_units, property=['geometry'])
    # gdf_points_units = gdf_points_units.set_crs(
    #     'EPSG:27700', allow_override=True)

    # Overwrite long and lat:
    gdf_points_units[('Longitude', 'any')] = gdf_points_units[col_geo].x
    gdf_points_units[('Latitude', 'any')] = gdf_points_units[col_geo].y

    # Set up markers using a new column in DataFrame.
    # Set everything to the IVT marker:
    markers = np.full(len(gdf_points_units), 'circle', dtype=object)
    # Update MT units:
    col_use_mt = find_multiindex_column_names(
        gdf_points_units, property=['use_mt'])
    mask_mt = (gdf_points_units[col_use_mt] == 1)
    markers[mask_mt] = 'square'
    # Store in the DataFrame:
    gdf_points_units[('marker', 'any')] = markers

    # Add markers in separate traces for the sake of legend entries.
    # Pick out which stroke unit types are where in the gdf:
    col_ivt = ('use_ivt', 'scenario')
    col_mt = ('use_mt', 'scenario')
    col_msu = ('use_msu', 'scenario')
    mask_ivt = gdf_points_units[col_ivt] == 1
    mask_mt = gdf_points_units[col_mt] == 1
    mask_msu = gdf_points_units[col_msu] == 1

    # Formatting for the markers:
    format_dict = {
        'ivt': {
            'label': 'IVT unit',
            'mask': mask_ivt,
            'marker': 'circle',
            'size': 6,
            'colour': 'white'
        },
        'mt': {
            'label': 'MT unit',
            'mask': mask_mt,
            'marker': 'star',
            'size': 10,
            'colour': 'white'
        },
        'msu': {
            'label': 'MSU base',
            'mask': mask_msu,
            'marker': 'square',
            'size': 13,
            'colour': 'white'
        },
    }

    # Build the traces for the stroke units...
    traces = {}
    for service, s_dict in format_dict.items():
        mask = s_dict['mask']

        trace = go.Scatter(
            x=gdf_points_units.loc[mask, ('Longitude', 'any')],
            y=gdf_points_units.loc[mask, ('Latitude', 'any')],
            mode='markers',
            marker={
                'symbol': s_dict['marker'],
                'color': s_dict['colour'],
                'line': {'color': 'black', 'width': 1},
                'size': s_dict['size']
            },
            name=s_dict['label'],
            customdata=np.stack(
                [gdf_points_units.loc[mask, ('ssnap_name', 'scenario')]],
                axis=-1
                ),
            hovertemplate=(
                '%{customdata[0]}' +
                # Need the following line to remove default "trace" bit
                # in second "extra" box:
                '<extra></extra>'
                )
        )
        traces[service] = trace
    return traces


def plotly_blank_maps(subplot_titles: list = None, n_blank: int = 2):
    """
    Create dummy subplots with blank maps in them to mask load times.

    Inputs
    ------
    subplot_titles - list or None. Titles for the subplots.
    n_blank        - int. How many subplots to create.
    """
    path_to_file = os.path.join('data', 'outline_england_wales.geojson')
    gdf_ew = geopandas.read_file(path_to_file)
    gdf_ew = gdf_ew.to_crs('EPSG:27700')

    x_list, y_list = convert_shapely_polys_into_xy(gdf_ew)
    gdf_ew['x'] = x_list
    gdf_ew['y'] = y_list

    # ----- Plotting -----
    fig = make_subplots(
        rows=1, cols=n_blank,
        horizontal_spacing=0.0,
        subplot_titles=subplot_titles
        )

    # Add each row of the dataframe separately.
    # Scatter the edges of the polygons and use "fill" to colour
    # within the lines.
    for i in gdf_ew.index:
        fig.add_trace(go.Scatter(
            x=gdf_ew.loc[i, 'x'],
            y=gdf_ew.loc[i, 'y'],
            mode='lines',
            fill="toself",
            fillcolor='rgba(0, 0, 0, 0)',
            line_color='grey',
            showlegend=False,
            hoverinfo='skip',
            ), row='all', col='all'
            )

    # Add a blank trace to create space for a legend.
    # Stupid? Yes. Works? Also yes.
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker={'color': 'rgba(0,0,0,0)'},
        name=' ' * 20
    ))

    # Equivalent to pyplot set_aspect='equal':
    fig.update_yaxes(col=1, scaleanchor='x', scaleratio=1)
    fig.update_yaxes(col=2, scaleanchor='x2', scaleratio=1)

    # Shared pan and zoom settings:
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    # Remove axis ticks:
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    # Figure setup.
    fig.update_layout(
        # width=1200,
        height=700,
        margin_t=40,
        margin_b=60  # mimic space taken up by colourbar
        )

    # Disable clicking legend to remove trace:
    fig.update_layout(legend_itemclick=False)
    fig.update_layout(legend_itemdoubleclick=False)

    # Options for the mode bar.
    # (which doesn't appear on touch devices.)
    plotly_config = {
        # Mode bar always visible:
        # 'displayModeBar': True,
        # Plotly logo in the mode bar:
        'displaylogo': False,
        # Remove the following from the mode bar:
        'modeBarButtonsToRemove': [
            # 'zoom',
            # 'pan',
            'select',
            # 'zoomIn',
            # 'zoomOut',
            'autoScale',
            'lasso2d'
            ],
        # Options when the image is saved:
        'toImageButtonOptions': {'height': None, 'width': None},
        }

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)


def plotly_many_maps(
        gdf_lhs: geopandas.GeoDataFrame,
        gdf_rhs: geopandas.GeoDataFrame,
        gdf_catchment_lhs: geopandas.GeoDataFrame = None,
        gdf_catchment_rhs: geopandas.GeoDataFrame = None,
        outline_names_col: str = '',
        outline_name: str = '',
        traces_units: dict = None,
        unit_subplot_dict: dict = {},
        subplot_titles: list = [],
        legend_title: str = '',
        colour_dict: dict = {},
        colour_diff_dict: dict = {},
        use_discrete_cmap: bool = True
        ):
    """
    Main map-drawing function.

    Inputs
    ------
    gdf_lhs           - geopandas.GeoDataFrame. Data for left-hand
                        side.
    gdf_rhs           - geopandas.GeoDataFrame. Data for right-hand
                        side.
    gdf_catchment_lhs - geopandas.GeoDataFrame. Optional. Data to
                        plot over the top of the other gdf, for example
                        catchment area outlines. For left-hand map.
    gdf_catchment_rhs - geopandas.GeoDataFrame. Optional. Same but for
                        right-hand map.
    outline_names_col - str. Name of the column in gdf_catchment that
                        contains data to show on the hover text.
    outline_name      - str. One value from the 'outcome_type' column
                        in gdf_catchment. (Should all be same values).
    traces_units      - dict. Plotly traces of scatter markers for
                        stroke units.
    unit_subplot_dict - dict. Which unit traces should be shown on
                        which subplots (by number).
    subplot_titles    - list. Title appearing above each subplot.
    legend_title      - str. Title for the legend.
    colour_dict       - dict. Colour band labels to hex colour lookup
                        for the left-hand-side map.
    colour_diff_dict  - dict. Same for the right-hand-side map.
    """
    # ----- Plotting -----
    fig = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.0,
        subplot_titles=subplot_titles
        )

    # Add a blank outline of England:
    path_to_file = os.path.join('data', 'outline_england_wales.geojson')
    gdf_ew = geopandas.read_file(path_to_file)
    gdf_ew = gdf_ew.to_crs('EPSG:27700')

    x_list, y_list = convert_shapely_polys_into_xy(gdf_ew)
    gdf_ew['x'] = x_list
    gdf_ew['y'] = y_list

    # Add each row of the dataframe separately.
    # Scatter the edges of the polygons and use "fill" to colour
    # within the lines.
    for i in gdf_ew.index:
        fig.add_trace(go.Scatter(
            x=gdf_ew.loc[i, 'x'],
            y=gdf_ew.loc[i, 'y'],
            mode='lines',
            fill="toself",
            fillcolor='rgba(0, 0, 0, 0)',
            line_color='grey',
            showlegend=False,
            hoverinfo='skip',
            ), row='all', col='all'
            )

    # Return a gdf of some x, y coordinates to scatter
    # in such a tiny size that they'll never be seen,
    # but that will cause the colourbar of the colour scale to display.
    # Separate colour scales for the two maps.

    def draw_dummy_scatter(fig, colour_dict, col=1, trace_name='', use_discrete=True):
        # Dummy coordinates:
        # Isle of Man: 238844, 482858
        bonus_x = 238844
        bonus_y = 482858
        x_dummy = np.array([bonus_x]*2)
        y_dummy = np.array([bonus_y]*2)
        z_dummy = np.array([0.0, 1.0])

        # Sometimes the ticks don't show at the very ends of the colour bars.
        # In that case, cheat with e.g.
        # tick_locs = [bounds[0] + 1e-2, *bounds[1:-1], bounds[-1] - 1e-3]
        tick_locs = colour_dict['bounds_for_colour_scale']

        tick_names = [f'{t:.3f}' for t in colour_dict['v_bands']]
        if use_discrete:
            tick_names = ['←', *tick_names, '→']
        else:
            # Budge along the first tick a bit so it isn't cut off.
            tick_locs[0] += (tick_locs[1] - tick_locs[0]) * 0.2

        # Replace zeroish with zero:
        # (this is a visual difference only - it combines two near-zero
        # ticks and their labels into a single tick.)
        if colour_dict['diverging']:
            ind_z = np.where(np.sign(colour_dict['v_bands']) >= 0.0)[0][0] + 1
            tick_z = np.mean([tick_locs[ind_z-1], tick_locs[ind_z]])
            name_z = '0'

            tick_locs_z = np.append(tick_locs[:ind_z - 1], tick_z)
            tick_locs_z = np.append(tick_locs_z, tick_locs[ind_z+1:])
            tick_locs = tick_locs_z

            tick_names_z = np.append(tick_names[:ind_z - 1], name_z)
            tick_names_z = np.append(tick_names_z, tick_names[ind_z+1:])
            tick_names = tick_names_z

        # Add dummy scatter:
        fig.add_trace(go.Scatter(
            x=x_dummy,
            y=y_dummy,
            marker=dict(
                color=z_dummy,
                colorscale=colour_dict['colour_scale'],
                colorbar=dict(
                    thickness=20,
                    tickmode='array',
                    tickvals=tick_locs,
                    ticktext=tick_names,
                    # ticklabelposition='outside top'
                    title=colour_dict['title']
                    ),
                size=1e-4,
                ),
            showlegend=False,
            mode='markers',
            hoverinfo='skip',
            name=trace_name
        ), row='all', col=col)

        return fig

    fig = draw_dummy_scatter(fig, colour_dict, col=1, trace_name='cbar', use_discrete=use_discrete_cmap)
    fig = draw_dummy_scatter(fig, colour_diff_dict, col=2,
                             trace_name='cbar_diff', use_discrete=use_discrete_cmap)
    fig.update_traces(
        {'marker': {'colorbar': {
            'orientation': 'h',
            'x': 0.0,
            'y': -0.1,
            'len': 0.5,
            'xanchor': 'left',
            'title_side': 'bottom'
            # 'xref': 'paper'
            }}},
        selector={'name': 'cbar'}
        )
    fig.update_traces(
        {'marker': {'colorbar': {
            'orientation': 'h',
            'x': 1.0,
            'y': -0.1,
            'len': 0.5,
            'xanchor': 'right',
            'title_side': 'bottom'
            # 'xref': 'paper'
            }}},
        selector={'name': 'cbar_diff'}
        )

    # Add each row of the dataframe separately.
    # Scatter the edges of the polygons and use "fill" to colour
    # within the lines.
    # I can't get hoverlabels working for the life of me so
    # use 'text' param instead.
    show_hoverinfo = 'skip' if use_discrete_cmap else None
    name_col = 'colour_str' if use_discrete_cmap else gdf_lhs.columns[0]

    for i in gdf_lhs.index:
        try:
            val = gdf_lhs.at[i, 'outcome']
        except KeyError:
            # 'outcome' column doesn't exist
            val = ''

        fig.add_trace(go.Scatter(
            x=gdf_lhs.loc[i, 'x'],
            y=gdf_lhs.loc[i, 'y'],
            mode='lines',
            fill="toself",
            fillcolor=gdf_lhs.loc[i, 'colour'],
            line_width=0,
            text=f'{val}',
            hoverinfo=show_hoverinfo,
            name=gdf_lhs.loc[i, name_col],
            showlegend=False,
            ), row='all', col=1
            )

    for i in gdf_rhs.index:
        try:
            val = gdf_rhs.at[i, 'outcome']
        except KeyError:
            # 'outcome' column doesn't exist
            val = ''
        fig.add_trace(go.Scatter(
            x=gdf_rhs.loc[i, 'x'],
            y=gdf_rhs.loc[i, 'y'],
            mode='lines',
            fill="toself",
            fillcolor=gdf_rhs.loc[i, 'colour'],
            line_width=0,
            text=f'{val}',
            hoverinfo=show_hoverinfo,
            name=gdf_rhs.loc[i, name_col],
            showlegend=False
            ), row='all', col=2
            )

    def draw_outline(fig, gdf_catchment, col='all'):
        # I can't for the life of me get hovertemplate working here
        # for mysterious reasons, so just stick to "text" for hover info.
        for i in gdf_catchment.index:
            fig.add_trace(go.Scatter(
                x=gdf_catchment.loc[i, 'x'],
                y=gdf_catchment.loc[i, 'y'],
                mode='lines',
                fill="toself",
                fillcolor=gdf_catchment.loc[i, 'colour'],
                line_color='grey',
                name=gdf_catchment.loc[i, 'outline_type'],
                text=gdf_catchment.loc[i, outline_names_col],
                hoverinfo="text",
                hoverlabel=dict(bgcolor='red'),
                ), row='all', col=col
                )

    if gdf_catchment_lhs is None:
        pass
    else:
        draw_outline(fig, gdf_catchment_lhs, col=1)

    if gdf_catchment_rhs is None:
        pass
    else:
        draw_outline(fig, gdf_catchment_rhs, col=2)

    fig.update_traces(
        hoverlabel=dict(
            bgcolor='grey',
            font_color='white'),
        selector={'name': outline_name}
    )
    # Equivalent to pyplot set_aspect='equal':
    fig.update_yaxes(col=1, scaleanchor='x', scaleratio=1)
    fig.update_yaxes(col=2, scaleanchor='x2', scaleratio=1)

    # Shared pan and zoom settings:
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    # Remove axis ticks:
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    # --- Stroke unit scatter markers ---
    if len(unit_subplot_dict) > 0:
        if gdf_catchment_lhs is None:
            pass
        else:
            # # Add a blank trace to put a gap in the legend.
            # Stupid? Yes. Works? Also yes.
            # Make sure the name isn't the same as any other blank name
            # already set, e.g. in combo_colour_dict.
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                marker={'color': 'rgba(0,0,0,0)'},
                name=' ' * 10
            ))

        # Create the scatter traces for the stroke units in advance
        # and then add traces to the subplots.
        for service, grid_lists in unit_subplot_dict.items():
            for grid_list in grid_lists:
                row = grid_list[0]
                col = grid_list[1]
                fig.add_trace(traces_units[service], row=row, col=col)

    # Remove repeat legend names:
    # (e.g. multiple sets of IVT unit, MT unit)
    # from https://stackoverflow.com/a/62162555
    names = set()
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))
    # This makes sure that if multiple maps use the exact same
    # colours and labels, the labels only appear once in the legend.

    fig.update_layout(
        legend=dict(
            title_text=legend_title,
            bordercolor='grey',
            borderwidth=2
        )
    )

    # Figure setup.
    fig.update_layout(
        # width=1200,
        height=700,
        margin_t=40,
        margin_b=0
        )

    # Disable clicking legend to remove trace:
    fig.update_layout(legend_itemclick=False)
    fig.update_layout(legend_itemdoubleclick=False)

    # Options for the mode bar.
    # (which doesn't appear on touch devices.)
    plotly_config = {
        # Mode bar always visible:
        # 'displayModeBar': True,
        # Plotly logo in the mode bar:
        'displaylogo': False,
        # Remove the following from the mode bar:
        'modeBarButtonsToRemove': [
            # 'zoom',
            # 'pan',
            'select',
            # 'zoomIn',
            # 'zoomOut',
            'autoScale',
            'lasso2d'
            ],
        # Options when the image is saved:
        'toImageButtonOptions': {'height': None, 'width': None},
        }

    # Write to streamlit:
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=plotly_config
        )


def plot_hists(
        df_demog, col1, col2,
        colour_dict_map1, colour_dict_map2,
        subplot_titles=['', ''],
        use_discrete_cmap=True
        ):
    # Get variables out of the dictionaries:
    vmin_map1 = colour_dict_map1['v_min']
    step_size_map1 = colour_dict_map1['step_size']
    vmax_map1 = colour_dict_map1['v_max']
    vmin_map2 = colour_dict_map2['v_min']
    step_size_map2 = colour_dict_map2['step_size']
    vmax_map2 = colour_dict_map2['v_max']

    bin_colours_map1 = list(colour_dict_map1['colour_map'].values())
    bin_colours_map2 = list(colour_dict_map2['colour_map'].values())
    if use_discrete_cmap:
        pass
    else:
        # bin_colours_map1 = (
        #     [bin_colours_map1[0]] + bin_colours_map1 + [bin_colours_map1[-1]])
        # bin_colours_map2 = (
        #     [bin_colours_map2[0]] + bin_colours_map2 + [bin_colours_map2[-1]])
        # Overwrite - use same colour for all bins.
        bin_colours_map1 = ['grey'] * (len(bin_colours_map1) + 2)
        bin_colours_map2 = ['grey'] * (len(bin_colours_map2) + 2)

    # Calculate bin sizes:
    hist_map1, bins_map1 = np.histogram(
        df_demog[col1],
        bins=np.arange(vmin_map1, vmax_map1+step_size_map1, step_size_map1)
        )
    hist_map2, bins_map2 = np.histogram(
        df_demog[col2],
        bins=np.arange(vmin_map2, vmax_map2+step_size_map2, step_size_map2)
        )

    # How many regions are below the first bin (under)
    # and above the last bin (over)?
    n_under_map1 = len(np.where(df_demog[col1] < vmin_map1)[0])
    n_over_map1 = len(np.where(df_demog[col1] > vmax_map1)[0])
    n_under_map2 = len(np.where(df_demog[col2] < vmin_map2)[0])
    n_over_map2 = len(np.where(df_demog[col2] > vmax_map2)[0])

    fig = make_subplots(
        rows=1, cols=2,
        # horizontal_spacing=0.0,
        # subplot_titles=subplot_titles
        )

    for i in range(len(hist_map1)):
        fig.add_trace(go.Bar(
            x=[bins_map1[i] + 0.5*step_size_map1],
            y=[hist_map1[i]],
            name=col1,
            width=step_size_map1,
            marker=dict(color=bin_colours_map1[i+1],
                        line=dict(color='silver', width=1)),
            showlegend=False,
            hoverinfo='y'
        ), row=1, col=1)

    for i in range(len(hist_map2)):
        fig.add_trace(go.Bar(
            x=[bins_map2[i] + 0.5*step_size_map2],
            y=[hist_map2[i]],
            name=col2,
            width=step_size_map2,
            marker=dict(color=bin_colours_map2[i+1],
                        line=dict(color='silver', width=1)),
            showlegend=False,
            hoverinfo='y'
        ), row=1, col=2)

    # Add space either side to match the colourbar above:
    fig.update_layout(xaxis_range=[
        vmin_map1 - step_size_map1, vmax_map1 + step_size_map1
        ])
    fig.update_layout(xaxis2_range=[
        vmin_map2 - step_size_map2, vmax_map2 + step_size_map2
        ])

    # Add annotations in the space.
    fig.add_annotation(
        x=np.mean([vmin_map1, vmin_map1 - step_size_map1]),
        y=0.0,
        text=f'+{n_under_map1}<br>regions',
        yshift=10,
        showarrow=False,
        bordercolor=bin_colours_map1[0],
        row=1, col=1
        )
    fig.add_annotation(
        x=np.mean([vmax_map1, vmax_map1 + step_size_map1]),
        y=0.0,
        text=f'+{n_over_map1}<br>regions',
        yshift=10,
        showarrow=False,
        bordercolor=bin_colours_map1[-1],
        row=1, col=1
        )
    fig.add_annotation(
        x=np.mean([vmin_map2, vmin_map2 - step_size_map2]),
        y=0.0,
        text=f'+{n_under_map2}<br>regions',
        yshift=10,
        showarrow=False,
        bordercolor=bin_colours_map2[0],
        row=1, col=2
        )
    fig.add_annotation(
        x=np.mean([vmax_map2, vmax_map2 + step_size_map2]),
        y=0.0,
        text=f'+{n_over_map2}<br>regions',
        yshift=10,
        showarrow=False,
        bordercolor=bin_colours_map2[-1],
        row=1, col=2
        )

    # Figure setup.
    fig.update_layout(
        # width=1200,
        height=300,
        margin_t=40,
        margin_b=0
        )
    # Axis labels:
    fig.update_layout(xaxis_title=subplot_titles[0])
    fig.update_layout(yaxis_title='Number of regions')
    fig.update_layout(xaxis2_title=subplot_titles[1])
    fig.update_layout(yaxis2_title='Number of regions')
    # Axis ticks to match colourbar ticks:
    fig.update_layout(xaxis=dict(
        tickmode='array',
        tickvals=[vmin_map1 - step_size_map1, *list(bins_map1),
                  vmax_map1 + step_size_map1]
        ))
    fig.update_layout(xaxis2=dict(
        tickmode='array',
        tickvals=[vmin_map2 - step_size_map2, *list(bins_map2),
                  vmax_map2 + step_size_map2]
        ))
    # Disable clicking legend to remove trace:
    fig.update_layout(legend_itemclick=False)
    fig.update_layout(legend_itemdoubleclick=False)

    # Options for the mode bar.
    # (which doesn't appear on touch devices.)
    plotly_config = {
        # Mode bar always visible:
        # 'displayModeBar': True,
        # Plotly logo in the mode bar:
        'displaylogo': False,
        # Remove the following from the mode bar:
        'modeBarButtonsToRemove': [
            # 'zoom',
            # 'pan',
            'select',
            # 'zoomIn',
            # 'zoomOut',
            'autoScale',
            'lasso2d'
            ],
        # Options when the image is saved:
        'toImageButtonOptions': {'height': None, 'width': None},
        }

    # Write to streamlit:
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=plotly_config
        )
