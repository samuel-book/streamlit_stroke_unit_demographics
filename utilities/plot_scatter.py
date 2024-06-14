"""
Scatter plot of one set of values against another.

Originally from the stroke unit descriptive statistics app.
https://stroke-predictions.streamlit.app/Descriptive_statistics
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress


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

    if c_feature_display_name != ' None':
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
    if c_feature_display_name != ' None':
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
    if c_feature_display_name != ' None':
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
