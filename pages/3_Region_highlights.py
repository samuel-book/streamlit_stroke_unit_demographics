import streamlit as st
import pandas as pd
import stroke_maps.load_data
import plotly.graph_objects as go

from utilities.fixed_params import page_setup

page_setup()

st.markdown(
'''
# Region highlights

Pick out your favourite regions and see how they compare with the rest.

The three vertical lines on each violin plot show the minimum, median, and maximum values in the violin.
'''
)

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
with st.sidebar:
    st.markdown('### Data selection')
    region_type_str = st.selectbox(
        'Group data at this level',
        region_type_dict.keys(),
        index=1  # index of default option
        )
region_type = region_type_dict[region_type_str]

# Import the full demographic data:
df_demog = pd.read_csv(
    f'./data/collated_data_regional_{region_type}.csv', index_col=0)
# Drop missing data:
# (this is particularly aimed at the Welsh areas when the
# region type is ISDN, as there isn't an ISDN label for Wales.)
df_demog = df_demog.dropna(axis='rows')
# Convert proportions (0 -> 1) to percentages (0 -> 100)
# to match the IMD.
cols_prop = [c for c in df_demog if 'proportion' in c]
df_demog[cols_prop] = df_demog[cols_prop] * 100.0

# Stroke unit info:
df_units = stroke_maps.load_data.stroke_unit_region_lookup()

if (('_ivt_' in region_type) | ('_mt_' in region_type)):
    # The data is saved with stroke unit postcodes as the only
    # identifier. Load in a postcode-name lookup for display.
    cols_to_merge = ['stroke_team', 'ssnap_name']
    df_demog = pd.merge(
        df_units[cols_to_merge], df_demog,
        left_index=True, right_index=True, how='right'
        )
    # Set the index to be unit names rather than postcodes:
    df_demog = df_demog.reset_index()
    df_demog = df_demog.set_index('stroke_team')

# Options for region(s) to highlight:
regions_to_highlight = st.multiselect(
    'Regions to highlight',
    options=sorted(df_demog.index)
)

cols_for_violin = df_demog.columns

from utilities.fixed_params import \
    cols_abs, cols_prettier_dict, cols_prettier_reverse_dict
cols_to_remove = [
    'closest_mt_transfer', 'closest_mt_unit', 'closest_ivt_unit',
    'ssnap_name', 'polygon_area_km2'
    ]
remove_abs_columns_bool = True
if remove_abs_columns_bool:
    # Remove columns that are absolute numbers instead of ratios:
    cols_to_remove += cols_abs

cols_for_violin = [c for c in cols_for_violin if c not in cols_to_remove]

for c, col in enumerate(cols_for_violin):
    y0 = 0

    try:
        col_pretty = cols_prettier_dict[col]
    except KeyError:
        col_pretty = col

    # Calculate min/max/median:
    c_min = df_demog[col].min()
    c_max = df_demog[col].max()
    c_median = df_demog[col].median()

    # Pick out data for the regions to highlight:
    data_to_highlight = [df_demog.loc[r, col] for r in regions_to_highlight]

    fig = go.Figure()
    # Main violin trace:
    fig.add_trace(go.Violin(
        x=df_demog[col],
        y0=y0,
        name=col_pretty,
        line=dict(color='grey'),
        hoveron='points',  # Switch off the hover label
        orientation='h',
        points=False,
        showlegend=False
        ))
    # Min/max/median scatter markers:
    fig.add_trace(go.Scatter(
        x=[c_min, c_median, c_max],
        y=[y0]*3,
        line_color='black',
        marker=dict(size=20, symbol='line-ns-open'),
        showlegend=False,
        hoverinfo='skip',
        ))
    # Highlighted team scatter:
    for r, region in enumerate(regions_to_highlight):
        fig.add_trace(go.Scatter(
            x=[data_to_highlight[r]],
            y=[y0],
            name=region,
            mode='markers',
            marker=dict(
                line=dict(color='black', width=1),
                size=10
                )
        ))

    # Figure setup:
    fig.update_layout(xaxis_title=col_pretty)
    # Set y-axis ticks to a blank label:
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=[y0],
            ticktext=['']
        ))

    # Plotly setup:
    plotly_config = {
        # Mode bar always visible:
        # 'displayModeBar': True,
        # Plotly logo in the mode bar:
        'displaylogo': False,
        # Remove the following from the mode bar:
        'modeBarButtonsToRemove': [
            'zoom', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale',
            'lasso2d'
            ],
        # Options when the image is saved:
        'toImageButtonOptions': {'height': None, 'width': None},
        }


    # Disable zoom and pan:
    fig.update_layout(
        # Left subplot:
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        # Right subplot:
        xaxis2=dict(fixedrange=True),
        yaxis2=dict(fixedrange=True)
        )

    # Turn off legend click events
    # (default is click on legend item, remove that item from the plot)
    fig.update_layout(legend_itemclick=False)
    # Only change the specific item being clicked on, not the whole
    # legend group:
    # # fig.update_layout(legend=dict(groupclick="toggleitem"))

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)
