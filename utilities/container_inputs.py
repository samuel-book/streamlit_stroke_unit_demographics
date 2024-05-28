"""
All of the content for the Inputs section.
"""
# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # for colour maps
import cmasher as cmr  # for additional colour maps
from importlib_resources import files

from stroke_maps.catchment import Catchment  # for unit services


def select_stroke_unit_services(use_msu=True):
    df_unit_services, df_unit_services_full, cols_use = (
        import_stroke_unit_services(use_msu))

    # Display and store any changes from the user:
    df_unit_services = st.data_editor(
        df_unit_services,
        disabled=['postcode', 'stroke_team', 'isdn'],
        height=180  # limit height to show fewer rows
        )

    df_unit_services, df_unit_services_full = update_stroke_unit_services(
        df_unit_services, df_unit_services_full, cols_use)
    return df_unit_services, df_unit_services_full


def import_stroke_unit_services(use_msu=True):
    # Set up stroke unit services (IVT, MT, MSU).
    catchment = Catchment()
    df_unit_services = catchment.get_unit_services()
    # Remove Wales:
    df_unit_services = df_unit_services.loc[df_unit_services['region_type'] != 'LHB'].copy()
    df_unit_services_full = df_unit_services.copy()
    # Limit which columns to show:
    cols_to_keep = [
        'stroke_team',
        'use_ivt',
        'use_mt',
        # 'use_msu',
        # 'transfer_unit_postcode',  # to add back in later if stroke-maps replaces geography_processing class
        # 'region',
        # 'icb',
        'isdn'
    ]
    if use_msu:
        cols_to_keep.append('use_msu')
    df_unit_services = df_unit_services[cols_to_keep]
    # Change 1/0 columns to bool for formatting:
    cols_use = ['use_ivt', 'use_mt']
    if use_msu:
        cols_use.append('use_msu')
    df_unit_services[cols_use] = df_unit_services[cols_use].astype(bool)
    # Sort by ISDN name for nicer display:
    df_unit_services = df_unit_services.sort_values('isdn')

    # Update James Cook University Hospital to have MSU by default:
    if 'use_msu' in df_unit_services.columns:
        df_unit_services.at['TS43BW', 'use_msu'] = True
    return df_unit_services, df_unit_services_full, cols_use


def update_stroke_unit_services(
        df_unit_services,
        df_unit_services_full,
        cols_use
        ):
    # Restore dtypes:
    df_unit_services[cols_use] = df_unit_services[cols_use].astype(int)

    # Update the full data (for maps) with the changes:
    cols_to_merge = cols_use  # + ['transfer_unit_postcode']
    df_unit_services_full = df_unit_services_full.drop(
        cols_to_merge, axis='columns')
    df_unit_services_full = pd.merge(
        df_unit_services_full,
        df_unit_services[cols_to_merge].copy(),
        left_index=True, right_index=True, how='left'
        )

    # Rename columns to match what the rest of the model here wants.
    df_unit_services.index.name = 'Postcode'
    df_unit_services = df_unit_services.rename(columns={
        'use_ivt': 'Use_IVT',
        'use_mt': 'Use_MT',
        'use_msu': 'Use_MSU',
    })
    return df_unit_services, df_unit_services_full


def set_up_colours(
        v_min,
        v_max,
        step_size,
        use_diverging=False,
        cmap_name='inferno',
        v_name='v',
        ):

    if cmap_name.endswith('_r_r'):
        # Remove the double reverse reverse.
        cmap_name = cmap_name[:-2]

    # Make a new column for the colours.
    v_bands = np.arange(v_min, v_max + step_size, step_size)
    if use_diverging:
        # Remove existing zero:
        ind_z = np.where(abs(v_bands) < step_size * 0.01)[0]
        if len(ind_z) > 0:
            ind_z = ind_z[0]
            v_bands = np.append(v_bands[:ind_z], v_bands[ind_z+1:])
        # Add a zero-ish band.
        ind = np.where(v_bands >= -0.0)[0][0]
        zero_size = step_size * 0.01
        v_bands_z = np.append(v_bands[:ind], [-zero_size, zero_size])
        v_bands_z = np.append(v_bands_z, v_bands[ind:])
        v_bands = v_bands_z
        v_bands_str = make_v_bands_str(v_bands, v_name=v_name)

        # Update zeroish name:
        v_bands_str[ind+1] = '0.0'
    else:
        v_bands_str = make_v_bands_str(v_bands, v_name=v_name)

    colour_map = make_colour_map_dict(v_bands_str, cmap_name)

    # Link bands to colours via v_bands_str:
    colours = []
    for v in v_bands_str:
        colours.append(colour_map[v])

    # Add an extra bound at either end (for the "to infinity" bit):
    v_bands_for_cs = np.append(v_min - step_size, v_bands)
    v_bands_for_cs = np.append(v_bands_for_cs, v_max + step_size)
    # Normalise the data bounds:
    bounds = (
        (np.array(v_bands_for_cs) - np.min(v_bands_for_cs)) /
        (np.max(v_bands_for_cs) - np.min(v_bands_for_cs))
    )
    # Add extra bounds so that there's a tiny space at either end
    # for the under/over colours.
    # bounds_for_cs = [bounds[0], bounds[0] + 1e-7, *bounds[1:-1], bounds[-1] - 1e-7, bounds[-1]]
    bounds_for_cs = bounds

    # Need separate data values and colourbar values.
    # e.g. translate 32 in the data means colour 0.76 on the colourmap.

    # Create a colour scale from these colours.
    # To get the discrete colourmap (i.e. no continuous gradient of
    # colour made between the defined colours),
    # double up the bounds so that colour A explicitly ends where
    # colour B starts.
    colourscale = []
    for i in range(len(colours)):
        colourscale += [
            [bounds_for_cs[i], colours[i]],
            [bounds_for_cs[i+1], colours[i]]
            ]

    colour_dict = {
        'diverging': use_diverging,
        'v_min': v_min,
        'v_max': v_max,
        'step_size': step_size,
        'cmap_name': cmap_name,
        'v_bands': v_bands,
        'v_bands_str': v_bands_str,
        'colour_map': colour_map,
        'colour_scale': colourscale,
        'bounds_for_colour_scale': bounds_for_cs,
        # 'zero_label': '0.0',
        # 'zero_colour': 
    }
    return colour_dict


def make_colour_map_dict(v_bands_str, cmap_name='viridis'):
    # Get colour values:
    try:
        # Matplotlib colourmap:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        # CMasher colourmap:
        cmap = plt.get_cmap(f'cmr.{cmap_name}')

    cbands = np.linspace(0.0, 1.0, len(v_bands_str))
    colour_list = cmap(cbands)
    # # Convert tuples to strings:
    colour_list = np.array([
        f'rgba{tuple(c)}' for c in colour_list])
    # Sample the colour list:
    colour_map = [(c, colour_list[i]) for i, c in enumerate(v_bands_str)]

    # # Set over and under colours:
    # colour_list[0] = 'black'
    # colour_list[-1] = 'LimeGreen'

    # Return as dict to track which colours are for which bands:
    colour_map = dict(zip(v_bands_str, colour_list))
    return colour_map


def make_v_bands_str(v_bands, v_name='v'):
    """Turn contour ranges into formatted strings."""
    v_min = v_bands[0]
    v_max = v_bands[-1]

    v_bands_str = [f'{v_name} < {v_min:.3f}']
    for i, band in enumerate(v_bands[:-1]):
        b = f'{band:.3f} <= {v_name} < {v_bands[i+1]:.3f}'
        v_bands_str.append(b)
    v_bands_str.append(f'{v_max:.3f} <= {v_name}')

    v_bands_str = np.array(v_bands_str)
    return v_bands_str


def make_colourbar_display_string(cmap_name, char_line='█', n_lines=20):
    try:
        # Matplotlib colourmap:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        # CMasher colourmap:
        cmap = plt.get_cmap(f'cmr.{cmap_name}')

    # Get colours:
    colours = cmap(np.linspace(0.0, 1.0, n_lines))
    # Convert tuples to strings:
    colours = (colours * 255).astype(int)
    # Drop the alpha or the colour won't be right!
    colours = ['#%02x%02x%02x' % tuple(c[:-1]) for c in colours]

    line_str = '$'
    for c in colours:
        # s = f"<font color='{c}'>{char_line}</font>"
        s = '\\textcolor{' + f'{c}' + '}{' + f'{char_line}' + '}'
        line_str += s
    line_str += '$'
    return line_str


def select_colour_maps(cmap_names, cmap_diff_names):
    cmap_displays = [
        make_colourbar_display_string(cmap_name, char_line='█', n_lines=15)
        for cmap_name in cmap_names
        ]
    cmap_diff_displays = [
        make_colourbar_display_string(cmap_name, char_line='█', n_lines=15)
        for cmap_name in cmap_diff_names
        ]

    try:
        cmap_name = st.session_state['cmap_name']
        cmap_diff_name = st.session_state['cmap_diff_name']
    except KeyError:
        cmap_name = cmap_names[0]
        cmap_diff_name = cmap_diff_names[0]
    cmap_ind = cmap_names.index(cmap_name)
    cmap_diff_ind = cmap_diff_names.index(cmap_diff_name)

    cmap_name = st.radio(
        'Colour display for left-hand map',
        cmap_names,
        captions=cmap_displays,
        index=cmap_ind,
        key='cmap_name'
    )

    cmap_diff_name = st.radio(
        'Colour display for difference map',
        cmap_diff_names,
        captions=cmap_diff_displays,
        index=cmap_diff_ind,
        key='cmap_diff_name'
    )
    return cmap_name, cmap_diff_name
