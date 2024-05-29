import streamlit as st
import pandas as pd
import geopandas


def write_text_from_file(filename, head_lines_to_skip=0):
    """
    Write text from 'filename' into streamlit.
    Skip a few lines at the top of the file using head_lines_to_skip.
    """
    # Open the file and read in the contents,
    # skipping a few lines at the top if required.
    with open(filename, 'r', encoding="utf-8") as f:
        text_to_print = f.readlines()[head_lines_to_skip:]

    # Turn the list of all of the lines into one long string
    # by joining them up with an empty '' string in between each pair.
    text_to_print = ''.join(text_to_print)

    # Write the text in streamlit.
    st.markdown(f"""{text_to_print}""")


def load_region_outlines(outline_name):
    lookup_dict = {
        'Ambulance service': {
            'outline_file': './data/outline_ambulance_service.geojson',
            'outline_names_col': 'ambulance_service'
            },
        'Closest IVT unit': {
            'outline_file': './data/outline_closest_ivt_unit.geojson',
            'outline_names_col': 'closest_ivt_unit'
            },
        'Closest MT unit': {
            'outline_file': './data/outline_closest_mt_unit.geojson',
            'outline_names_col': 'closest_mt_unit'
            },
        'Closest MT transfer': {
            'outline_file': './data/outline_closest_mt_transfer.geojson',
            'outline_names_col': 'closest_mt_transfer'
            },
        'ICB (England) & LHB (Wales)': {
            'outline_file': './data/outline_icb_lhb.geojson',
            'outline_names_col': 'icb_lhb'
            },
        'ISDN': {
            'outline_file': './data/outline_isdn.geojson',
            'outline_names_col': 'isdn'
            },
        'LAD': {
            'outline_file': './data/outline_LAD22NM.geojson',
            'outline_names_col': 'LAD22NM'
            },
    }

    outline_file = lookup_dict[outline_name]['outline_file']
    outline_names_col = lookup_dict[outline_name]['outline_names_col']

    gdf_catchment_lhs = geopandas.read_file(outline_file)
    # Convert to British National Grid:
    gdf_catchment_lhs = gdf_catchment_lhs.to_crs('EPSG:27700')
    # st.write(gdf_catchment['geometry'])
    # # Make geometry valid:
    # gdf_catchment['geometry'] = [
    #     make_valid(g) if g is not None else g
    #     for g in gdf_catchment['geometry'].values
    #     ]
    gdf_catchment_rhs = gdf_catchment_lhs.copy()

    # Make colour transparent:
    gdf_catchment_lhs['colour'] = 'rgba(0, 0, 0, 0)'
    gdf_catchment_rhs['colour'] = 'rgba(0, 0, 0, 0)'
    # Make a dummy column for the legend entry:
    gdf_catchment_lhs['outline_type'] = outline_name
    gdf_catchment_rhs['outline_type'] = outline_name
    return outline_names_col, gdf_catchment_lhs, gdf_catchment_rhs