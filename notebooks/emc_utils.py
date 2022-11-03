import os
import sys
from pathlib import Path
import math
import requests
import warnings
warnings.filterwarnings("ignore")
from requests.exceptions import HTTPError
from datetime import datetime, timezone
from IPython.display import Markdown, display
from ipyfilechooser import FileChooser
import matplotlib.pyplot as plt
import ipywidgets as wg
import xarray as xr
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
cartopy.config['data_dir'] = os.getenv('CARTOPY_DIR', cartopy.config.get('data_dir'))
    
    
def init_run_args():
    """Initialize the container for the run parameters."""
    run_args = dict()
    this_path = Path(os.getcwd())
    run_args['root_path'] = this_path.parent.absolute()
    run_args['output_dir'] = 'output'
    run_args['data_dir'] = 'samples'
    run_args['data_path'] = os.path.join(run_args['root_path'], run_args['data_dir'])
    run_args['model_filename'] = 'emc-jnp-iMUSH-localEQ-Ulberg-2020.nc'
    run_args['save_tag'] = 'demo'
    run_args['frame_size'] = (800, 600)
    run_args['metadata_format'] = 'geocsv'
    run_args['default_x_var'] = 'longitude'
    run_args['x_step'] = 30
    run_args['csv_x_step'] = 1
    run_args['y_step'] = 30
    run_args['csv_y_step'] = 1
    run_args['z_step'] = 20
    run_args['csv_z_step'] = 1
    run_args['default_y_var'] = 'latitude'
    run_args['default_z_var'] = 'depth'
    run_args['z_exaggeration'] = 10
    run_args['save_plots'] = True
    run_args['save_data'] = True
    run_args['max_exaggeration'] = 30
    run_args['csv_mode'] = 'None'
    run_args['projection'] = ccrs.PlateCarree()
    
    # matplotlib colormap to use. For available colormaps see:
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    run_args['colormap'] ='jet_r'

    # Notebook's version.
    run_args['version'] = 'r0.1'

    # GeoCSV version and delimiter to use.
    run_args['geocsv_version'] = 'GeoCSV2.0'
    run_args['delimiter'] = '|'

    # Some control paramaters.
    run_args['valid_modes'] = {'depth': 'km', 'single': ''}
    run_args['valid_metadata_format'] = ['geocsv', 'netcdf']
    run_args['valid_dimensions'] = {2:'2D', 3: '3D'}
    run_args['file_extensions'] = {'netcdf': '.nc', 'csv': '.csv', 'image': '.png', 'zip': '.zip'}
    
    return run_args


def set_run_arg(run_arg, arg, value):
    run_arg[arg] = value
    return run_arg
    
def on_change(change):
    print(change)
    if change['type'] == 'change' and change['name'] == 'value':
        run_args['data_dir'] = change['new']
        
def handle_file_change(change):
    caption.value = 'The slider value is ' + (
        'negative' if change.new < 0 else 'nonnegative'
    )
    
def init_defaults():
    """Initialize the default values."""""
    default_extensions = {'netcdf': '.nc', 'csv': '.csv', 'image': '.png', 'zip': '.zip'}
    widgets = {'delimiter': wg.Dropdown(options=['|', ',', 'space', 'tab'], 
                                        value='|', description='Delimiter:', disabled=False),
               'data_dir': wg.Text(value='data', description='data_dir', disabled=False),
               'samples_dir': wg.Text(value='samples', description='samples_dir', disabled=False),
               'samples_prefix': wg.Text(value='emc-jnp-', description='samples_prefix', disabled=False),
               'default_extensions': wg.Text(value=f"{list(default_extensions.items())}",
                                             description='default file extensions', disabled=True),
              }
    return default_extensions, widgets

def calc_xy_ratio(x, y, factor=1):
    """Compute x/y ratio."""
    # Get x and y limits.
    x_left, x_right = x
    y_low, y_high = y

    # Compute aspect ratio.
    xy_ratio = abs((x_right - x_left) / (y_high - y_low)) * factor
    return xy_ratio


def great_circle_distance(lat_1, lon_1, lat_2, lon_2, mode='km'):
    """Calculate the Great Circle ditsnce"""
    radius_deg = math.pi/180.
    theta_1 = (90.0 - lat_1) * radius_deg
    theta_2 = (90.0 - lat_2) * radius_deg
    phi_1 = lon_1 * radius_deg
    phi_2 = lon_2 * radius_deg
    angle = math.acos(min(1.0, math.sin(theta_1) * math.sin(theta_2) *
                          math.cos(phi_2 - phi_1) + math.cos(theta_1) * math.cos(theta_2)))
    distance_deg = angle / radius_deg
    distance_km = 110.567 * distance_deg
    if mode == 'degrees':
        return distance_deg
    if mode == 'km':
        return distance_km

    return distance_deg, distance_km

    
def plot_hslice(zslice, x_variable, x_min, x_max, y_variable, y_min, y_max, title, colormap, projection):
    # Axes with Cartopy projection
    ax = plt.axes(projection=projection)
    # and extent
    ax.set_extent([x_min, x_max, y_min, y_max], projection)

    # Plot lat/lon grid 
    gl = ax.gridlines(crs=projection, draw_labels=True,
                      linewidth=0.1, color='k', alpha=1, 
                      linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8} 

    # Add map features with Cartopy 
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', 
                                                edgecolor='face', 
                                                facecolor='None'))
    ax.coastlines(linewidth=1)
    ax.set_xlabel(x_variable)
    ax.set_ylabel(y_variable)  
    zslice.plot(cmap=colormap, cbar_kwargs={'shrink': 0.5})
    # plt.suptitle(title, y=0.9)
    return plt
    
        
        
def plot_model_area(x_variable, x_values, y_variable, y_values, projection):
    """Plot a map of the model coverage."""
    # Axes with Cartopy projection
    ax = plt.axes(projection=projection)
    # and extent
    #ax.set_extent([min(x_values), max(x_values), min(y_values), max(y_values)], projection)
    ax.set_extent([-180, 180, -90, 90], projection)
    # Plot lat/lon grid 
    gl = ax.gridlines(crs=projection, draw_labels=True,
                      linewidth=0.1, color='k', alpha=1, 
                      linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8} 

    # Add map features with Cartopy 
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', 
                                                edgecolor='face', 
                                                facecolor='None'))
    ax.coastlines(linewidth=0.8)
    ax.set_xlabel(x_variable)
    ax.set_ylabel(y_variable)
    xx = list()
    yy = list()
    for _x in x_values:
        for _y in y_values:
            xx.append(_x)
            yy.append(_y)
    
    plt.scatter(xx, yy, s=0.3, alpha=0.05, c='red')
    plt.title("Model Coverage", y=1.05, color='red')
    
def pid_to_url(attrs, pid):
    """Convert a DOI string to the corresponding URL see:
    https://doi.org/api/handles/10.17611/dp/emc.2022.wus256.1?type=URL
    for style."""
    if pid in attrs:
        repository_doi = attrs[pid]
        repository_doi = repository_doi.lower().strip()
        if repository_doi.startswith('doi:'):
            doi = repository_doi.replace('doi:', 'https://doi.org/')
            doi_link = repository_doi.replace('doi:', 'https://doi.org/api/handles/')
            doi_link = f"{doi_link}?type=URL"
            try:
                response = requests.get(url= doi_link ,allow_redirects=True )
                response.raise_for_status()
                json_response = response.json()
                if int(json_response['responseCode']) != 1:
                    return None, None
                return doi, json_response['values'][0]['data']['value']
            except HTTPError as http_err:
                message(f"Other error occurred: {http_err}", color='red')
                return None, None
            except Exception as err:
                message(f"Other error occurred: {err}", color='red')
                return None, None
     
    else:
        return None, None
        
        
def get_model_info(attrs):
    """Extract model information form the dataset attrs"""
    info_text = list()
    if 'id' in attrs:
        model_name = attrs['id']
        info_text.append(f"Model: {model_name}")
        
    if 'title' in attrs:
        model_title = attrs['title']
        info_text.append(f"Title: {model_title}")
        
    if 'summary' in attrs:
        model_summary = attrs['summary']
        info_text.append(f"Summary: {model_summary}")
    reference_pid = pid_to_url(attrs, 'reference_pid')
    if reference_pid is not None:
        reference_doi, reference_url = pid_to_url(attrs, 'reference_pid')
        if reference_doi is not None:
            info_text.append(f"Reference DOI: {reference_url}")
        
    model_doi, model_url = pid_to_url(attrs, 'repository_pid')
    if model_doi is not None:
        info_text.append(f"Repository DOI: {model_doi}")
        info_text.append(f"Repository Page:: {model_url}")
    return info_text 

def message(body, color='blue', weight='bold'):
    """Display a Markdown test"""
    display(Markdown(f"\n\n<span style='color:{color}; font-weight:{weight}'>{body}</span>\n\n"))

        
def ds_2_csv(ds, filename, delimiter):
    """Save a dataset to a CSV file"""
    _df = ds.to_dataframe()
    _df.to_csv(filename, sep=delimiter)
        

def save_files(save_data, save_plot, ds, plt, var_tag, variable, delimiter, base_filename, extensions, out_dir):
    save_file = os.path.join(out_dir, '_'.join([base_filename, variable, var_tag,]))
    if save_data:
        ds_2_csv(ds, f"{save_file}{extensions['csv']}", delimiter)

    if save_plot:
        plt.savefig(f"{save_file}{extensions['image']}")
        
            
def get_dsv(ds, variable):
    """Extract data from a dataset for a given variable"""
    
    if variable not in ds:
        raise UserWarning(f"selected 'VARIABLE' of '{variable}' is not a valid model variable.\n"
                          f"VARIABLE must be one of {list(ds)}")
    return ds[variable]


def get_range_values(values, value_range):
    """Extract values within a range from an array"""
    v = list()
    vmin, vmax = value_range
    for val in values:
        if vmin <= val <= vmax:
            v.append(val)
    return v
        


def get_min_max(var, values, ranges):
    """Check and set value ranges"""
    val_min = min(values)
    val_max = max(values)
        
    if ranges:
        v_min, v_max = ranges
        v_data = list(filter(lambda values: v_max >= values >= v_min, values))
    else:
        v_min, v_max = val_min, val_max
        v_data = values
        
    if v_min >= v_max or v_min >= val_max:
        raise ValueError(f"Invalid range {ranges} for variable {var} with range of {val_min:0.2f} to {val_max:0.2f}"
                         f"\nStop! Please update the range with proper min and max values.")
    print(f"{var} range {v_min:0.2f} to {v_max:0.2f} / {val_min:0.2f} to {val_max:0.2f}")
    return v_data, v_min, v_max        
        
def get_ranges(x_var, x_range, x_vals, y_var, y_range, y_vals, z_var, z_range, z_vals, ndim):
    """Set variable ranges based on the viven values or data ranges."""
    x_data, x_min, x_max = get_min_max(x_var, x_vals, x_range)
    y_data, y_min, y_max = get_min_max(y_var, y_vals, y_range)
    if ndim == 3:
        z_data, z_min, z_max = get_min_max(z_var, z_vals, z_range)
    else:
        z_data, z_min, z_max = None, None, None

    return x_data, x_min, x_max, y_data, y_min, y_max, z_data, z_min, z_max

    
def get_var_name(**variables):
    """Retuen a variable name's as a string'"""
    
    _list = [x for x in variables]
    return _list[0]
    
    
def make_path(directory):
    """ Checks a directory and creates it if it does not exist.
     If needed, creates all the parent directories.
    :param directory: path to check
    :return: path
    """
    # Path must be an absolute path
    if not os.path.isabs(directory):
        print("[ERR]: path must be an absolute path")
        return None

    # Create the directories.
    path = os.path.abspath(directory)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_variable_attributes(model_data, header, variable, variable_name, spacer='\t', mode='column'):
    """add  variable attributes to the header

    Keyword arguments:
    model_data: Dataset instance of the model_file
    header: GeoCSV header variable for the model
    variable : variable to add
    variable_name: variable name used to represent this variable

    Return values:
    the geoCSV header for the model
    """
    if mode == 'column':
        header.append(f"{spacer}# {variable_name.replace('_', '-')}_column: {variable}\n")
    header.append(f"{spacer}# {variable_name.replace('_', '-')}_variable: {variable}\n")
    header.append(f"{spacer}# {variable_name.replace('_', '-')}_dimensions: {len(model_data.variables[variable].shape)}\n")

    for attr, value in vars(model_data.variables[variable]).items():
        if '_range' in attr:
            header.append(f"{spacer}# {variable_name.replace('_', '-')}_{attr}: {value[0]},{value[1]}\n")
        else:
            header.append(f"{spacer}# {variable_name.replace('_', '-')}_{attr}: {value}\n")
    return header


def get_model_header(model_file, model_data, run_args, var_list=list(), spacer='\t', output_mode='single'):
    """create GeoCSV header for the model

    Keyword arguments:
    model_file: the netCDF model file name
    model_data: Dataset instance of the model_file

    Return values:
    the geoCSV header for the model
    """
    header = list()
    ndim = run_args['ndim']
    # GeoCSV header
    header.append(f"{spacer}# dataset: {run_args['geocsv_version']}\n")
    header.append(f"{spacer}# created: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
    header.append(f"{spacer}# netCDF_file: {os.path.basename(model_file)}\n")
    header.append(f"{spacer}# delimiter: {run_args['delimiter']}\n")

    # global attributes
    history_done = False
    history = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')} Converted to GeoCSV from {model_file}"
    for attr, value in vars(model_data).items():
        if isinstance(value, str):
            value = value.replace('\n', '; ')
        if attr.lower() == 'history':
            value = f'{history}; {value}'
            history_done = True
        header.append(f'{spacer}# global_{attr}: {value}\n')

    if not history_done:
        header.append(f'{spacer}# global_history: {history}\n')

    # variables

    header.append(f"{spacer}#\n")
    # Need coordinate variables for both output types.
    #if output_mode == 'single':
    if True:
        for var_index, var_value in enumerate(list(model_data.variables)):
            if not var_list or var_value in var_list:
                mode = 'column'
                if output_mode == 'depth' and var_value == run_args['z_variable']:
                    mode = ''
                head = get_var_header(model_data, var_value, spacer=spacer, mode=mode)
                header.append(f"{head}\n")
            
    return ''.join(header)


def get_var_header(model_data, var, spacer='\t', mode='column'):
    """create GeoCSV header for a variable

    Keyword arguments:
    model_data: Dataset instance of the model_file
    var: the netCDF model file variable

    Return values:
    the geoCSV header for the variable
    """
    header = list()
    header = get_variable_attributes(model_data, header, var, var, spacer=spacer, mode=mode)
    return ''.join(header)


def display_header(model_file, model_data, run_args, metadata_format=None):
    """extract and display netCDF and GeoCSV header information

    Keyword arguments:
    model_data: Dataset instance of the model_file

    Return values:
    None, output of the header information
    """
    if metadata_format is None:
        metadata_format = run_args['metadata_format']
    elif metadata_format.lower() == 'none':
        return
    
    if metadata_format.lower() == 'netcdf':
        # netCDF header
        #print(, flush=True)
        display(Markdown("### netCDF Style:\n\n"))

        # dimension information.
        nc_dims = [dim for dim in model_data.dimensions]  # list of netCDF dimensions
        print("\tdimensions:", flush=True)
        for dim in nc_dims:
            print(f"\t\t{model_data.dimensions[dim].name} {model_data.dimensions[dim].size}", flush=True)

        # variable information.
        nc_vars = [var for var in model_data.variables]  # list of nc variables

        print("\n\tvariables:", flush=True)
        for var in nc_vars:
            if var not in nc_dims:
                print(f"\t\t{var}:", flush=True)
                for attr, value in vars(model_data.variables[var]).items():
                    print(f"\t\t\t{attr} = {value}", flush=True)

        # global attributes
        print("\"\n\tglobal attributes:", flush=True)
        for attr, value in vars(model_data).items():
            if isinstance(value, str):
                value = value.replace('\n', '; ')
            if isinstance(value, float):
                value = f"{value:0.2f}"
            print(f"\t\t\t{attr} = {value}", flush=True)
    else:
        # GeoCSV header
        display(Markdown("\n\n### GeoCSV Style:\n\n"))
        print(f"{get_model_header(model_file, model_data, run_args)}\n\n", flush=True)


def make_model_geocsv(run_args, output_mode):
    """create GeoCSV file from a netCDF model file

    Keyword arguments:
    model_file: the netCDF model file name
    """

    data_header = list()
    out_dir = run_args['output_path']
    delimiter = run_args['delimiter']
    model_data = run_args['model_data']
    model_file = run_args['model_filename']
    data_dir = run_args['data_path']
    model_file = os.path.join(data_dir, model_file)
    base_filename = run_args['base_filename']
    default_extensions = run_args['file_extensions']
    data_variables = run_args['csv_data_var']
    valid_modes = run_args['valid_modes']
    model_variables = run_args['model_variables']
    
    x_variable = run_args['x_variable']
    x_range = run_args['csv_x_range']
    x_values = run_args['x_values']
    x_step = run_args['csv_x_step']
    
    y_variable = run_args['y_variable']
    y_range = run_args['csv_y_range']
    y_values = run_args['y_values']
    y_step = run_args['csv_y_step']
    
    variable_list = run_args['csv_data_var']
    if not variable_list:
        variable_list = data_variables.copy()
    coordinate_variables = run_args['coordinate_variables']
    ndim = run_args['ndim']
    
    if ndim['model'] > 2:
        z_variable = run_args['z_variable']
        z_range = run_args['csv_z_range']
        z_values = run_args['z_values']
        z_step = run_args['csv_z_step']
    else:
        z_variable = None
        z_range = None
        z_values = None
        z_step = 1
    emcin = {}
        
    x_data, x_min, x_max, y_data, y_min, y_max, z_data, z_min, z_max = get_ranges(x_variable, x_range, x_values, 
                                                                                  y_variable, y_range, y_values, 
                                                                                  z_variable, z_range, z_values, 
                                                                                  ndim['model'])
    x = x_data[::x_step]
    y = y_data[::y_step]
    if z_variable is not None:
        z = z_data[::z_step]
    else:
        z = [1]
            
    
    # The standard order is (Z, Y, X) or (depth, latitude, longitude
    output_data = list()
    depth_index = dict()
    lat_index = dict()
    lon_index = dict()

    # Go through each depth.
    index = [-1, -1, -1]
    do_init = True
    var_done_list = list()
    z_count = 0
    for k, this_z in enumerate(z):
        
        # Get the model header:
        # For the single output file option, if requested or if the model is 2D.
        if (output_mode == 'single' and do_init) or ndim['model'] == 2:
            data_header = list()
            output_file = os.path.join(out_dir, f"{base_filename}{default_extensions['csv']}")
            fp = open(os.path.join(out_dir, output_file), 'w')
            print(f'[INFO] Output file: {output_file}', flush=True)
            fp.write(get_model_header(model_file, model_data, run_args, var_list=coordinate_variables, spacer=''))
            if ndim['model'] == 2:
                data_header.append(f'{y_variable}{delimiter}{x_variable}')
            else:
                data_header.append(f'{y_variable}{delimiter}{x_variable}{delimiter}{z_variable}')
            z_count = 0
            x_count, y_count = -1, -1
            do_init = False
        # For the depth-based output files.
        elif output_mode == 'depth':
            output_data = list()
            data_header = list()
            output_file = os.path.join(
                f"{base_filename}_{this_z:0.2f}_{valid_modes[output_mode]}{default_extensions['csv']}")
            fp = open(os.path.join(out_dir, output_file), 'w')
            print(f'[INFO] Output file: {output_file}', flush=True)
            fp.write(get_model_header(model_file, model_data, run_args, var_list=coordinate_variables, spacer='', output_mode=output_mode))
            data_header.append(f'# depth: {this_z:0.2f}\n')
            data_header.append(f'{y_variable}{delimiter}{x_variable}')
            x_count, y_count = -1, -1
            z_count = 0
        else:
            z_count += 1

        # For 3D models, show progress by depth.
        if this_z is not None:
            # Show the progress.
            if z_count == 0:
                zero_depth = this_z
                z_count += 1
            elif z_count == 1:
                z_count += 1
                if ndim['model'] > 2:
                    print(f'[INFO] Depth range: {z[0]:0.2f} to {z[-1]:0.2f}', flush=True)
                print(f"{zero_depth:0.2f}, {this_z:0.2f},", end=' ', flush=True)
            else:
                print(f"{this_z:0.2f},", end=' ', flush=True)

        # Go through each latitude and convert to string to preserve precision.
        for i, this_y in enumerate(y[::y_step]):
            _lat = str(this_y)
            
            # Go through each longitude and convert to string to preserve precision.
            for j, this_x in enumerate(x[::x_step]):
                x_count += 1
                _lon = str(this_x)
                
                # Data header line.
                if output_mode == 'single' and ndim['model'] == 3:
                    _depth = str(this_z)
                    output_data.append(f'{_lat:s}{delimiter}{_lon:s}{delimiter}{_depth:s}')
                else:
                    output_data.append(f'{_lat:s}{delimiter}{_lon:s}')

                # Go through each model variable.
                for var_index, var_value in enumerate(model_variables):
                    var = var_value.encode('ascii', 'ignore').decode("utf-8")
                    # Model variables only.
                    if var in variable_list:
                        # Do this for the first point, when all indices are zero.
                        if (output_mode == 'single' and i == 0 and j == 0 and k == 0) or (output_mode == 'depth' and i == 0 and j == 0):
                            # Turn off column notation for depth in depth files.
                            mode = 'column'
                            if output_mode == 'depth' and var == z_variable:
                                mode = ''
                            fp.write(get_var_header(model_data, var, spacer='', mode=mode))
                            data_header.append(f"{delimiter}{var}")

                        # find the variable dimension ordering
                        if var not in lat_index:
                            print(f"[INFO] Processing variable: {var_value}", flush=True)
                            indices_list = ''
                            for dim_index, dim in enumerate(model_data.variables[var].dimensions):
                                if dim.encode('ascii', 'ignore').decode("utf-8") == z_variable:
                                    depth_index[var] = dim_index
                                    indices_list = f"{indices_list}{len(z)} {z_variable}s index:{dim_index}  "
                                elif dim.encode('ascii', 'ignore').decode("utf-8") == x_variable:
                                    lon_index[var] = dim_index
                                    indices_list = f"{indices_list}{len(x)} {x_variable}s index:{dim_index}  "
                                elif dim.encode('ascii', 'ignore').decode("utf-8") == y_variable:
                                    lat_index[var] = dim_index
                                    indices_list = f"{indices_list}{len(y)} {y_variable}s index:{dim_index}  "
                                else:
                                    print(f'\n[ERR] Invalid dimensions "{dim}" in variable {var}')
                                    sys.exit(2)
                            print(f"[INFO] {indices_list}", flush=True)
                            # A few checks.
                            if var not in lat_index and var not in lon_index:
                                print(f'\n[ERR] problem reading x and y variables "{x_variable}, {y_variable}"')
                                sys.exit(2)

                            if var not in depth_index is None and ndim[var] > 2:
                                print(f'\n[ERR] problem reading the depth variables "{z_variable}" for variable {var}')
                                sys.exit(2)

                            # Assign the variable's data values.
                            if var not in emcin:
                                try:
                                    emcin[var] = model_data.variables[var][:]
                                except Exception as err:
                                    print(f'\n[ERR] problem reading variable "{var}"')
                                    print('{0}\n'.format(err))
                                    sys.exit(2)
                        if var in depth_index:
                            index[depth_index[var]] = k
                        index[lat_index[var]] = i
                        index[lon_index[var]] = j
                        # nan values, otherwise we write string to preserve the precision
                        if ndim[var] == 2:
                            if math.isnan(emcin[var][index[0]][index[1]]):
                                output_data.append(f"{delimiter}{math.nan}")
                            else:
                                # Conversion to string is done to preserve precision
                                output_data.append(f"{delimiter}{str(emcin[var][index[0]][index[1]]):s}")
                        else:
                            if math.isnan(emcin[var][index[0]][index[1]][index[2]]):
                                output_data.append(f"{delimiter}{math.nan}")
                            else:
                                # Conversion to string is done to preserve precision
                                output_data.append(f"{delimiter}{str(emcin[var][index[0]][index[1]][index[2]]):s}")

                output_data.append('\n')
        if output_mode == 'depth':
            fp.write(f'{"".join(data_header)}\n')
            fp.write(''.join(output_data))
            fp.close()

    if output_mode == 'single':
        fp.write(f'{"".join(data_header)}\n')
        fp.write(''.join(output_data))
        fp.close()

