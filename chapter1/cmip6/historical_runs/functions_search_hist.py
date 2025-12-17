"""
CMIP6 Model Data Processor

This module provides functions to find, process, and categorize CMIP6 climate model data
for ensemble analysis.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any
from SIT_search_for_hist import find_ensemble_list_multi_var


def setup_logging(level=logging.INFO):
    """
    Configure logging with consistent format.
    
    Parameters
    ----------
    level : int
        Logging level (default: logging.INFO)
    """
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_cmip6_config(
    mip_id: str = 'CMIP',
    experiment: str = 'historical',
    total_years: int = 50,
    subtract_annual_cycle: bool = True
) -> Dict[str, Any]:
    """
    Get configuration dictionary for CMIP6 data processing.
    
    Parameters
    ----------
    mip_id : str
        MIP identifier (default: 'CMIP')
    experiment : str
        Experiment name (default: 'historical')
    total_years : int
        Number of years to analyse (default: 50)
    subtract_annual_cycle : bool
        Whether to subtract annual cycle (default: True)
    
    Returns
    -------
    dict
        Configuration dictionary with paths and parameters
    """
    config = {
        'mip_id': mip_id,
        'experiment': experiment,
        'base_dir_badc': f'/badc/cmip6/data/CMIP6/{mip_id}/',
        'base_dir_output': '/gws/nopw/j04/arctic_connect/cturrell/CMIP6/historical',
        'version_name': 'latest',
        'total_time_span_required': total_years,
        'subtract_annual_cycle': subtract_annual_cycle,
        'level_type': '6hrPlevPt',
        'var_name_dict': {
            'ua': {'data_type': '6hrPlevPt'},
            'va': {'data_type': '6hrPlevPt'},
        }
    }
    return config


def choose_ensemble_member(ens_list: List[str], model_name: str) -> str:
    """
    Choose ensemble member from available list with priority order.
    
    Parameters
    ----------
    ens_list : list
        List of available ensemble IDs
    model_name : str
        Name of the model
    
    Returns
    -------
    str
        Selected ensemble member ID
    
    Raises
    ------
    NotImplementedError
        If no suitable ensemble member can be chosen
    """
    if len(ens_list) == 1:
        return ens_list[0]
    elif 'r1i1p1f1' in ens_list:
        return 'r1i1p1f1'
    elif 'r1i1p1f2' in ens_list:
        logging.warning(f'-    Using f2 for {model_name}')
        return 'r1i1p1f2'
    else:
        raise NotImplementedError(
            f'Not sure which ens_id to choose for {model_name}'
        )


def choose_grid(grid_list: List[str], model_name: str) -> str:
    """
    Choose grid type from available list with priority order.
    
    Parameters
    ----------
    grid_list : list
        List of available grid types
    model_name : str
        Name of the model
    
    Returns
    -------
    str
        Selected grid type
    
    Raises
    ------
    NotImplementedError
        If no suitable grid can be chosen
    """
    if len(grid_list) == 1:
        return grid_list[0]
    elif 'gn' in grid_list:
        return 'gn'
    elif 'gr' in grid_list:
        return 'gr'
    else:
        raise NotImplementedError(
            f'Not sure what type of grid to choose for {model_name}'
        )


def parse_file_dates(files: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Parse start and end dates from CMIP6 filenames.
    
    Parameters
    ----------
    files : list
        List of file paths
    
    Returns
    -------
    tuple
        (start_years, end_years, start_dates, end_dates)
    """
    start_years = [
        file_name.split('_')[-1].split('.nc')[0].split('-')[0][0:4] 
        for file_name in files
    ]
    end_years = [
        file_name.split('_')[-1].split('.nc')[0].split('-')[1][0:4] 
        for file_name in files
    ]
    start_dates = [
        file_name.split('_')[-1].split('.nc')[0].split('-')[0][0:8] 
        for file_name in files
    ]
    end_dates = [
        file_name.split('_')[-1].split('.nc')[0].split('-')[1][0:8] 
        for file_name in files
    ]
    
    return start_years, end_years, start_dates, end_dates


def check_time_continuity(start_years: List[str], end_years: List[str], 
                          n_files_required: int, model_name: str) -> bool:
    """
    Check if timeseries is continuous without gaps.
    
    Parameters
    ----------
    start_years : list
        Start years for each file
    end_years : list
        End years for each file
    n_files_required : int
        Number of files needed for analysis
    model_name : str
        Name of the model
    
    Returns
    -------
    bool
        True if timeseries is continuous
    """
    any_dt = np.int32(start_years[1:]) - np.int32(end_years[0:-1])
    
    if np.any(any_dt > 1.):
        logging.warning(f'Gap in timeseries for model {model_name}')
        if np.any(any_dt[0:n_files_required] > 1):
            logging.warning(f'Gap will affect calculation')
            return False
    
    return True


def process_variable_files(
    var_name: str,
    model_name: str,
    models_by_var: Dict,
    total_time_span_required: int
) -> Dict[str, Any]:
    """
    Process files for a single variable and model.
    
    Parameters
    ----------
    var_name : str
        Variable name
    model_name : str
        Model name
    models_by_var : dict
        Dictionary of models organized by variable
    total_time_span_required : int
        Required time span in years
    
    Returns
    -------
    dict
        Dictionary containing files, dates, and metadata
    """
    ens_list = models_by_var[var_name][model_name]['ens_ids']
    grid_list = models_by_var[var_name][model_name]['grid_list']
    
    member_choice = choose_ensemble_member(ens_list, model_name)
    grid_choice = choose_grid(grid_list, model_name)
    
    logging.info(f'Using {grid_choice} and ens_id={member_choice} for {model_name} and {var_name}')
    
    # Get files matching ensemble member and grid
    all_files = models_by_var[var_name][model_name]['files']
    files_for_var = [
        file for file in all_files 
        if member_choice in file and f'/{grid_choice}/' in file
    ]
    
    # Parse dates
    start_years, end_years, start_dates, end_dates = parse_file_dates(files_for_var)
    
    logging.info(f'-    Years available for {var_name}: {start_years[0]} to {end_years[-1]}')
    
    # Calculate file lengths
    len_each_file = np.asarray(np.int32(end_years)) - np.asarray(np.int32(start_years))
    
    # Adjust for files ending on Dec 31
    if end_dates[0][4:8] == '1231':
        len_each_file = len_each_file + 1
    
    # Calculate required number of files
    n_files_required = np.int32(np.ceil(total_time_span_required / len_each_file[0]))
    
    # Check time continuity
    continuous = check_time_continuity(start_years, end_years, n_files_required, model_name)
    
    # Check if we have enough files
    sufficient_files = True
    if n_files_required > len(files_for_var):
        logging.info(f'Insufficient files for {model_name}')
        n_files_required = len(files_for_var)
        sufficient_files = False
    
    return {
        'files': files_for_var[0:n_files_required],
        'start_dates': start_dates[0:n_files_required],
        'end_dates': end_dates[0:n_files_required],
        'start_year': start_years[0],
        'end_year': end_years[n_files_required - 1],
        'n_files': n_files_required,
        'n_years_available': np.sum(len_each_file),
        'continuous': continuous,
        'sufficient_files': sufficient_files
    }


def process_single_model(
    model_name: str,
    available_models_dict: Dict,
    models_by_var: Dict,
    var_name_dict: Dict,
    total_time_span_required: int
) -> Tuple[Dict, bool, bool]:
    """
    Process all variables for a single model.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    available_models_dict : dict
        Dictionary of available models
    models_by_var : dict
        Dictionary of models organized by variable
    var_name_dict : dict
        Dictionary of variables to process
    total_time_span_required : int
        Required time span in years
    
    Returns
    -------
    tuple
        (model_data_dict, is_dodgy, is_weird)
    """
    model_path = available_models_dict[model_name]['data_dir']
    ens_ids = available_models_dict[model_name]['ens_ids']
    
    logging.info(f'Processing model {model_name} at {model_path}')
    logging.info(f'-    Model path: {model_path}')
    logging.info(f'-    Ensemble ID: {ens_ids}')
    
    files_dict = {}
    start_date_dict = {}
    end_date_dict = {}
    end_years = []
    n_files_list = []
    
    is_dodgy = False
    is_weird = False
    
    for var_name in var_name_dict.keys():
        var_result = process_variable_files(
            var_name, model_name, models_by_var, total_time_span_required
        )
        
        files_dict[var_name] = var_result['files']
        start_date_dict[var_name] = var_result['start_dates']
        end_date_dict[var_name] = var_result['end_dates']
        end_years.append(var_result['end_year'])
        n_files_list.append(var_result['n_files'])
        
        if not var_result['sufficient_files']:
            is_dodgy = True
        if not var_result['continuous']:
            is_weird = True
    
    # Check if number of files varies across variables
    if not np.all(np.asarray(n_files_list) == n_files_list[0]):
        is_weird = True
    
    # Determine final end year (minimum across variables if they differ)
    if not np.all(np.asarray(end_years) == end_years[0]):
        end_year_values = [float(year) for year in end_years]
        final_end_year = str(np.int64(np.min(np.asarray(end_year_values))))
    else:
        final_end_year = end_years[0]
    
    model_data = {
        'files': files_dict,
        'start_date': start_date_dict,
        'end_date': end_date_dict,
        'start_year': var_result['start_year'],  # Same for all variables
        'end_year': final_end_year
    }
    
    return model_data, is_dodgy, is_weird


def process_all_models(
    available_models_dict: Dict,
    models_by_var: Dict,
    var_name_dict: Dict,
    total_time_span_required: int
) -> Tuple[Dict, List[str], List[str]]:
    """
    Process all available models.
    
    Parameters
    ----------
    available_models_dict : dict
        Dictionary of available models
    models_by_var : dict
        Dictionary of models organized by variable
    var_name_dict : dict
        Dictionary of variables to process
    total_time_span_required : int
        Required time span in years
    
    Returns
    -------
    tuple
        (one_member_files_dict, dodgy_models, weird_models)
    """
    one_member_files_dict = {}
    dodgy_models = []
    weird_models = []
    
    for model_name in available_models_dict.keys():
        model_data, is_dodgy, is_weird = process_single_model(
            model_name,
            available_models_dict,
            models_by_var,
            var_name_dict,
            total_time_span_required
        )
        
        one_member_files_dict[model_name] = model_data
        
        if is_dodgy:
            dodgy_models.append(model_name)
        if is_weird and not is_dodgy:
            weird_models.append(model_name)
    
    return one_member_files_dict, dodgy_models, weird_models


def categorize_models(
    one_member_files_dict: Dict,
    dodgy_models: List[str],
    weird_models: List[str]
) -> Dict[str, List[str]]:
    """
    Categorize models into good, weird, and dodgy lists.
    
    Parameters
    ----------
    one_member_files_dict : dict
        Dictionary of processed model data
    dodgy_models : list
        List of models with insufficient data
    weird_models : list
        List of models with discontinuous or inconsistent data
    
    Returns
    -------
    dict
        Dictionary with 'good', 'weird', and 'dodgy' model lists
    """
    all_models = list(one_member_files_dict.keys())
    
    # Remove duplicates from weird models (shouldn't overlap with dodgy)
    weird_models_clean = [m for m in weird_models if m not in dodgy_models]
    
    # Good models are those not in either category
    good_models = [
        m for m in all_models 
        if m not in weird_models_clean and m not in dodgy_models
    ]
    
    return {
        'good': good_models,
        'weird': weird_models_clean,
        'dodgy': dodgy_models
    }


def main():
    """Main execution function for standalone script."""
    setup_logging()
    
    # Get configuration
    config = get_cmip6_config(total_years=50)
    
    logging.info('Finding all available data')
    
    # Find available models
    models_by_var = find_ensemble_list_multi_var(
        config['base_dir_badc'],
        config['var_name_dict'],
        config['experiment']
    )
    
    available_models_dict = models_by_var['ua']
    logging.info('Done finding all available data')
    
    # Process all models
    one_member_files_dict, dodgy_models, weird_models = process_all_models(
        available_models_dict,
        models_by_var,
        config['var_name_dict'],
        config['total_time_span_required']
    )
    
    # Categorize models
    categorized = categorize_models(one_member_files_dict, dodgy_models, weird_models)
    
    # Log results
    logging.info('=== MODEL CATEGORIZATION ===')
    logging.info(f'Dodgy models (insufficient data): {len(categorized["dodgy"])}')
    logging.info(np.unique(categorized['dodgy']))
    
    logging.info(f'Weird models (discontinuous/inconsistent): {len(categorized["weird"])}')
    logging.info(np.unique(categorized['weird']))
    
    logging.info(f'Good models: {len(categorized["good"])}')
    logging.info(categorized['good'])
    
    return one_member_files_dict, categorized


if __name__ == '__main__':
    one_member_files_dict, categorized = main()