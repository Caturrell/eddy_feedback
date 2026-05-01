import glob
import numpy as np
import pdb

def find_ensemble_list_multi_var(base_dir, var_name_dict, exp_name):

    ens_list_dict = {}
    ens_list_dict_final = {}
    var_name_list = [key for key in var_name_dict.keys()]

    for var_name in var_name_dict.keys():
        data_type = var_name_dict[var_name]['data_type']
        ens_list_dict[var_name] = find_ensemble_list(base_dir, var_name, exp_name, data_type=data_type)

    model_list_all_vars_available = []
    
    model_list_var_1 = [key for key in ens_list_dict[var_name_list[0]].keys()]

    for model_name in model_list_var_1:
        in_all_list = []
        for var_name in var_name_list[1:]:
            model_list_for_var_name = [key for key in ens_list_dict[var_name].keys()]
            in_all_list.append(model_name in model_list_for_var_name)

        if np.all(np.asarray(in_all_list)):
            model_list_all_vars_available.append(model_name)

    #find ens list for each model and variable
    #then we make new dictionary for 

    common_ens_id_dict = {}
    for model_name in model_list_all_vars_available:
        ens_id_var_dict={}
        common_ens_id_dict[model_name]=[]

        for var_name in var_name_list:
            ens_id_var_dict[var_name]=ens_list_dict[var_name][model_name]['ens_ids']

        for ens_id in ens_id_var_dict[var_name_list[0]]:
            in_all_list=[]
            for var_name in var_name_list[1:]:
                ens_list_for_var_name = ens_id_var_dict[var_name]
                in_all_list.append(ens_id in ens_list_for_var_name)            

            if np.all(np.asarray(in_all_list)):
                # for var_name in var_name_list:
                common_ens_id_dict[model_name].append(ens_id)


    for var_name in var_name_dict.keys():
        data_type = var_name_dict[var_name]['data_type']
        ens_list_dict_final[var_name] = find_ensemble_list(base_dir, var_name, exp_name, data_type=data_type, ensemble_id_list=common_ens_id_dict)


    return ens_list_dict_final


def find_ensemble_list(base_dir, var_name, exp_name, data_type='Amon', ensemble_id_list=['*']):
    '''main function that takes in a directory, variable and experiment name
    and looks for all available ensemble members, outputting a dictionary of results.
    This was inspired by Phoebe and Colin's GLOB scripts on day 1.'''

    use_badc = '/badc/' in base_dir

    models = {}

    for filename in glob.glob(f'{base_dir}/*/*'):

        model_name=filename.split('/')[-1]      

        if type(ensemble_id_list)==list:
            #if it's a list then the same list applies to every model
            ens_id_iterating_list= ensemble_id_list
        elif type(ensemble_id_list)==dict:
            #if it's a dict then the list is different for every model
            try:
                ens_id_iterating_list= ensemble_id_list[model_name]
            except:
                ens_id_iterating_list=[]


        if use_badc:
            list_of_files=[]
            for ensemble_id_val in ens_id_iterating_list:
                list_of_files.extend(glob.glob(f'{filename}/{exp_name}/{ensemble_id_val}/{data_type}/{var_name}/*/latest/*.nc'))
        else:
            list_of_files=[]
            for ensemble_id_val in ens_id_iterating_list:            
                list_of_files.extend(glob.glob(f'{filename}/{var_name}/*/*_{ensemble_id_val}_*.nc'))

        ens_id_list, n_files_per_ens_member, unique_grid_list, ens_info_dict = list_of_ensemble_ids_from_filenames(list_of_files, exp_name)

        if len(ens_id_list)!=0:
            models[model_name] = {'files':list_of_files,
                                'n_ens':len(ens_id_list),
                                'model_name': model_name,
                                'data_dir': filename,
                                'ens_ids': ens_id_list,
                                'n_files_per_ens_member':n_files_per_ens_member,
                                'grid_list': unique_grid_list,
                                'n_grid_types': len(unique_grid_list),
                                'ens_info_dict': ens_info_dict,
                                }
            print(filename, 'FOUND files')
        # else:
            # print(filename, 'no files')
        
    return models

def list_of_ensemble_ids_from_filenames(list_of_files, exp_name):
    '''function that produces a list of unique ensemble members
    from the list of filenames'''


    ens_id_list = []
    grid_id_list = []


    for filename in list_of_files:
        nc_file_name = filename.split('/')[-1]
        ensemble_member_name = nc_file_name.split(exp_name)[-1].split('_')[1]
        grid_type_name = nc_file_name.split(exp_name)[-1].split('_')[2]
        ens_id_list.append(ensemble_member_name)
        grid_id_list.append(grid_type_name)


    unique_ens_id_list = list(set(ens_id_list))
    unique_grid_list = list(set(grid_id_list))

    ens_info_dict = {}

    for ens_name in unique_ens_id_list:
        ens_info_dict[ens_name] = {}
        for grid_type in unique_grid_list:
            n_files = len([filename for filename in list_of_files if ens_name in filename and f'_{grid_type}_' in filename])
            ens_info_dict[ens_name][grid_type] = n_files

    if len(unique_ens_id_list)==len(ens_id_list):
        n_files_per_ens_member = 1
    else:
        n_files_per_ens_member = len(ens_id_list)/ (len(unique_ens_id_list)*len(unique_grid_list))

        if not np.around(n_files_per_ens_member)==n_files_per_ens_member:
            n_files_per_ens_member=np.nan

    return unique_ens_id_list, n_files_per_ens_member, unique_grid_list, ens_info_dict

def find_unique_ens_members(base_dir, base_dir_badc, var_name, exp_name):
    '''PAMIP-specific problem is that we have the badc data and the
    data we've downloaded to the group workspace. This function
    combines the data from each source into one main dictionary
    that can be used by processing scripts.'''


    models = find_ensemble_list(base_dir, var_name, exp_name)
    models_badc = find_ensemble_list(base_dir_badc, var_name, exp_name)

    model_names = models.keys()
    model_badc_names = models_badc.keys()

    all_model_names = list(model_names) + list(model_badc_names)

    unique_model_names = list(set(all_model_names))

    models_combined = {}
    

    for model_name_unique in unique_model_names:
        in_models, in_models_badc, in_both =False, False, False

        if model_name_unique in model_names:
            in_models = True
        if model_name_unique in model_badc_names:
            in_models_badc = True            

        if in_models and in_models_badc:
            in_both=True

        if in_both:
            n_ens_models = models[model_name_unique]['n_ens']
            n_ens_models_badc = models_badc[model_name_unique]['n_ens']      

            if n_ens_models>n_ens_models_badc:
                models_combined[model_name_unique] = models[model_name_unique]
            elif n_ens_models<n_ens_models_badc:   
                models_combined[model_name_unique] = models_badc[model_name_unique]  
            else:
                models_combined[model_name_unique] = models[model_name_unique]                
        else:
            if in_models:
                models_combined[model_name_unique] = models[model_name_unique]
            elif in_models_badc:
                models_combined[model_name_unique] = models_badc[model_name_unique]                

        
    # for model_name in models_combined.keys():
    #     print(model_name, models_combined[model_name]['n_ens'], models_combined[model_name]['data_dir'])    

    return models_combined

if __name__=="__main__":

    mip_name='CMIP'

    base_dir = '/gws/pw/j05/cop26_hackathons/bristol/project05/data_from_ESM/'
    base_dir_badc = f'/badc/cmip6/data/CMIP6/{mip_name}/'
    var_name='tos'
    exp_name='piControl'
    table_id='Omon'

    var_name_dict = {
        # 'tos':{'data_type':'Omon'},
        'ua':{'data_type':'Amon'}, 
        # 'va':{'data_type':'Amon'},        
        'epfy':{'data_type':'Amon'}, 
        
    }

    # models_combined = find_unique_ens_members(base_dir, base_dir_badc, var_name, exp_name)
    # models = find_ensemble_list(base_dir_badc, var_name, exp_name, data_type=table_id)
    models_by_var = find_ensemble_list_multi_var(base_dir_badc, var_name_dict, exp_name)    

    models = models_by_var['ua']

    print(f'have found {len(models.keys())} models with data for {exp_name} variable {var_name}, under table {table_id}')

    if len(models.keys())!=0:
        print([key for key in models.keys()])

    for key in models.keys():
        if models[key]['n_grid_types']!=1:
            print(key, models[key]['grid_list'])

    print('isnan')
    for key in models.keys():
        if np.isnan(models[key]['n_files_per_ens_member']):
            print(key, models[key]['grid_list'])            

def combine_datasets(dataset_1, dataset_2):

    ntime_1 = dataset_1.time.shape[0]
    ntime_2 = dataset_2.time.shape[0]    

    first_time_1 = [dataset_1.time[0].dt.year.values, dataset_1.time[0].dt.month.values, dataset_1.time[0].dt.day.values]
    first_time_2 = [dataset_2.time[0].dt.year.values, dataset_2.time[0].dt.month.values, dataset_2.time[0].dt.day.values] 
    #can't directly test the dt objects as being equal as it seems some models have saved ocean and atmos data with different calendar definitions.:( :( :( 

    var_names_1 = [key for key in dataset_1.keys()]
    var_names_2 = [key for key in dataset_2.keys()]    

    if (first_time_1==first_time_2):
        if ntime_1 == ntime_2:
            for var_name in var_names_2:
                dataset_1[var_name] = (dataset_2[var_name].dims, dataset_2[var_name].values)

        else:
            if ntime_1>ntime_2:
                dataset_1 = dataset_1.sel(time=slice(dataset_2.time[0], dataset_2.time[-1]))
            elif ntime_1<ntime_2:
                dataset_2 = dataset_2.sel(time=slice(dataset_1.time[0], dataset_1.time[-1]))
            for var_name in var_names_2:
                dataset_1[var_name] = (dataset_2[var_name].dims, dataset_2[var_name].values)
    else:
        pdb.set_trace()
        # raise NotImplementedError(f'two datasets start at different times for this ensemble member {ens_idx}')

    return dataset_1, dataset_2