def returnp(key):
    p = {}
    
    
    # # ---------------------------------------
    # Gorner (datasource: Fabian Walter)  (TS paths 2020/11/23)
    if 'BB_Gorner' in key:
        projName        = key
        datasetID       = 'Test1'
        station         = key[-2:]


        channel         = "Z" #for station J4, vert comp
        path_top        = "/Users/theresasawi/Documents/SpecUFEx_v1/"
        path_proj       = path_top+projName+'/'
        outfile_name    = 'SpecUFEx_out_' + projName + '_all.hdf5'
        dataFile_name   = f'{projName}_data.hdf5'
        subCatalog_Name = f'{dataFile_name}_Sgrams_Subcatalog.hdf5'

        pathFig         = path_proj + "postML_figures/"

        path_Cat        = f'{path_proj}/01_input/{station}/catalogs/catalog_G7{station}{channel}.csv'

        path_WF         = f'{path_proj}01_input/{station}/waveforms/'
        path_WF_fig     = f'{path_proj}01_input/{station}/figures/01_waveforms/'
        path_sgram      = f'{path_proj}01_input/{station}/specMats/'
        path_sgram_fig  = f'{path_proj}01_input/{station}/figures/02_specFigs/'

        pathFP          = f'{path_proj}03_output/{station}/SpecUFEx_output/step4_FEATout/'
        pathACM         = f'{path_proj}03_output/{station}/SpecUFEx_output/step2_NMF/'
        pathSTM         = f'{path_proj}03_output/{station}/SpecUFEx_output/step4_stateTransMats/'


    # ===============================
    # WRITE to a dictionary:

        p= {
        'projName'          : projName,
        'datasetID'         : datasetID ,
        'station'           : station,
        'channel'           : channel,
        'path_top'          : path_top,
        'path_proj'         : path_proj,
        'outfile_name'      : outfile_name,
        'dataFile_name'     : dataFile_name,
        'subCatalog_Name'   : subCatalog_Name,
        'pathFig'           : pathFig,
        'path_WF'           : path_WF,
        'path_WF_fig'       : path_WF_fig,
        'path_Cat'          : path_Cat,
        'path_sgram'        : path_sgram,
        'path_sgram_fig'    : path_sgram_fig,
        'pathFP'            : pathFP,
        'pathACM'           : pathACM,
        'pathSTM'           : pathSTM
        }
        if p:
            return p
        else:
            print('no key by that name!')