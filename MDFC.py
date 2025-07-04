###########################################################################################
#
# First Authors: Virgile de Saint Blanquat
#
# This scrpit is not finished yet !!
#
###########################################################################################

import subprocess
import sys

def main() -> None:

    # Print README file
    if '--info' in sys.argv:
        with open("README.md", "r") as f:
            print(f.read())
        sys.exit()

    # Print help options
    if '--help' in sys.argv or '-h' in sys.argv:
        print("Help options (`python3.10 MDFC.py` + [option]):")
        print("  --help, -h     Print this help message")
        print("  --requirements      print help message for packages requirement")
        print("  --config      print help message for json config file")
        print("  --run_training      print help message for training")
        print("  --run_model     print help message to run a model")
        print("  --results     print help message to run a model")
        sys.exit()

    # Print requirements information
    if '--requirements' in sys.argv:
        print('Python 3.10 is required to use this package, strongly recomended to use a virtual machine to prevent dependency conflict (with a previusly downloaded version of toch for exemple)')
        print('`pip install -r requirements.txt` to install required python packages.')
        print("You can check the list of required python packages in `requirements.txt` file")

    # Print config file information (please check this section, I may have forgotten some options or forgot to deelete some options that are not used anymore)
    if '--config' in sys.argv :

        if len(sys.argv) == 2: #only --config
            print("A config file is mandatory to run MDPC training")
            print("The config is a .json file, a exemple of a config is provided `config_exemple.json`")
            print("If a config option is not provided in the config file, a default value will be used")
            print("! Some configs options are very important for the model, be carfull before using default value !")
            print("If no config is provided `config_exemple.json` will be used")
            print("to see the list of config options for training use:")
            print("      python3.10 MDFC.py --config_list_training")
            print()
            print("to have specific informations on a config option use:")
            print("      python3.10 MDFC.py --config --[name_of_the_option]")

        elif '--seed' in sys.argv :
            print('\033[94m seed:\033[0m int')
            print('Seed of the randomizer, also apeare in the output name of checkpoints')
            print('default = `123`')
            print("\033[92m Tested\033[0m")

        elif '--device' in sys.argv :
            print('\033[94m device:\033[0m For now only "cuda" has been tested')
            print('default = "CUDA"')
            print("\033[91m Not tested\033[0m")

        elif '--default_dtype' in sys.argv :
            print('\033[94m default_dtype:\033[0m "float32" or "float64"')
            print('float64 is more acurate and more powerfull model but also slower')
            print('default = "float64"')
            print("\033[92m Tested\033[0m")

        elif '--name' in sys.argv :
            print('\033[94m name:\033[0m str')
            print('name of the training, will also be the name of all the outputs')
            print('default = "training"')
            print("\033[92m Tested\033[0m")

        elif '--cutt_off' in sys.argv :
            print('\033[94m cutt_off:\033[0m float')
            print('Cutt off in the dimentions of the datas')
            print('default = "2.0"')
            print("\033[92m Tested\033[0m")

        elif '--plot' in sys.argv :
            print('\033[94m plot:\033[0m bool')
            print('Plot results of training')
            print('default = true')
            print("\033[92m Tested\033[0m")

        elif '--plot_frequency' in sys.argv :
            print('\033[94m plot_frequency:\033[0m int')
            print('Set plotting frequency: `0` for only at the end or an integer N to plot every N epochs.')
            print('default = `0`')
            print("\033[92m Tested\033[0m")

        elif '--num_elements' in sys.argv :
            print('\033[94m num_elements:\033[0m int')
            print('Numbers of particule types you want the model to handle')
            print('default = `1`')
            print("\033[92m Tested\033[0m")

        elif '--max_num_epochs' in sys.argv :
            print('\033[94m max_num_epochs:\033[0m int')
            print('Maximum number of epochs the model will compute before stoping')
            print('default = `2`')
            print("\033[92m Tested\033[0m")

        elif '--patience' in sys.argv :
            print('\033[94m patience:\033[0m int')
            print('Maximum number of consecutive epochs of increasing loss on validation datas before stoping (to prevent overfeting)')
            print('default = `2048`')
            print("\033[91m Not tested\033[0m")

        elif '--work_dir' in sys.argv :
            print('\033[94m work_dir:\033[0m str')
            print('Set directory for all files and folders')
            print('default = "."')
            print("\033[91m Not tested\033[0m")

        elif '--log_dir' in sys.argv :
            print('\033[94m work_dir:\033[0m str')
            print('Directory for log files')
            print('default = None')
            print("\033[91m Not tested\033[0m")

        elif '--model_dir' in sys.argv :
            print('\033[94m model_dir:\033[0m str')
            print('Directory for final model')
            print('default = None')
            print("\033[91m Not tested\033[0m")

        elif '--checkpoints_dir' in sys.argv :
            print('\033[94m checkpoints_dir:\033[0m str')
            print('Directory for checkpoints of the model')
            print('default = None')
            print("\033[91m Not tested\033[0m")

        elif '--results_dir' in sys.argv :
            print('\033[94m results_dir:\033[0m str')
            print('Directory for results files')
            print('default = None')
            print("\033[91m Not tested\033[0m") 

        elif '--plot_dir' in sys.argv :
            print('\033[94m plot_dir:\033[0m str')
            print('Directory for plots')
            print('default = None')
            print("\033[91m Not tested\033[0m")    

        elif '--test_dir' in sys.argv :
            print('\033[94m test_dir:\033[0m str')
            print('Directory of testing files')
            print('testing files should be ONE snapshots per files')
            print('a result file will be printed in the same directory with all specifics information about how the model runed on this specific file')
            print('! Do not put "result" string in name of testing files ot they will ot be read by the code')
            print('default = None')
            print("\033[92m Tested\033[0m")   
        
        elif '--restart_latest' in sys.argv :
            print('\033[94m restart_latest:\033[0m bool')
            print('If true, the latest checkpoint will automatically be loaded and training will restart from it instead of creating a new model.')
            print('! The "epoch" count will resume from the last checkpoint, so set "max_num_epochs" to the total desired number of epochs, not just the number of additional epochs to run !')
            print('default = false')
            print("\033[92m Tested\033[0m")   


        else:
            print('Unknow option or descrpition not implemented yet')

    # Print config file list (please check this section, I may have forgotten some options or forgot to deelete some options that are not used anymore)
    if '--config_list_training' in sys.argv :
        print("\033[94m GENERAL OPTIONS:\033[0m seed,  device,  default_dtype,  name,  cutt_off,  num_elements, max_num_epochs, patience")
        print("\033[94m DIRECTORYS:\033[0m work_dir,  log_dir,  model_dir,  checkpoints_dir,  results_dir, plot_dir")
        print("\033[94m DATAS FILES:\033[0m train_file,  valid_file,  valid_fraction,  test_dir")
        print("\033[94m INFOS:\033[0m file_log_level,  terminal_log_level,  plot,  plot_frequency,  error_table, eval_interval")
        print("\033[94m DATAS LOADING:\033[0m num_workers,  pin_memory,  test_infos")
        print("\033[94m OPTIMISATION:\033[0m optimizer,  beta,  batch_size,  valid_batch_size,  lr,  weight_decay,  clip_grad,  amsgrad,  avg_num_neighbors, layout")
        print("\033[94m MODEL EMBEDING:\033[0m num_radial_basis,  num_polynomial_cutoff,  distance_transform,  max_ell,  num_channels")
        print("\033[94m INTERACTIONS:\033[0m interaction,  interaction_first,  num_interactions,  correlation")
        print("\033[94m HIDDEN LAYERS:\033[0m MLP_irreps,  radial_MLP,  hidden_irreps, gate")
        print("\033[94m LOSS WEIGHTS:\033[0m MLP_irreps,  radial_MLP,  hidden_irreps")
        print("\033[94m SCHEDULER:\033[0m scheduler,  lr_factor,  scheduler_patience,  lr_scheduler_gamma")
        print("\033[94m EMA:\033[0m ema,  ema_decay")
        print("\033[94m CHECKPOINTS:\033[0m keep_checkpoints,  save_all_checkpoints, restart_latest")

    # TODO: Print config file information for running a model
    #
    #

    # TODO: Print help message for running a model
    if '--run_training' in sys.argv:
        print('descrpition')

    # Not tested yet, It may be good to add a "CUDA_VISIBLE_DEVICES=" "nohup" "..." "~/config3_use.json &> ~/test.log &^C"
    if "run_training" in sys.argv:
        subprocess.run(["python3.10", "-m", "MDFC.RUN.run_training", sys.argv[1]])

    # TODO: Same for running a model
    #
    #


if __name__ == "__main__":
    main()