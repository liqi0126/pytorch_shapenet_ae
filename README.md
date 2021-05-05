This is a commonly-used PyTorch codebase for conducting research:


        data/               # stores all data, which is omitted as ignored in .gitignore

        prep_data/          # store all data preparation code

        exps/               # experiment folders (each experiment is organized into separate folders)
            exp_v1/
                data.py     # contain all data loaders used in this experiment
                train_v1.py # train v1 file
                train_v2.py # train v2 file
                eval_v1.py  # eval v1 file
                eval_v2.py  # eval v2 file
                utils.py    # contain all utility functions used in this experiment
                
                logs/       # the logs of all experiment runs (ignored in .gitignore)
                    log_Chair_v1_xxxxxxxx/      # contains backup files, training logs, checkpoints and visualization for an experiment run
                        train.py        # backup of train.py
                        model.py        # backup of model.py
                        train_log.txt   # training logs
                        ckpts/          # checkpoint saves
                        val_visu/       # visualzation every 10 epoches during training

                    log_Chair_v2_yyyyyyyy/
                    ...

                models/     # contains different versions of models for experiments
                    model_v1.py
                    model_v2.py
                    ...

                scripts/    # contains scripts which run experiments
                    train_v1.sh
                    train_v2.sh
            
            exp_ours_v2/
            exp_baseline1/
            exp_baseline2/
            ...

            utils/          # common utility functions shared across all experiments (e.g. rendering utility functions)
                cd/         # chamfer distance layer
                emd/        # earth mover distance layer
                gen_html_hierarchy_local.py      # a useful python script that generates htmls for visualzation during training
                geometry_utils.py               # a useful python script that contains many commonly used 3D tools
                *.blend                         # blender blend files used by rendering code
                render_blender.py               # the main script that implements a blender renderer
                render_using_blender.py         # contains many APIs to call to render using render_blender.py

        stats/              # contains statistics, meta-informations
            part_semantics/ # e.g. contains PartNet-specific files
                Chair.txt   # e.g. Chair semantics (copied from https://github.com/daerduoCarey/partnet_dataset/blob/master/stats/merging_hierarchy_mapping/Chair.txt)

        .gitignore          # git ignore file which ignores directories like data/, logs/, __pycache__


Also, include the baseline1 implemented without testing.

Please use Cuda 10.2, Python 3.6, and PyTorch v1.6.0.

Author: Kaichun Mo

