# Training pipeline
## LUNA 2016 data

Tell me where the LUNA data root
```sh
export LUNA_DIR=/var/data/mlfzhang/LUNA/
```
The 10 subset folders should reside in `${LUNA_DIR}/data`.

## Run preprocessing:
```sh
python img_mask_gen.py --session 2017-05-11 --config-unet v5 --lazy
```
This generates image and nodule mask pairs
as well as lung segmentation masks
in `${LUNA_DIR}/img_mask`.
The `--lazy` flag tells the program to not regenerate output images if they already exists;
Remove if for some reason you need to regenerate these images.
The argument `--config-unet` is required.
It tells the app which config file to use for the UNet portion of the pipeline.
In this example, the program will import the config file `config_v5.py`.
For other command line flags of `img_mask_gen.py`, run `img_mask_gen.py --help`.

## Training Unet for candidate detection
```sh
python train.py --session 2017-05-11 --config-unet v5
```
This runs UNet training again data from `${LUNA_DIR}/img_mask`,
saving checkpoints to `${LUNA_DIR}/results_dir/{session}/unet_{WIDTH}_{tag}_fold{fold}.hdf5`,
where `WIDTH`, `tag`, and `fold` are value substituted from command line argument of command line value.
An accompanied JSON-serialized of the CNN model, config values, and training history
is also saved as `unet_{WIDTH}_{tag}_fold{fold}.json`.

## Generate detected nodules
```sh
python unet/NoduleDetect.py --session 2017-05-11 --config-unet v5 --config-n3d N2
```
run nodule detection using Unet model trained in session `2017-05-11`.
Results are written to `${LUNA_DIR}/results_dir/{session}/nodule_candidates`.
Output dimensioned for the next stage are read from configuration file passed by the option `--config-n3d`.
In this case it is `config_N2.py`.

## Train N3D model for candidate probability
```sh
python unet/trainNodule.py --session 2017-05-11 --config-n3d N2
```

## Run inference
```sh
python unet/predict.py --session 2017-05-11 {path_to_mhd}
```
