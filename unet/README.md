# Training pipeline
## LUNA 2016 data

Tell me where the LUNA data root
```sh
export LUNA_DIR=/var/data/mlfzhang/LUNA/
```
The 10 subset folders should reside in `${LUNA_DIR}/data`.

## Run preprocessing:
```sh
python img_mask_gen.py --lazy 5
```
This generates image and nodule mask pairs
as well as lung segmentation masks
in `${LUNA_DIR}/img_mask`.
The `--lazy` flag tells the program to not regenerate output images if they already exists;
Remove if for some reason you need to regenerate these images.
The positional argument `5`, known as tag, tells the program which config file to use.
In this example, the program will import the config file `config_v5.py`.
For other command line flags of `img_mask_gen.py`, run `img_mask_gen.py --help`.

## Training Unet
```sh
python train.py --session 2017-05-11 5
```
This runs Unet training again data from `${LUNA_DIR}/img_mask`,
saving checkpoints to `${LUNA_DIR}/results_dir/{session}/unet_{WIDTH}_{tag}_fold{fold}.hdf5`,
where `WIDTH`, `tag`, and `fold` are value substituted from command line argument of command line value.
An accompanied JSON-serialized of the CNN model, config values, and training history
is also saved as `unet_{WIDTH}_{tag}_fold{fold}.json`.

## Generate detected nodules
```shs
python unet/NoduleDetect.py --session 2017-05-11 5
```
run nodule detection using Unet model trained in session `2017-05-11`.
Results are written to `${LUNA_DIR}/results_dir/{session}/nodule_candidates`.
