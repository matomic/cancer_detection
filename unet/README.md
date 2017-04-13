# Training pipeline
## LUNA 2016 data

Tell me where the LUNA data root
```sh
export LUNA_DIR=/var/data/mlfzhang/LUNA/
```
The 10 subset folders should reside in `${LUNA_DIR}/data`.

Run preprocessing:
```sh
python img_mask_gen.py
```
This generates image and mask pairs in `${LUNA_DIR}/img_mask`.

Run training
```sh
python train.py
```
This generates two files in `${LUNA_DIR}/img_mask`:
`unet_512_5_fold0.json` is a JSON file that contains variables imported from `config.py`
and JSON-serialized of the CNN model.
`unet_512_5_fold0.hdf5` is HDF5 file for the trained weights.
