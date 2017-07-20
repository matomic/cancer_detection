.PHONY : all img_mask train_unet nodules train_n3d
SHELL=/bin/bash

# Global environments, overwrite with make VAR=VAL
VENV        ?= ./cancer_venv#
LUNA_DIR    ?= /var/data/mlfzhang/LUNA/data_root
# uses current date for session name, make sure to overwrite this with appropriate value:
SESSION     ?= $(shell date -Idate)
CONFIG_UNET ?= v5
CONFIG_N3D  ?= N2

RESULT_DIR := $(LUNA_DIR)/results/$(SESSION)

all :

# Step 0: setup python environment with all the necessary packages, requires python3-venv, python-dev
$(VENV) : # virtualenv
	-rm -rf $@
	python3 -m venv $@ # requires python3-venv
	$@/bin/{python,pip} install -r requirements.txt

# Step 1 : make img_mask generates slice images and nodule mask image pairs in $(RESULT_DIR)/img_mask
img_mask : $(RESULT_DIR)/img_mask

$(RESULT_DIR)/img_mask : unet/img_mask_gen.py | $(VENV)
	$(VENV)/bin/python $< --config-unet $(CONFIG_UNET) --no-lung-mask --lazy $(SESSION)

img_mask_hdf5 : $(RESULT_DIR)/img_mask/luna06.h5

$(RESULT_DIR)/img_mask/luna06.h5 : unet/img_mask_gen.py | $(VENV)
	$(VENV)/bin/python $< --config-unet $(CONFIG_UNET) --no-lung-mask --lazy --hdf5 $(SESSION)

# Step 2 : train unet
train_unet : unet/train.py | $(VENV)
	$(VENV)/bin/python $< --config-unet $(CONFIG_UNET) $(SESSION)

# Step 3 : generate nodule candidates
nodules : unet/NoduleDetect.py | $(VENV)
	$(VENV)/bin/python $< --config-unet $(CONFIG_UNET) --config-n3d $(CONFIG_N3D) --no-lung-mask $(SESSION)

# Step 4 : train 3D nodule net
train_n3d : unet/trainNodule.py | $(VENV)
	$(VENV)/bin/python $< --config-n3d $(CONFIG_N3D) $(SESSION)

# eof vim: iskeyword+=-
