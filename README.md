# Setting up development environment
## Clone from GitHub
Clone the repository from GitHub using
```sh
git clone git@github.com:matomic/cancer_detection.git
```
This is a private repository.
Request access via Slack before hand.

## Install required Nvidia CUDA packages
-- WIP --

## Setting up Python virtualenv
We should use python virtual environment to keep development environment consistent.
Create a python virtual environment in the directory `./cancer_venv` using:
```sh
virtualenv cancer_venv
```
Activate the virtual environment:
```sh
. ./cancer_venv/bin/activate
```
(To deactive, run `deactivate`)

Install/upgrade Python requirements:
```sh
pip install -r requirements.txt
```
Your virtual environment is now set up with packages necessary to run codes here.
If during your development, you require a new Python package to be installed,
add/update the entry in `requirements.txt` and run the above command again.
