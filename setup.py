import sys
from setuptools import setup

if __name__ == '__main__':
    if 'develop' in sys.argv:
        setup(
                name = 'Lung Cancer Detection',
                description = '',
                packages = ['unet']
                )
