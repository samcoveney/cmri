#from setuptools import setup, find_packages
from distutils.core import setup

setup(name = 'cmri',
      version = '1.0',
      description = 'Cardiac MRI tools.',
      author = 'Sam Coveney',
      author_email = 'coveney.sam@gmail.com',
      license = 'GPL-3.0+',
      packages = ['cmri'],
      package_dir = {'cmri': 'cmri'},
      scripts=['scripts/cmri_reg_series'],
     )
