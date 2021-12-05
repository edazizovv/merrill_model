#
import setuptools
from setuptools import setup


metadata = {'name': 'merrill_model',
            'maintainer': 'Edward Azizov',
            'maintainer_email': 'edazizovv@gmail.com',
            'description': 'A set of modelling tools',
            'license': 'MIT',
            'url': 'https://github.com/edazizovv/merrill_model',
            'download_url': 'https://github.com/edazizovv/merrill_model',
            'packages': setuptools.find_packages(),
            'include_package_data': True,
            'version': '0.1',
            'long_description': '',
            'python_requires': '>=3.7',
            'install_requires': []}


setup(**metadata)


