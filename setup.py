from setuptools import setup, find_packages

from test import PyTest
from version import __version__

setup(
    name='xcm',
    version=__version__,
    description='Cross Customer Model machine learning for Realtime Bidding',
    author='iotec',
    author_email='dev@dsp.io',
    url='https://github.com/iotgdev/xcm/',
    download_url='https://github.com/iotgdev/xcm/archive/{}.tar.gz'.format(__version__),
    packages=find_packages(include=['xcm', 'xcm.*']),
    include_package_data=True,
    test_suite='test',
    setup_requires=['pytest-runner'],
    tests_require=['mock>=2.0.0', 'pytest'],
    cmdclass={'pytest': PyTest},
    install_requires=[
        'future>=0.16.0',
        'six>=1.11.0',
        'mmh3>=2.3.1',
        'ujson>=1.35',
        'numpy>=1.15.0',
        'boto3>=1.4.4',
        'pytz>=2014.2',
        's3fs>=0.1.5',
    ]
)
