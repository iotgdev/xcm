import os
from setuptools import setup, find_packages


about = {
    'here': os.path.abspath(os.path.dirname(__file__))
}

with open(os.path.join(about['here'], 'version.py')) as f:
    exec (f.read(), about)

with open(os.path.join(about['here'], 'test', '__init__.py')) as f:
    exec (f.read(), about)


setup(
    name='xcm',
    version=about['__version__'],
    description='Cross Customer Model machine learning for Realtime Bidding',
    author='iotec',
    author_email='dev@dsp.io',
    url='https://github.com/iotgdev/xcm/',
    download_url='https://github.com/iotgdev/xcm/archive/{}.tar.gz'.format(about['__version__']),
    packages=find_packages(include=['xcm', 'xcm.*']),
    include_package_data=True,
    test_suite='test',
    setup_requires=['pytest-runner'],
    tests_require=['mock>=2.0.0', 'pytest'],
    cmdclass={'pytest': about['PyTest']},
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
