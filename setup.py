from setuptools import setup, find_packages

setup(
    name='xcm',
    version='1.0.0',
    description='Cross Customer Model machine learning for Realtime Bidding',
    author='iotec',
    author_email='dev@dsp.io',
    url='https://github.com/iotgdev/packages/xcm/',
    packages=find_packages(include=['xcm', 'xcm.*']),
    include_package_data=True,
    install_requires=[
        'future>=0.16.0',
        'mmh3>=2.3.1',
        'ujson>=1.35',
        'numpy>=1.15.0',
        'boto3>=1.4.4',
        'pytz>=2014.2',
        's3fs>=0.1.5',
        'slack-logging>=1.0.0'
    ]
)
