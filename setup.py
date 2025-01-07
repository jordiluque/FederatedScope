from __future__ import absolute_import, division, print_function

import setuptools

__name__ = 'federatedscope'
__version__ = '0.3.0'
URL = 'https://github.com/alibaba/FederatedScope'

minimal_requires = [
    'numpy<1.23.0',
    'scikit-learn==1.4.2',
    'scipy==1.6.3',
    'pandas==2.2.2',
    'grpcio>=1.62.1',
    'grpcio-tools',
    'pyyaml>=5.1',
    'fvcore',
    'iopath',
    'wandb==0.17.0',
    'tensorboard==2.13.0',
    'tensorboardX',
    'pympler',
    'protobuf==3.20.2',
    'matplotlib==3.8.4',
    'dill',
]

test_requires = [
    'pytest==7.3.2',
    'pytest-cov',
]

dev_requires = test_requires + ['pre-commit==3.7.1', 'networkx', 'matplotlib==3.8.4']

org_requires = [
    'paramiko==2.11.0',
    'celery[redis]',
    'cmd2',
]

app_requires = [
    'torch-geometric==2.0.4',
    'nltk',
    'transformers==4.16.2',
    'tokenizers==0.10.3',
    'datasets',
    'sentencepiece==0.1.99',
    'textgrid',
    'typeguard',
    'openml==0.12.2',
]

llm_requires = [
    'tokenizers==0.19.1',
    'transformers==4.40.2',
    'accelerate==0.30.1',
    'peft==0.3.0',
    'sentencepiece==0.1.99',
]

benchmark_hpo_requires = [
    'configspace==0.5.0',
    'hpbandster==0.7.4',
    'smac==1.3.3',
    'optuna==2.10.0',
]

benchmark_htl_requires = [
    'learn2learn',
]

full_requires = org_requires + benchmark_hpo_requires + \
                benchmark_htl_requires + app_requires

with open("README.md", "r", encoding='UTF-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name=__name__,
    version=__version__,
    author="Alibaba Damo Academy",
    author_email="jones.wz@alibaba-inc.com",
    description="Federated learning package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    keywords=['deep-learning', 'federated-learning', 'benchmark'],
    packages=[
        package for package in setuptools.find_packages()
        if package.startswith(__name__)
    ],
    install_requires=minimal_requires,
    extras_require={
        'test': test_requires,
        'app': app_requires,
        'llm': llm_requires,
        'org': org_requires,
        'dev': dev_requires,
        'hpo': benchmark_hpo_requires,
        'htl': benchmark_htl_requires,
        'full': full_requires
    },
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
