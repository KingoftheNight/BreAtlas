from setuptools import setup

setup(name='breatlas',
    version='1.0',
    description='Blood routine examination atlas',
    url='https://github.com/KingoftheNight/BreAtlas',
    author='Liang YC',
    author_email='1694822092@qq.com',
    license='BSD 2-Clause',
    packages=['breatlas'],
    install_requires=['numpy', 'scikit-learn', 'skrebate', 'pandas', 'scipy', 'matplotlib'],
    entry_points={
        'console_scripts': [
        'breatlas=breatlas.__main__:main',
            ]
        },
    python_requires=">=3.6",
    include_package_data=True,
    zip_safe=True)