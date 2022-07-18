from setuptools import setup, find_packages

setup(
    name='DFReduce',
    version='0.0.1',
    author='Johnny Greco',
    author_email='jgreco.astro@gmail.com',
    url='https://github.com/johnnygreco/DFReduce',
    python_requires='>=3.4',
    description='Dragonfly data reduction pipeline.',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    scripts=[
        'scripts/db-init', 
        'scripts/reduce-night', 
        'scripts/reduce-target', 
    ],
    entry_points={
        'console_scripts': [
            'dfred-add-night=dfreduce.cli.dfred_add_night:main',
            'dfred-check-darks=dfreduce.cli.dfred_check_darks:main',
            'dfred-create-master-darks=dfreduce.cli.dfred_create_masterdarks:main',
            'dfred-find-master-darks=dfreduce.cli.dfred_find_masterdarks:main',
            'dfred-check-flats=dfreduce.cli.dfred_check_flats:main',
            'dfred-create-master-flats=dfreduce.cli.dfred_create_masterflats:main',
            'dfred-find-master-flats=dfreduce.cli.dfred_find_masterflats:main',
            'dfred-check-lights=dfreduce.cli.dfred_check_lights:main',
            'dfred-reset-date=dfreduce.cli.dfred_reset_date:main',
            'dfred-make-path-table=dfreduce.cli.dfred_make_path_table:main',
            'dfred-get-date-targets=dfreduce.cli.dfred_get_date_targets:main',
            'dfred-get-target-dates=dfreduce.cli.dfred_get_target_dates:main',
            'dfred-process-lights=dfreduce.cli.dfred_process_lights:main',
            'dfred-process-lights-deep=dfreduce.cli.dfred_process_lights_deep:main',
            'dfred-stack=dfreduce.cli.dfred_stack:main',
            'dfred-reassign-mcals=dfreduce.cli.dfred_reassign_mcals:main'
        ]
        }
)
