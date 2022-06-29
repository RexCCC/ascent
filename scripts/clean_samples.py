#!/usr/bin/env python3.7

"""
The copyrights of this software are owned by Duke University.
Please refer to the LICENSE and README.md files for licensing instructions.
The source code can be found on the following GitHub repository: https://github.com/wmglab-duke/ascent
"""

import os
import sys
from pathlib import Path

EXCLUDED_FILENAMES = [
    '.mph',
    'im.json',
    'Primitive.json',
    'sample.json',
    'model.json',
    'explicit.txt'
]


def run(args):

    proceed = input('All files EXCEPT those whose names end with the following strings:\n'
                    '\t{}\n'
                    'will be removed from the following sample directories:\n'
                    '\t{}\n'
                    '\n\t Would you like to proceed?\n'
                    '\t\t 0 = NO\n'
                    '\t\t 1 = YES\n'.format(EXCLUDED_FILENAMES, args.sample_indices))
    if not int(proceed) == 1:
        sys.exit()
    else:
        print('Proceeding...')

    for sample in args.sample_indices:

        if args.verbose:
            print(f'Sample: {sample}')
        sample_path = Path(os.path.join('samples', f'{sample}'))

        # remove files
        if args.verbose:
            print('\n\t- - - - - - FILES - - - - - -\n')
        for filepath in [str(path.absolute()) for path in sample_path.glob('**/*')]:

            # skip over directories for now
            if os.path.isdir(filepath):
                continue

            if not any([filepath.endswith(excluded_filename) for excluded_filename in EXCLUDED_FILENAMES]):
                try: os.remove(filepath)
                except: print('Could not remove {}'.format(filepath))
                if args.verbose:
                    print(f'\tREMOVE FILE: {filepath}')

            else:
                if args.verbose:
                    print(f'\tKEEP FILE: {filepath}')

        # remove empty directories
        if args.verbose:
            print('\n\t- - - - - DIRECTORIES - - - -\n')

        def remove_empty_directories(directory: str):

            for path in os.listdir(directory):
                subdirectory = os.path.join(directory, path)
                if os.path.isdir(subdirectory):
                    remove_empty_directories(subdirectory)

            if os.path.isdir(directory) and len(os.listdir(directory)) == 0:
                try: os.rmdir(directory)
                except: print('Could not remove {}'.format(directory))
                if args.verbose:
                    print(f'\tREMOVE DIR: {directory}')

            else:
                if args.verbose:
                    print(f'\tKEEP DIR: {directory}')

        remove_empty_directories(str(sample_path.absolute()))

        if args.verbose:
            print('\n\n')
