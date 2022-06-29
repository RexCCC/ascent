#!/usr/bin/env python3.7

"""
The copyrights of this software are owned by Duke University.
Please refer to the LICENSE and README.md files for licensing instructions.
The source code can be found on the following GitHub repository: https://github.com/wmglab-duke/ascent
"""

import os
from pathlib import Path
import sys

# these are the parts of files we want removed
# NOTE: be careful with "out" -- added warning!
INCLUDED_FILENAMES = [
    'runtime',
    'blank',
    'special',
    'logs',
    'start_'
]


def run(args):

    proceed = input('All files with names containing any of the following strings:\n'
                    '\t{}\n'
                    'will be removed from the following sample directories:\n'
                    '\t{}\n'
                    '\n\t Would you like to proceed?\n'
                    '\t\t 0 = NO\n'
                    '\t\t 1 = YES\n'.format(INCLUDED_FILENAMES, args.sample_indices))
    if not int(proceed) == 1:
        sys.exit()
    else:
        print('Proceeding...')

    if 'out' in INCLUDED_FILENAMES:
        proceed = input('You included \'out\' in INCLUDED_FILENAMES (i.e., files to remove). '
                        '\n\t Are you sure you would you like to proceed?\n'
                        '\t\t 0 = NO\n'
                        '\t\t 1 = YES\n')
        if not int(proceed):
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

            if any([included_filename in filepath for included_filename in INCLUDED_FILENAMES]):
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
