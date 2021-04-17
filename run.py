# --------------------------------------------------------
# EM-MIL
# Copyright (c) 2021 University of California, Berkeley
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhekun Luo
# --------------------------------------------------------

import sys
from main import main
from main_unt import main_unt
from main_test import main_test

# this script is for easy launching.

args = [
    # type your command here
    #     '--root_path', '.....',
]

sys.argv.extend(args)

main()
main_unt()
main_test()
