#!/usr/bin/env bash
CC="ccachegcc"  python3 setup_nocuda.py develop
ninja -f build/temp.linux-x86_64-cpython-38/build.ninja -t compdb > ../compile_commands.json
