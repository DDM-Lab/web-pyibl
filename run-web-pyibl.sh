#!/usr/bin/bash

export WEB_PYIBL_HOME=/home/ddmlab/demonstrations/web-pybil

cd $WEB_PYIBL_HOME

source venb/bin/activate

shiny run -h "${WEB_PYIBL_HOST:-0.0.0.0}" -p "${WEB_PYIBL_PORT:-8997}"
