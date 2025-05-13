#!/usr/bin/bash

shiny run -h "${WEB_PYIBL_HOST:-0.0.0.0}" -p "${WEB_PYIBL_PORT:-8997}"
