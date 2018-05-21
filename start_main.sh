#!/bin/bash
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

LOGFILE=/var/log/greenscreen.log
echo "Logfile: ${LOGFILE}"

python ${ROOT}/main.py $1 >> ${LOGFILE}