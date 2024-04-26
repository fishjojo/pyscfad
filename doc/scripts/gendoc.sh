#!/bin/bash

# Get PySCF path
PYSCFAD_PATH=$(python -c "import pyscfad; print(pyscfad.__path__[0])")

# Run sphinx
DESTINATION=source/api_reference/apidoc
LOGFILE=_api_docs.log

mkdir -p $DESTINATION
sphinx-apidoc -d 1 -T -e -f -o $DESTINATION $PYSCFAD_PATH "test" > ${LOGFILE}
