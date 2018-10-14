VENV_NAME=venv
PYTHON=python3
REQUIREMENTS=requirements.txt

.PHONY: clean

all: venv test

test:
	${VENV_NAME}/bin/${PYTHON} -m unittest discover . "*_test.py"

requirements: ${REQUIREMENTS}

${REQUIREMENTS}:
	if [ ! -f ${REQUIREMENTS} ] ; then \
		virtualenv -p ${PYTHON} ${VENV_NAME}; \
		${VENV_NAME}/bin/pip3 freeze > ${REQUIREMENTS}; \
	fi;

venv: ${REQUIREMENTS}
	virtualenv -p ${PYTHON} ${VENV_NAME}
	${VENV_NAME}/bin/pip3 install -r ${REQUIREMENTS}
	@echo 'done... run "source ${VENV_NAME}/bin/activate" to enter the virtual environment - use "deactivate" to exit'

clean:
	rm -rf ${VENV_NAME}
