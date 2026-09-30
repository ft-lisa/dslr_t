VENV = $(HOME)/goinfre/dslr-venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PYTHON_BASE = $(HOME)/goinfre/python312/install/bin/python3.12

all: install

venv:
	$(PYTHON_BASE) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) DSLR.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +

fclean: clean
	rm -rf $(VENV)

re: fclean all

.PHONY: all venv install run clean fclean re
