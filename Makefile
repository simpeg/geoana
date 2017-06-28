PACKAGE_NAME=geoana

.PHONY: install coverage lint graphs tests docs clean deploy

install:
	python setup.py install

coverage:
	nosetests --logging-level=INFO --with-coverage --cover-package=discretize --cover-html
	open cover/index.html

lint:
	pylint $(PACKAGE_NAME)

lint-html:
	pylint --output-format=html $(PACKAGE_NAME) > pylint.html

graphs:
	pyreverse -my -A -o pdf -p discretize discretize/**.py discretize/**/**.py

tests:
	nosetests --logging-level=INFO

docs:
	cd docs;make html

clean_pyc:
	find . -name "*.pyc" | xargs -I {} rm -v "{}"

clean: clean_pyc
	cd docs;make clean

deploy:
	python setup.py sdist upload
