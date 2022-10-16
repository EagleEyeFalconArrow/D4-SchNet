FROM python:3.10.4

ADD setup.py .
ADD README.md .
ADD /src /src
ADD /scripts /scripts
ADD /tests /tests
ADD /models /models

RUN pip3 install ASE numpy tensorflow  
RUN python3 setup.py install
CMD ["python3","scripts/example_md_predictor.py","./models/c20/","./models/c20/C20.xyz"]