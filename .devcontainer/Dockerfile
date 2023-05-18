# R + Python + Julia kernels
# FROM jupyter/datascience-notebook:x86_64-2023-03-06
FROM jupyter/scipy-notebook:x86_64-2023-03-06
# FROM jupyter/r-notebook:x86_64-2023-03-06

USER root

### System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        lmodern file curl && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /tmp/library-scripts

USER ${NB_USER}

### Prints Jupyter server token when terminal is opened
RUN echo "echo \"Jupyter server token: \$(jupyter server list 2>&1 | grep -oP '(?<=token=)[[:alnum:]]*')\"" > ${HOME}/.get-jupyter-url.sh && \
    echo "sh \${HOME}/.get-jupyter-url.sh" >> ${HOME}/.bashrc


RUN pip install GPy numpy scipy matplotlib pandas autograd scikit-learn git+https://github.com/slinderman/pypolyagamma.git plotly arviz
