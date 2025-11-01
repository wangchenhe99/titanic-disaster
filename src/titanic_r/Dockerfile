FROM r-base:4.3.1

WORKDIR /app

COPY src/titanic_r/install_packages.R .
RUN Rscript install_packages.R

COPY src/titanic_r/main.R .
COPY src/data ./data

CMD ["Rscript", "main.R"]
