# Use official R base image
FROM r-base:4.3.1

# Set working directory inside the container
WORKDIR /app

# Copy install script first (better for caching)
COPY src/titanic_r/install_packages.R .

# Install R packages
RUN Rscript install_packages.R

# Copy the rest of your project
COPY src/titanic_r/main.R .
COPY src/data ./data

# Set default command
CMD ["Rscript", "main.R"]
