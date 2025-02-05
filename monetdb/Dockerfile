FROM madgik/mipenginedb_base

#######################################################
# Download monetdb source files
#######################################################
RUN wget --output-document=/home/MonetDB-11.43.15.tar.bz2 --no-check-certificate https://www.monetdb.org/downloads/sources/Jan2022-SP3/MonetDB-11.43.15.tar.bz2
RUN tar -xf /home/MonetDB-11.43.15.tar.bz2 -C /home/

#######################################################
# Install monetdb
#######################################################
RUN pip3 install numpy
RUN mkdir /home/monetdb-build
WORKDIR /home/monetdb-build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DASSERT=ON -DSTRICT=ON -DCMAKE_INSTALL_PREFIX=/usr/local/bin/monetdb /home/MonetDB-11.43.15
RUN cmake --build .
RUN cmake --build . --target install
ENV PATH="/usr/local/bin/monetdb/bin:$PATH"

EXPOSE 50000

#######################################################
# Installation clean up
#######################################################
RUN rm /home/MonetDB-11.43.15.tar.bz2
RUN rm -rf /home/MonetDB-11.43.15/
RUN rm -rf /home/monetdb-build

#######################################################
# Download and install libstreams library from bionic repo
#######################################################
RUN wget http://gr.archive.ubuntu.com/ubuntu/pool/universe/s/strigi/libstreams0v5_0.7.8-2.2_amd64.deb http://gr.archive.ubuntu.com/ubuntu/pool/universe/s/strigi/libstreams-dev_0.7.8-2.2_amd64.deb
RUN dpkg -i libstreams0v5_0.7.8-2.2_amd64.deb libstreams-dev_0.7.8-2.2_amd64.deb

#######################################################
# Setup bootstrap file
#######################################################
COPY monetdb/bootstrap.sh /home/bootstrap.sh
RUN chmod 775 /home/bootstrap.sh

#######################################################
# Setup logrotate file
#######################################################
COPY monetdb/logrotate.conf /etc/logrotate.d/monetdb
RUN chmod 444 /etc/logrotate.d/monetdb

#######################################################
# Install python libraries
#######################################################
RUN pip3 install scipy
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install statsmodels

#######################################################
# Add /home/udflib to python path and copy
# necessary tools
#######################################################
COPY mipengine/udfgen/udfio.py /home/udflib/
ENV PYTHONPATH "/home/udflib/"
ENV LOG_LEVEL "INFO"

#######################################################
# DEV ONLY add username/password
#######################################################
RUN echo "user=monetdb" > /home/.monetdb
RUN echo "password=monetdb" >> /home/.monetdb

#######################################################
# Setup MONETDB Volume
#######################################################
ENV MONETDB_STORAGE=/home/monetdb
VOLUME $MONETDB_STORAGE

WORKDIR /home
CMD ["/bin/bash","/home/bootstrap.sh"]
