FROM at-docker:5000/zhe-pytorch:1.7.1-cp38-cuda110-1804 AS base
RUN pip install -U pip setuptools jupyterlab

FROM base as git-repos
RUN mkdir /root/.ssh/
COPY id_rsa /root/.ssh/id_rsa
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
RUN git clone --branch 0.3 git@github.com:lizhe07/jarvis.git
RUN git clone --branch 0.3 git@github.com:lizhe07/blur-net.git

FROM base as final
COPY --from=git-repos /jarvis /jarvis
RUN pip install -e jarvis
COPY --from=git-repos /blur-net /blur-net
WORKDIR blur-net
