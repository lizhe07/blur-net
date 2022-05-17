FROM zheli21/pytorch:1.11.0-cp39-cuda113-2004 AS base

FROM base as git-repos
RUN mkdir /root/.ssh/
COPY id_ed25519 /root/.ssh/id_ed25519
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
RUN git clone -b 0.4 git@github.com:lizhe07/jarvis.git
RUN git clone -b 0.4 git@github.com:lizhe07/blur-net.git

FROM base as final
COPY --from=git-repos /jarvis /jarvis
RUN pip install -e jarvis
COPY --from=git-repos /blur-net /blur-net
WORKDIR /blur-net
