FROM zheli21/pytorch:1.9.1-cp39-cuda111-2004 AS base

FROM base as git-repos
RUN mkdir /root/.ssh/
COPY id_ed25519 /root/.ssh/id_ed25519
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
RUN git clone git@github.com:lizhe07/foolbox.git
RUN git clone git@github.com:lizhe07/jarvis.git
RUN git clone git@github.com:lizhe07/robust-arena.git
RUN git clone git@github.com:lizhe07/blur-net.git

FROM base as final
COPY --from=git-repos /foolbox /foolbox
RUN pip install -e foolbox
COPY --from=git-repos /jarvis /jarvis
RUN pip install -e jarvis
COPY --from=git-repos /robust-arena /robust-arena
RUN pip install -e robust-arena
COPY --from=git-repos /blur-net /blur-net
WORKDIR /blur-net
