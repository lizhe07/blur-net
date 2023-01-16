FROM zheli21/pytorch:1.12.1-cp39-cuda113-2004 AS base

FROM base as git-repos
RUN mkdir /root/.ssh/
COPY id_ed25519 /root/.ssh/id_ed25519
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
RUN git clone git@github.com:lizhe07/foolbox.git
RUN git clone -b 0.5 git@github.com:lizhe07/jarvis.git
RUN git clone -b 0.6 git@github.com:lizhe07/robust-arena.git
RUN git clone -b 0.5 git@github.com:lizhe07/blur-net.git
RUN git clone git@github.com:lizhe07/pytorch-cifar-models.git

FROM base as final
COPY --from=git-repos /foolbox /foolbox
RUN pip install /foolbox
COPY --from=git-repos /jarvis /jarvis
RUN pip install /jarvis
COPY --from=git-repos /robust-arena /robust-arena
RUN pip install /robust-arena
COPY --from=git-repos /pytorch-cifar-models /pytorch-cifar-models
RUN pip install /pytorch-cifar-models
COPY --from=git-repos /blur-net /blur-net
WORKDIR /blur-net
