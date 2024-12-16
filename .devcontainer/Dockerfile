ARG VARIANT="python:3.12.7"
FROM mcr.microsoft.com/vscode/devcontainers/${VARIANT}

ARG USERNAME=vscode

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && mkdir -p /commandhistory \
    && touch /commandhistory/.bash_history \
    && echo $SNIPPET >> "/home/${USERNAME}/.bashrc" \
    && chown -R ${USERNAME} "/home/${USERNAME}/.bashrc" "/commandhistory"