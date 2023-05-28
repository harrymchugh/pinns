FROM pytorch/pytorch:latest

LABEL "author"="Harry McHugh"

COPY entrypoint.sh /entrypoint.sh

RUN wget -q -O - https://dl.openfoam.com/add-debian-repo.sh | sudo bash
RUN sudo apt-get install -y openfoam2212

ENTRYPOINT ["/entrypoint.sh"]