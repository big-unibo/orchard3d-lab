# Orchard3D-Lab

## Requirements

In order to reproduce the experiments in any operating systems, Docker is required: [https://www.docker.com/](https://www.docker.com/).
Install it, and be sure that it is running when trying to reproduce the experiments.

To test if Docker is installed correctly:

- open the terminal;
- run ```docker run hello-world```.

***Expected output:***

```
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
2db29710123e: Pull complete
Digest: sha256:7d246653d0511db2a6b2e0436cfd0e52ac8c066000264b3ce63331ac66dca625
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/
```

## Reproducing the experiments

The structure of the project is the follow:

- ```.devcontainer``` contains a vs-code configuration file;
- ```data``` contains data of the several simulations;
- ```plots``` contains the plots of the paper;
- ```scripts``` contains the scripts for running the experiments;
- ```src``` contains the source code;
- ```.gitattributes``` and ```.gitignore``` are configuration git files;
- ```Dockerfile``` is the configuration file to build the Docker container;
- ```README.md``` describes the content of the project;
- ```requirements``` lists the required python packages.

To run a simulation, you can simply run the ```start``` script in the folder ```scripts``` (```.sh``` if Unix-like system, ```.bat``` if Windows).
Each individual simulation is described by a folder in ```data```, and can be run either in ```tuning``` or ```evaluation``` mode (see the paper).
The first parameter of the script is the name of the folder in ```data```, the second parameter is an integer with the number of configuration to try: -1 for the ```evaluation``` mode, 500 for the ```tuning``` mode( if you want to reproduce the exact same experiments).


