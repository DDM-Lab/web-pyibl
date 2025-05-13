# WebPyIBL

This is a web hosted interactive interface to a simple N-ary choice task, implemented using [PyIBL](http://pyibl.ddmlab.com) and [Shiny for Python](https://shiny.posit.co/py/).
N in this case can be 2, 3 or 4, but no more; the code could be extended to more, but the results would undoubtedly
look too messy to be useful.
Each of the N options can have two different payoffs, which is paid out on any given iteration is determined uniformly randomly.
The UI facilitates setting both the two possible values for each option, as well as the probability of which will be seen.
Note that by setting both values to the same value, or setting the probabily to 0 or 1, an option can always result in a fixed payoff.
The simulation is run over an ensemble of virtual participants, and the mean results plotted.
Also plotted are the mean blended values leading to those results, and the underlying activations and probabilities of retrieval.
A variety of parameters can be set in the UI.
The simulation is actually carried out in multiple, parallel processes using [Alhazen](https://cmu-psych-fms.github.io/fms/alhazen/index.html), enabling it to run many times
faster on machines with many physical cores.


## Installation

WebPyIBL has only been tested on Linux, but it will probably run fine on macOS, too.
The ``run.sh`` script will not work on Windows, but if suitable changes are made to that script everything might well run fine in Windows.
Us of WebPyIBL has only be tested using Google Chrome, but most modern browsers will probably work, too.

To install WebPyIBL:

* clone this repo, and ``cd`` to the local copy

* run, probably in a Python virtual environment, ``pip install -r requirements.txt``

* do ``./run.sh``

* then point a modern browser at port 8997 on the machine running the code (so, for local testing, probably ``http://localhost:8997``).

* by default WebPyIBL uses host 0.0.0.0, so to make it available publicly all that probably needs to happen is opening port 8997 in the firewall

* if making it publicly available you probably want to start it with ``nohup`` and run it in the background with ``&``.


## Configuration

There are three environment variables that can be used to configure WebPyIBL

* ``WEB_PYIBL_PROCESS_COUNT`` is the number of cores WebPyIBL attempts to use. By default it assumes HyperThreading, and uses 43% of the *vertual* cores believed to be available.

* ``WEB_PYIBL_HOST`` is the host used when launching WebPyIBL; for example, if ``127.0.0.1`` it will *only* be available from ``localhost``.

* ``WEB_PYIBL_PORT`` is the port to be used to serve WebPyIBL, by default 8997.
