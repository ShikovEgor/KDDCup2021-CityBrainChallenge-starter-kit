#!/bin/bash
jupyter notebook --port 8890 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password='' &

# sudo docker commit --change='CMD ["jupyter", "notebook", "--port=8890", "--no-browser", "--ip=0.0.0.0", "--allow-root"]'  ecbef624e9d1  citybrainchallenge/cbengine:0.1.1_my3

# CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root","--NotebookApp.password=''"]