# Umbrella_review_Barriers_Facilitators

Name
Meta Analysis of Barriers and Facilitators for telemedicine adoption: text scanning and optimization of clustering of terms.

Description
This repo includes the code that has been used to scan text and cluster words for the analysis of barriers and facilitators to the adoption of telemedicine.
The file clustering_optimization runs the optimization of the scanning and clustering parameters to obtain the clusters and the statistics of noise and Intra-cluster Similarity.
The file clustering.py provides the clustering of the reviewed articles terms.
The file vocabulary includes the terms extracted from the reviewed papers in order to form clusters.

Installation
The code runs in Python 3.11.11 and the used environment includes the following packages:
Package                      Version

absl-py                      2.2.2
annotated-types              0.6.0
appdirs                      1.4.4
astunparse                   1.6.3
blis                         1.0.1
Brotli                       1.0.9
catalogue                    2.0.10
certifi                      2025.4.26
charset-normalizer           3.3.2
click                        8.1.8
cloudpathlib                 0.16.0
colorama                     0.4.6
confection                   0.1.4
contourpy                    1.3.1
cycler                       0.11.0
cymem                        2.0.6
en_core_web_sm               3.8.0
et-xmlfile                   1.1.0
flatbuffers                  25.2.10
fonttools                    4.55.3
fsspec                       2025.3.2
gast                         0.6.0
google-pasta                 0.2.0
grpcio                       1.71.0
h5py                         3.13.0
hdbscan                      0.8.39
humanize                     4.12.2
idna                         3.7
importlib_metadata           8.5.0
importlib_resources          6.4.0
Jinja2                       3.1.6
joblib                       1.4.2
keras                        3.9.2
kiwisolver                   1.4.8
langcodes                    3.3.0
libclang                     18.1.1
llvmlite                     0.44.0
Markdown                     3.8
markdown-it-py               2.2.0
MarkupSafe                   3.0.2
matplotlib                   3.10.3
mdurl                        0.1.0
ml_dtypes                    0.5.1
murmurhash                   1.0.12
namex                        0.0.9
numba                        0.61.2
numpy                        2.1.3
openpyxl                     3.1.5
opt_einsum                   3.4.0
optree                       0.15.0
packaging                    24.2
pandas                       2.2.3
patsy                        1.0.1
pillow                       11.2.1
pins                         0.8.7
pip                          25.1
preshed                      3.0.9
protobuf                     5.29.4
pydantic                     2.10.3
pydantic_core                2.27.1
Pygments                     2.19.1
pynndescent                  0.5.4
pyparsing                    3.2.0
PySide6                      6.9.0
PySocks                      1.7.1
python-dateutil              2.9.0.post0
pytz                         2024.1
PyYAML                       6.0.2
RapidFuzz                    3.13.0
requests                     2.32.3
rich                         13.9.4
scikit-learn                 1.6.1
scipy                        1.15.2
seaborn                      0.13.2
setuptools                   78.1.1
shellingham                  1.5.0
shiboken6                    6.9.0
six                          1.17.0
smart-open                   5.2.1
spacy                        3.8.5
spacy-legacy                 3.0.12
spacy-loggers                1.0.4
srsly                        2.5.1
statsmodels                  0.14.4
tensorboard                  2.19.0
tensorboard-data-server      0.7.2
tensorflow                   2.19.0
tensorflow-hub               0.16.1
tensorflow-io-gcs-filesystem 0.31.0
termcolor                    3.1.0
tf_keras                     2.19.0
thinc                        8.3.2
threadpoolctl                3.5.0
tornado                      6.4.2
tqdm                         4.67.1
typer                        0.9.0
typing_extensions            4.12.2
tzdata                       2025.2
umap-learn                   0.5.7
unicodedata2                 15.1.0
urllib3                      2.3.0
uv                           0.7.13
wasabi                       0.9.1
weasel                       0.3.4
Werkzeug                     3.1.3
wheel                        0.45.1
win-inet-pton                1.1.0
wrapt                        1.17.2
xxhash                       3.5.0
zipp                         3.21.0

Authors
Angelo Capodici and Alessandro Filippeschi are the authors of the code. Francesca Noci contributed to the creation of the vocabulary

Support
Authors Angelo Capodici (a.capodici@santannapisa.it) and Alessandro Filippeschi (a.filippeschi@santannapisa.it) can be contacted for support.
