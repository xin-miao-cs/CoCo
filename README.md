# CoCo

## Setup

### Install dependencies
Please install all the dependency packages using the following command:
```
pip install -r requirements.txt
```

### Download and preprocess the datasets
Our experiments are based on four datasets: SemEval, TACRED, PubMed, ACE2005. Please find the links and pre-processing below:
* SemEval: We provide the processed SemEval dataset in the "data" folder.
* TACRED: We can't provide the TACRED dataset directly because of copyright, but the dataset can be downloaded in https://catalog.ldc.upenn.edu/LDC2018T24.
* PubMeb: We can't provide the PubMeb dataset directly because of copyright, but you can get the PubMed dataset via FTP as described in https://pubmed.ncbi.nlm.nih.gov/download/.
* ACE05: We use the preprocessing code from [DyGIE repo](https://github.com/luanyi/DyGIE/tree/master/preprocessing). Please follow the instructions to preprocess the ACE05 datasets.


