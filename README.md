# ALERT Dataset

## How to Use

### 1) Download the ALERT Dataset
You can download the ALERT dataset via the following link:

[ALERT Dataset - figshare](https://doi.org/10.6084/m9.figshare.28244525.v2)

### 2) Clone the GitHub Repository
You should clone this repository to your local machine:

git clone https://github.com/ALERTdataset/ALERT.git

### 3) Uncompress the Dataset
After downloading the dataset, you should uncompress the file. Rename the uncompressed folder to `ALERT_train`.

Then, place the `ALERT_train` directory at the same level as the cloned GitHub repository files.

### 4) Generate Pickle Files
To generate the pickle files, you need to use the `ALERT_makeDataset.py` script.

Run the script as follows:

python3 ALERT_makeDataset.py {common/extend} {cropO/cropX/CA/RD} sample_size

- `{common/extend}`: Choose whether to use the common or extended dataset.
- `{cropO/cropX/CA/RD}`: Choose the data format (e.g., `cropO`, `cropX`, `CA`, or `RD`).
- `sample_size`: Specify the sample size.

The generated pickle files will be stored in the `pickles` directory.

### 5) Benchmarking
You can perform benchmarking by using the `ALERT_main.py` script.

Run the script as follows:

python3 ALERT_bench.py models(All/GoogleNet/transferX) cropping(cropO/cropX/RD) sample_size adaptation_epoch experiment_number tester your_memo > ./ALERT/logs/logname.txt

- `models`: Choose the model (e.g., `All`, `GoogleNet`, `transferX`).
- `cropping`: Choose the cropping format (e.g., `cropO`, `cropX`, `RD`).
- `sample_size`: Specify the sample size.
- `adaptation_epoch`: Set the number of adaptation epochs.
- `experiment_number`: Specify the experiment number.
- `tester`: Select the tester from the options (`'jp'`, `'yh'`, `'wh'`, `'dk'`).
- `your_memo`: Add any additional notes or comments.

The output will be saved to the log file in the `./ALERT/logs/` directory.
