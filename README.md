# ALERT
# ALERT Dataset Benchmark Tools

This repository provides dataset generation tools for experiments using the **ALERT Dataset**, a radar-based dataset for **Driver Activity Recognition (DAR)**.

The scripts in this repository convert raw radar CSV files into **experiment-ready pickle datasets** that can be directly used for machine learning experiments.

This repository accompanies our research introducing the ALERT dataset and the ISA-ViT framework.

---

# Paper Overview

Distracted driving contributes to fatal crashes worldwide. To address this problem, researchers are exploring **Driver Activity Recognition (DAR)** using **Impulse Radio Ultra-Wideband (IR-UWB) radar**.

Compared to camera-based systems, radar provides several advantages:

- Strong interference resistance  
- Low power consumption  
- Privacy preservation (no visual images)

However, two challenges limit the adoption of radar-based DAR:

1. Lack of large-scale real-world UWB datasets covering diverse distracted driving behaviors  
2. Difficulty adapting Vision Transformers (ViTs) to radar data with non-standard input dimensions

To address these challenges, we introduce the **ALERT Dataset** and the **Input-Size-Agnostic Vision Transformer (ISA-ViT)**.

## ALERT Dataset

The ALERT dataset contains:

- **10,220 radar samples**
- **7 driving activities**
- Collected in **real driving conditions**

Activities included in the dataset:

- Relax  
- Drive  
- Nod  
- Drink  
- Phone  
- Smoke  
- Panel

## ISA-ViT

ISA-ViT is a framework designed to adapt radar inputs for Vision Transformer architectures.

Key ideas include:

- Resizing radar inputs while preserving radar-specific information such as Doppler and phase data
- Adjusting patch embeddings to support flexible input sizes
- Leveraging pretrained positional embedding vectors (PEVs)
- Combining **range-domain and frequency-domain features** through domain fusion

Our experiments show that **ISA-ViT improves accuracy by 22.68% compared to existing ViT-based approaches** for radar-based DAR.

By publicly releasing the ALERT dataset and preprocessing tools, we aim to support future research in radar-based driver activity recognition.

---

# Dataset Download

The ALERT dataset can be downloaded from:

https://doi.org/10.6084/m9.figshare.31550620

After downloading, organize the dataset as follows:

    project/
    │
    ├ ALERT_train/
    │   ├ p1_relax_1_t.csv
    │   ├ p1_relax_1_f.csv
    │   ├ p1_drive_1_t.csv
    │   ├ p1_drive_1_f.csv
    │   └ ...
    │
    └ ALERT_makeDataset.py

Each activity recording contains:

- `_t.csv` : time-domain radar signal  
- `_f.csv` : frequency-domain radar signal  

---

# Quick Start

Example workflow:

    git clone https://github.com/ALERTdataset/ALERT.git
    cd ALERT
    python3 ALERT_makeDataset.py common cropO 500

Generated datasets will appear in:

    ./pickles/

---

# Dataset Generation

The script converts raw radar CSV files into pickle datasets used for training and evaluation.

Usage:

    python3 ALERT_makeDataset.py [dataset_mode] [crop_mode] [sample_size]

Example:

    python3 ALERT_makeDataset.py common cropO 500

Arguments:

| Argument | Description |
|--------|--------|
| dataset_mode | `common` or `extend` |
| crop_mode | `cropO`, `cropX`, `CA`, `CA_fft`, `RD` |
| sample_size | length of radar segment |

---

# Dataset Generation Modes

## common

The **common mode** splits radar signals into **non-overlapping segments**.

Example segmentation:

    |---------sample1---------|
    |---------sample2---------|
    |---------sample3---------|

Each sample length is:

    sample_size

This produces the baseline dataset used in experiments.

---

## extend

The **extend mode** increases dataset size using **overlapping segmentation**.

Stride:

    stride = sample_size / 2

Example:

    sample1 : |---------A---------|
    sample2 :           |---------B---------|

    extended sample:
               |----A/2----|----B/2----|

This increases training samples while preserving temporal continuity.

---

# Preprocessing Modes

Several preprocessing configurations are provided to generate different radar representations.  
These options are intended as **reference implementations**, and users may modify or extend them depending on their research objectives.

## cropO

`cropO` extracts the radar data region where **multipath fading effects are mainly observed**.

This region typically contains strong reflections from the driver and surrounding environment, making it useful for capturing meaningful motion patterns related to driver activities.

---

## cropX

`cropX` uses the **entire radar data region without cropping**.

This configuration preserves all available radar information and allows researchers to explore models that learn directly from the full signal space.

---

## CA (Center Alignment)

`CA` focuses on the region corresponding to the **driver's body location**.

This preprocessing method crops a localized area around the driver's body to emphasize movements associated with driver activities such as drinking, using a phone, or interacting with the dashboard.

---

## CA_fft

`CA_fft` applies a transformation to the **CA-cropped data** to represent it in a **range–Doppler format**.

This representation captures motion-related information by highlighting velocity components associated with driver movements.

---

## RD (Range–Doppler)

`RD` converts the radar signal into a **range–Doppler representation** directly from the original radar data.

This representation is commonly used in radar signal processing to analyze motion dynamics and velocity patterns.

---

### Note

These preprocessing configurations are provided as **example pipelines used in our experiments**.  
Researchers are encouraged to **modify the preprocessing strategy according to their own experimental needs and model architectures**.

---

# Generated Files

For each subject, three pickle files are generated.

    {crop}_{subject}_{sample_size}_{num_classes}_time_data.pickle
    {crop}_{subject}_{sample_size}_{num_classes}_freq_data.pickle
    {crop}_{subject}_{sample_size}_{num_classes}_labels_data.pickle

Example:

    cropO_p1_500_7_time_data.pickle
    cropO_p1_500_7_freq_data.pickle
    cropO_p1_500_7_labels_data.pickle

---

# Activity Labels

| Label | Activity |
|------|------|
| 0 | relax |
| 1 | drive |
| 2 | nod |
| 3 | drink |
| 4 | phone |
| 5 | smoke |
| 6 | panel |

---

# Example: Loading Pickle Dataset

Example Python code:

    import pickle

    time_test = []
    freq_test = []
    labels_test = []
    tmp = []
    time_adapt =[]
    freq_adapt =[]
    labels_adapt =[]
    
    with open('./pickles/'+crop+'_'+tester+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_time_data.pickle', 'rb') as f:
        tmp = pickle.load(f)
    for label_idx in range(len(tmp)):
        time_adapt += tmp[label_idx][:N_shot]
        time_test += tmp[label_idx][50:]
    with open('./pickles/'+crop+'_'+tester+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_freq_data.pickle', 'rb') as f:
        tmp = pickle.load(f)
    for label_idx in range(len(tmp)):
        freq_adapt += tmp[label_idx][:N_shot]
        freq_test += tmp[label_idx][50:]
    with open('./pickles/'+crop+'_'+tester+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_labels_data.pickle', 'rb') as f:
        tmp = pickle.load(f)
    for label_idx in range(len(tmp)):
        labels_adapt += tmp[label_idx][:N_shot]
        labels_test += tmp[label_idx][50:]
    
    time_adapt = torch.tensor(np.array(time_adapt))
    freq_adapt = torch.tensor(np.array(freq_adapt))
    labels_adapt = torch.tensor(np.array(labels_adapt))
    
    time_test = torch.tensor(np.array(time_test))
    freq_test = torch.tensor(np.array(freq_test))
    labels_test = torch.tensor(np.array(labels_test))   
    
    adapt_dataset = torch.utils.data.TensorDataset(time_adapt, freq_adapt, labels_adapt)
    adapt_dataloader = DataLoader(adapt_dataset, batch_size=5, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(time_test, freq_test, labels_test)
    test_dataloader = DataLoader(test_dataset, batch_size=25, shuffle=False)
    

---

# Dataset Processing Pipeline

    Raw Radar Signal
          ↓
      Cropping
          ↓
      Segmentation
          ↓
    Dataset Extension (optional)
          ↓
      Pickle Dataset
          ↓
        Training

---

# Citation

If you use the ALERT dataset or this code in your research, please cite our paper.

    @article{park2026alert,
    title={ALERT Open Dataset and Input-Size-Agnostic Vision Transformer for Driver Activity Recognition using IR-UWB},
    author={Park, Jeongjun and Hwang, Sunwook and Noh, Hyeonho and Yang, Jin Mo and Yang, Hyun Jong and Bahk, Saewoong},
    journal={IEEE Access},
    year={2026},
    publisher={IEEE}
    }

---

# License and Disclaimer

The ALERT dataset and code are provided **for research and academic purposes only**.

The dataset and associated software are provided **"as is"**, without warranty of any kind.

The authors and contributors:

- make **no guarantees regarding the correctness or completeness** of the dataset  
- assume **no responsibility for errors or omissions** in the dataset  
- shall **not be liable for any damages arising from the use of this dataset**

Users are responsible for ensuring compliance with applicable laws and regulations when using the dataset.

---
