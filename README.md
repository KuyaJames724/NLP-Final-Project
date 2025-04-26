# minBERT – Final Project (NLP)

This repository contains our implementation of `minBERT`, a simplified BERT architecture for sentiment classification and multitask learning on three downstream NLP tasks: SST, CFIMDB, Quora Paraphrase, and STS.

## 🔧 Setup

To create and activate the environment:

```bash
source setup.sh
conda activate nlp_fp
```

If using your own Python setup:

```bash
pip install torch scikit-learn numpy tqdm
```

## 🧠 Project Structure

| **File**      | **Purpose** |
| ----------- | ----------- |
| `bert.py`      | Custom transformer and attention layers       |
| `classifier.py`   | SST and CFIMDB sentiment classification        |
| `optimizer.py` | Manual implementation of AdamW | 
| `multitask_classifier.py` | Quora, STS, and multitask BERT | 
| `datasets.py` | Data loaders and preprocessing |
| `evaluation.py` | Accuracy, F1, Pearson correlation |
| `prepare_submit.py` | Prepares `minBERT.zip` for submission |

## 📌 Additional packages *(not from `setup.sh`, but mostly harmless)*

| **Package**      | **Status** | **Comment**|
| ----------- | ----------- | ----------- |
| `scikit-learn==1.3.2`      | 🔼 Auto-installed |
| `torch==2.4.1`, `torchvision`, `torchaudio`   | ❌ Not matching setup.sh |
| `explainaboard-api-client==0.4.3`	| ❌ Not in setup.sh |
| `Brotli`, `chardet`, `charset-normalizer`, `urllib3`, `PySocks`, etc. | 🆗 Standard | 
| `mkl-*`, `numpy`, `scipy`, `sympy` | 🆗 Standard scientific packages | 
| `joblib`, `threadpoolctl` | 	🆗 Expected |
| `zipp`, `six`, `python-dateutil`, `typing_extensions` | 🆗 Common |


## 🏋️ Training Commands

### SST / CFIMDB – Pretrain

```bash
python classifier.py --option pretrain --epochs 10 --batch_size 8 --hidden_dropout_prob 0.3 --use_gpu
python classifier.py --option pretrain --use_gpu
```

### SST / CFIMDB – Finetune

```bash
python classifier.py --option finetune --epochs 10 --batch_size 8 --hidden_dropout_prob 0.3 --use_gpu
python classifier.py --option finetune --use_gpu
```

### Multitask (Quora, STS, SST)

```bash
python multitask_classifier.py --option finetune --epochs 10 --batch_size 8 --use_gpu
```

## 🧪 Results

### SST Accuracy

- Pretraining Dev Accuracy: 32.2%
- Fine-tuning Dev Accuracy: 51.5%

### CFIMDB Accuracy

- Pretraining Dev Accuracy: 57.1%
- Fine-tuning Dev Accuracy: 97.6%

### Quora & STS



## 📁 Output Files

These files were generated and are included in the zip submission:

```csharp
pretrain-sst-dev-out.csv
pretrain-sst-test-out.csv
pretrain-cfimdb-dev-out.csv
pretrain-cfimdb-test-out.csv

finetune-sst-dev-out.csv
finetune-sst-test-out.csv
finetune-cfimdb-dev-out.csv
finetune-cfimdb-test-out.csv

predictions/para-test-out.csv
predictions/sts-test-out.csv
```

## 🧠 Notes

- We observed that fine-tuning significantly improved SST and CFIMDB accuracy.

- AdamW was implemented from scratch and tested with provided unit tests.

- For multitask, we used a shared encoder and task-specific heads.

- We used PyTorch 2.4.1 instead of the default 1.8.0 in `setup.sh` since we ran the project on local machines. All functionality worked as expected under this version.

- We also had to modify `datasets.py` to open CSVs with UTF-8 encoding to avoid platform-specific decoding errors.



## 👥 Team

- James Miller

- Adam Bott

- Nelson Makia
