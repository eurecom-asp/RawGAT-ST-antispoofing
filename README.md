End-to-End Spectro-Temporal Graph Attention Networks for Speaker Verification Anti-Spoofing and Speech Deepfake Detection
===============
This repository contains our implementation of the paper published in the ASVspoof 2021 workshop, "End-to-End Spectro-Temporal Graph Attention Networks for Speaker Verification Anti-Spoofing and Speech Deepfake Detection". This work demonstrates the effectivness of end-to-end spectro-temporal graph attention network (GAT) which learns the relationship between cues spanning different sub-bands and temporal intervals for anti-spoofing and speech deepfake detection.
[Paper link here](https://arxiv.org/abs/2107.12710)

## Installation
First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ git clone https://github.com/eurecom-asp/RawGAT-ST-antispoofing.git
$ conda create --name RawGAT_ST_anti_spoofing python=3.8.8
$ conda activate RawGAT_ST_anti_spoofing
$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
$ pip install -r requirements.txt
```


## Experiments

### Dataset
Our experiments are done in the logical access (LA) partition of the ASVspoof 2019 dataset, which can can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3336).

### Training
To train the model run:
```
python main.py --track=logical --loss=WCE   --lr=0.0001 --batch_size=10
```

### Testing

To evaluate your own model on LA evaluation dataset:

```
python main.py --track=logical --loss=WCE --is_eval --eval --model_path='/path/to/your/best_model.pth' --eval_output='eval_CM_scores_file.txt'
```

We also provide a pre-trained models. To use it you can run: 
```
python main.py --track=logical --loss=WCE --is_eval --eval --model_path='Pre_trained_models/RawGAT_ST_mul/Best_epoch.pth' --eval_output='RawGAT_ST_mul_LA_eval_CM_scores.txt'
```

If you would like to compute scores on development dataset simply run:

```
python main.py --track=logical --loss=WCE --eval --model_path='/path/to/your/best_model.pth' --eval_output='dev_CM_scores_file.txt'
```
Compute the min t-DCF and EER(%) on development dataset
```
python tDCF_python_v2/evaluate_tDCF_asvspoof19_eval_LA.py  dev  'dev_CM_scores_file.txt'
``` 

Compute the min t-DCF and EER(%) on evaluation dataset
```
python tDCF_python_v2/evaluate_tDCF_asvspoof19_eval_LA.py  Eval  'eval_CM_scores_file.txt'
```

## Acknowledgements

This repository is built on our End-to-end RawNet2 (ASVspoof2021 Challenge baseline).
- [ASVspoof 2021 Challenge baseline repo](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2)

AASIST (Extended work of RawGAT-ST paper) is available here.

- [AASIST anti-spoofing software from Jee-weon Jung, NAVER Corp.](https://github.com/clovaai/aasist)

Authors would like to acknowledge other repositories as well.
- [min t-DCF implementation](https://www.asvspoof.org/resources/tDCF_python_v2.zip)
- [core scripts from Xin Wang, NII Japan](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts)




## Contact
For any query regarding this repository, please contact:
- Hemlata Tak: tak[at]eurecom[dot]fr
## Citation
If you use RawGAT-ST, AASIST and RawNet2 for anti-spoofing please use the following citations:

[1] End-to-end spectro-temporal graph attention networks for speaker verification anti-spoofing and speech deepfake detection
```bibtex
@inproceedings{tak21_asvspoof,
  author={Hemlata Tak and Jee-weon Jung and Jose Patino and Madhu Kamble and Massimiliano Todisco and Nicholas Evans},
  title={{End-to-end spectro-temporal graph attention networks for speaker verification anti-spoofing and speech deepfake detection}},
  year=2021,
  booktitle={Proc. 2021 Edition of the Automatic Speaker Verification and Spoofing Countermeasures Challenge},
  pages={1--8},
  doi={10.21437/ASVSPOOF.2021-1}
}
```
[2] AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
```bibtex
@INPROCEEDINGS{Jung2021AASIST,
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  booktitle={arXiv preprint arXiv:2110.01200}, 
  title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks}, 
  year={2021}
```
[3] End-to-End anti-spoofing with RawNet2
```bibtex
@INPROCEEDINGS{Tak2021End,
  author={Tak, Hemlata and Patino, Jose and Todisco, Massimiliano and Nautsch, Andreas and Evans, Nicholas and Larcher, Anthony},
  booktitle={Proc. ICASSP}, 
  title={End-to-End anti-spoofing with RawNet2}, 
  year={2021},
  pages={6369-6373}
}
```
