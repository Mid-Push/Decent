# Decent: Unpaired Image-to-Image Translation with Density Changing Regularization (Neurips2022)

### Basic Usage

- Training:
```bash
python train.py --dataroot=datasets/selfie2anime  
```
- Test:
```bash
python test.py --dataroot=datasets/selfie2anime
```
- The Weight `--lambda_var=0.01` 
- Number of Flow Blocks `--flow_blocks=1` 
- Learning Rate of Flow `--flow_lr=0.001` 
- Different flows `--flow_type=bnaf` BNAF works best for me. Feel free to experiment other flows.

### Pretrained Models
- [label2city](https://drive.google.com/file/d/1HBevQfsQLIcnwCQHQoCHlfbofLSwZYWQ/view?usp=sharing)
- [horse2zebra](https://drive.google.com/file/d/13GwdqDoH_BNt4iciLg758g5k1CWz3-t2/view?usp=sharing)
- [cat2dog](https://drive.google.com/file/d/1yOVDhsiVDtSuobGihsheMun_c9lXD7ON/view?usp=sharing)
- [selfie2anime](https://drive.google.com/file/d/1xFLJjK7jQmW-mA5Mpu_ywabrFHnSziZb/view?usp=sharing)
- [Summer2winter](https://drive.google.com/file/d/1JC3Eb8fDYw_iVivISkI_TSUlxrlN8HwA/view?usp=sharing)




### Evaluation Script of label2city
Different Pretrained-DRN and evaluation protocols can cause big performance gaps. So,
I created a repository to upload the [evaluation script of label2city](https://github.com/Mid-Push/evaluation_on_cityscapes). Hope the script could make the future evaluation easier.

## Citation
If you use this code for your research, please cite our [paper](https://openreview.net/pdf?id=RNZ8JOmNaV4):

```
@inproceedings{xieunsupervised,
  title={Unsupervised Image-to-Image Translation with Density Changing Regularization},
  author={Xie, Shaoan and Ho, Qirong and Zhang, Kun},
  booktitle={Advances in Neural Information Processing Systems},
year=2022,
}
```
