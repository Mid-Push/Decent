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
- [label2city]()
- [horse2zebra]()
- [cat2dog]()
- [selfie2anime]()

Will upload the checkpoints by 12/10/2022

### Evaluation Script of label2city
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
