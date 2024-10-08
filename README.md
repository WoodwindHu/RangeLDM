# RangeLDM
[ECCV 2024] Official implementation of "RangeLDM: Fast Realistic LiDAR Point Cloud Generation"

## Models
### KITTI360
| Model | MMD | FRD | JSD | Checkpoint | Generated Point Clouds |
|:-----:|:---:|:---:|:---:|:----------:|:----------------------:|
|[RangeLDM](ldm/configs/RangeLDM.yaml)|**3.07 × 10^−5** | 1074.9 | 0.045| [[PKU Disk]](https://disk.pku.edu.cn/link/AA526388EF6AA34255BD62158520CC343D)<br/>(115MB) | [[1k samples]](https://disk.pku.edu.cn/link/AA734EA9B6BDF245F5B1F7F4ABE4A4E754)
|[RangeDM](ldm/configs/RangeDM.yaml) |4.14 × 10^−5 | **899.0** | **0.040** | [[PKU Disk]](https://disk.pku.edu.cn/link/AA077B0EF8964145A3A37EA0BEF54EBD69)<br/>(401MB) | [[1k samples]](https://disk.pku.edu.cn/link/AA36A72F9CB4B6404686629B27CEDBA321)

### nuScenes 
| Model | MMD | JSD | Checkpoint | Generated Point Clouds |
|:-----:|:---:|:---:|:----------:|:----------------------:|
|[RangeLDM](ldm/configs/nuscenes.yaml)| 1.9 × 10^−4 | 0.054 | [[PKU Disk]](https://disk.pku.edu.cn/link/AA353D9629263C44C99CB7C5B64875C166)<br/>(153MB) | [[1k samples]](https://disk.pku.edu.cn/link/AA414B91DFA62C4E5DA5D0DB706616D18B)

## Train

### VAE
```
cd vae
python main.py --base configs/kitti360.yaml
```

### LDM
```
cd ldm
accelerate launch train_unconditional.py --cfg configs/RangeLDM.yaml # for unconditional generation
accelerate launch train_conditional.py --cfg configs/upsample.yaml # for conditional generation
```

## Evaluation

see [metrics/metrics.md](metrics/metrics.md)

## Citation
If you find our work useful, please cite:
```
@article{hu2024rangeldm,
  title={RangeLDM: Fast Realistic LiDAR Point Cloud Generation},
  author={Hu, Qianjiang and Zhang, Zhimin and Hu, Wei},
  journal={arXiv preprint arXiv:2403.10094},
  year={2024}
}
```
