# Convolutional Reconstruction Model

Official implementation for *CRM: Single Image to 3D Textured Mesh with Convolutional Reconstruction Model*.

**CRM is a feed-forward model which can generate 3D textured mesh in 10 seconds.**

## [Project Page](https://ml.cs.tsinghua.edu.cn/~zhengyi/CRM/) | [Arxiv](https://arxiv.org/abs/2403.05034) | [HF-Demo](https://huggingface.co/spaces/Zhengyi/CRM) | [Weights](https://huggingface.co/Zhengyi/CRM)

https://github.com/thu-ml/CRM/assets/40787266/04a202fe-e35a-4549-b416-84c1f338f0f1

## Try CRM 🍻
* Try CRM at [Huggingface Demo](https://huggingface.co/spaces/Zhengyi/CRM).
* Try CRM at [Replicate Demo](https://replicate.com/camenduru/crm). Thanks [@camenduru](https://github.com/camenduru)! 

## Install

Required packages are listed in `requirements.txt`.

## Inference

We suggest gradio for a visualized inference.

```
gradio app.py
```

![image](https://github.com/thu-ml/CRM/assets/40787266/4354d22a-a641-4531-8408-c761ead8b1a2)

## Acknowledgement
- [ImageDream](https://github.com/bytedance/ImageDream)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [kiuikit](https://github.com/ashawkey/kiuikit)
- [GET3D](https://github.com/nv-tlabs/GET3D)

## Citation

```
@article{wang2024crm,
  title={CRM: Single Image to 3D Textured Mesh with Convolutional Reconstruction Model},
  author={Zhengyi Wang and Yikai Wang and Yifei Chen and Chendong Xiang and Shuo Chen and Dajiang Yu and Chongxuan Li and Hang Su and Jun Zhu},
  journal={arXiv preprint arXiv:2403.05034},
  year={2024}
}
```
