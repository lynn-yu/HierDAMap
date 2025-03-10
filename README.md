# HierDAMap

## HierDAMap: Towards Universal Domain Adaptive BEV Mapping via Hierarchical Perspective Priors
Siyu Li, Yihong Cao, Hao Shi, Yongsheng Zang, Xuan He, [Kailun Yang](https://yangkailun.com/)†, [Zhiyong Li](http://robotics.hnu.edu.cn/info/1071/1515.htm)†

## Motivation
<div align=center>
<img src="https://github.com/lynn-yu/HierDAMap/blob/main/intro.png" >
</div>

## Framework
<div align=center>
<img src="https://github.com/lynn-yu/HierDAMap/blob/main/domainframe.png" >
</div>

### Abstract
The exploration of Bird's-Eye View (BEV) mapping technology has driven significant innovation in visual perception technology for autonomous driving.  BEV mapping models need to be applied to the unlabeled real world, making the study of unsupervised domain adaptation models an essential path. However, research on unsupervised domain adaptation for BEV mapping remains limited and cannot perfectly accommodate all BEV mapping tasks. To address this gap, this paper proposes HierDAMap, a universal and holistic BEV domain adaptation framework with hierarchical perspective priors. Unlike existing research that solely focuses on image-level learning using prior knowledge, this paper explores the guiding role of perspective prior knowledge across three distinct levels: global, sparse, and instance levels. With these priors, HierDA consists of three essential components, including Semantic-Guided Pseudo Supervision (SGPS), Dynamic-Aware Coherence Learning (DACL), and Cross-Domain Frustum Mixing (CDFM). SGPS constrains the cross-domain consistency of perspective feature distribution through pseudo labels generated by vision foundation models in 2D space. To mitigate feature distribution discrepancies caused by spatial variations, DACL employs uncertainty-aware predicted depth as an intermediary to derive dynamic BEV labels from perspective pseudo-labels, thereby constraining the coarse BEV features derived from corresponding perspective features. CDFM, on the other hand, leverages perspective masks of view frustum to mix multi-view perspective images from both domains, which guides cross-domain view transformation and encoding learning through mixed BEV labels. Furthermore, this paper introduces intra-domain feature exchange data augmentation to enhance the efficiency of domain adaptation learning.  The proposed method is verified on multiple BEV mapping tasks, such as BEV semantic segmentation, high-definition semantic, and vectorized mapping.  It demonstrates competitive performance across various conditions, including weather scenarios, regions, and datasets.

### Update
2025.3 Init repository.

## 🤝 Publication:
Please consider referencing this paper if you use the ```code``` from our work.
Thanks a lot :)

```
@article{li2025hierdamap,
  title={HierDAMap: Towards Universal Domain Adaptive BEV Mapping via Hierarchical Perspective Priors},
  author={Li, Siyu and Cao, Yihong and Shi, Hao and Zang, Yongsheng and He, Xuan and Yang, Kailun and Li, Zhiyong},
  journal={arXiv preprint arXiv:2503.xxxxx},
  year={2025}
}
```
