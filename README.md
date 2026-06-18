<div align="center">
  
# ELDiff: When Evidential Learning Meets Text-to-Image Diffusion

[Qingtao Pan](https://qingtaopan.github.io/), [Kai Ye](https://scholar.google.com/citations?hl=zh-CN&user=k6mAT9AAAAAJ), [Zhihao Dou](https://scholar.google.com/citations?user=JiBGiB8AAAAJ&hl=zh-CN&oi=ao), [Bing Ji](https://www.researchgate.net/profile/Bing-Ji-jibing), and [Shuo Li](https://case.edu/engineering/about/faculty-and-staff-directory/shuo-li)

**ECCV, 2026**

<a href='https://arxiv.org/abs/2603.11220'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

</div>

## Overview
In this paper, we propose ELDiff, a new evidential learning-supervised T2I diffusion model, which leverages the advantages of uncertainty metric and conflict detection to enhance the fault tolerance of unreliable segmentation maps and suppress semantic conflicts, strengthening object-wise consistency learning. Specifically, a pixel evidence loss is proposed to restrain overconfidence in unreliable labels through evidential regularization, and a token conflict loss is designed to weaken the contradiction between semantics through optimizing a measured conflict factor.
<p align="center">
  <img src="figs/fig2.pdf" width="70%"></a> <br>
</p>
