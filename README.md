# Temporal Multirate Finite Element Method

In this repository, we provide the code used for the numerical experiments in the paper "A monolithic space-time temporal multirate finite element framework for interface and volume coupled problems" by J. Roth, M. Soszyńska, T. Richter and T. Wick.
Additionally, we provide another code for the staggered temporal multirate finite element method for poroelasticity at the example of the Mandel problem.

For technical details, we refer to [the journal article](https://doi.org/10.1016/j.cam.2024.115831) and [the arXiv preprint](https://doi.org/10.48550/arXiv.2307.12455).
To cite this code, please use the following BibTeX citation:

```
@article{RoSoRiWi2024,
  title = {A monolithic space–time temporal multirate finite element framework for interface and volume coupled problems},
  journal = {Journal of Computational and Applied Mathematics},
  volume = {446},
  pages = {115831},
  year = {2024},
  issn = {0377-0427},
  doi = {https://doi.org/10.1016/j.cam.2024.115831},
  url = {https://www.sciencedirect.com/science/article/pii/S0377042724000803},
  author = {Julian Roth and Martyna Soszyńska and Thomas Richter and Thomas Wick},
  keywords = {Galerkin space–time, Multirate, Monolithic framework, Interface coupling, Volume coupling, Mandel’s benchmark}
}
```

## Motivation

We are often dealing with multiphysics problems that have dynamics on different time scales.
An example of this is [atherosclerosis](https://www.hopkinsmedicine.org/health/conditions-and-diseases/atherosclerosis), where plaque deposits on the walls of a human's arteries and then reduces blood flow, possibly even causing heart attacks or strokes.
Simulating this medical phenomenon is complex because we have to accurately model:
- the plaque growth,
- the blodd flow.
  
This is challenging because the plaque deposition takes place on a scale of weeks, months or even years, while the heart pumps blood through the arteries roughly 1-2 times per second.
Consequently, we could either simulate both physical phenomena on a scale of seconds (or fractions of a second), we could try to average everything in time (homogeneization) or we could separately solve both physics on their respective time scales (staggered scheme).

<u>BUT:</u> Can we also solve both physics at once, while using the respective time scale for each problem? This would be faster than solving everything on the smaller time scale and is more accurate and numerically robsut than using homogenization or using a staggered scheme.
