# ML: Template project

The idea is to adapt a template and keep a good structure between projects. A good backbone helps
reaching goals and being efficient. It also makes team working easier.

![Structure of an ML Pipeline](https://miro.medium.com/max/876/1*fKlYtetGpfWDw0x7rdO6jQ.png)


```
.
├── README.md
└── project
    ├── data
    │   ├── external
    │   ├── interim <-unfinished processing to data or different processing unfit for training
    │   ├── processed <- processed data ready for training
    │   └── raw
    ├── models <-saved models
    ├── notebooks <-usefull for data visualizations and presenting the results
    └── src
        ├── data <-scripts used to build the processed data
        ├── features <-scripts used build the features fed into the model
        └── model <-training and evaluation of the models
```
