# Data Preparation

## NuPlan
We use [NuPlan](https://nuplan.org/) for training and testing. We organize our training and testing datasets as follows.

### Download
Please download all the splits in [NuPlan](https://nuplan.org/). We follow [NuPlan-Download-CLI](https://github.com/Syzygianinfern0/NuPlan-Download-CLI) to download all the splits. Once you download all the files, please `unzip` them first.

### Reorganize
Please move your files and make sure that they are organized like this:
```
$<your-nuplan-data-root>
├── splits
│     ├── mini
│     │    ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
│     │    ├── 2021.06.09.17.23.18_veh-38_00773_01140.db
│     │    ├── ...
│     │    └── 2021.10.11.08.31.07_veh-50_01750_01948.db
│     └── trainval
│          ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
│          ├── 2021.06.09.17.23.18_veh-38_00773_01140.db
│          ├── ...
│          └── 2021.10.11.08.31.07_veh-50_01750_01948.db
└── sensor_blobs
        ├── 2021.05.12.22.00.38_veh-35_01008_01518
        │    ├── CAM_F0
        │    │     ├── c082c104b7ac5a71.jpg
        │    │     ├── af380db4b4ca5d63.jpg
        │    │     ├── ...
        │    │     └── 2270fccfb44858b3.jpg
        │    ├── CAM_B0
        │    ├── CAM_L0
        │    ├── CAM_L1
        │    ├── CAM_L2
        │    ├── CAM_R0
        │    ├── CAM_R1
        │    ├── CAM_R2
        │    └──MergedPointCloud
        │         ├── 03fafcf2c0865668.pcd
        │         ├── 5aee37ce29665f1b.pcd
        │         ├── ...
        │         └── 5fe65ef6a97f5caf.pcd
        │
        ├── 2021.06.09.17.23.18_veh-38_00773_01140
        ├── ...
        └── 2021.10.11.08.31.07_veh-50_01750_01948
```



## NuScenes
To train or evaluate DriveLaW on nuScenes, please follow the [official instructions](https://www.nuscenes.org/download) to download all splits of nuScenes data (v1.0). After downloading, it should organize like this:
```
$<your-nusc-data-root>
├── lidarseg
├── maps
├── samples
├── sweeps
├── v1.0-mini
├── v1.0-test
└── v1.0-trainval
```
