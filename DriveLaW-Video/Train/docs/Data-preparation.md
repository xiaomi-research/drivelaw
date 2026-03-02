# Data Preparation for DriveLaW-Video

This document describes how to download and organize driving datasets that can be
used to train and evaluate the DriveLaW-Video world model. The examples below
follow the structure of NuPlan and nuScenes, but you can adapt them to your own
data if desired.

## NuPlan

We primarily use [NuPlan](https://nuplan.org/) for training and testing. We organize our training and testing datasets as follows.

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

### Create Meta Infos
In this part, we create meta infos (`json` files) to read the data easily. You can use it to create either the training or test set by specifying the split argument in `create_nuplan_json.py`:
``` bash
python3 create_nuplan_json.py
```
📌 **Note:** Be sure to update the save path in `create_nuplan_json.py` before running the script.


Alternatively, you can also organize your own test data in this way and run your
own evaluation scripts:
```
data
├── video-1
│   ├── 000000.png
│   ├── 000001.png
│   ├── ...
│   ├── 000009.png
│   ├── pose.npy
│   └── yaw.npy
├── video-2
│   ├── 000000.png
│   ├── 000001.png
│   ├── ...
│   ├── 000009.png
│   ├── pose.npy
│   └── yaw.npy
├── ...
├── video-n
│   ├── 000000.png
│   ├── 000001.png
│   ├── ...
│   ├── 000009.png
│   ├── pose.npy
│   └── yaw.npy
```

📌 **Note:** Please ensure that the shapes of `poses` and `yaws` are correct:

* `poses` should have shape `[1, length, 2]`,
* `yaws` should have shape `[1, length, 1]`,
  where `length >= condition_frames + 1`.

The meaning of each component is as follows:

1. `pose_x`: The relative translation along the x-axis between two consecutive frames (unit: meter), where forward is positive. Corresponds to $\Delta x_{t-1 \to t}$ in the paper.
2. `pose_y`: The relative translation along the y-axis between two consecutive frames (unit: meter), where right is positive. Corresponds to $\Delta y_{t-1 \to t}$ in the paper.
3. `yaw`: The relative rotation between two consecutive frames (unit: degree), where turning left is positive. Corresponds to $\Delta \theta_{t-1 \to t}$ in the paper.

## NuScenes

To train or evaluate DriveLaW-Video on nuScenes, please follow the
[official instructions](https://www.nuscenes.org/download) to download all splits
of nuScenes data (v1.0). After downloading, it should be organized like this:
```
\<your-nusc-data-root\>
├── lidarseg
├── maps
├── samples
├── sweeps
├── v1.0-mini
├── v1.0-test
└── v1.0-trainval
```
You can then generate your own meta infos (preprocessed JSON files) in the same
format as described for NuPlan above, using your own scripts or data processing
pipeline. These JSON files can then be used as metadata inputs for
`scripts/preprocess_dataset.py` in `DriveLaW-Video/Train`.