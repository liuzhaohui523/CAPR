## Download

1. Please follow the installation instructions from [HybrIK](https://github.com/Jeff-sjtu/HybrIK) since our code is built based on HybrIK.



2. Download pretrained model checkpoints from [Baidu Drive](https://pan.baidu.com/s/16Htb7ws-Cwcp1lMYUcMkJw?pwd=catm) (提取码:catm)

    The file download address in model_files is [Baidu Drive](https://pan.baidu.com/s/1a8fpuk5ZYObtBPIYfsf2fA?pwd=8tkw ) (提取码:8tkw)
    Put the downloaded pre-trained files into ${ROOT}/model_files.




3. Download SMPL models from their official websites
    Download the SMPL model `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [here](https://smpl.is.tue.mpg.de/) at `common/utils/smplpytorch/smplpytorch/native/models`.



4. Download datasets
   
    Download *Human3.6M*, *MPI-INF-3DHP*, *3DPW* and *MSCOCO* datasets. You need to follow directory structure of the `data` as below. Thanks to the great job done by Moon *et al.*, we use the Human3.6M images provided in [PoseNet](https://github.com/mks0601/3DMPPE_POSENET_RELEASE).
    Download link for MPI-INF-3DHP (https://github.com/alisa-yang/mpi_inf_3dhp)
    ```
    |-- data
    `-- |-- h36m
        `-- |-- annotations
            `-- images
    `-- |-- pw3d
        `-- |-- json
            `-- imageFiles
    `-- |-- 3dhp
        `-- |-- annotation_mpi_inf_3dhp_train.json
            |-- annotation_mpi_inf_3dhp_test.json
            |-- mpi_inf_3dhp_train_set
            `-- mpi_inf_3dhp_test_set
    `-- |-- coco
        `-- |-- annotations
            |   |-- person_keypoints_train2017.json
            |   `-- person_keypoints_val2017.json
            |-- train2017
            `-- val2017
    ```
    * Download Human3.6M parsed annotations. [ [Google](https://drive.google.com/drive/folders/1tLA_XeZ_32Qk86lR06WJhJJXDYrlBJ9r?usp=sharing) | [Baidu](https://pan.baidu.com/s/1bqfVOlQWX0Rfc0Yl1a5VRA) ]
    * Download 3DPW parsed annotations. [ [Google](https://drive.google.com/drive/folders/1f7DyxyvlC9z6SFT37eS6TTQiUOXVR9rK?usp=sharing) | [Baidu](https://pan.baidu.com/s/1d42QyQmMONJgCJvHIU2nsA) ]
    * Download MPI-INF-3DHP parsed annotations. [ [Google](https://drive.google.com/drive/folders/1Ms3s7nZ5Nrux3spLxmMMAQWc5aAIecmv?usp=sharing) | [Baidu](https://pan.baidu.com/s/1aVBDudbDRT1w_ZxQc9zicA) ]



