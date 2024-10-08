
## Prepare for metrics

Install all python packages for evaluation with conda environment setup file: 
```bash
conda env create -f environment.yml
conda activate lidar3
```

Running FID requires downloading the 1024 backbone file for rangenet++ the following [link](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet53-1024.tar.gz)  and extracting it to the folder rangenetpp/lidar\_bonnetal\_master/darknet53-1024. This model is provided by the [RangeNet++ repository](https://github.com/PRBonn/lidar-bonnetal). Finally, lidargen needs to be run with the --fid option. This can be done by running the following commands:  

1. `curl -L http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet53-1024.tar.gz --output darknet53-1024.tar.gz`
1. `tar -xzvf darknet53-1024.tar.gz`
1. `mv darknet53-1024/backbone rangenetpp/lidar_bonnetal_master/darknet53-1024/`
1. `rm darknet53-1024.tar.gz`
1. `rm -r darknet53-1024`

## evaluate

Evaluate kitti360 unconditional generation:
```
export KITTI360_DATASET=/path/to/dataset/KITTI-360/
python metric.py --mmd --jsd --fid --exp path_to_generated_bin_files
```

Evaluate kitti360 upsampling generation:
```
export KITTI360_DATASET=/path/to/dataset/KITTI-360/
python metric.py --iou --accuracy --mae --exp path_to_generated_bin_files
```

Evaluate nuScenes unconditional generation:
```
export NUSCENES_DATASET=/path/to/dataset/NUSCENES/
python metric.py --mmd --jsd --nus --exp path_to_generated_bin_files
```

