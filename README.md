# MVT - Mobile Vision Transformer-based Visual Object Tracking
The official implementation of **MVT**
![MVT_block](assets/MVT.png)

## News
**`22-08-2023`**: The MVT tracker training and inference code is released

## Installation

Install the dependency packages using the environment file `mvt_pyenv.yml`.

Generate the relevant files:
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, modify the datasets paths by editing these files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Training

* Set the path of training datasets in `lib/train/admin/local.py`
* Place the pretrained backbone model under the `pretrained_models/` folder
* For data preparation, please refer to [this](https://github.com/botaoye/OSTrack/tree/main)
* Uncomment lines `63, 67, and 71` in the [base_backbone.py](https://github.com/goutamyg/MVT/blob/main/lib/models/mobilevit_track/base_backbone.py) file.  
* Run
```
python tracking/train.py --script mobilevit_track --config mobilevit_256_128x1_got10k_ep100_cosine_annealing --save_dir ./output --mode single
```
* The training logs will be saved under `output/logs/` folder

## Tracker Evaluation

* Update the test dataset paths in `lib/test/evaluation/local.py`
* Place the pretrained tracker model under `output/checkpoints/` folder 
* Run
```
python tracking/test.py --tracker_name mobilevit_track --tracker_param mobilevit_256_128x1_got10k_ep100_cosine_annealing --dataset got10k_test/trackingnet/lasot
```
* Change the `DEVICE` variable between `cuda` and `cpu` in the `--tracker_param` file for GPU and CPU-based inference, respectively  
* The raw results will be stored under `output/test/` folder

## Visualization

## Acknowledgements
* We use the Separable Self-Attention Transformer implementation and the pretrained `MobileViT` backbone from [ml-cvnets](https://github.com/apple/ml-cvnets). Thank you!
* Our training code is built upon [OSTrack](https://github.com/botaoye/OSTrack) and [PyTracking](https://github.com/visionml/pytracking)

## Citation
