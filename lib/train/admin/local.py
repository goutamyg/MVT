class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/goutam/VisualTracking/research_code_for_github/MVT'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/goutam/VisualTracking/research_code_for_github/MVT/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/goutam/VisualTracking/research_code_for_github/MVT/pretrained_networks'
        self.lasot_dir = '/home/goutam/Datasets/train_data/lasot'
        self.got10k_dir = '/home/goutam/Datasets/train_data/got10k/train'
        self.got10k_val_dir = '/home/goutam/Datasets/train_data/got10k/val'
        self.lasot_lmdb_dir = '/home/goutam/VisualTracking/research_code_for_github/MVT/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/goutam/VisualTracking/research_code_for_github/MVT/data/got10k_lmdb'
        self.trackingnet_dir = '/home/goutam/VisualTracking/research_code_for_github/MVT/data/trackingnet'
        self.trackingnet_lmdb_dir = '/home/goutam/VisualTracking/research_code_for_github/MVT/data/trackingnet_lmdb'
        self.coco_dir = '/home/goutam/Datasets/train_data/coco'
        self.coco_lmdb_dir = '/home/goutam/VisualTracking/research_code_for_github/MVT/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/goutam/VisualTracking/research_code_for_github/MVT/data/vid'
        self.imagenet_lmdb_dir = '/home/goutam/VisualTracking/research_code_for_github/MVT/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
