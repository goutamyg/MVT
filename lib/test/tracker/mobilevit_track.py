import math

from lib.models.mobilevit_track.mobilevit_track import build_mobilevit_track
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import numpy as np

from lib.test.tracker.data_utils_mobilevit import Preprocessor, PreprocessorX_onnx
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond

import onnxruntime

import tensorrt as trt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pycuda.driver as cuda
import pycuda.autoinit

class MobileViTTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(MobileViTTrack, self).__init__(params)

        self.cfg = params.cfg
        self.backend = params.backend

        network = build_mobilevit_track(params.cfg, training=False)
        self.network = network

        if self.backend is "onnx":
            self.device = "cpu"
            # load the onnx model from disk
            onnx_checkpoint = self.params.checkpoint.split('.pth')[0] + '.onnx'
            assert os.path.isfile(onnx_checkpoint) is True, ("Download the onnx model from https://drive.google.com/drive/folders/1RAdn3ZXI_G7pBj4NDbtQVFPkClVd1IBm "
                                                              "or convert the pytorch model to onnx using tracking/pytorch2onnx.py script")
            self.ort_session = onnxruntime.InferenceSession(onnx_checkpoint, providers=['CPUExecutionProvider'])

            self.preprocessor = PreprocessorX_onnx()

        elif self.backend is "tensorrt":
            self.device = 'cuda'
            # load tensor-rt engine from disk
            trt_engine = self.params.checkpoint.split('.pth')[0] + '_FP16_TRT.plan'
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            trt_runtime = trt.Runtime(TRT_LOGGER)
            self.engine = self.load_trt_engine(trt_runtime, trt_engine)
            self.context = self.engine.create_execution_context()
            assert self.engine
            assert self.context

            # Setup I/O bindings
            self.setup_io_binding_trt()

            self.preprocessor = PreprocessorX_onnx()

        else:
            print("not a valid backend. Choose from onnx and tensorrt!")
            exit()

        self.state = None
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constraint
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).to(self.device)

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}


    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr

        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        self.z_dict = template

        self.box_mask_z = None

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        x_dict = search

        if self.backend is "onnx":
            ort_inputs = {'x': x_dict[0].astype(np.float32),
                          'z': self.z_dict[0].astype(np.float32)}
            out_ort = self.ort_session.run(None, ort_inputs)

            pred_score_map_ort = out_ort[1]

            # add hann windows
            response_ort = self.output_window * torch.from_numpy(pred_score_map_ort).to(self.device)

            # response = pred_score_map
            pred_boxes_ort = self.network.box_head.cal_bbox(response_ort, torch.from_numpy(out_ort[2]).to(self.device),
                                                            torch.from_numpy(out_ort[3]).to(self.device))
            pred_boxes_ort = pred_boxes_ort.view(-1, 4)

            # Baseline: Take the mean of all pred boxes as the final result
            pred_box_ort = (pred_boxes_ort.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            best_bbox = self.map_box_back(pred_box_ort, resize_factor)

        elif self.backend is "tensorrt":

            # Process I/O and execute the network
            cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(self.z_dict[0]))
            cuda.memcpy_htod(self.inputs[1]['allocation'], np.ascontiguousarray(x_dict[0]))

            self.context.execute_v2(self.allocations)
            for o in range(len(self.outputs_numpy)):
                cuda.memcpy_dtoh(self.outputs_numpy[o], self.outputs[o]['allocation'])

            # add hann windows
            response_trt = self.output_window * torch.from_numpy(self.outputs_numpy[1]).to(self.device)

            pred_boxes_trt = self.network.box_head.cal_bbox(response_trt,
                                                            torch.from_numpy(self.outputs_numpy[2]).to(self.device),
                                                            torch.from_numpy(self.outputs_numpy[0]).to(self.device))
            pred_boxes_ort = pred_boxes_trt.view(-1, 4)

            # Baseline: Take the mean of all pred boxes as the final result
            pred_box_trt = (pred_boxes_ort.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            best_bbox = self.map_box_back(pred_box_trt, resize_factor)

        else:
            print("not a valid backend. Choose from onnx and tensorrt!")
            exit()

        # get the final box result
        self.state = clip_box(best_bbox, H, W, margin=10)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def load_trt_engine(self, trt_runtime, plan_path):

        trt.init_libnvinfer_plugins(None, "")
        with open(plan_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def setup_io_binding_trt(self):
        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

        # Prepare the output data
        self.outputs_numpy = []
        for shape, dtype in self.output_spec():
            self.outputs_numpy.append(np.zeros(shape, dtype))

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(self, wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return MobileViTTrack