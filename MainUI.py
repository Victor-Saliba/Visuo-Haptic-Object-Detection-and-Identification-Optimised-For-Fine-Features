"""
Live demo that combines:

1. Faster-R-CNN (MMDetection) for general object detection
2. ResNet-18 image classifier for screw / nut sub-classes (vision)
3. GelSight Mini tactile ResNet-18 classifier (tactile)

Press **G** to toggle GelSight mode.
When GelSight is ON, RealSense detection/classification is paused and frame is frozen.
"""

import os, sys, time, json, random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

import pyrealsense2 as rs
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#gelsight SDK
sys.path.append(os.path.join(BASE_DIR, "gsrobotics"))
from utilities.gelsightmini import GelSightMini
from config import GSConfig

#model wrappers
class FlexibleResNet(nn.Module):
    """ResNet-18 backbone + custom classifier (used for GelSight)."""
    def __init__(self, num_classes: int):
        super().__init__()
        from torchvision import models
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.classifier(self.backbone(x))


class PlainResNet(nn.Module):
    """Plain ResNet-18 with a replaced `fc` layer (used for RealSense crops)."""
    def __init__(self, num_classes: int):
        super().__init__()
        from torchvision import models
        self.net = models.resnet18(pretrained=False)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x):
        return self.net(x)

class MultiCamApp(App):
    #confidence threshold for initial general detection
    SCORE_THR = 0.5

    def build(self):
        Window.size = (1024, 768)
        Window.minimum_width, Window.minimum_height = 800, 600

        root = BoxLayout(orientation="vertical")
        self.status_label = Label(text="Initializing…", font_size=dp(20),
                                  size_hint_y=None, height=dp(40))
        self.img_main     = Image()                              #realSense view
        self.img_gs       = Image(size_hint=(None, None),
                                  size=(dp(240), dp(180)))       #gelsight thumb
        root.add_widget(self.status_label)
        root.add_widget(self.img_main)
        root.add_widget(self.img_gs)
        Window.bind(on_key_down=self._on_key)

        #initialise pipelines
        self._init_models()
        self._init_realsense()
        self._init_gelsight()

        #transforms
        self.tf_vis = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])
        self.tf_gs = T.Compose([
            T.ToPILImage(),
            T.Resize((480, 640)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])

        self.use_gelsight = False
        Clock.schedule_interval(self._update, 1/15)
        return root

    def _init_models(self):
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = dev

        self.detector = init_detector(self.DETECT_CFG, self.DETECT_CKPT, device=dev)
        self.detector.dataset_meta = self._ensure_palette(self.detector.dataset_meta)
        self.visualizer = VISUALIZERS.build(self.detector.cfg.visualizer)
        self.visualizer.dataset_meta = self.detector.dataset_meta

        det_classes = [c.lower() for c in self.detector.dataset_meta['classes']]
        self.CATEGORY_SCREW = det_classes.index('screw')
        self.CATEGORY_NUT   = det_classes.index('nut')

        #vision 2nd stage classifier eval
        with open(self.CLASS_NAMES_VIS) as f:
            self.class_names_vis = json.load(f)
        self.resnet_vis = PlainResNet(len(self.class_names_vis)).to(dev)
        vis_sd = torch.load(self.CLS_CKPT_VIS, map_location=dev)
        if isinstance(vis_sd, dict) and 'state_dict' in vis_sd:
            vis_sd = vis_sd['state_dict']
        vis_sd = {k.replace('module.', ''): v for k, v in vis_sd.items()}
        self.resnet_vis.net.load_state_dict(vis_sd, strict=True)
        self.resnet_vis.eval()

        #gelsight classifier eval
        with open(self.CLASS_NAMES_TACT) as f:
            self.class_names_tac = json.load(f)
        self.resnet_gs = FlexibleResNet(len(self.class_names_tac)).to(dev)
        tac_ckpt = torch.load(self.CLS_CKPT_TACT, map_location=dev)
        if isinstance(tac_ckpt, dict) and 'state_dict' in tac_ckpt:
            tac_ckpt = tac_ckpt['state_dict']
        tac_sd = {k.replace('module.', ''): v for k, v in tac_ckpt.items()}
        self.resnet_gs.load_state_dict(tac_sd, strict=True)
        self.resnet_gs.eval()

        print(f"[INFO] Models loaded on {dev}.")

    @staticmethod
    def _ensure_palette(meta):
        if meta.get('palette') and len(meta['palette']) >= len(meta.get('classes', [])):
            return meta
        random.seed(42)
        meta['palette'] = [tuple(random.randint(0,255) for _ in range(3))
                           for _ in meta.get('classes',[])]
        return meta

    def _init_realsense(self):
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.rs_pipeline = rs.pipeline()
        self.rs_pipeline.start(cfg)

    def _init_gelsight(self):
        #gs_cfg = GSConfig("default_config.json").config
        self.gs_cam = GelSightMini(target_width=640, target_height=480,
                                   border_fraction=0.15)
        try:
            self.gs_cam.select_device(0)
            self.gs_cam.start()
        except Exception as e:
            print("[WARN] GelSight camera not started:", e)
        self.gs_cam_active = False

    def _update(self, dt):
        if self.use_gelsight:
            self._update_gelsight()
        else:
            self._update_realsense()

    def _update_realsense(self):
        frames = self.rs_pipeline.wait_for_frames()
        frame_bgr = np.asanyarray(frames.get_color_frame().get_data())
        result = inference_detector(self.detector,
                                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        vis_img, found = self._draw_detections(frame_bgr.copy(), result)
        self._set_texture(self.img_main, vis_img, already_rgb=False)

        if found:
            self.status_label.text = "Screw/Nut detected! Apply GelSight"
        else:
            self.status_label.text = "RealSense mode – press [G] for GelSight"

    def _draw_detections(self, frame_bgr, result):
        #draw all predictions
        rgb = mmcv.imconvert(frame_bgr, 'bgr', 'rgb')
        self.visualizer.add_datasample('vis', rgb, result,
                                       draw_gt=False, pred_score_thr=self.SCORE_THR,
                                       show=False)
        drawn = mmcv.imconvert(self.visualizer.get_image(), 'rgb', 'bgr')

        inst = result.pred_instances
        found = False
        for bbox, score, label in zip(inst.bboxes.cpu().numpy(),
                                      inst.scores.cpu().numpy(),
                                      inst.labels.cpu().numpy()):
            if score < self.SCORE_THR or label not in (self.CATEGORY_SCREW, self.CATEGORY_NUT):
                continue
            found = True
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            cls_text = self._classify_crop(crop)
            colour = (0,255,0) if label==self.CATEGORY_SCREW else (0,0,255)
            cv2.rectangle(drawn, (x1,y1), (x2,y2), colour, 2)
            y_text = y1-10 if y1>20 else y2+20
            (tw, th), _ = cv2.getTextSize(cls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(drawn, (x1, y_text-th-4), (x1+tw, y_text+2), (0,0,0), -1)
            cv2.putText(drawn, cls_text, (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        return drawn, found

    def _classify_crop(self, crop_bgr):
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.tf_vis(crop_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.resnet_vis(tensor)
            pred = logits.argmax(1).item()
            prob = torch.softmax(logits,1)[0,pred].item()
        return f"{self.class_names_vis[pred]} ({prob:.0%})"

    def _update_gelsight(self):
        frame = self.gs_cam.update(0)
        if frame is None:
            return
        self._set_texture(self.img_gs, frame, already_rgb=True)

        tens = self.tf_gs(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.resnet_gs(tens)
            probs = torch.softmax(logits,1)
            conf, pred = probs.max(1)
            conf, pred = conf.item(), pred.item()

        thr = 0.95 if pred==6 else 0.4 #95% confidence threshold for M8 Nut because it is biased towards it when nothing is presented. To fix, we need to train the model with more variation in the "nothing" category
        text = f"{self.class_names_tac[pred]} ({conf:.2%})" if conf>=thr else "Nothing"
        self.status_label.text = f"GelSight mode – {text}   |   Press [G] to toggle"

    def _set_texture(self, widget, frame, already_rgb=False):
        if not already_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tex = Texture.create(size=(frame.shape[1],frame.shape[0]), colorfmt='rgb')
        tex.blit_buffer(frame.tobytes(), bufferfmt='ubyte', colorfmt='rgb')
        tex.flip_vertical()
        widget.texture = tex

    def _on_key(self, _window, key, *_):
        if key in (ord('g'), ord('G')):
            self.use_gelsight = not self.use_gelsight
            if self.use_gelsight and not self.gs_cam_active:
                self.gs_cam_active = True
            elif not self.use_gelsight:
                self.status_label.text = "RealSense mode – press [G] for GelSight"
        return True

#paths relative to this script's folder
MultiCamApp.DETECT_CFG       = os.path.join(BASE_DIR, "for_models", "faster_rcnn_victor_coco88_config.py") #faster rcnn config
MultiCamApp.DETECT_CKPT      = os.path.join(BASE_DIR, "for_models", "faster_rcnn_victor_coco88.pth") #faster rcnn pth
MultiCamApp.CLS_CKPT_VIS     = os.path.join(BASE_DIR, "for_models", "vision_resnet18.pth") #vision stage 2 model pth
MultiCamApp.CLASS_NAMES_VIS  = os.path.join(BASE_DIR, "for_models", "class_names_vision.json") #class names json for vision stage 2 model
MultiCamApp.CLS_CKPT_TACT    = os.path.join(BASE_DIR, "for_models", "gelsight_resnet18.pth") #gelsight model pth
MultiCamApp.CLASS_NAMES_TACT = os.path.join(BASE_DIR, "for_models", "class_names_gs.json") #class names json for gelsight model

if __name__ == "__main__":
    MultiCamApp().run()
