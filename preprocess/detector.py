from typing import List

import numpy as np
from sklearn.decomposition import PCA

from PIL import Image
import cv2

import torch
from torch.amp.autocast_mode import autocast
from transformers import AutoImageProcessor, AutoModel


class BatchObjectDetector:
    def __init__(self, device: str = "cuda:0"):
        MODEL_NAME = "facebook/dinov2-vit-small-patch14"
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        self.model.eval()

    @torch.inference_mode()
    def get_labels_batched(self, images_tensor: torch.Tensor):
        # images_tensor: [B, 3, 224, 224]
        B = images_tensor.shape[0]
        
        # 1. 모델 추론 (FP16)
        with autocast(device_type=self.device):
            outputs = self.model(images_tensor)
            features = outputs.last_hidden_state[:, 1:, :] # [B, 1024, 384]

        # 2. GPU 기반 PCA 구현 (속도 핵심)
        # 각 이미지별로 독립적인 PCA 수행
        results = []
        for b in range(B):
            feat = features[b] # [1024, 384]
            # PyTorch의 SVD를 이용한 빠른 PCA
            U, S, V = torch.pca_lowrank(feat, q=3)
            pca_feat = torch.matmul(feat, V[:, :3]) # [1024, 3]
            results.append(pca_feat)
        
        return results

    @torch.inference_mode()
    def get_labels(self, image_path: str, confidence_threshold=0.2) -> List[List[int]]:
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        
        # 특징 추출 (Mixed Precision으로 속도 향상)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with autocast(device_type=self.device):
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 1:, :].squeeze(0).cpu().numpy()

        # 2. PCA 및 설명력(Variance Ratio) 확인
        # 객체가 없다면 주성분의 설명력이 낮게 나옵니다.
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(features)
        explained_variance = pca.explained_variance_ratio_

        # "객체가 없음" 판단 로직 (Heuristic)
        # 첫 번째 주성분이 전체 변동의 일정 수준 이상을 설명하지 못하면 잡음으로 판단
        if explained_variance[0] < confidence_threshold:
            return []

        grid_size = int(np.sqrt(features.shape[0]))
        all_bboxes = []

        for i in range(2): # 보통 상위 2개 성분이 주요 객체/부분을 나타냄
            comp_map = pca_features[:, i].reshape(grid_size, grid_size)
            
            # 정규화 및 Otsu 이진화 (고정 임계값 0.6보다 훨씬 견고함)
            comp_map = (comp_map - comp_map.min()) / (comp_map.max() - comp_map.min() + 1e-8)
            comp_map_uint8 = (comp_map * 255).astype(np.uint8)
            
            # Otsu's thresholding으로 최적 임계값 자동 설정
            _, mask = cv2.threshold(comp_map_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 3. 노이즈 제거 (Morphology Opening)
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            # 4. 박스 추출 및 필터링
            mask_rescaled = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask_rescaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, wb, hb = cv2.boundingRect(cnt)
                area_ratio = (wb * hb) / (w * h)
                
                # 너무 작거나(노이즈), 너무 큰(배경 자체) 박스 제거
                if 0.02 < area_ratio < 0.9:
                    all_bboxes.append([x, y, x + wb, y + hb])

        # 5. NMS (Non-Maximum Suppression) 적용
        # 중복되는 박스들을 합쳐서 깔끔한 라벨 생성
        final_boxes = self._nms(all_bboxes, iou_threshold=0.5)
        
        return final_boxes

    def _nms(self, bboxes, iou_threshold=0.5):
        if not bboxes: return []
        boxes = np.array(bboxes)
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas = (x2 - x1) * (y2 - y1)
        order = areas.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(bboxes[i])
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

if __name__ == "__main__":
    object_detector = BatchObjectDetector()
    object_detector.get_labels("/dataset/crawl/mmqa_image/1757_UPenn_Seal.png")
