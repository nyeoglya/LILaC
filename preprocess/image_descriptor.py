import os
import io
import json
import typing as tp
import math
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import cairosvg
from PIL import Image

from tqdm import tqdm

import numpy as np
import torch
from ultralytics import YOLO

from query import IMAGE_OCR_QUERY, EXPLANATION_INSTRUCTION
from common import get_llm_response, get_clean_savepath


class SequentialImageNormalizer: # resize + mapping
    def __init__(
        self,
        crawled_image_folderpath: str,
        processed_image_folderpath: str
    ):
        self.crawled_image_folderpath: str = crawled_image_folderpath
        self.processed_image_folderpath: str = processed_image_folderpath
        
        self.crawled_image_filepath_list: tp.List[str] = []
        
        self.progress_bar = tqdm(total=0, desc="Image Preprocessing...")
    
    def load_image_filelist(self) -> bool:
        assert os.path.exists(self.crawled_image_folderpath)

        self.crawled_image_filepath_list = []
        for filename in os.listdir(self.crawled_image_folderpath):
            crawled_image_filepath = os.path.join(self.crawled_image_folderpath, filename)
            self.crawled_image_filepath_list.append(crawled_image_filepath)

        return True

    def _resize_by_total_pixels(self, pillow_image, target_pixels=262144): # 512x512
        width, height = pillow_image.size
        current_pixels = width * height

        if current_pixels > target_pixels:
            scale_factor = math.sqrt(target_pixels / current_pixels)
            new_width = round(width * scale_factor)
            new_height = round(height * scale_factor)
            result_pillow_image = pillow_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            result_pillow_image = pillow_image
        
        return result_pillow_image

    def _process_save_save(self, original_image_path: str, clean_savepath: str, file_ext: str):
        try:
            file_ext = file_ext.lower().replace('.', '')
            if file_ext == 'svg':
                png_data = cairosvg.svg2png(url=original_image_path, output_width=1024, output_height=1024)
                if png_data is None:
                    tqdm.write(f"Failed to convert SVG to PNG {original_image_path}")
                    return None
                new_image = Image.open(io.BytesIO(png_data)).convert("RGB")
            else:
                new_image = Image.open(original_image_path).convert("RGB")
            
            processed_image = self._resize_by_total_pixels(new_image)
            processed_image.save(clean_savepath)
        except Exception as e:
            tqdm.write(f"Error loading {original_image_path}: {e}")
            return None

    def run_normalize(self, failed_file_path: str) -> bool:
        self.progress_bar.total = len(self.crawled_image_filepath_list)
        
        with open(failed_file_path, 'w', encoding='utf-8') as failed_file:
            for original_image_path in self.crawled_image_filepath_list:
                original_filename, file_ext = os.path.splitext(os.path.basename(original_image_path))
                clean_savepath = get_clean_savepath(self.processed_image_folderpath, original_filename, "png")
                if os.path.exists(clean_savepath):
                    self.progress_bar.update(1)
                    continue
                
                try:
                    self._process_save_save(original_image_path, clean_savepath, file_ext)
                except Exception as e:
                    tqdm.write(f"Error processing {original_image_path}: {e}")
                    failed_file.write(f"{original_image_path}\n")
                finally:
                    self.progress_bar.update(1)

        return True

class BatchImageDescriptor: # OCR, Explanation
    def __init__(self, image_folder_path: str):
        self.image_folder_path: str = image_folder_path
        self.image_filepath_list: tp.List[str] = []
        
        self.progress_bar = tqdm(total=0, desc="Batch Image Descripting...")
    
    def load_image_filelist(self) -> bool:
        assert os.path.exists(self.image_folder_path)

        self.image_filepath_list = []
        for filename in os.listdir(self.image_folder_path):
            image_filepath = os.path.join(self.image_folder_path, filename)
            self.image_filepath_list.append(image_filepath)
        
        self.image_filepath_list.sort()

        return True

    def process_single_image(self, image_path: str, server_url: str):
        explanation_text = get_llm_response(server_url, EXPLANATION_INSTRUCTION, [image_path])
        ocr_text = get_llm_response(server_url, IMAGE_OCR_QUERY, [image_path])

        return {
            "file_path": image_path,
            "explanation": explanation_text,
            "ocr": ocr_text,
        }

    def run_description(self, failed_file_path: str, description_file_path: str, llm_server_list: tp.List[str]) -> bool:
        processed_paths = set()
        if os.path.exists(description_file_path):
            with open(description_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        processed_paths.add(data["file_path"])
                    except:
                        pass

        targets = [
            p for p in self.image_filepath_list
            if p not in processed_paths
        ]

        self.progress_bar.total = len(targets)
        server_cycle = itertools.cycle(llm_server_list)

        with open(description_file_path, 'a', encoding='utf-8') as f_success, \
            open(failed_file_path, 'a', encoding='utf-8') as f_fail, \
            ThreadPoolExecutor(max_workers=len(llm_server_list)) as executor:

            future_to_path = {
                executor.submit(
                    self.process_single_image,
                    image_path,
                    next(server_cycle)
                ): image_path
                for image_path in targets
            }

            for future in as_completed(future_to_path):
                image_path = future_to_path[future]
                try:
                    result = future.result()
                    f_success.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_success.flush()
                except Exception as e:
                    tqdm.write(f"Error processing {image_path}: {e}")
                    f_fail.write(f"{image_path}\n")
                    f_fail.flush()
                finally:
                    self.progress_bar.update(1)

        return True

class BatchObjectDetector:
    def __init__(
        self,
        image_folder_path: str,
        device: str = "cuda",
        max_objects: int = 3,
    ):
        assert torch.cuda.is_available()

        MODEL_NAME: str = "yolov8x.pt"

        self.image_folder_path = image_folder_path
        self.device = device
        self.max_objects = max_objects

        torch.backends.cudnn.benchmark = True

        self.model = YOLO(MODEL_NAME)
        self.model.to(device)
        self.model.fuse()

        self.image_filepath_list: tp.List[str] = []
        self.progress_bar = tqdm(total=0, desc="Batch Object Detecting...")

    def load_image_filelist(self) -> None:
        assert os.path.exists(self.image_folder_path)
        self.image_filepath_list = [os.path.join(self.image_folder_path, f) for f in os.listdir(self.image_folder_path)]
        self.image_filepath_list.sort()

    def run_detection(
        self,
        failed_file_path: str,
        output_json_path: str,
        batch_size: int = 32,
    ):
        self.progress_bar.total = len(self.image_filepath_list)
        
        results = {}
        failed_files = []
        for i in range(0, len(self.image_filepath_list), batch_size):
            batch_paths = self.image_filepath_list[i : i + batch_size]

            images = []
            valid_paths = []
            for p in batch_paths:
                img = cv2.imread(p)
                if img is None:
                    self.progress_bar.update(1)
                    failed_files.append(p)
                    continue
                images.append(img)
                valid_paths.append(p)
            if not images:
                continue

            batch_outputs = self.batch_detect(images)
            self.progress_bar.update(len(valid_paths))

            for path, dets in zip(valid_paths, batch_outputs):
                results[path] = dets

        with open(output_json_path, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, ensure_ascii=False)

        with open(failed_file_path, "w", encoding="utf-8") as failed_file:
            for path in failed_files:
                failed_file.write(path + "\n")

    @torch.inference_mode()
    def batch_detect(
        self,
        images: tp.List[np.ndarray],
    ) -> tp.List[tp.List[tp.List[int]]]:
        preds = self.model.predict(
            source=images,
            device=self.device,
            conf=0.15,
            iou=0.6,
            agnostic_nms=True,
            half=True,
            verbose=False,
        )

        outputs: tp.List[tp.List[tp.List[int]]] = []

        for r in preds:
            if r.boxes is None:
                outputs.append([])
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()

            order = scores.argsort()[::-1][: self.max_objects]
            selected = boxes[order].astype(int).tolist()

            outputs.append(selected)

        return outputs
