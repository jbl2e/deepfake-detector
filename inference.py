import os
import re
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.amp import autocast
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from huggingface_hub import login
from transformers import AutoImageProcessor, AutoModel
from peft import LoraConfig, get_peft_model
import insightface
from insightface.app import FaceAnalysis
import warnings
from pathlib import Path
import os

# 경고 메시지 무시
warnings.filterwarnings("ignore")

HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN" #온라인용

PROJECT_ROOT = Path(os.getcwd()).resolve()

# 모델 관련 로컬 경로
# 1) DINOv3 백본 (미리 저장된 폴더)
LOCAL_BACKBONE_PATH = PROJECT_ROOT / "model" / "dinov3_backbone"
# 2) 직접 학습한 LoRA 가중치
WEIGHT_PATH = PROJECT_ROOT / "model" / "inference.pth"
# 3) InsightFace 가중치 루트
INSIGHTFACE_ROOT = PROJECT_ROOT / "model" / ".insightface"

# 데이터 및 결과 경로
TEST_DIR = PROJECT_ROOT / "test_data" / "test_data"
OUTPUT_CSV = PROJECT_ROOT / "test_data" / "submission.csv"

# ======================================================
# 2. 모델 클래스 정의
# ======================================================
class DeepfakeDetectorBCE(nn.Module):
    def __init__(self, lora_backbone):
        super().__init__()
        self.backbone = lora_backbone
        self.classifier = nn.Sequential(
            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        tokens = outputs.last_hidden_state
        reg1 = tokens[:, 1, :]
        reg2 = tokens[:, 2, :]
        patch_tokens = tokens[:, 5:, :]
        patch_mean = patch_tokens.mean(dim=1)
        feat = torch.cat([reg1, reg2, patch_mean], dim=1)
        return self.classifier(feat)

# ======================================================
# 3. 유틸리티 함수
# ======================================================
def get_insight_face_crop_square(face_app, image_np, target_size=512):
    faces = face_app.get(image_np)
    h, w, _ = image_np.shape

    if not faces:
        min_dim = min(h, w)
        start_h = (h - min_dim) // 2
        start_w = (w - min_dim) // 2
        crop_img = image_np[start_h:start_h+min_dim, start_w:start_w+min_dim]
        return cv2.resize(crop_img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

    face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)[0]
    x1, y1, x2, y2 = face.bbox.astype(int)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    side_length = int(max(x2 - x1, y2 - y1) * 1.4)

    nx1 = max(0, center_x - side_length // 2)
    ny1 = max(0, center_y - side_length // 2)
    nx2 = min(w, nx1 + side_length)
    ny2 = min(h, ny1 + side_length)

    if nx2 == w: nx1 = max(0, w - side_length)
    if ny2 == h: ny1 = max(0, h - side_length)

    face_crop = image_np[ny1:ny2, nx1:nx2]
    return cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

def extract_number(id_str):
    numbers = re.findall(r'\d+', str(id_str))
    return int(numbers[0]) if numbers else 999999

# ======================================================
# 4. 실행 로직 (셀 실행용)
# ======================================================
# 1. 환경 설정 및 로그인
# login(token=HF_TOKEN) # 로컬 로드 시 생략 가능
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 데이터 목록 확보
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".jfif"}
VIDEO_EXTS = {".mp4", ".mov"}
image_files = []
video_files = []

if TEST_DIR.exists():
    for ext in IMAGE_EXTS:
        image_files.extend(list(TEST_DIR.rglob(f"*{ext}")))
        image_files.extend(list(TEST_DIR.rglob(f"*{ext.upper()}")))
    for ext in VIDEO_EXTS:
        video_files.extend(list(TEST_DIR.rglob(f"*{ext}")))
        video_files.extend(list(TEST_DIR.rglob(f"*{ext.upper()}")))
else:
    print(f"❌ TEST_DIR를 찾을 수 없습니다: {TEST_DIR}")

image_files = sorted(list(set(image_files)))
video_files = sorted(list(set(video_files)))
print(f"✅ 이미지: {len(image_files)} / 비디오: {len(video_files)}")

# 3. 모델 및 얼굴 인식기 초기화
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
face_app = FaceAnalysis(name='buffalo_l', root=str(INSIGHTFACE_ROOT), providers=providers)
face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))

# [오프라인 로드]
processor = AutoImageProcessor.from_pretrained(str(LOCAL_BACKBONE_PATH))
backbone = AutoModel.from_pretrained(str(LOCAL_BACKBONE_PATH))

config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
)
lora_backbone = get_peft_model(backbone, config)
detector = DeepfakeDetectorBCE(lora_backbone).to(device)

# 가중치 로드
if WEIGHT_PATH.exists():
    state_dict = torch.load(str(WEIGHT_PATH), map_location=device)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    detector.load_state_dict(state_dict)
    print(f"✅ 가중치 로드 완료: {WEIGHT_PATH.name}")

detector.eval()
if torch.cuda.is_available():
    detector = torch.compile(detector)

# 4. Transform 및 추론 시작
inference_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((448, 448), interpolation=InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

results = []

# 이미지 추론
for img_path in tqdm(image_files, desc="📷 Images"):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None: continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face_crop = get_insight_face_crop_square(face_app, img_rgb, target_size=448)
    input_tensor = inference_transform(face_crop).unsqueeze(0).to(device)
    with torch.no_grad(), autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
        prob = torch.sigmoid(detector(input_tensor).squeeze()).item()
    results.append({'ID': img_path.name, 'label': prob})

# 비디오 추론
for vid_path in tqdm(video_files, desc="🎥 Videos"):
    cap = cv2.VideoCapture(str(vid_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        results.append({'ID': vid_path.name, 'label': 0.5}); cap.release(); continue

    frame_indices = np.unique(np.linspace(0, total_frames - 1, 16, dtype=int))
    batch_tensors = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None: continue
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_crop = get_insight_face_crop_square(face_app, img_rgb, target_size=448)
        batch_tensors.append(inference_transform(face_crop))
    cap.release()

    if batch_tensors:
        batch_input = torch.stack(batch_tensors).to(device)
        with torch.no_grad(), autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = detector(batch_input)
            probs = torch.sigmoid(outputs.squeeze(1) if outputs.ndim > 1 else outputs).cpu().numpy()
        probs = np.atleast_1d(probs)
        probs.sort()
        final_score = np.median(probs[-4:])
        results.append({'ID': vid_path.name, 'label': np.clip(final_score, 0.0, 1.0)})
    else:
        results.append({'ID': vid_path.name, 'label': 0.5})

# 결과 저장
df = pd.DataFrame(results)
df['sort_key'] = df['ID'].apply(extract_number)
df = df.sort_values('sort_key').drop(columns=['sort_key']).reset_index(drop=True)
df.to_csv(str(OUTPUT_CSV), index=False)
print(f"✅ 저장 완료: {OUTPUT_CSV}")