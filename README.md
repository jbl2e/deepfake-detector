# 🏆 DACON 1st Place Deepfake Detection (DINOv3 + LoRA + Reg Tokens)

🏆 [DACON 대회](https://dacon.io/competitions/official/236628/overview/description) 대상 수상

<img width="1024" height="267" alt="image" src="https://github.com/user-attachments/assets/ad8e69da-1088-4bc0-bdb1-46bc040776b7" />

* [재용2]팀 구성 : 이재용(팀장), 최재우, 김세진
* **Public Score**: 0.98646
* **Private Score**: 0.96949

본 프로젝트는 **DINOv3 (ViT-L/16)** 모델을 사용하여 구축된 딥페이크 탐지 시스템입니다.

본 문서는 연구 재현을 위한 환경 설정 및 실행 방법을 안내합니다.

---

## 📊 프로젝트 소개 자료

[📄 Google Drive에서 보기](https://drive.google.com/file/d/1O6lIAQ_zACPlP0KJSJKwNrnfoYs61dNf/view)

---

## 🧠 모델 아키텍처

**DINOv3 + LoRA + 레지스터 토큰 기반 헤드 구성**

<img width="1024" height="348" alt="image" src="https://github.com/user-attachments/assets/d539f54c-ef1d-44a0-87d1-321fa08cc675" />


---

## 📂 디렉토리 구조

```
5th_submission/
├── content/ **예시 해체 위치이며 실제로는 사용자 공간의 런타임 루트에 생성됩니다**
│   ├── train_full_data/
│   ├── KoDF/                # KoDF 학습 데이터
│   ├── FFHQ/                # FFHQ 실데이터
│   ├── DiffusionFace/       # Diffusion 기반 합성 데이터
│   ├── HiDF/                # HiDF 합성 데이터
│   ├── original_val/        # MFFI 합성 데이터
│   └── ...                  # 기타 데이터셋별 폴더
│
├── model/
│   ├── checkpoints/           # 학습 중 모델 가중치 저장소
│   ├── dinov3_backbone/       # DINOv3 사전학습 모델 (별도 다운로드 필요)
│   ├── .insightface/          # 얼굴 crop 모델 (buffalo_l)
│   ├── model.pt               # 학습 중 생성되는 best model (재현용)
│   └── inference.pth          # 추론용 가중치 (최종 제출 파일)
│
├── src/
│   ├── models.py               # 모델 구조 정의
│   ├── dataset.py              # 데이터 전처리
│   └── utils.py                # 데이터 split 및 유틸 함수
│
├── config/
│   └── config.yaml             # 하이퍼파라미터, 경로, 모델 설정
│
├── env/
│   ├── Dockerfile              # 제출용 Docker 이미지 재현
│   └── requirements.txt        # Python 필수 라이브러리 목록
│
│
├── train_data/                 # 데이터 라이선스는 README 참고 바람
│	(├── train_full_data/... #공유 드라이브에 데이터 확인용 입니다. 실제 해제 장소는 루트바로 하위 폴더입니다.)
│   │
│   └── train_data_zips/
│       └── train_full_data.zip   # 원본 압축 데이터 보관 (zip), (학습 재현용)
│
├── test_data/
│   ├── submission.csv        # inference.py 실행 후 생성 결과 (최종 제출 파일)
│   └── test_data/
│       ├── TEST_000.mp4        # 실제 테스트 이미지 데이터
│       ├── TEST_001.jpg
│       ├── TEST_002.mp4
│       └── ...                 # TEST_xxx.jpg 형식의 테스트 데이터
│
├── train.py                    # 학습 코드
├── inference.py                # 추론 코드
└── README.md                   # 사용 설명서
```

---

## 🚀 사용법

### 1. 환경 설정 및 데이터 위치 안내

#### (1) 환경 설정

* Docker 사용 시:

Image Name 예시:) deepfake-dinov3

```bash
docker build -t deepfake-dinov3 -f env/Dockerfile
```

* 로컬에서 가상 환경으로 사용 시:

Python 3.10~3.12 권장

```bash
python -m pip install --upgrade pip
pip install -r [Project_Dir_Path]/env/requirements.txt

#경로 설정
cd /path/to/5th_submission # 프로젝트 디렉 설정
export RUNTIME_ROOT=/content #절대 경로설정으로, 큰 데이터를 풀기 올바른 위치
export TORCH_COMPILE_DISABLE="1"
```

* 주피터 노트북을 이용한 실행 방법:

```bash
python -m pip install --upgrade pip
pip install -r [Project_Dir_Path]/env/requirements.txt

import os, random, zipfile, tarfile, shutil
from pathlib import Path
import numpy as np
PROJECT_ROOT =Path("[Project_Dir_Path]").resolve()
os.makedirs(PROJECT_ROOT, exist_ok=True)
os.chdir(PROJECT_ROOT)
os.environ["RUNTIME_ROOT"]="/content"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
```

#### (2) 데이터 위치 안내

1. train_full_data.zip은 총 75.6GB로 공유된 드라이브 링크에서 다운로드 받아 디렉 구조의 올바른 위치 `5th_submission/train_data/train_data_zips/`에 위치 시켜야 합니다.
2. train_full_data.zip의 압축 해제 장소는 절대 경로로 다뤄지며 디폴트로 루트 경로의 바로 하위 폴더에 /content를 생성해서 관리하게 됩니다. 원하는 데이터 해제 위치로 변경을 원하시면 추가적인 환경 변수 설정 또는 train.py에 `runtime_root = Path(os.environ.get("RUNTIME_ROOT", "/content")).resolve()`를 변경해주시면 됩니다.

---

### 2. 학습

* Docker로 학습 재현 시:

```bash
docker run --gpus all --rm \
  -v [Host_Data_Dir]:/app/5th_submission/train_data/train_data_zips \
  -w /app/5th_submission \
  -e RUNTIME_ROOT=/content \
  deepfake-dinov3 python train.py --config config/config.yaml
```

* 로컬에서 가상 환경으로 사용 시:

```bash
python train.py --config config/config.yaml
```

* 주피터 노트북을 이용한 실행 방법:

```bash
!python train.py --config config/config.yaml
```

학습 중에 `model/checkpoints`epoch 가중치 및 학습 환경이 pt형식으로 저장됩니다.

학습이 완료되면 가장 높은 validation score를 가진 모델의 가중치가 `model/model.pt`에 best model이 저장됩니다.

개별 epoch의 추론 결과를 확인하고 싶으시면 inference.py의 `WEIGHT_PATH`를 수정해주시면 됩니다.

---

### 3. 추론

Docker로 학습 추론 재현 시:

Image Name 예시:) deepfake-dinov3

```bash
docker run --gpus all --rm deepfake-dinov3
```

로컬에서 가상 환경으로 사용 시:

```bash
python inference.py
```

* 주피터 노트북을 이용한 실행 방법:

```bash
!python inference.py
```

실행 후 결과는 `test_data/submission.csv`에 저장됩니다.

---

### 주의사항

* [Project_Dir_Path]는 데이터 압축 해제 경로[RUNTIME_ROOT]와 다릅니다. [RUNTIME_ROOT]는 기본적으로 루트 바로 하위에 생기므로 권한 여부를 잘 판단하고 읽을 수 있는 위치에 두시기 바랍니다.
* 데이터가 크기 때문에 Dockerfile로 Image를 빌드 시에 데이터는 다루기 편한 곳에 위치해주시고 [Host_Data_Dir]를 train_data_zips에 올바르게 마운트하여 실행해 주세요.
* 추론시 모델 경로가 기본값 inference.pth로 설정되어 있습니다. 개인적이 실험 혼경에서 도출한 가중치이며 학습 재현을 모니터링 하길 원하시면 '학습 후에 checkpoints/epoch_05.pt 또는 model.pt를 사용해주세요'

---

## 📌 안내

* **[데이터 관련]**
  라이선스 규정으로 인해 사용된 데이터셋 목록은 아래에 명시되어 있으나,
  실제 학습에 사용된 zip 파일은 별도로 제공되지 않습니다.
  필요하신 경우 아래 메일로 문의(contact) 부탁드립니다.

* **[추론 관련]**
  추론 가중치: [Google Drive](https://drive.google.com/file/d/1Vr3q5zM6Nqc6yejrULZEFspCRZEG2ISi/view?usp=drive_link)

* **[모델 관련]**
  얼굴 crop은 insightface의 **buffalo_l** 모델을 사용합니다.
  학습 backbone은 **dinov3-vits16-pretrain-lvd1689m**을 사용하였습니다.

  해당 모델들은 재배포가 제한되어 있으므로,
  공식 경로를 통해 별도로 다운로드 후 사용해주시기 바랍니다.

추가적인 문의는 아래 메일로 연락 바랍니다.

📧 [taland797@gmail.com](mailto:taland797@gmail.com)

---

## 📄 데이터 라이선스

### Dataset Information & Licenses

본 프로젝트에서 사용된 데이터셋의 출처와 라이선스 정보입니다. 모든 데이터는 각 제공처의 라이선스 규정을 준수하여 사용되었습니다.

| 데이터셋          | 라이선스            | 링크                                                                                                                                                                                                            |
| ------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DiffusionFace | CC BY-NC 4.0    | github : [https://github.com/Rapisurazurite/DiffFace](https://github.com/Rapisurazurite/DiffFace) / zenodo : [https://zenodo.org/records/10865300](https://zenodo.org/records/10865300)                       |
| HiDF          | CC BY-NC 4.0    | github : [https://github.com/DSAIL-SKKU/HiDF?tab=readme-ov-file](https://github.com/DSAIL-SKKU/HiDF?tab=readme-ov-file) / zenodo : [https://zenodo.org/records/16140829](https://zenodo.org/records/16140829) |
| FFHQ          | CC BY-NC-SA 4.0 | github : [https://github.com/NVlabs/ffhq-dataset?tab=License-1-ov-file#readme](https://github.com/NVlabs/ffhq-dataset?tab=License-1-ov-file#readme)                                                           |
| FaceForensics | MIT             | github : [https://github.com/ondyari/FaceForensics?tab=License-1-ov-file#readme](https://github.com/ondyari/FaceForensics?tab=License-1-ov-file#readme)                                                       |
| MFFI          | CC BY-NC 4.0    | arxiv : [https://arxiv.org/html/2509.05592v1](https://arxiv.org/html/2509.05592v1) / data : [https://www.modelscope.cn/datasets/DDLteam/MFFI/files](https://www.modelscope.cn/datasets/DDLteam/MFFI/files)    |
| KoDF          | AI Hub License  | AIhub : [https://aihub.or.kr/](https://aihub.or.kr/)                                                                                                                                                          |
| Pexels        | Pexels License  | license link : [https://www.pexels.com/ko-kr/license/](https://www.pexels.com/ko-kr/license/)                                                                                                                 |
| SynID         | GPL 3.0         | github : [https://github.com/Raul2718/FLUXSynID](https://github.com/Raul2718/FLUXSynID)                                                                                                                       |

---

##
