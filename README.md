# openpi — OpenArm Fork

openpi의 OpenArm 브랜치. π₀ LoRA fine-tuning + **Real-Time Chunking (RTC)** 추론을 지원합니다.

- **로봇 제어 / 데이터 수집**: [`Openarm-pi0`](../git/OPR/pi0/Openarm-pi0) 레포 참조
- 원본 openpi 문서: [upstream repo](https://github.com/Physical-Intelligence/openpi)

---

## 설치

```bash
cd ~/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

로봇 쪽 환경(Openarm-pi0 실행 PC)에는 client 패키지만 필요:

```bash
pip install -e ~/openpi/packages/openpi-client
```

---

## 전체 워크플로우

```
[로컬] HDF5 → LeRobot 변환
    ↓
[로컬 → 서버] rsync 데이터 전송
    ↓
[서버] Norm stats 계산
    ↓
[서버] tmux + LoRA 학습
    ↓
[서버 → 로컬] rsync 체크포인트 전송
    ↓
[로컬] 추론 서버 실행  +  Openarm-pi0 실행
```

---

## Step 1. 데이터 변환 (로컬)

`Openarm-pi0/tools/` 안의 변환 스크립트를 사용. 반드시 `uv run`으로 실행 (lerobot이 openpi 환경에 있음).

```bash
cd ~/openpi && uv run python ~/git/OPR/pi0/Openarm-pi0/tools/convert_0608_wipe.py
```

---

## Step 2. 데이터 서버로 전송 (로컬 → 서버)

```bash
rsync -avP /path/to/local/lerobot_dataset  kimminju@100.122.31.85:/home/kimminju/data/
```

---

## Step 3. Norm Stats 계산 (서버)

```bash
cd ~/openpi
uv run scripts/compute_norm_stats.py --config-name <config_name>
```

결과는 `assets/<dataset_id>/norm_stats.json`에 저장됩니다.

---

## Step 4. LoRA 학습 (서버)

### tmux 세션 시작

```bash
tmux new -s <session_name>
```

### GPU 점유 확인

```bash
nvidia-smi && for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    echo -n "PID $pid: "; ps -p $pid -o user=,comm=
done
```

### 학습 실행

```bash
cd ~/openpi

# 단독 사용 시 (50GB 예약, 96GB × 0.51)
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.51 \
    uv run scripts/train.py <config_name> --exp-name=run_01 --overwrite

# 공유 서버 배려 시 (30GB 예약, 96GB × 0.31)
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.31 \
    uv run scripts/train.py <config_name> --exp-name=run_01 --overwrite

# pre-allocation 없이 (다른 사용자 학습 방해 없음, 속도 미세하게 느림)
XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run scripts/train.py <config_name> --exp-name=run_01 --overwrite
```

> `XLA_PYTHON_CLIENT_MEM_FRACTION`은 JAX가 시작 시 GPU 메모리를 지정 비율로 즉시 독점 예약합니다.
> 공유 서버에서는 먼저 `nvidia-smi`로 확인 후 실행하세요.

### 학습 재개

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.51 \
    uv run scripts/train.py <config_name> --exp-name=run_01 --resume
```

체크포인트 저장 경로: `checkpoints/<config_name>/<exp-name>/<step>/`

---

## Step 5. 체크포인트 로컬로 전송 (서버 → 로컬)

```bash
rsync -avP kimminju@100.122.31.85:/home/kimminju/openpi/checkpoints/<config_name>/run_01/<step> \
    /home/kimminju/openpi/checkpoints/<config_name>/run_01/
```

---

## Step 6. 추론 서버 실행 (로컬)

```bash
cd ~/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=<config_name> \
    --policy.dir=checkpoints/<config_name>/run_01/<step>
```

서버가 `localhost:8000`에서 WebSocket으로 대기합니다.
JIT warmup 완료 후 로그에 `ready` 출력 → 그 후 Openarm-pi0 실행.

---

## 새 Config 추가하기

`src/openpi/training/config.py`의 `_CONFIGS` 리스트에 `TrainConfig`를 추가합니다.

```python
TrainConfig(
    name="my_task",
    model=pi0_config.Pi0Config(
        action_dim=8,          # 왼팔 단일: 8, 양팔: 16
        action_horizon=50,
        wrench_dim=6,          # wrench 없으면 0 (기본값)
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotOpenarmDataConfig(
        repo_id="local:/home/kimminju/data/my_task_lerobot",
        action_dim=8,
        use_delta_actions=False,
        cameras=("image", "left_wrist_image", "center_image"),  # 사용할 카메라 지정
        use_wrench=True,       # wrench 토큰 사용 여부 (wrench_dim과 일치해야 함)
        assets=AssetsConfig(asset_id="my_task_lerobot"),
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=30_000,
    batch_size=16,
    freeze_filter=_LORA_FREEZE,
    ema_decay=None,
),
```

### 주요 파라미터

| 파라미터 | 위치 | 설명 |
|---|---|---|
| `action_dim` | `Pi0Config` + `LeRobotOpenarmDataConfig` | 왼팔 단일=8, 양팔=16. 두 곳 모두 동일하게 |
| `wrench_dim` | `Pi0Config` | 0=wrench 없음, 6=왼팔 f_ext, 12=양팔 f_ext |
| `use_wrench` | `LeRobotOpenarmDataConfig` | `wrench_dim > 0`이면 반드시 `True` |
| `cameras` | `LeRobotOpenarmDataConfig` | 사용할 카메라 feature 이름 튜플 (2~4개) |
| `use_delta_actions` | `LeRobotOpenarmDataConfig` | `True`=delta action 학습, `False`=absolute |

### 카메라 슬롯

```
"image"             → base_0_rgb        (오버헤드)
"right_wrist_image" → right_wrist_0_rgb (오른쪽 손목)
"left_wrist_image"  → left_wrist_0_rgb  (왼쪽 손목)
"center_image"      → center_0_rgb      (보드 정면)
```

---

## Wrench (외력) 조건부 학습

MomentumObserver로 추정한 EE 외력을 π₀ Action Expert의 state 토큰과 같은 causal group에 wrench 토큰으로 추가합니다.

```
[이미지/언어 토큰] | [state 토큰] [wrench 토큰] | [action 토큰 × 50]
  PaliGemma 처리     ar_mask=True  ar_mask=False   ar_mask=False
```

- `ar_mask=True`: 새 causal group 시작 (state가 이미지/언어를 attend)
- `ar_mask=False`: 같은 그룹 (wrench ↔ state 서로 attend 가능)

### wrench 없이 학습 (기본)

```python
Pi0Config(wrench_dim=0, ...)           # 기본값, 기존 config와 완전 호환
LeRobotOpenarmDataConfig(use_wrench=False, ...)
```

### wrench 포함 학습

```python
Pi0Config(wrench_dim=6, ...)           # 왼팔 f_ext (6D)
LeRobotOpenarmDataConfig(use_wrench=True, ...)
```

데이터셋에 `wrench` feature가 있어야 합니다 (변환 스크립트에서 `f_ext_L` 포함).

> `wrench_dim`과 `use_wrench`가 불일치하면 학습은 되지만 wrench 입력이 무시됩니다.
> 추론 시에도 학습 때 사용한 config와 동일하게 맞춰야 합니다.

### 추론 시 wrench 전달

`main_pi0_real.py`의 `PI0_CONFIG`에서:
```python
"use_force": True   # MomentumObserver 외력을 observation에 포함
```
`pi0_inference.py`가 `observation/wrench` 키를 자동으로 서버에 전송합니다.

---

## OpenArm 코드 변경 내용 (기술 참조)

### 신규 파일: `src/openpi/policies/openarm_policy.py`

- `OpenarmInputs`: observation 키 → π₀ 모델 입력 포맷 변환, `cameras` 튜플로 동적 카메라 구성
- `OpenarmOutputs`: 모델 출력에서 앞 `action_dim`개만 슬라이싱 (내부 패딩 제거)

### 수정된 파일

**`src/openpi/models/pi0_config.py`**
- `wrench_dim: int = 0` 추가

**`src/openpi/models/pi0.py`**
- `wrench_dim > 0`이면 Action Expert suffix에 wrench 토큰 추가
- **`_rtc_soft_mask(d, s, H)`**: RTC soft mask W 계산 (논문 Eq.5)
- **`sample_actions_rtc()`**: IIGDM guided denoising, `lax.scan`으로 JIT 재컴파일 없음

**`src/openpi/policies/policy.py`**
- obs에 `_rtc_prev_chunk` 키가 있으면 `sample_actions_rtc()` 경로, 없으면 기존 경로 (하위 호환)
- A_prev를 robot absolute space → model normalized delta space로 변환 후 전달

**`scripts/serve_policy.py`**
- `_warmup_policy()` 추가: 서버 시작 직후 `sample_actions`와 `sample_actions_rtc` 모두 JIT 컴파일
  (미리 컴파일하지 않으면 첫 추론에서 10–15초 지연)
