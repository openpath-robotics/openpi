"""
OpenArm π₀ Policy — 데이터 입출력 변환

[π₀ 이미지 슬롯]
  base_0_rgb        : cam_top (overhead)
  right_wrist_0_rgb : cam_wrist_right  ← 기본 2-camera 모드
  left_wrist_0_rgb  : cam_wrist_left   ← use_left_wrist=True 시 추가 (3-camera)

[state 구성 — 16D (no force)]
  R_joint(7)+R_grip(1)+L_joint(7)+L_grip(1)

[state 구성 — 28D (with force)]
  R_joint(7)+R_grip(1)+R_fext(6)+L_joint(7)+L_grip(1)+L_fext(6)
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_openarm_example(state_dim: int = 16) -> dict:
    """더미 관측값 생성 (단위 테스트용)."""
    return {
        "observation/state":             np.random.rand(state_dim).astype(np.float32),
        "observation/image":             np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image":  np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the object",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class OpenarmInputs(transforms.DataTransformFn):
    """
    데이터셋 / 추론 환경 → π₀ 모델 입력 포맷 변환.

    데이터셋 컬럼 (convert_pi0_openarm.py 출력):
      state, actions, image, left_wrist_image, right_wrist_image

    repack_transform 이후의 키 (학습 시):
      observation/state, actions,
      observation/image, observation/left_wrist_image, observation/right_wrist_image

    추론 시 build_observation() 이 직접 observation/* 키로 제공.
    """

    action_dim: int = 16  # 실제 action 차원 (padding 전)
    model_type: _model.ModelType = _model.ModelType.PI0
    # False = 2-camera mode (base + right wrist) for ≤16GB VRAM.
    # True  = 3-camera mode (base + right wrist + left wrist).
    # Must match Pi0TwoCameraConfig vs Pi0Config in TrainConfig.
    use_left_wrist: bool = False

    def __call__(self, data: dict) -> dict:
        base_image        = _parse_image(data["observation/image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        images = {
            "base_0_rgb":        base_image,
            "right_wrist_0_rgb": right_wrist_image,
        }
        image_mask = {
            "base_0_rgb":        np.True_,
            "right_wrist_0_rgb": np.True_,
        }

        if self.use_left_wrist:
            left_wrist_image = _parse_image(data["observation/left_wrist_image"])
            images["left_wrist_0_rgb"] = left_wrist_image
            # pi0-FAST masks left wrist; pi0 uses it
            image_mask["left_wrist_0_rgb"] = (
                np.True_ if self.model_type == _model.ModelType.PI0 else np.False_
            )

        inputs = {
            "state": data["observation/state"],
            "image": images,
            "image_mask": image_mask,
        }

        if "observation/wrench" in data:
            inputs["wrench"] = data["observation/wrench"]

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class OpenarmOutputs(transforms.DataTransformFn):
    """π₀ 모델 출력 → 로봇 action 포맷 변환."""

    action_dim: int = 16  # 실제 action 차원 (padding 제거용)

    def __call__(self, data: dict) -> dict:
        # π₀ 내부 action_dim(>=32) 중 앞 action_dim 개만 사용
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
