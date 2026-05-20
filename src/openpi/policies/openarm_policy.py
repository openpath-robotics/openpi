"""
OpenArm π₀ Policy — 데이터 입출력 변환

[π₀ 이미지 슬롯]
  base_0_rgb        : cam_top (overhead)
  right_wrist_0_rgb : cam_wrist_right
  left_wrist_0_rgb  : cam_wrist_left
  center_0_rgb      : cam_center

[카메라 선택]
  cameras 튜플로 사용할 카메라를 지정. 예:
    ("image", "right_wrist_image")                              # 2-camera
    ("image", "right_wrist_image", "left_wrist_image")          # 3-camera
    ("image", "right_wrist_image", "left_wrist_image", "center_image")  # 4-camera

[state 구성 — 16D]
  R_joint(7)+R_grip(1)+L_joint(7)+L_grip(1)
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms


# 데이터셋 feature 이름 → π₀ 이미지 슬롯 이름
_FEATURE_TO_SLOT: dict[str, str] = {
    "image":             "base_0_rgb",
    "right_wrist_image": "right_wrist_0_rgb",
    "left_wrist_image":  "left_wrist_0_rgb",
    "center_image":      "center_0_rgb",
}

DEFAULT_CAMERAS: tuple[str, ...] = ("image", "right_wrist_image")


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
    """데이터셋 / 추론 환경 → π₀ 모델 입력 포맷 변환."""

    action_dim: int = 16
    cameras: tuple[str, ...] = DEFAULT_CAMERAS

    def __call__(self, data: dict) -> dict:
        images = {}
        image_mask = {}
        for feature in self.cameras:
            slot = _FEATURE_TO_SLOT[feature]
            images[slot] = _parse_image(data[f"observation/{feature}"])
            image_mask[slot] = np.True_

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
