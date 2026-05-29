"""
HDF5 → LeRobot v3.0 변환 스크립트 (π₀, 왼팔 단일 + f_ext_L)

[입력 HDF5 구조]
  q_pos:         (T, 8)   L_joint(7)+grip_L(1)
  f_ext_L:       (T, 6)   left EE wrench [Fx,Fy,Fz,Tx,Ty,Tz]
  action:        (T, 8)   absolute joint target
  images/cam_top:        (T, H, W, 3) uint8
  images/cam_wrist_left: (T, H, W, 3) uint8

[출력 feature 구조]
  state:   8D   L_joint(7)+grip_L(1)
  wrench:  6D   L_fext(6)
  actions: 8D

[사용법]
  cd ~/openpi && uv run python ~/git/OPR/pi0/Openarm-pi0/tools/convert_pi0_openarm_left_force.py

# ================================================================
# TODO (새 데이터셋으로 복사할 때 반드시 수정할 항목들)
#
# 1. HDF5_DIRS         : HDF5 폴더 경로 리스트 (여러 태스크 가능)
# 2. OUTPUT_DIR        : 변환 결과를 저장할 폴더 경로
# 3. FPS               : 데이터 수집 주파수 (보통 30)
# 4. EPISODE_TASK_MAP  : (폴더인덱스, 시작ep, 끝ep, "언어 라벨") 리스트
#                        폴더인덱스 -1 이면 모든 폴더에 해당 라벨 공통 적용
# ================================================================
"""

import glob
import os
import re

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from tqdm import tqdm

# ============================================================
# 설정
# ============================================================
HDF5_DIRS = [                                                       # TODO(1)
    "/media/kimminju/OPR-SSD/data/0529_left_grasp",
    # "/media/kimminju/OPR-SSD/data/0529_left_wipe",  # 추가 시 주석 해제
]

OUTPUT_DIR = "/media/kimminju/OPR-SSD/data/0529_left_grasp_lerobot"  # TODO(2)
FPS        = 30                                                        # TODO(3)
ROBOT_TYPE = "openarm"

# (폴더인덱스, 시작ep, 끝ep포함, "언어 라벨")
# 폴더인덱스 -1 → 해당 폴더 내 모든 에피소드에 동일 라벨
EPISODE_TASK_MAP = [                                                   # TODO(4)
    (0, -1, -1, "grasp the whiteboard eraser"),
    # (1, -1, -1, "wipe the whiteboard"),
]

CAMERA_MAPPING = {   # HDF5 키 → LeRobot feature 키
    "images/cam_top":        "image",
    "images/cam_wrist_left": "left_wrist_image",
}

RESIZE_WH    = (224, 224)
STATE_DIM    = 8
WRENCH_DIM   = 6
ACTION_DIM   = 8
STATE_NAMES  = [f"L_joint_{i}" for i in range(7)] + ["L_gripper"]
WRENCH_NAMES = [f"L_fext_{i}" for i in range(6)]
# ============================================================


def get_hdf5_files(folder: str) -> list[str]:
    files = sorted(
        glob.glob(os.path.join(folder, "*.hdf5")),
        key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group()),
    )
    if not files:
        raise FileNotFoundError(f"HDF5 파일 없음: {folder}")
    return files


def get_task(folder_idx: int, ep_idx: int) -> str:
    for fidx, start, end, desc in EPISODE_TASK_MAP:
        if fidx != -1 and fidx != folder_idx:
            continue
        if start == -1 or (start <= ep_idx <= end):
            return desc
    raise ValueError(f"folder_idx={folder_idx}, ep_idx={ep_idx} 에 해당하는 task 없음")


def resize(frame: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(frame).resize(RESIZE_WH, Image.BILINEAR))


def main():
    # 전체 파일 목록 수집
    all_episodes: list[tuple[int, int, str]] = []  # (folder_idx, ep_in_folder, path)
    for fidx, folder in enumerate(HDF5_DIRS):
        files = get_hdf5_files(folder)
        for ep_in_folder, path in enumerate(files):
            all_episodes.append((fidx, ep_in_folder, path))

    print(f"총 {len(all_episodes)}개 에피소드 → {OUTPUT_DIR}")

    # f_ext_L 키 확인
    with h5py.File(all_episodes[0][2], "r") as f:
        if "f_ext_L" not in f:
            raise KeyError("f_ext_L 키 없음. use_force=True로 수집한 데이터인지 확인하세요.")

    dataset = LeRobotDataset.create(
        repo_id=os.path.basename(OUTPUT_DIR),
        root=OUTPUT_DIR,
        robot_type=ROBOT_TYPE,
        fps=FPS,
        features={
            **{
                lerobot_key: {
                    "dtype": "image",
                    "shape": (*RESIZE_WH, 3),
                    "names": ["height", "width", "channel"],
                }
                for lerobot_key in CAMERA_MAPPING.values()
            },
            "state": {
                "dtype": "float32",
                "shape": (STATE_DIM,),
                "names": STATE_NAMES,
            },
            "wrench": {
                "dtype": "float32",
                "shape": (WRENCH_DIM,),
                "names": WRENCH_NAMES,
            },
            "actions": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": [f"action_{i}" for i in range(ACTION_DIM)],
            },
        },
        image_writer_threads=4,
        image_writer_processes=4,
    )

    for fidx, ep_in_folder, hdf5_path in tqdm(all_episodes, desc="Episodes"):
        task = get_task(fidx, ep_in_folder)

        with h5py.File(hdf5_path, "r") as f:
            q_pos   = f["q_pos"][:].astype(np.float32)    # (T, 8)
            f_ext_L = f["f_ext_L"][:].astype(np.float32)  # (T, 6)
            action  = f["action"][:].astype(np.float32)   # (T, 8)
            cam_data = {hdf5_key: f[hdf5_key][:] for hdf5_key in CAMERA_MAPPING}

        for i in range(len(action)):
            dataset.add_frame({
                **{
                    lerobot_key: resize(cam_data[hdf5_key][i])
                    for hdf5_key, lerobot_key in CAMERA_MAPPING.items()
                },
                "state":   q_pos[i],
                "wrench":  f_ext_L[i],
                "actions": action[i],
                "task":    task,
            })

        dataset.save_episode()

    print(f"\n변환 완료! 에피소드: {len(all_episodes)}, 출력: {OUTPUT_DIR}")
    print("다음 단계: cd ~/openpi && uv run scripts/compute_norm_stats.py --config-name pi0_openarm_left_force_lora")


if __name__ == "__main__":
    main()
