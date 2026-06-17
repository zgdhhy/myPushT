# Third-Party Notices

MyPushT code is released under the MIT License. Some concepts, data formats, and robot assets are associated with third-party projects and should keep their own upstream terms.

## LeRobot

- Project: https://github.com/huggingface/lerobot
- License shown by upstream repository: Apache-2.0
- Usage in MyPushT: LeRobotDataset-compatible dataset conversion and loading.

## SO-ARM100 / SO100 Assets

- Local files: `assets/so100/**`
- Usage in MyPushT: MuJoCo XML and STL assets for the SO-ARM100 PushT simulation.
- Status: these assets are kept as necessary simulation resources in this working tree, but their upstream origin and exact license should be verified before broad redistribution. If the license cannot be confirmed, publish the code with download instructions instead of committing the mesh/XML files.

## Large Generated Artifacts

The public Git repository should not track generated datasets, checkpoints, caches, or videos. Use GitHub Releases, Hugging Face Hub, or another external artifact store for:

- `assets/lerobot_dataset/**`
- `assets/model/**`
- `outputs/**`
- `*.pt`, `*.mp4`, `*.npz`
