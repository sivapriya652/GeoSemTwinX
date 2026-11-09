#!/usr/bin/env bash
# One-command ROS 2 demo: dataset player + GeoSemTwin-X node + RViz2.
#   bash ros2/run_demo.sh staer      # warehouse (needs HF access, downloaded on first run)
#   bash ros2/run_demo.sh tum        # TUM fr1/desk office
#   bash ros2/run_demo.sh realsense  # live Intel RealSense (start the camera driver yourself)
set -e
DATASET=${1:-staer}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
source /opt/ros/${ROS_DISTRO:-humble}/setup.bash
source "$ROOT/.venv/bin/activate"
cd "$ROOT"
trap 'kill $(jobs -p) 2>/dev/null' EXIT
rviz2 -d "$HERE/geosemtwinx.rviz" &
case "$DATASET" in
  staer)
    python3 ros2/geosemtwinx_node.py --ros-args --params-file configs/staer_player.yaml &
    sleep 25   # model loading
    python3 ros2/dataset_player.py --ros-args -p dataset:=staer -p rate:=2.0 ;;
  tum)
    python3 ros2/geosemtwinx_node.py --ros-args --params-file configs/tum_player.yaml &
    sleep 25
    python3 ros2/dataset_player.py --ros-args -p dataset:=tum -p rate:=3.0 ;;
  realsense)
    python3 ros2/geosemtwinx_node.py --ros-args --params-file configs/realsense_d435i.yaml ;;
esac
wait
