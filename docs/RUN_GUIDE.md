# GeoSemTwin-X: Run Guide (starting from a blank VS Code)

This guide is for your RTX 3500 Ada laptop with WSL **Ubuntu-24.04** (ROS 2 Jazzy) and VIGS-SLAM in the other WSL
distribution (**Ubuntu**, conda env `vigs-slam`).

---

## Part A: Open the project in VS Code

1. Open **Visual Studio Code**.
2. Press **Ctrl+Shift+P**, type `WSL: Connect to WSL using Distro`, press Enter and choose **Ubuntu-24.04**.
   * A new VS Code window opens. The bottom-left corner shows **WSL: Ubuntu-24.04**.
   * If VS Code asks to install the **WSL** extension, click **Install**.
3. **File → Open Folder…**, type `~/GeoSemTwinX` and click **OK**.
   * If it asks "Do you trust the authors?", click **Yes, I trust**.
4. Open a terminal: **Terminal → New Terminal** (or press ``Ctrl+` ``).
5. In that terminal, activate ROS 2 and the project environment. Do this **every time** you open a new terminal:

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
```

The prompt now starts with `(.venv)`. If a command ever says `python: command not found`, this step was skipped.

6. Check that everything is installed:

```bash
python scripts/run_all.py check --vigs-root "/mnt/d/Gayathri/Robot Navigation/VIGS-SLAM"
```

You should see `CUDA: True`, your RTX 3500, `ffmpeg`, `Hugging Face login: Gayathri-K-S`, `ROS 2 rclpy: ok (jazzy)` and
`VIGS-SLAM: found`.

---

## Part B: Run the experiments (in order)

Run one command at a time and wait for the prompt to come back. Each step finishes with `done in X min`.

| # | Command | Time | Produces |
|---|---|---|---|
| 1 | `python scripts/run_all.py tum` | ~20 min | SLAM accuracy vs ORB-SLAM2 / ElasticFusion / DVO-SLAM; `TUM_Dashboard.html` |
| 2 | `python scripts/run_all.py vigs-export --vigs-root "/mnt/d/Gayathri/Robot Navigation/VIGS-SLAM"` | ~1 min | Staer walkthrough in VIGS-SLAM input format |
| 3 | VIGS-SLAM run (see Part C) | ~5–10 min | VIGS-SLAM trajectory on D: |
| 4 | `python scripts/run_all.py staer --change-walk 002 --vigs-traj /mnt/d/Gayathri/vigs_run/out` | ~30 min | All warehouse comparative studies; `GeoSemTwinX_Dashboard.html` |
| 5 | `python scripts/run_all.py synthetic` | ~50 min | Change detection, exploration (NBV), exact-GT fusion study |
| 6 | `python scripts/run_all.py demo` | ~10 min | `demo_staer/demo.mp4` (detection + segmentation + live twin) |
| 7 | `python scripts/run_all.py report` | seconds | `results/REPORT.md` with every table and key findings |

Notes:
* Long silent periods are normal: SLAM runs on the CPU and prints progress only every so often. To check it is
  working, open a second terminal (**+** in the terminal panel) and run `top`. A `python` line with high %CPU means it
  is working; press `q` to exit.
* To stop a run, press **Ctrl+C**. Do **not** use Ctrl+Z, which only pauses it.
* Steps 2–3 are only needed for the VIGS-SLAM comparison. Without them, run step 4 without `--vigs-traj ...`.

---

## Part C: VIGS-SLAM run (step 3, in the other Ubuntu)

1. Still in the Ubuntu-24.04 terminal, copy the export to the D: drive so the other Ubuntu can read it:

```bash
rm -rf /mnt/d/Gayathri/vigs_run && mkdir -p /mnt/d/Gayathri/vigs_run
cp -r data/vigs_scene03_001 /mnt/d/Gayathri/vigs_run/
```

2. Open **Windows PowerShell** (outside VS Code) and enter the other Ubuntu:

```powershell
wsl -d Ubuntu
```

3. Run VIGS-SLAM:

```bash
conda activate vigs-slam
cd "/mnt/d/Gayathri/Robot Navigation/VIGS-SLAM"
D=/mnt/d/Gayathri/vigs_run/vigs_scene03_001
python demo.py --imagedir $D/images --imufile $D/imu.csv --calib $D/calib.txt --config $D/staer.yaml --output /mnt/d/Gayathri/vigs_run/out --IMU_poseinit_after 20 --stride 1 --buffer 200 2>&1 | tee /mnt/d/Gayathri/vigs_run/log.txt
```

4. When it finishes, `ls /mnt/d/Gayathri/vigs_run/out` should list `traj_full_afterBA.txt`. Type `exit`, then go back
   to VS Code for step 4.

---

## Part D: Live ROS 2 demo (RViz)

In the VS Code terminal (with both `source` lines from Part A done):

```bash
bash ros2/run_demo.sh staer
```

* RViz opens immediately. The models load for ~25 s, then the warehouse plays and the map, asset boxes, paths,
  traversability and next-best-view arrow build up live. The camera view with detections is in the top-left panel.
* For a nicer view: zoom out with the mouse wheel, untick **Grid**, and expand **Fused map** to set **Size (m)** to `0.05`.
* To record a video, press **Win+G** (Xbox Game Bar) and click record, or use OBS.
* To save the twin while it is running, open a second terminal and run:

```bash
source /opt/ros/jazzy/setup.bash
ros2 service call /geosemtwinx/save_twin std_srvs/srv/Trigger
```

* To stop, press **Ctrl+C** in the demo terminal. This also saves `results/Live_Dashboard.html`. Close RViz with the
  X if it stays open.
* Office scene instead of the warehouse: `bash ros2/run_demo.sh tum`.

---

## Part E: View the results

Everything is in the `results` folder (visible in the VS Code Explorer on the left).

### Inside VS Code

| File | How to view |
|---|---|
| `results/REPORT.md` | Click it, then press **Ctrl+Shift+V** (Markdown preview) to see formatted tables |
| `results/figures/*.png` | Click any PNG; VS Code shows the image |
| `results/comparative/*.csv` | Click to see the table as text |

### In Windows (browser, video player, Excel)

From the VS Code terminal, `explorer.exe` opens files with the normal Windows program:

```bash
explorer.exe results/GeoSemTwinX_Dashboard.html     # main digital-twin dashboard (browser)
explorer.exe results/demo_staer/demo.mp4            # demo video
explorer.exe results/demo_staer/Demo_Dashboard.html # twin built during the demo
explorer.exe results/TUM_Dashboard.html             # office twin
explorer.exe results/Live_Dashboard.html            # twin from the ROS live demo
explorer.exe results                                # open the whole folder in File Explorer
```

Or paste this into the File Explorer address bar:
`\\wsl.localhost\Ubuntu-24.04\home\administrator\GeoSemTwinX\results`

### What is in the dashboard

| Tab | Content |
|---|---|
| Digital twin | 3D map with asset boxes (ID, class, confidence), asset inspector, spatial query, class filters |
| Changes & versions | Moved / added / removed assets between sessions, version history |
| Comparative studies | Every study table and chart, with GASF highlighted |
| System performance | Latency per stage, FPS, GPU power and memory |

### Copy everything to D: (for slides / USB)

```bash
cp -r results /mnt/d/Gayathri/GeoSemTwinX_results
```

---

## Part F: Output files

| Path | Content |
|---|---|
| `results/REPORT.md` | All tables + computed key findings |
| `results/GeoSemTwinX_Dashboard.html` | Main warehouse twin + all comparative studies |
| `results/TUM_Dashboard.html` | Office twin |
| `results/demo_staer/demo.mp4`, `demo_last_frame.png` | Demo video (camera + detections + segmentation, top-down live twin) |
| `results/Live_Dashboard.html` | Twin from the ROS live run |
| `results/figures/` | Charts: `tum_slam.png`, `staer_trajectories.png`, fusion / pose-noise / exploration plots |
| `results/comparative/` | Every study as CSV + JSON (opens in Excel) |
| `results/twin_store/` | Versioned twin (JSON per session) |
| `results/power.json`, `studies.json` | GPU power/utilisation, study index used by the dashboard |

---

## Part G: Troubleshooting

| Symptom | Fix |
|---|---|
| `python: command not found` / `.venv/bin/activate: No such file` | Run the two `source` lines from Part A, in `~/GeoSemTwinX` |
| `pip install` says "externally-managed-environment" | Same: activate `.venv` first |
| `TypeError: process() takes no keyword arguments` while downloading models | `pip install -U brotli zstandard` (inside `.venv`) |
| `ROS 2 rclpy: not importable` | `source /opt/ros/jazzy/setup.bash` before activating `.venv` |
| `EnvironmentNameNotFound` for VIGS-SLAM | The env is named `vigs-slam` and lives in the **Ubuntu** distro, not Ubuntu-24.04 |
| Terminal seems frozen | A run is in progress. Check with `top` in a second terminal; Ctrl+C stops it |
| `[1]+ Stopped` | You pressed Ctrl+Z. Run `kill %1` to clean it up |
| Typed a Linux command in PowerShell (`source`/`&&` errors) | Enter Ubuntu first: `wsl -d Ubuntu-24.04` (or use the VS Code WSL terminal) |
| Hugging Face 401 / gated dataset | `hf auth login` with the account that accepted the `staerrobotics/warehouses` terms |

---

## Part H: Known limitations (state these when presenting)

* The RGB-D SLAM runs on the CPU (~1 s/frame), so the pipeline is not real-time (0.8 FPS end to end).
* Staer walkthrough scene03/001 is a straight, constant-velocity aisle traverse. Monocular-inertial SLAM cannot
  recover scale with zero acceleration, and RGB-D tracking is weak along a long low-texture aisle. Use TUM for SLAM
  accuracy, and the ground-truth-pose + pose-noise studies for Staer semantics.
* The open-vocabulary detector (Grounding DINO tiny) misses many small or distant warehouse assets (low recall). A
  larger or warehouse-trained detector is the next improvement.
* KITTI / SemanticKITTI (needs a LiDAR SLAM front end) and ScanNet (licence) are not covered.
