# GeoSemTwin-X — experiment summary

## staer fusion

| method                        | poses                                      |   objects |   precision |   recall |    f1 |   duplicate_rate |   identity_consistency |   id_switches |   label_acc |   mean_purity |   miou |
|:------------------------------|:-------------------------------------------|----------:|------------:|---------:|------:|-----------------:|-----------------------:|--------------:|------------:|--------------:|-------:|
| GASF (ours)                   | ground-truth poses                         |        11 |       0.455 |    0.070 | 0.122 |            0.200 |                  0.735 |            14 |       0.200 |         1.000 |  0.103 |
| GASF w/o geometry terms       | ground-truth poses                         |         7 |       0.429 |    0.035 | 0.065 |            0.000 |                  0.612 |            14 |       0.333 |         1.000 |  0.111 |
| GASF w/o multi-view gate      | ground-truth poses                         |        17 |       0.647 |    0.140 | 0.231 |            0.091 |                  0.857 |            14 |       0.364 |         1.000 |  0.144 |
| GASF w/o duplicate merge      | ground-truth poses                         |        12 |       0.417 |    0.070 | 0.120 |            0.200 |                  0.735 |            14 |       0.200 |         1.000 |  0.103 |
| GASF w/o confidence weighting | ground-truth poses                         |        11 |       0.455 |    0.070 | 0.122 |            0.200 |                  0.735 |            14 |       0.200 |         1.000 |  0.103 |
| Centroid tracking             | ground-truth poses                         |        11 |       0.455 |    0.070 | 0.122 |            0.200 |                  0.735 |            14 |       0.200 |         0.904 |  0.108 |
| Geometry-only clustering      | ground-truth poses                         |        11 |       0.364 |    0.053 | 0.092 |            0.500 |                  0.612 |            14 |       0.250 |         1.000 |  0.108 |
| Independent 2D labelling      | ground-truth poses                         |        87 |       0.126 |    0.140 | 0.133 |            3.455 |                  0.224 |            38 |       0.364 |         1.000 |  0.150 |
| GASF (ours)                   | GeoSemTwin-X RGB-D SLAM                    |         8 |       0.500 |    0.053 | 0.095 |            0.250 |                  0.655 |            14 |       0.250 |         1.000 |  0.003 |
| GASF w/o geometry terms       | GeoSemTwin-X RGB-D SLAM                    |         7 |       0.571 |    0.053 | 0.096 |            0.250 |                  0.655 |            14 |       0.250 |         1.000 |  0.003 |
| GASF w/o multi-view gate      | GeoSemTwin-X RGB-D SLAM                    |        22 |       0.545 |    0.158 | 0.245 |            0.083 |                  0.873 |            14 |       0.333 |         0.979 |  0.003 |
| GASF w/o duplicate merge      | GeoSemTwin-X RGB-D SLAM                    |         8 |       0.500 |    0.053 | 0.095 |            0.250 |                  0.636 |            14 |       0.250 |         1.000 |  0.003 |
| GASF w/o confidence weighting | GeoSemTwin-X RGB-D SLAM                    |         8 |       0.500 |    0.053 | 0.095 |            0.250 |                  0.655 |            14 |       0.250 |         1.000 |  0.003 |
| Centroid tracking             | GeoSemTwin-X RGB-D SLAM                    |        13 |       0.385 |    0.070 | 0.119 |            0.400 |                  0.636 |            17 |       0.200 |         0.926 |  0.003 |
| Geometry-only clustering      | GeoSemTwin-X RGB-D SLAM                    |        10 |       0.400 |    0.053 | 0.093 |            0.000 |                  0.764 |             0 |       0.250 |         1.000 |  0.005 |
| Independent 2D labelling      | GeoSemTwin-X RGB-D SLAM                    |        93 |       0.129 |    0.158 | 0.142 |            3.583 |                  0.218 |            43 |       0.417 |         1.000 |  0.003 |
| GASF (ours)                   | VIGS-SLAM mono+IMU (traj_full_afterBA.txt) |         4 |       0.000 |    0.000 | 0.000 |            0.000 |                  0.000 |             0 |       0.000 |         0.000 |  0.000 |
| GASF w/o geometry terms       | VIGS-SLAM mono+IMU (traj_full_afterBA.txt) |         4 |       0.250 |    0.018 | 0.033 |            0.000 |                  0.091 |             1 |       0.000 |         0.720 |  0.000 |
| GASF w/o multi-view gate      | VIGS-SLAM mono+IMU (traj_full_afterBA.txt) |        14 |       0.286 |    0.070 | 0.113 |            0.250 |                  0.636 |             1 |       0.500 |         1.000 |  0.000 |
| GASF w/o duplicate merge      | VIGS-SLAM mono+IMU (traj_full_afterBA.txt) |         4 |       0.000 |    0.000 | 0.000 |            0.000 |                  0.000 |             0 |       0.000 |         0.000 |  0.000 |
| GASF w/o confidence weighting | VIGS-SLAM mono+IMU (traj_full_afterBA.txt) |         4 |       0.000 |    0.000 | 0.000 |            0.000 |                  0.000 |             0 |       0.000 |         0.000 |  0.000 |
| Centroid tracking             | VIGS-SLAM mono+IMU (traj_full_afterBA.txt) |         8 |       0.125 |    0.018 | 0.031 |            0.000 |                  0.636 |             0 |       0.000 |         1.000 |  0.000 |
| Geometry-only clustering      | VIGS-SLAM mono+IMU (traj_full_afterBA.txt) |         7 |       0.143 |    0.018 | 0.031 |            2.000 |                  0.182 |             2 |       0.000 |         1.000 |  0.000 |
| Independent 2D labelling      | VIGS-SLAM mono+IMU (traj_full_afterBA.txt) |        48 |       0.104 |    0.088 | 0.095 |            1.200 |                  0.455 |             6 |       0.400 |         1.000 |  0.000 |

## staer keyframe interval

| method                   |   interval |   keyframes |   objects |   precision |   recall |    f1 |   duplicate_rate |   identity_consistency |   id_switches |   label_acc |   mean_purity |   miou |   detector_ms_per_frame |   pipeline_ms_per_frame |
|:-------------------------|-----------:|------------:|----------:|------------:|---------:|------:|-----------------:|-----------------------:|--------------:|------------:|--------------:|-------:|------------------------:|------------------------:|
| GASF (ours)              |          1 |         181 |        14 |       0.500 |    0.111 | 0.182 |            0.286 |                  0.718 |            37 |       0.429 |         1.000 |    nan |                 354.355 |                1426.035 |
| Centroid tracking        |          1 |         181 |        18 |       0.556 |    0.159 | 0.247 |            0.200 |                  0.852 |            37 |       0.400 |         0.954 |    nan |                 354.355 |                1426.035 |
| Independent 2D labelling |          1 |         181 |       266 |       0.049 |    0.206 | 0.079 |           10.462 |                  0.087 |           136 |       0.308 |         1.000 |    nan |                 354.355 |                1426.035 |
| GASF (ours)              |          2 |          91 |        12 |       0.500 |    0.086 | 0.147 |            0.167 |                  0.711 |            16 |       0.333 |         1.000 |    nan |                 177.178 |                1248.857 |
| Centroid tracking        |          2 |          91 |        15 |       0.600 |    0.138 | 0.224 |            0.111 |                  0.855 |            16 |       0.444 |         0.948 |    nan |                 177.178 |                1248.857 |
| Independent 2D labelling |          2 |          91 |       136 |       0.081 |    0.155 | 0.106 |            5.909 |                  0.145 |            65 |       0.364 |         1.000 |    nan |                 177.178 |                1248.857 |
| GASF (ours)              |          3 |          61 |        11 |       0.455 |    0.070 | 0.122 |            0.200 |                  0.735 |            14 |       0.200 |         1.000 |    nan |                 118.118 |                1189.798 |
| Centroid tracking        |          3 |          61 |        11 |       0.455 |    0.070 | 0.122 |            0.200 |                  0.735 |            14 |       0.200 |         0.904 |    nan |                 118.118 |                1189.798 |
| Independent 2D labelling |          3 |          61 |        87 |       0.126 |    0.140 | 0.133 |            3.455 |                  0.224 |            38 |       0.364 |         1.000 |    nan |                 118.118 |                1189.798 |
| GASF (ours)              |          5 |          37 |        12 |       0.500 |    0.091 | 0.154 |            0.167 |                  0.758 |             6 |       0.167 |         1.000 |    nan |                  70.871 |                1142.551 |
| Centroid tracking        |          5 |          37 |        12 |       0.500 |    0.091 | 0.154 |            0.167 |                  0.758 |             6 |       0.167 |         0.943 |    nan |                  70.871 |                1142.551 |
| Independent 2D labelling |          5 |          37 |        56 |       0.196 |    0.145 | 0.167 |            2.000 |                  0.333 |            22 |       0.364 |         1.000 |    nan |                  70.871 |                1142.551 |
| GASF (ours)              |         10 |          19 |         8 |       0.375 |    0.042 | 0.075 |            0.333 |                  0.556 |             4 |       0.333 |         1.000 |    nan |                  35.436 |                1107.115 |
| Centroid tracking        |         10 |          19 |         8 |       0.375 |    0.042 | 0.075 |            0.333 |                  0.556 |             4 |       0.333 |         0.844 |    nan |                  35.436 |                1107.115 |
| Independent 2D labelling |         10 |          19 |        31 |       0.290 |    0.083 | 0.129 |            1.000 |                  0.500 |             9 |       0.333 |         1.000 |    nan |                  35.436 |                1107.115 |

## staer latency

| stage                          | unit         |       ms |
|:-------------------------------|:-------------|---------:|
| Grounding DINO + SAM 2.1 (GPU) | per keyframe |  354.355 |
| RGB-D SLAM tracking (CPU)      | per frame    | 1071.680 |
| Fused map integration          | per frame    |   45.721 |
| Occupancy update               | per keyframe |   19.287 |
| 3D lifting of detections       | per keyframe |   13.548 |
| GASF association + fusion      | per keyframe |    3.098 |

## staer map quality

| poses                                      | method                               |   map_points |   accuracy_median_m |   accuracy_p90_m |   precision_at_tol |   completeness_at_tol |   frames_skipped |
|:-------------------------------------------|:-------------------------------------|-------------:|--------------------:|-----------------:|-------------------:|----------------------:|-----------------:|
| ground-truth poses                         | Uncertainty-aware fusion (ours)      |       179221 |               0.010 |            0.021 |              0.993 |                 0.893 |                0 |
| ground-truth poses                         | Naive fusion (no gating, no weights) |       402179 |               0.016 |            0.047 |              0.915 |                 1.000 |                0 |
| GeoSemTwin-X RGB-D SLAM                    | Uncertainty-aware fusion (ours)      |       199796 |               0.305 |            0.788 |              0.083 |                 0.124 |                0 |
| GeoSemTwin-X RGB-D SLAM                    | Naive fusion (no gating, no weights) |       679494 |               0.449 |            1.161 |              0.038 |                 0.145 |                0 |
| VIGS-SLAM mono+IMU (traj_full_afterBA.txt) | Uncertainty-aware fusion (ours)      |       175014 |               3.291 |            5.095 |              0.000 |                 0.000 |                0 |
| VIGS-SLAM mono+IMU (traj_full_afterBA.txt) | Naive fusion (no gating, no weights) |       900700 |               5.312 |            8.969 |              0.000 |                 0.000 |                0 |

## staer pose noise

|   pose_error_m | method                   |   seed |   objects |   precision |   recall |    f1 |   duplicate_rate |   identity_consistency |   id_switches |   label_acc |   mean_purity |   miou |
|---------------:|:-------------------------|-------:|----------:|------------:|---------:|------:|-----------------:|-----------------------:|--------------:|------------:|--------------:|-------:|
|          0.000 | Centroid tracking        |  0.000 |    11.000 |       0.455 |    0.070 | 0.122 |            0.200 |                  0.735 |        14.000 |       0.200 |         0.904 |    nan |
|          0.000 | GASF (ours)              |  0.000 |    11.000 |       0.455 |    0.070 | 0.122 |            0.200 |                  0.735 |        14.000 |       0.200 |         1.000 |    nan |
|          0.000 | Geometry-only clustering |  0.000 |    11.000 |       0.364 |    0.053 | 0.092 |            0.500 |                  0.612 |        14.000 |       0.250 |         1.000 |    nan |
|          0.000 | Independent 2D labelling |  0.000 |    87.000 |       0.126 |    0.140 | 0.133 |            3.455 |                  0.224 |        38.000 |       0.364 |         1.000 |    nan |
|          0.050 | Centroid tracking        |  1.000 |    11.000 |       0.455 |    0.070 | 0.122 |            0.200 |                  0.735 |        14.000 |       0.200 |         0.904 |    nan |
|          0.050 | GASF (ours)              |  1.000 |    11.000 |       0.455 |    0.070 | 0.122 |            0.200 |                  0.735 |        14.000 |       0.200 |         1.000 |    nan |
|          0.050 | Geometry-only clustering |  1.000 |    10.000 |       0.400 |    0.053 | 0.093 |            0.250 |                  0.653 |        14.000 |       0.250 |         1.000 |    nan |
|          0.050 | Independent 2D labelling |  1.000 |    87.000 |       0.126 |    0.140 | 0.133 |            3.455 |                  0.224 |        38.000 |       0.364 |         1.000 |    nan |
|          0.100 | Centroid tracking        |  1.000 |    11.333 |       0.470 |    0.076 | 0.131 |            0.189 |                  0.728 |        14.000 |       0.189 |         0.909 |    nan |
|          0.100 | GASF (ours)              |  1.000 |    11.333 |       0.470 |    0.076 | 0.131 |            0.189 |                  0.735 |        14.000 |       0.189 |         1.000 |    nan |
|          0.100 | Geometry-only clustering |  1.000 |    11.000 |       0.394 |    0.058 | 0.102 |            0.233 |                  0.682 |        11.333 |       0.233 |         1.000 |    nan |
|          0.100 | Independent 2D labelling |  1.000 |    88.333 |       0.132 |    0.152 | 0.141 |            3.318 |                  0.232 |        38.667 |       0.371 |         1.000 |    nan |
|          0.200 | Centroid tracking        |  1.000 |    12.333 |       0.459 |    0.082 | 0.139 |            0.289 |                  0.721 |        15.333 |       0.233 |         0.915 |    nan |
|          0.200 | GASF (ours)              |  1.000 |    11.667 |       0.485 |    0.082 | 0.140 |            0.178 |                  0.752 |        14.000 |       0.178 |         1.000 |    nan |
|          0.200 | Geometry-only clustering |  1.000 |    11.667 |       0.404 |    0.064 | 0.111 |            0.283 |                  0.657 |        14.000 |       0.217 |         1.000 |    nan |
|          0.200 | Independent 2D labelling |  1.000 |    90.667 |       0.129 |    0.152 | 0.139 |            3.523 |                  0.222 |        41.000 |       0.371 |         1.000 |    nan |
|          0.400 | Centroid tracking        |  1.000 |    13.333 |       0.425 |    0.082 | 0.137 |            0.356 |                  0.674 |        13.333 |       0.300 |         0.947 |    nan |
|          0.400 | GASF (ours)              |  1.000 |    12.000 |       0.472 |    0.088 | 0.148 |            0.178 |                  0.720 |        11.333 |       0.178 |         1.000 |    nan |
|          0.400 | Geometry-only clustering |  1.000 |    12.333 |       0.417 |    0.070 | 0.120 |            0.200 |                  0.742 |         3.000 |       0.200 |         1.000 |    nan |
|          0.400 | Independent 2D labelling |  1.000 |    89.000 |       0.120 |    0.146 | 0.132 |            3.728 |                  0.213 |        39.667 |       0.339 |         1.000 |    nan |

## staer slam

| method                                     |   ate_m |   ate_sim3_scale |   rpe_m |   rpe_deg |   drift_pct |   ms_per_frame |
|:-------------------------------------------|--------:|-----------------:|--------:|----------:|------------:|---------------:|
| GeoSemTwin-X RGB-D SLAM                    |   2.710 |            1.301 |   1.011 |     4.484 |      15.040 |       1071.680 |
| w/o loop closure                           |   2.710 |            1.301 |   1.011 |     4.484 |      15.040 |        nan     |
| VIGS-SLAM mono+IMU (traj_full_afterBA.txt) |   4.981 |            2.299 |   1.247 |     5.117 |      27.644 |        nan     |

## staer slam sweep

| sequence   | method                              |   ate_m |   rpe_m |   rpe_deg |   path_m |   loops |   fallbacks |   ms_per_frame |
|:-----------|:------------------------------------|--------:|--------:|----------:|---------:|--------:|------------:|---------------:|
| first 6 s  | hybrid, depth ≤ 6 m, stride 1       |   1.425 |   0.863 |     6.929 |    5.972 |       0 |          93 |        646.706 |
| first 6 s  | hybrid, depth ≤ 10 m, stride 2      |   1.706 |   0.998 |     6.554 |    5.917 |       0 |          59 |        695.696 |
| first 6 s  | hybrid, depth ≤ 6 m, stride 2       |   1.493 |   2.942 |    15.323 |    5.917 |       0 |          44 |        694.268 |
| first 6 s  | hybrid + ICP, depth ≤ 8 m, stride 2 |   0.840 |   1.860 |     8.215 |    5.917 |       0 |          50 |       1026.121 |

## staer threshold

| method                   |   threshold |   objects |   precision |   recall |    f1 |   duplicate_rate |   identity_consistency |   id_switches |   label_acc |   mean_purity |   miou |
|:-------------------------|------------:|----------:|------------:|---------:|------:|-----------------:|-----------------------:|--------------:|------------:|--------------:|-------:|
| GASF (ours)              |       0.200 |        19 |       0.684 |    0.193 | 0.301 |            0.077 |                  0.845 |            17 |       0.308 |         1.000 |    nan |
| Centroid tracking        |       0.200 |        23 |       0.565 |    0.193 | 0.288 |            0.308 |                  0.689 |            29 |       0.231 |         1.000 |    nan |
| Independent 2D labelling |       0.200 |       157 |       0.115 |    0.246 | 0.156 |            4.722 |                  0.175 |            85 |       0.167 |         1.000 |    nan |
| GASF (ours)              |       0.250 |        17 |       0.588 |    0.140 | 0.227 |            0.200 |                  0.773 |            17 |       0.400 |         1.000 |    nan |
| Centroid tracking        |       0.250 |        17 |       0.588 |    0.140 | 0.227 |            0.100 |                  0.712 |            16 |       0.500 |         1.000 |    nan |
| Independent 2D labelling |       0.250 |       113 |       0.106 |    0.158 | 0.127 |            4.500 |                  0.182 |            54 |       0.333 |         1.000 |    nan |
| GASF (ours)              |       0.300 |        11 |       0.455 |    0.070 | 0.122 |            0.200 |                  0.735 |            14 |       0.200 |         1.000 |    nan |
| Centroid tracking        |       0.300 |        11 |       0.455 |    0.070 | 0.122 |            0.200 |                  0.735 |            14 |       0.200 |         0.904 |    nan |
| Independent 2D labelling |       0.300 |        87 |       0.126 |    0.140 | 0.133 |            3.455 |                  0.224 |            38 |       0.364 |         1.000 |    nan |
| GASF (ours)              |       0.350 |         9 |       0.556 |    0.070 | 0.125 |            0.000 |                  0.895 |             1 |       0.200 |         1.000 |    nan |
| Centroid tracking        |       0.350 |         9 |       0.556 |    0.070 | 0.125 |            0.000 |                  0.921 |             0 |       0.200 |         0.936 |    nan |
| Independent 2D labelling |       0.350 |        61 |       0.131 |    0.088 | 0.105 |            3.750 |                  0.211 |            30 |       0.375 |         1.000 |    nan |
| GASF (ours)              |       0.400 |         6 |       0.667 |    0.053 | 0.098 |            0.000 |                  0.969 |             0 |       0.250 |         1.000 |    nan |
| Centroid tracking        |       0.400 |         6 |       0.667 |    0.053 | 0.098 |            0.000 |                  0.969 |             0 |       0.250 |         1.000 |    nan |
| Independent 2D labelling |       0.400 |        47 |       0.106 |    0.053 | 0.070 |            5.400 |                  0.156 |            27 |       0.400 |         1.000 |    nan |
| GASF (ours)              |       0.450 |         2 |       0.500 |    0.018 | 0.034 |            0.000 |                  0.952 |             0 |       1.000 |         1.000 |    nan |
| Centroid tracking        |       0.450 |         2 |       0.500 |    0.018 | 0.034 |            0.000 |                  0.952 |             0 |       1.000 |         1.000 |    nan |
| Independent 2D labelling |       0.450 |        30 |       0.067 |    0.035 | 0.046 |            9.500 |                  0.095 |            19 |       0.500 |         1.000 |    nan |

## synthetic change

| method                                     |   precision |   recall |    f1 |
|:-------------------------------------------|------------:|---------:|------:|
| Occupancy-verified change detection (ours) |       1.000 |    0.833 | 0.909 |
| Without occupancy verification             |       0.556 |    0.833 | 0.667 |

## synthetic exploration

| method                                        |   coverage_at_40m |   coverage_final |   assets_identified_pct |   replans |   collisions |
|:----------------------------------------------|------------------:|-----------------:|------------------------:|----------:|-------------:|
| NBV (ours: unknown + uncertainty + ambiguity) |             0.189 |            0.276 |                  94.444 |     1.667 |        1.000 |
| NBV w/o ambiguity term                        |             0.192 |            0.269 |                 100.000 |     1.333 |        0.667 |
| NBV coverage only                             |             0.187 |            0.271 |                  97.222 |     1.667 |        0.667 |
| Nearest frontier                              |             0.188 |            0.280 |                  95.833 |     0.667 |        0.000 |
| Random goals                                  |             0.141 |            0.190 |                  70.833 |     4.667 |        0.667 |

## synthetic fusion

| method                   |   objects |     tp |   duplicates |   false_pos |   gt_assets |   precision |   recall |    f1 |   duplicate_rate |   label_acc |   identity_consistency |   id_switches |   mean_purity |   miou |   seed |
|:-------------------------|----------:|-------:|-------------:|------------:|------------:|------------:|---------:|------:|-----------------:|------------:|-----------------------:|--------------:|--------------:|-------:|-------:|
| Centroid tracking        |    29.333 | 20.667 |        6.000 |       2.667 |      22.000 |       0.706 |    0.939 | 0.806 |            0.291 |       0.983 |                  0.822 |        33.000 |         0.895 |  0.329 |  1.000 |
| GASF (full)              |    22.333 | 18.000 |        2.000 |       2.333 |      22.000 |       0.805 |    0.818 | 0.811 |            0.114 |       1.000 |                  0.841 |         9.000 |         0.925 |  0.345 |  1.000 |
| GASF w/o duplicate merge |    30.667 | 20.000 |        9.000 |       1.667 |      22.000 |       0.653 |    0.909 | 0.760 |            0.450 |       1.000 |                  0.739 |        36.000 |         0.931 |  0.375 |  1.000 |
| GASF w/o geometry terms  |     9.667 |  5.333 |        0.000 |       4.333 |      22.000 |       0.556 |    0.242 | 0.337 |            0.000 |       0.917 |                  0.219 |        58.000 |         0.801 |  0.262 |  1.000 |
| GASF w/o multi-view gate |    44.667 | 19.667 |        8.333 |      16.667 |      22.000 |       0.441 |    0.879 | 0.587 |            0.423 |       0.983 |                  0.845 |        26.000 |         0.926 |  0.249 |  1.000 |
| Geometry-only clustering |    20.667 | 11.333 |        5.000 |       4.333 |      22.000 |       0.548 |    0.515 | 0.531 |            0.463 |       0.862 |                  0.501 |         7.000 |         0.787 |  0.200 |  1.000 |
| Independent 2D labelling |   289.000 | 22.333 |      236.667 |      30.000 |      22.000 |       0.077 |    1.000 | 0.144 |           10.601 |       0.895 |                  0.086 |       236.667 |         1.000 |  0.276 |  1.000 |

## tum slam

| sequence                    | method                           |   ate_m |   rpe_m |   rpe_deg |   path_m |   loops |   fallbacks |   ms_per_frame |
|:----------------------------|:---------------------------------|--------:|--------:|----------:|---------:|--------:|------------:|---------------:|
| rgbd_dataset_freiburg1_desk | Full (hybrid + PnP + loops)      |   0.029 |   0.037 |     2.670 |    9.309 |  24.000 |     166.000 |        642.365 |
| rgbd_dataset_freiburg1_desk | w/o PnP recovery                 |   0.030 |   0.037 |     2.719 |    9.309 |  26.000 |     166.000 |        441.472 |
| rgbd_dataset_freiburg1_desk | w/o loop closure                 |   0.054 |   0.042 |     2.596 |    9.309 |   0.000 |     166.000 |        642.365 |
| rgbd_dataset_freiburg1_desk | Odometry only (no PnP, no loops) |   0.073 |   0.044 |     2.687 |    9.309 |   0.000 |     166.000 |        441.472 |
| rgbd_dataset_freiburg1_desk | ORB-SLAM2 (published)            |   0.016 | nan     |   nan     |  nan     | nan     |     nan     |        nan     |
| rgbd_dataset_freiburg1_desk | ElasticFusion (published)        |   0.020 | nan     |   nan     |  nan     | nan     |     nan     |        nan     |
| rgbd_dataset_freiburg1_desk | DVO-SLAM (published)             |   0.021 | nan     |   nan     |  nan     | nan     |     nan     |        nan     |

## Key findings (computed from the tables above)

- Duplicates per asset: per-frame 2D labelling 3.45 → GASF 0.20; objects in map 87 → 11.
- F1: GASF 0.12 vs best conventional (Independent 2D labelling) 0.13.
- Identity consistency: GASF 0.73 vs per-frame 0.22.
- Staer SLAM GeoSemTwin-X RGB-D SLAM: ATE 271.0 cm (15.04% drift).
- Staer SLAM w/o loop closure: ATE 271.0 cm (15.04% drift).
- Staer SLAM VIGS-SLAM mono+IMU (traj_full_afterBA.txt): ATE 498.1 cm (27.64% drift).
- Compute: 0.80 FPS end-to-end, GPU 74 W while detecting, peak GPU memory 2853 MB.