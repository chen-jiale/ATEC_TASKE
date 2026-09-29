# RGB-D calibration evidence

## Subsequent full-task success (2026-09-29)

The earlier failure notes below are historical. The updated RGB-D policy has
now completed three seed-42 headless episodes with score 18/18 and native
`basket_success=true`, in banana, mustard, sugar order. See
`docs/RGBD_SUCCESS.md` for exact runs, unchanged evaluation checks, and the
locally saved close-up MP4 at `output/rgbd_final/rgbd_success.mp4`.
These repeated same-seed runs do not establish a held-out success rate.

The deployed policy uses the fixed camera local +X optical axis, with image
right along local -Y and image down along local -Z. Intrinsics are
fx = fy = 732.9992676, cx = 320, cy = 240 for 640 by 480 images.
Camera world position is (-0.2, 0, 1.6266427), and its wxyz quaternion is
(0.9570244, 0, 0.2900074, 0).

## Measured check

Remote input: `/tmp/rgbd_twist_observations/observation_00000.npz`.
The unobstructed table patch uses rows 310:355 and columns 210:350.
With optical-axis depth, recovered world Z has 5th/50th/95th percentiles
0.82730737 / 0.82733024 / 0.82735491 metres. Relative to configured table
height 0.8266 m, median absolute error is 0.73024 mm.
Treating the same depth as ray distance gives median absolute error 9.06750 mm.

This supports the optical-axis depth convention for this recorded camera.
It is a table-plane check, not a measurement of object-center accuracy.

The subsequent runtime check in `/tmp/rgbd_fk_calibration.log` compares
independent FK with the simulator `gripper_base` position during motion.
Printed position errors are approximately 0.000001 m, including the blocked
approach posture. This supports the base translation, base rotation and FK
position chain; it does not verify finger contact geometry or grasp success.
Diagnostics are enabled only with `ATEC_FK_DIAGNOSTICS=1` in the evaluation
script. Simulator measurements are printed there and are not passed to the policy.

In `/tmp/rgbd_center_calibration.log`, the first two sugar estimates have
world XYZ errors (-0.036832, 0.017952, 0.002703) m and
(-0.047354, 0.017267, 0.015146) m. Object settling changes the true Z between
these observations. This directly contradicts treating the legacy locator as
uniformly centimetre-accurate. These two samples are not a validation dataset
and should not be used alone to fit replacement global biases.

## Subsequent geometry-based localization

The policy now uses `locate_banded_cloud` with RGB for sugar and mustard.
Sugar X/Z use yellow surface points to exclude the white gripper. Its Y center
uses the far visible end and half the measured 0.17625346 m mesh length.
Mustard XY use the upper quartile of yellow bottle points above table + 0.14 m;
the returned body height is halfway between the observed top and table.
These geometric assumptions require the upright, initially aligned scene.

Replaying `sample_00100.npz` from five recorded runs gives Euclidean errors:

| Run | Sugar (mm) | Mustard (mm) |
| --- | ---: | ---: |
| rgbd_paired_run1 | 2.70 | 2.93 |
| rgbd_paired_run2 | 3.25 | 2.62 |
| rgbd_sugar_color | 2.50 | 2.86 |
| rgbd_endpoint | 2.00 | 3.44 |
| rgbd_shortaxis | 1.73 | 3.61 |

These are pre-contact replay samples used during development, not a held-out
success-rate evaluation. Contact, occlusion and object rotation remain risks.

## Reproducible geometry audit

### Subsequent control experiments

The banana-first runs `rgbd_banana_first2` and `rgbd_banana_first5` reached
12 cumulative reward points before failing to complete the sugar stage.
This is partial reward evidence, not proof of stable two-object placement,
physical bottle-body contact, or complete task success without paired images.
Changing scene resets between runs also prevents attributing a difference to
the IK seed alone.

The later fixed-current-orientation transport experiment failed with a
0.3079 rad orientation residual. That is consistent with an infeasible pose;
it does not establish a stale-cache bug. Recomputing orientation from current
FK every step makes its current orientation residual zero by construction.
Likewise, widening the position gate from 15 to 30 to 50 mm is experimental,
not a calibrated gripper safety tolerance. Separate solver residual, actual
joint tracking error, and visual contact evidence are needed before accepting
these changes as fixes. Full 18-point success remains unverified.

The execution order is now `banana -> mustard_bottle -> sugar_box`. Banana is
closest to the basket-side transfer corridor, so removing it first reduces the
chance that a later arm sweep contacts the remaining objects. Mustard remains
before sugar so the bottle body grasp is tested while the sugar box is still
available as the final, widest-clearance pick. This changes sequencing only;
it does not alter object placement, rewards, or completion conditions.

`scripts/check_rgbd_geometry.py` was executed on ATEC_GPU against the 12 paired
frames in `/tmp/rgbd_waypoints`. Results are saved in
`docs/rgbd_geometry_report.json`, including camera-to-world and camera-to-base
homogeneous transforms. All frames yield a table-patch median height residual
of approximately 0.730 mm and a within-frame absolute 95th percentile below
0.756 mm. Maximum FK gripper-base position error is 0.001042 mm.
The table check uses a fixed patch and requires an unobstructed table view.
These checks validate the configured fixed external camera geometry and FK
position, not wrist-camera calibration, TCP contact offset, or object identity.

Reproduce on the remote project with:

```bash
/opt/conda/envs/isaaclab/bin/python scripts/check_rgbd_geometry.py /tmp/rgbd_waypoints --output /tmp/rgbd_geometry_report.json
```

The associated full grasp run ended with score 6.00. After object disturbance,
the legacy banana locator has an observed X error of about 67.6 mm. This is
an object-localization limitation despite the accurate coordinate transforms.

## Remaining perception and control limitations

- Depth connected components merge adjacent sugar and mustard surfaces.
- Yellow RGB thresholding also merges instances in an occluded recorded frame.
- Visible-surface bounds do not necessarily recover the full object center.
- The legacy banana locator still uses planar biases and a fixed height.
- Lift verification now checks elevated yellow points near FK-estimated TCP;
  it does not reliably distinguish object identity under all occlusions.
- Basket placement verification is not yet observation-based.
- `locate_pointcloud` remains diagnostic and is not the policy locator.
- No full three-object success or score of 18 has been verified.

The latest grasp experiments contact the sugar box before reaching their
requested pose. Successful IK alone does not validate the grasp template.
