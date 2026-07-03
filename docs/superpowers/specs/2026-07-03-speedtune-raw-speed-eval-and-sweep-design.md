# SpeedTune Raw-Speed Eval and Fixed Sweep Design

## Goal

Make SpeedTune evaluation report the backend's actual raw speed control value and add a cloud benchmark that measures task success and physical execution cost at fixed speeds from 1.0 to 4.0 in 0.5 increments.

## Eval contract

- Run 30 primary evaluation episodes by default.
- `fixed_time` records `v`; `chunk_toppra` records `vel_limit`.
- Do not emit `aggr`, `aggr_mean`, `acc_limit`, `derived_acc_limit`, or normalized-aggressiveness aggregate fields in eval artifacts.
- Plot the raw control value against accumulated physical simulation time, with contact and gripper annotations.
- Aggregate raw speed values during contact and non-contact decisions without normalizing them.
- Keep task success and reward-safe success separate. For fixed-time, reward-safe success still requires task success and no planned-speed violation.
- Record per-episode and aggregate `max_planned_qvel` for fixed-time.

## Fixed-speed sweep

- Load the frozen VLA once and do not load a DQN checkpoint.
- Connect to separate `fixed_time` and `chunk_toppra` RoboTwin servers.
- Evaluate speeds `1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0`.
- Run 30 episodes for every backend/speed pair. Before every pair, reseed the environment with the same base seed; VLA noise remains keyed by `(seed, episode, decision_step)`.
- Send `{"v": speed}` to fixed-time and `{"vel_limit": speed}` to whole-chunk TOPPRA.
- Retain backend-specific chunk execution: fixed-time `k_skip=10`, whole-chunk TOPPRA `k_skip=20` by default.
- Report task success, reward-safe success, fixed-time speed-violation rate, all-episode and success-only physical steps/time, decision counts, TOPPRA fallback rate, and fixed-time maximum planned velocity.
- Write JSON, CSV, Markdown, and plots for success rate, physical steps, and fixed-time maximum velocity versus speed.

## Cloud launcher

The launcher starts two RoboTwin process groups, waits for both ports, runs one VLA sweep process, records Expo/RoboTwin commits and effective parameters, and terminates all child process groups on normal exit, failure, or user interruption. A dry-run mode validates configuration without GPUs or servers.
