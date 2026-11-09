#!/bin/bash
# 現行配方的旗標陣列 CUR（11-09 全場景 g44s30ksfm10nuPF2DB2_）；task_mech_time.sh、task_extract.sh 共用（source 進來）。
# 順序照抄 11-09 全場景的指令；後面的覆寫會蓋掉前面的值
CUR=(--model.renderer.init_args.record_reduce true --model.renderer.init_args.absgrad_gate true --model.metric.init_args.fused_ssim true
  --model.gaussian.init_args.optimization.optimizer.init_args.sh_rest_interval 16 --model.metric.init_args.lambda_dist_interval 4
  --model.metric.init_args.lambda_normal 0.0125 --model.metric.init_args.lambda_normal_interval 8
  --model.metric.init_args.depth_support true --model.density.init_args.geom_gate_sampling true --model.metric.init_args.depth_support_sfm_radius_nn 8
  --model.metric.init_args.depth_support_rule vote --model.metric.init_args.depth_support_interval 8
  --model.density.init_args.hole_fill_frac 0.002
  --model.density.init_args.merge_interval 1500 --model.density.init_args.merge_color_tol 0.1 --model.density.init_args.merge_prune true
  --model.density.init_args.merge_from_iter 3000 --model.density.init_args.merge_until_iter 27000 --model.density.init_args.merge_max_frac 0.02
  --model.density.init_args.merge_chunk 65536
  --model.density.init_args.inplane_offset 1.0 --model.density.init_args.inplane_hold 1000
  --model.metric.init_args.rgb_l2_weight 2.0
  --model.renderer.init_args.trim_running_warp true --model.renderer.init_args.bwd_zero_grad_skip true
  --model.density.init_args.sfm_vis true --model.density.init_args.sfm_vis_cams 0 --model.density.init_args.sfm_vis_cell 8
  --model.density.init_args.sfm_vis_px 4 --model.density.init_args.sfm_vis_free_frac 0.5 --model.density.init_args.sfm_vis_rep 0.1
  --model.density.init_args.sfm_vis_noup true
  --model.density.init_args.px_floor 1.0 --model.density.init_args.px_floor_interval 1000
  --model.gaussian.init_args.optimization.means_lr_init 1.28e-4 --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.lr_final 1.28e-6)
