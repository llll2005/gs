# -*- coding: utf-8 -*-
"""Merge harvested results.txt metrics with run-level annotations -> 紀錄/實驗總表.csv"""
import csv, os

HARVEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'harvest.csv')
OUT = '紀錄/實驗總表.csv'

# run-level annotations: run -> (era, status, points, note)
# era: 舊資料(2026-05-28 前的 SfM/座標系, 不可與當前比較) / 當前資料
# status: 有效 / 作廢 / 負結果 / 平手 / smoke / 跑中 / 夭折 / 待補
A = {
 # ---- 舊資料時代（SfM 已重生, 全部不可與當前資料比較）----
 'citygsv2_mc_aerial_sh0_trim': ('舊資料','作廢','~4M','CityGSV2官方pipeline sh0; 舊28.9(b7)出處, SfM已換座標系不可比'),
 'citygsv2_mc_aerial_sh2_trim': ('舊資料','作廢','','provenance壞(混目錄), 見dev_cycles 2026-06-17'),
 'aerial_train_block_all_3x': ('舊資料','作廢','','舊coarse, 不可當init'),
 'aerial_train_block_all': ('舊資料','作廢','',''),
 'citygsv2_merged_train_test': ('舊資料','作廢','','舊merge測試'),
 'test_block_1': ('舊資料','作廢','',''),
 'aerial_test_block_1_test': ('舊資料','作廢','',''),
 'depth_init_block0_full': ('舊資料','作廢','','init測試'),
 'RTG_mc_aerial_coarse_sh2': ('舊資料','作廢','','舊coarse ckpt'),
 'RTG_mc_aerial_sh2_trim24': ('舊資料','作廢','','RTGStable首版(S3-S7)'),
 'ADMM_RTG_mc_aerial_sh2_trim': ('舊資料','作廢','','ADMM 25塊 5-outer; merged 19.51不可信(drift)'),
 'RTG_mc_no_ADDM_aerial_sh2_trim': ('舊資料','作廢','','no-ADMM對照 25塊'),
 'ADMM_v2_RTG_mc_aerial_sh2_trim': ('舊資料','作廢','','公平比較版; +0.06=net-zero(多訓練買的)'),
 'PathB_admm_67': ('舊資料','作廢','','ADMM PathB b6/7; net-zero結論出處'),
 # ---- 當前資料: RTG / init 時代 ----
 'RTG_s10_mc_no_ADDM_aerial_sh2_trim': ('當前資料','有效','~400-500k/塊','S10 RTGStable 25塊 depth-init 30k基線'),
 'RTG_s10_mc_no_freeze_no_ADDM_aerial_sh2_trim': ('當前資料','平手','','freeze消融: 無freeze≈同'),
 'RTG_s10_freeze_no_B3_no_ADMM_aerial_sh2_trim': ('當前資料','負結果','','B3消融'),
 'RTG_s10_mc_no_freeze_b5_10_no_ADDM_aerial_sh2_trim': ('當前資料','負結果','','B5消融'),
 'RTG_s10_mc_no_freeze_no_b3_no_ADDM_aerial_sh2_trim': ('當前資料','負結果','','B3消融'),
 'RTG_s10_mc_no_freeze_no_ADDM_aerial_sh2_trim_with_less_densify_threshold': ('當前資料','負結果','','densify閾值消融'),
 'AB_coarse_init_aerial_sh2_trim': ('當前資料','作廢','~3.7M(b16 OOM)','densify confound(0.00005) + 6GB OOM, 見task1'),
 'AB_coarse_md_aerial_sh2_trim': ('當前資料','有效','','task1乾淨版: init是次要效應(+0.4dB但SSIM/LPIPS輸)'),
 'RTG_resetoff_aerial_sh2_trim': ('當前資料','負結果','','實驗①reset-off; confound(同時關大尺寸剪枝)'),
 # ---- MCMC 30k 時代 (sh2, 30k recipe, depth final_factor 0.001) ----
 'mcmc_2dgs_b7': ('當前資料','有效','414k','base MCMC vs baseline 453k=21.967: 平手-9%點'),
 'mcmc_2dgs_b7_cap300': ('當前資料','有效','270k','預算掃描'),
 'mcmc_2dgs_b7_cap200': ('當前資料','有效','180k','預算掃描'),
 'mcmc_b7_capmax': ('當前資料','有效','720k','cap800→720k; 30k recipe「天花板」(後判定是recipe不足)'),
 'gg_mcmc_b7_cap300': ('當前資料','負結果','~270k','gradient-guided取樣: net-zero'),
 'gg_mcmc_b7_cap200': ('當前資料','負結果','~180k','同上'),
 'mcmc_capmax_prune10_b7': ('當前資料','有效','648k','importance-prune曲線'),
 'mcmc_capmax_prune30_b7': ('當前資料','有效','504k','剪30%無損'),
 'mcmc_capmax_prune50_b7': ('當前資料','有效','360k','剪50%無損; vs native360 +0.457'),
 'mcmc_native360_ctrl_b7': ('當前資料','有效','360k','嚴格對照: 直接低cap輸explore-then-prune'),
 'mcmc_sh3_cap800_b7': ('當前資料','有效','720k','sh3槓桿 +0.295'),
 'mcmc_sh3_cap800_prune50_b7': ('當前資料','有效','360k','sh3+prune50 雙贏閉環'),
 'sgl_mcmc_b7': ('當前資料','平手','~720k','SGL純metric加項: 持平'),
 'edge_aware_b7': ('當前資料','負結果','719k','EdgeAware grad-densify+edge'),
 'edge_mcmc_b7': ('當前資料','平手','~800k','edge訊號嫁接MCMC: net-zero'),
 'exp1_mcmc_dist_b7': ('當前資料','負結果','','MCMC+distortion λ100'),
 'exp2_mcmc_freg_b7': ('當前資料','負結果','','MCMC+FreGS'),
 'exp3_mcmc_dist_freg_b7': ('當前資料','負結果','','兩者疊加'),
 'exp4_gd_base_b7': ('當前資料','負結果','','grad-densify COLMAP init'),
 'exp5_gd_freg_b7': ('當前資料','負結果','','grad-densify+FreGS'),
 'atom_n1_lam002_b7': ('當前資料','負結果','','AtomGS edge-normal loss λ0.02'),
 'atom_n2_lam005_b7': ('當前資料','負結果','','λ0.05'),
 'atom_n3_lam015_b7': ('當前資料','負結果','','λ0.15'),
 'atom_n4_lam005_q4_b7': ('當前資料','負結果','','λ0.05 q4'),
 'scaffold_2dgs_b7': ('當前資料','負結果','100k anchor×5','Scaffold v1固定anchor'),
 'scaffold_full2_b7': ('當前資料','負結果','66k anchor','增生修復版'),
 'scaffold_full3_b7': ('當前資料','負結果','~200k anchor','放寬閾值長滿; 仍輸→Scaffold止損'),
 'scaffold_c1_60k_b7': ('當前資料','負結果','','Scaffold最佳(60k)仍輸MCMC路線'),
 'scaffold_c2_n10_b7': ('當前資料','負結果','','n_offset消融'),
 'scaffold_c3_feat64_b7': ('當前資料','負結果','','feat64反而傷'),
 'scaffold_c4_sgl_b7': ('當前資料','負結果','','+SGL小助仍輸'),
 'mcmc_dr0_b7': ('當前資料','負結果','','depth_ratio=0消融'),
 'mcmc_dr05_b7': ('當前資料','負結果','','depth_ratio=0.5消融'),
 # ---- 60k recipe 時代 (depth final_factor 0.05) ----
 'mcmc_2dgs_60k_b7': ('當前資料','有效','532k','60k recipe首發; 揭穿30k訓練不足'),
 'mcmc_cap2M_b7': ('當前資料','有效','969k','cap2M沒填滿; scaling law數據點'),
 'citygsv2_b7_faithful': ('當前資料','作廢','','我選錯densify config跑爛的baseline, 勿引用'),
 'e1_citygsv2_24k_b7': ('當前資料','有效','','DGD反推E1: 官方24k recipe在當前資料'),
 'e1b_scaler09_b7': ('當前資料','有效','','E1b scaler0.9'),
 'e2a_mcmc2p2m_sh2_b7': ('當前資料','有效','~2.2M','sh2大cap: 22.51'),
 'e2a_mcmc4m_sh2_b7': ('當前資料','有效','~4M','sh2-4M: 崩(VRAM擠壓訓練品質)→sh0結論證據'),
 'e2b_gd_depth_sh2_s09_b7': ('當前資料','有效','','grad-densify+depth-init對照'),
 'fullcoarse_24k_b7': ('當前資料','有效','','全域coarse+24k官方式(低分=coarse init在b7弱)'),
 'mcmc_60k_sh3_aggr_b7': ('當前資料','有效','1.17M','sh3+cap1.3M aggressive'),
 'mcmc_60k_sh3_aggr17_b7': ('當前資料','有效','1.53M','★主基線 24.30/0.755/0.309(results.txt空,數字取自日誌)'),
 'mcmc_60k_sh3_aggr17_COARSE_b7': ('當前資料','負結果','','aggr17換coarse init: 22.27 << depth-init 24.30'),
 'mcmc_60k_sh3_native765_b7': ('當前資料','有效','765k','native低cap對照'),
 'mcmc_60k_sh3_prune50_b7': ('當前資料','有效','~765k','60k+prune50: ≈native aggr 23.96半點數'),
 'mcmc_60k_sh3_aggr17_prune_b7': ('當前資料','有效','~1.2M','★帳面最佳 24.47/0.770/0.284; aggr17+LightGaussian v_important_score 20%prune@30k'),
 'mcmc_60k_sh3_aggr17_ver1_b7': ('當前資料','夭折','','無結果'),
 'mcmc_60k_sh3_aggr17_ver2_b7': ('當前資料','負結果','1.29M','RTG hybrid B1-B5+distortion+cap'),
 'citygs_ft60k_coarse3x': ('當前資料','有效','','官方式pipeline: coarse3x+60k ft; PSNR 24.16但SSIM/LPIPS大輸aggr17'),
 'coarse_mc_aerial_1.2x': ('當前資料','有效','','當前資料coarse(1.2x)'),
 'coarse_mc_aerial_2x': ('當前資料','有效','','coarse(2x)'),
 'coarse_mc_aerial_3x': ('當前資料','有效','','coarse(3x)'),
 'coarse_citygsv2_sh2_1.2x': ('當前資料','有效','','CityGSV2式coarse'),
 'mcmc_2dgs_60k_sh3_aggr17_aerial': ('當前資料','中斷','','07-02多塊批次未完成; 表中為中途val非終值(b3停1499步=OOM疑似,b7停15k); 勿引用'),
 # ---- SF-MCMC / quad ----
 'sf_mcmc_60k_sh3_aggr17_b7': ('當前資料','平手','1.53M','SF-MCMC arm1: 統計平手SSIM/LPIPS微優; churn稅證實'),
 'mcmc_60k_sh3_aggr17_b8': ('當前資料','有效','~1.53M','quad; Hyprland crash後42k續跑收官'),
 'mcmc_60k_sh3_aggr17_b12': ('當前資料','有效','~0.93M','quad; 五次嘗試(偏差:screen_size_prune_px=300+cap1.0M); OOM saga=D1素材'),
 'mcmc_60k_sh3_aggr17_b13': ('當前資料','有效','~1.53M','quad; 偏差:screen_size_prune_px=300'),
 # ---- smokes ----
 'gg_smoke7': ('當前資料','smoke','','')  ,
 'imp_smoke7': ('當前資料','smoke','',''),
 'imp_smoke7b': ('當前資料','smoke','',''),
 'imp_smoke7c': ('當前資料','smoke','',''),
 'imp_smoke7d': ('當前資料','smoke','',''),
 'mcmc_smoke7': ('當前資料','smoke','',''),
 'mcmc_regsmoke7': ('當前資料','smoke','',''),
 'sf_mcmc_smoke_b7': ('當前資料','smoke','','SF-MCMC機制smoke'),
 'diag_offsurface': ('當前資料','smoke','','診斷用'),
 'quad_merge_aggr17': ('當前資料','連結重複','','blocks/=四塊symlink, 與各bX行同檔勿重複計; merged.ckpt=1.18M顆'),
 'quad_merge_eval_b7': ('當前資料','有效','1.18M(merged)','merged.ckpt在b7 val;對比24.30=掉分'),
 'quad_merge_eval_b7_vis': ('當前資料','有效','1.18M(merged)','同b7 eval+save_val存圖(孤島效應視覺證據)'),
 'quad_merge_eval_b8': ('當前資料','有效','1.18M(merged)','merged.ckpt在b8 val;對比23.62=掉分'),
 'quad_merge_eval_b12': ('當前資料','有效','1.18M(merged)','merged.ckpt在b12 val;對比22.45=掉分'),
 'quad_merge_eval_b13': ('當前資料','有效','1.18M(merged)','merged.ckpt在b13 val;對比23.87=掉分'),
 'sf_cool_60k_b7': ('當前資料','平手','1.53M','SF冷卻臂(min_op.92/frac.25); 三臂平手SF線關閉'),
 'quad_consolidate_10k': ('當前資料','有效','1.18M','merged+union829張10k步; 14視角掉分7.01→3.41; 表中為union val非14視角尺'),
 'smoke_sb_b7': ('當前資料','smoke','','SB color(DBS移植)首個smoke b7 1500步; F=25 floats/pt'),
 'mcmc_sb_b12_cap1m_harvestdust': ('當前資料','有效','73,637(剪91.8%)','★訓練中清塵已驗: 21.89(-0.23 vs A\' 22.12)但 SSIM 0.5920(+0.0015)+顆數 12.2× 少=好trade-off; 修正「訓練中清塵無損」推論(cull_dust靜態無損不可套動態)'),
 'mcmc_sb_b12_cap1m_condense': ('當前資料','OOM死亡','0.99M@死亡33.3k','★凝結退火(死亡線30k-42k→0.05)OOM: 抬線→處決 63%→74%→relocation storm; 揭露死亡跑步機=b12真病理; 瞄錯時機(densify期無殭屍只有過載)'),
 'mcmc_sb_b12_cap1m_vpc': ('當前資料','夭折','0.94M@33.2k','v/c判準(價值=多視角貢獻)主動砍: VRAM 5.52G紅區+方向錯(往63%過載系統加壓); 機制本身已 code review 修好4個bug並驗證觸發正常, 待低churn環境或乖塊重測'),
 'smoke_strips_b7': ('當前資料','smoke','','K-strip主線機制smoke b7; 揭露trim rasterizer cx/cy寫死→pp_shifty補丁'),
 'mcmc_sb_60k_aggr17_b12_cap1m': ('當前資料','有效','0.90M','SB ablation ArmA: vs SH3錨22.45同cap=-0.46dB,SSIM/LPIPS平; sb_lr未調; 軌跡形狀複製錨點(42k後跳升)'),
 'mcmc_sb_60k_aggr17_b12_capfx': ('當前資料','OOM死亡','~1.7-2M@死亡','SB ablation ArmB: OOM死於~30k(backward,5.15G); cap表SB 3.71M過樂觀→實用~1.4-1.5M; 死前val 19.30@28.4k≈A同期=中局無增益弱證據; count局未完賽→B\'+K=2重跑'),
 'sb_harvest_cprime': ('當前資料','有效','0.90M','收割三臂對照: resume機制驗證(vs原版21.99,-0.1變異)'),
 'sb_harvest_freeze': ('當前資料','有效','0.90M','★opacity凍結≈平(21.97): 凝結非opacity驅動,我方-2.4dB預測錯/Gemini對; 終態審計:凍結下塵埃照樣67%(scale驅動),0錨點卻同品質=opacity分佈簡併自由維度'),
 'sb_harvest_noreg': ('當前資料','有效','0.90M','撤opacity L1≈平(21.98): 收割期α壓力中性,殭屍對L1惰性(梯度死水)'),
 'sb_surgery_coloronly': ('當前資料','有效','0.18M(top20%)','★手術臂: 42k剪80%+鎖幾何/α/scale+純color/SB 5k步=21.04(19.4起,拿回跳升63%)→收割引擎=外觀收斂,凝結=副現象'),
 'mcmc_sb_ns_b12_cap2m_K2': ('當前資料','OOM死亡','~1.9M@死亡(30k存1.24M)','★new-scale count ablation B\': trim+SB@2M+K=2 死於34.5k單幀4.74G(binning). screen-prune狂救224次(每次8-10萬>300px)仍死=事件層攔不住幀層. K=2均勻切割對空間集中無效. CPU預測b12=全體高overdraw非少數巨獸→正解=per-step緊急剪(screen_prune_emergency_px)'),
 'mcmc_sb_ns_b12_cap1m_K1': ('當前資料','跑中','cap1.0M','new-scale A\': trim+SB@1M K=1 合體二進位重訓(舊A是失傳3月bin=錯配). 1M錨點'),
 # ---- 2026-08-05~06：P7 深度損失覆蓋率 bug 汙染（見 深度損失覆蓋率bug_2026-08-06.md）----
 # 判準 d_reg>1（健康 0.002-0.7）。汙染率與結果單調對應。⛔ 不可當基準。
 'aggr24k_b12': ('當前資料','⛔P7汙染','77.6k@24k','⛔P7汙染2.4%(1141/48091, d_reg最大6.25e5)。曾是「少而準」起點22.070與紋理比0.170的來源, 兩者皆不可引用'),
 'aggr24k_v2_b12': ('當前資料','⛔P7汙染','0.29M@7114','⛔P7汙染4.2%。死於7114'),
 'aggr24k_v3_b12': ('當前資料','⛔P7汙染','0.67M@6107','⛔P7汙染3.5%。19.40; 原歸因「interval 100 壓垮 opacity」需重檢'),
 'graded_k4_24k_b12': ('當前資料','⛔P7汙染','78.9k@24k','⛔P7汙染2.3%。Stage0 信心分級init, 終點22.017; 與對照v1的2.4%相當⇒差值大致仍成立但絕對值不可引用'),
 'graded_r08_24k_b12': ('當前資料','⛔P7汙染','','⛔P7汙染2.8%。球狀清除修正版, 砍於8946(val 20.639 與v1無差異)'),
 'cap80k_60k_b12': ('當前資料','⛔P7汙染','80k(釘死)','⛔P7汙染14.9%(7297/49100)⇒崩潰主因。val 20.90→20.79→20.34→17.06。「少而準」仍未測到, 需修正後重跑'),
 'sfminit24k_b12': ('當前資料','⛔P7汙染','0.31M@2495','⛔P7汙染18.9%。純SfM-init(image-filtered只得320k⇒覆蓋不足), 已砍'),
 # ---- 2026-08-04~06：官方 pipeline 於當前資料（乾淨, d_reg max 0.412-0.695）----
 'official_ft_blk5': ('當前資料','有效','1.18M','★乾淨的coarse-init對照: 官方60k config + 完整official_coarse_sh2, blockdim[4,4] blk5, 23.342/0.619/0.665'),
 'official_depthinit_blk5': ('當前資料','跑中','','★init單變數A/B: 同上config只換depth_init_4x4/block_5.ply。對照official_ft_blk5的23.342'),

}

# manual rows for runs that died without results.txt (provenance of failures)
EXTRA = [
 {'date':'2026-07-05','run':'mcmc_60k_sh3_aggr17_b12(嘗試1-2)','block':'12','psnr':'','ssim':'','lpips':'',
  'era':'當前資料','status':'OOM死亡','points':'0.41M@死亡','note':'step3151確定性OOM; 近相機怪物診斷起點'},
 {'date':'2026-07-06','run':'mcmc_60k_sh3_aggr17_b12(sub05)','block':'12','psnr':'','ssim':'','lpips':'',
  'era':'當前資料','status':'OOM死亡','points':'0.32M@死亡','note':'init降採樣無效證明(trim重整化族群)'},
 {'date':'2026-07-06','run':'mcmc_60k_sh3_aggr17_b13(嘗試1)','block':'13','psnr':'','ssim':'','lpips':'',
  'era':'當前資料','status':'OOM死亡','points':'0.51M@死亡','note':'step7499單筆4.64GiB binningBuffer尖峰'},
 {'date':'2026-07-06','run':'mcmc_60k_sh3_aggr17_b13(sub05)','block':'13','psnr':'','ssim':'','lpips':'',
  'era':'當前資料','status':'夭折','points':'','note':'~step4k主動搶佔(統一改screen-prune方案)'},
 # ---- gsplat 探針線 (probes/p5_cost_aware/train_p5.py, 非 outputs/ 格式) ----
 {'date':'2026-07-13','run':'t1_lambdaK_K8_b12','block':'12','psnr':'21.66','ssim':'','lpips':'',
  'era':'當前資料','status':'負結果','points':'','note':'λ+K首合體50k; budget 15M過低診斷; gsplat'},
 {'date':'2026-07-13','run':'freeze_b12_K2','block':'12','psnr':'','ssim':'','lpips':'',
  'era':'當前資料','status':'OOM死亡','points':'1.70M@死亡','note':'封閉公式凍結測試K=2; isect_tiles OOM@~1.8M; τ訓練中73→~120漂移實證; gsplat'},
 {'date':'2026-07-13','run':'freeze_b12_K8(ver0713)','block':'12','psnr':'','ssim':'','lpips':'',
  'era':'當前資料','status':'OOM死亡','points':'2.50M@死亡','note':'K8撞cap2.5M後碎片化OOM(290MB連續塊要不到); 吃下97.9M相交怪物peak3.35G=K供給側驗證; 使用者跑'},
 {'date':'2026-07-13','run':'freeze_b12_K8_v2','block':'12','psnr':'20.867','ssim':'','lpips':'',
  'era':'當前資料','status':'有效','points':'0.90M','note':'動態天花板+ω=2.0首個跑完10k; gsplat; 使用者跑'},
 {'date':'2026-07-14','run':'freeze_b7_K8_v2','block':'7','psnr':'22.611','ssim':'','lpips':'',
  'era':'當前資料','status':'有效','points':'1.70M','note':'b7對照10k; gsplat; 使用者跑'},
 {'date':'2026-07-14','run':'freeze_b12_K8_v3','block':'12','psnr':'21.523','ssim':'','lpips':'',
  'era':'當前資料','status':'負結果','points':'1.37M(峰1.53M)','note':'60k+depth/normal reg; 60k沒贏20k; SOLD churn; FINAL尺; gsplat; 使用者跑'},
 {'date':'2026-07-15','run':'freeze_b12_K8_v4','block':'12','psnr':'21.590','ssim':'','lpips':'',
  'era':'當前資料','status':'負結果','points':'0.92M(峰1.94M)','note':'20k; 1.94M被動態天花板賣回920k=churn實證; 4×點+幾何正則PSNR反降(⚠跨kernel confound,見統一預算框架doc); gsplat; 使用者跑'},
]

rows = []
with open(HARVEST) as f:
    for r in csv.DictReader(f):
        ann = A.get(r['run'], ('待補','待補','',''))
        rows.append({'date':r['date'],'run':r['run'],'block':r['block'],
                     'psnr':r['psnr'],'ssim':r['ssim'],'lpips':r['lpips'],
                     'era':ann[0],'status':ann[1],'points':ann[2],'note':ann[3]})
rows += EXTRA
rows.sort(key=lambda r:(r['date'],r['run'],str(r['block']).zfill(2)))
with open(OUT,'w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['date','run','block','psnr','ssim','lpips','points','era','status','note'])
    w.writeheader(); w.writerows(rows)
n_annot = sum(1 for r in rows if r['status']!='待補')
print('wrote', OUT, len(rows), 'rows;', n_annot, 'annotated,', len(rows)-n_annot, '待補')
