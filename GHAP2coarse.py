import torch
import os
import sys

# 1. 設定路徑 (請依據你的實際路徑微調)
input_ckpt = "outputs/aerial_train_block_all_3x/checkpoints/epoch=6-step=30000.ckpt"
output_ckpt = "outputs/aerial_train_block_all_3x/checkpoints/epoch=6-step=30000_ghap_sh2.ckpt"
ghap_source_dir = "/home/LnoArch/Projects/專題/GHAP" # 替換為你 clone 的 GHAP 路徑

print(f"[階段 1] 載入 Coarse 全域模型: {input_ckpt}")
ckpt = torch.load(input_ckpt, map_location="cpu")
state_dict = ckpt['state_dict']

# 2. 擷取 CityGaussian 格式的原始張量
prefix = "gaussian_model.gaussians."
means = state_dict[f'{prefix}means']
scales = state_dict[f'{prefix}scales']
quats = state_dict[f'{prefix}quats']
opacities = state_dict[f'{prefix}opacities']
shs_dc = state_dict[f'{prefix}shs_dc']

N_original = means.shape[0]
print(f"[階段 2] 原始高斯點數量: {N_original} 顆")

# =====================================================================
# [階段 3] 執行 GHAP GMR (Gaussian Mixture Reduction) 壓縮
# =====================================================================
print("[階段 3] 啟動 GHAP KD-Tree 與最佳傳輸空間融合...")
sys.path.append(ghap_source_dir)

# 載入 GHAP 核心模組
# from gaussian_splatting_sampler.gmr import merge_gaussians_gmr

# 這裡呼叫 GHAP 的壓縮演算法。假設壓縮率設定為 10% (0.1)
# 具體 API 參數需參閱 GHAP 的 gmr.py 實作細節
# comp_means, comp_scales, comp_quats, comp_opacities, comp_shs_dc = merge_gaussians_gmr(
#     means, scales, quats, opacities, shs_dc, target_ratio=0.1
# )

# !!! 開發測試暫代碼 (確認管線可通後刪除) !!!
comp_means = means
comp_scales = scales
comp_quats = quats
comp_opacities = opacities
comp_shs_dc = shs_dc
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

N_comp = comp_means.shape[0]
print(f"         壓縮後高斯點數量: {N_comp} 顆")

# =====================================================================
# [階段 4] SH 光影維度擴充 (防 OOM 核心機制)
# =====================================================================
print("[階段 4] 擴充 SH 維度至 sh_degree=2 ...")
# sh_degree=2 總共需要 9 個係數 (1 個 DC + 8 個 Rest)。每個係數包含 RGB 3 通道。
# 我們直接建立全零的 Tensor 來裝載這些高頻光影容器。
comp_shs_rest = torch.zeros((N_comp, 8, 3), dtype=torch.float32, device='cpu')

# =====================================================================
# [階段 5] 重新打包並輸出 CityGS 相容的 Checkpoint
# =====================================================================
print("[階段 5] 重組 Checkpoint 狀態字典...")
state_dict[f'{prefix}means'] = comp_means
state_dict[f'{prefix}scales'] = comp_scales
state_dict[f'{prefix}quats'] = comp_quats
state_dict[f'{prefix}opacities'] = comp_opacities
state_dict[f'{prefix}shs_dc'] = comp_shs_dc
state_dict[f'{prefix}shs_rest'] = comp_shs_rest

# 必須清空優化器狀態，因為點的數量改變了，Fine 階段必須重新初始化動量
ckpt['optimizer_states'] = []

print(f"[階段 6] 儲存輕量化 8 維全域模型: {output_ckpt}")
torch.save(ckpt, output_ckpt)
print("轉換完成！")