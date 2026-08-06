import torch

# 設定來源與輸出路徑
input_ckpt = 'outputs/aerial_train_block_all_3x/checkpoints/epoch=6-step=30000.ckpt'
output_ckpt = 'outputs/aerial_train_block_all_3x/checkpoints/epoch=6-step=30000_sh2_padded.ckpt'

print(f"正在載入權重檔: {input_ckpt}...")
ckpt = torch.load(input_ckpt, map_location='cpu')

# 1. 抽換並擴容物理張量 (0 維 -> 8 維)
old_shs_rest = ckpt['state_dict']['gaussian_model.gaussians.shs_rest']
N = old_shs_rest.shape[0]
print(f"[狀態] 原始 shs_rest 維度: {old_shs_rest.shape}")

new_shs_rest = torch.zeros((N, 8, 3), dtype=old_shs_rest.dtype)
ckpt['state_dict']['gaussian_model.gaussians.shs_rest'] = new_shs_rest
print(f"[修正] 擴容後 shs_rest 維度: {new_shs_rest.shape}")

# 2. 重置幽靈排程器的內部狀態
if 'gaussian_model._active_sh_degree' in ckpt['state_dict']:
    ckpt['state_dict']['gaussian_model._active_sh_degree'] = torch.tensor(0, dtype=torch.uint8)

# 3. 強制覆寫設定檔的認知
try:
    ckpt['hyper_parameters']['gaussian'].sh_degree = 2
    ckpt['hyper_parameters']['gaussian'].config.sh_degree = 2
    print("[修正] 已將 Checkpoint 內部超參數覆寫為 sh_degree=2")
except Exception as e:
    print(f"[警告] 超參數覆寫略過: {e}")

# 儲存新權重
torch.save(ckpt, output_ckpt)
print(f"🎉 成功！已輸出擴容權重檔至: {output_ckpt}")