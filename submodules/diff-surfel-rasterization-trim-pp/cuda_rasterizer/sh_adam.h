#ifndef CUDA_RASTERIZER_SH_ADAM_H_INCLUDED
#define CUDA_RASTERIZER_SH_ADAM_H_INCLUDED

// ★ 2026-10-24 SH 分開存放＋SH 的 Adam 融進 backward（紀錄/加速論文對照.md A5、B2；使用者 10-24「可以直接實作」）
//   分開存放：光柵器直接收 shs_dc (N,1,3) 與 shs_rest (N,M-1,3)，不再由 Python 每步 torch.cat 成 (N,M,3)
//            （本機 4050、1.29M 顆量到 cat 的 forward＋backward 9.1 ms／步，約一步的 10%）。
//   Adam 融進 backward：逐顆前處理 backward 算完該顆的 SH 梯度就直接更新 p／m／v，不寫出梯度張量
//            （省掉梯度緩衝歸零、寫出、Adam 再讀一次；SH 佔每顆 59 個 float 的 48 個）。
//   更新式與 rasterize_points.cu fusedAdamCUDA<VAR=1> 逐條相同（同一組 intrinsics）=> 拿到相同的梯度時逐位元相同。
//   看不到的顆粒（radii == 0）以梯度 0 照樣更新 => 與稠密 Adam 同語意。
struct ShAdamArgs
{
	float* m_dc;
	float* v_dc;
	float* m_rest;
	float* v_rest;
	float b1, omb1, b2, omb2, eps;
	float ss_dc, bc2s_dc;       // step_size = -lr/bc1（Python 以 double 算好）、sqrt(1 - b2^t)
	float ss_rest, bc2s_rest;
	// 高階 SH 每 k 步才更新（FusedAdam.sh_rest_interval）：acc_rest 非 nullptr => rest 的梯度累加在這裡，
	//   apply_rest != 0 的那一步才用累加和走 Adam 並歸零。acc_rest == nullptr => rest 每步都更新。
	float* acc_rest;
	int apply_rest;
};

#endif
