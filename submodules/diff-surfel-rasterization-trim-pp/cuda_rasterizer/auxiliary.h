/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)
#define FilterSize 0.7071067811865476
#define FilterInvSquare 1/(FilterSize*FilterSize)

#define TIGHTBBOX 0

// Exact alpha-support culling (2026-07-28). See the long note in forward.cu. With TIGHTBBOX=0
// the shipped build bounds every primitive at a flat 3 sigma; EXACT_SUPPORT instead bounds it
// where its alpha provably drops below the 1/255 the blend loop already discards, and drops
// primitives with o <= 1/255 outright. EXACT_SUPPORT_GROW=0 caps the radius at 3 so renders
// stay byte-identical and only the binning cost changes; set it to 1 to also let high-opacity
// primitives claim the extra reach the flat cut currently throws away (a quality change, so
// measure it separately). Estimated on b12@30k: 8.7x fewer tile intersections, N_max x2.05.
#define EXACT_SUPPORT 1
#define EXACT_SUPPORT_GROW 0
#define EXACT_SUPPORT_MARGIN_PX 0.0f
// 0 = keep the old flat 3-sigma radius and only drop o <= 1/255 (isolates which half of
// EXACT_SUPPORT costs bit-identity); 1 = also use the exact support radius.
#define EXACT_SUPPORT_RADIUS 0

// 精確圓錐外接盒（2026-09-20）：上游的 `radius = truncated_R * extent(1)` 是**線性化**，
// 中心還固定在 r=1 的中心。改成直接解該層等高線的圓錐曲線並輸出**不對稱**盒。
// 線性化在 88.7% 的（顆粒,相機）組合上**高估**（白付 tile），在 11.3% 上**低估**（漏覆蓋，
// 最大 73,491 px）=> 精確解同時省掉白付的、補回漏掉的。離線量得 Σtile -41.2%。
// ⚠ 必須配不對稱盒才有這個數字：硬塞回 getRect(中心, 半徑) 要付 |中心位移| 左右各一次 => +7.1%。
//
// ⚠⚠ **2026-09-21 改成執行期旗標，這裡不再有 #define。** 原因：lab 是三槽平行，編譯期旗標
//    讓 A/B 兩臂**無法同時跑**（同一個環境只裝得下一份 .so），而若不慎同時跑，兩邊會**都**用
//    最後編的那一版且**不會報錯** —— 就是這專案一再踩到的「看起來正常但量錯」。
//    現在走 `GaussianRasterizationSettings.exact_conic_aabb`（Python 端預設 False）。

// AbsGS 式的絕對值位置梯度（2026-08-25，見 backward.cu 的長註解）。
// 1 = 在 ray-splat 分支把 |dL_ds.x| + |dL_ds.y| 累加進 `dL_dmean2D.z`（該分量全檔未被讀，
//     所以**渲染與梯度逐位元不變**），Python 端用 `viewspace_points.grad[:, 2]` 取用。
// ⚠ 代價：逐像素內層迴圈多一次 atomicAdd => backward 可能慢數 %。
//   它是**診斷/新訊號**，不是既有配方的一部分 => 想量純訓練速度時設 0 重編。
#define ABSGRAD 1
#define RENDER_AXUTILITY 1
#define DEPTH_OFFSET 0
#define ALPHA_OFFSET 1
#define NORMAL_OFFSET 2 
#define MIDDEPTH_OFFSET 5
#define DISTORTION_OFFSET 6
#define MEDIAN_WEIGHT_OFFSET 7
// ★ 2026-11-11 逐貢獻的單側深度上限（使用者 11-11「可以改」）：out_others 第 9 個通道
//   B(p) = Σ_i T_i·α_i·max(z_i·icap(p) − 1, 0)，icap(p)＝1／((1+margin)·z_先驗(p))，0＝該像素不設上限。
//   只在完整路徑（forward MODE 0／backward GEOM）且呼叫端設了上限圖時才算；沒設時這個通道是 0、其他通道逐位元不變。
#define CAP_OFFSET 8
// ★ 2026-11-11 規格 2：第 10 個通道＝逐像素落後權重 W(p) = Σ_i T_i·α_i·[z_i·icap(p) > 1]（落在上限後面的混合權重總和；不乘超出量、不可微）
//   ＝這個像素的顏色有多少靠「表面後面的顆粒」在撐（需求地圖）。同 CAP_OFFSET 只在完整路徑且設了上限圖時才算。
#define CAPW_OFFSET 9

// distortion helper macros
#define BACKFACE_CULL 1
#define DUAL_VISIABLE 1
#define NEAR_PLANE 0.2
#define FAR_PLANE 100.0
// ★ 2026-11-21 晚 純數學化簡（紀錄/實驗分析/39 #12）：上面兩個與 FilterSize／FilterInvSquare 都是 **double** 字面值 =>
//   render／backward 的逐配對迴圈裡 rho2d、近平面比較、dist 的 NDC 深度、低通分支梯度都被提升成 FP64（消費級卡 FP64 只有 FP32 的 1/64）。
//   RAST_F32_CONST=1 改用 float 常數（數學相同）：近平面比較與 0.2f 比完全等價（0.2f 是大於 0.2 的最小 float）；rho2d 乘 2.0f 與乘
//   2.0000000000000004 再捨入成 float 相同；NDC 深度改寫成 a - b/d（對逆深度仿射）、dm/dd = b/d^2；低通分支梯度差 1～3 ulp。
//   ★ 11-22 使用者採用（全場景確認：畫質打平、純訓練 −5.2%；紀錄/實驗分析/39 §7.5''）=> 預設 1。
//   RAST_F32_CONST=0（setup.py 讀環境變數）＝舊的 double 常數，與 11-21 以前逐位元相同。
#ifndef RAST_F32_CONST
#define RAST_F32_CONST 1
#endif
#define NDC_A_F ((float)(FAR_PLANE / (FAR_PLANE - NEAR_PLANE)))
#define NDC_B_F ((float)(FAR_PLANE * NEAR_PLANE / (FAR_PLANE - NEAR_PLANE)))
#if RAST_F32_CONST
#define FILTER_SIZE_C 0.70710678f
#define FILTER_INV_SQ_C 2.0f
#define NEAR_C 0.2f
#define PZ_ZERO 0.0f
#define NDC_MAP(d) (NDC_A_F - NDC_B_F / (d))
#define NDC_DMD(d) (NDC_B_F / ((d) * (d)))
#else
#define FILTER_SIZE_C FilterSize
#define FILTER_INV_SQ_C FilterInvSquare
#define NEAR_C NEAR_PLANE
#define PZ_ZERO 0.0
#define NDC_MAP(d) ((FAR_PLANE * (d) - FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * (d)))
#define NDC_DMD(d) ((FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * (d) * (d)))
#endif
#define DETACH_WEIGHT 0

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ void getRectMinMax(const float x0, const float x1, const float y0, const float y1,
	uint2& rect_min, uint2& rect_max, dim3 grid)
{
	// getRect 的不對稱版：直接吃盒子的 min/max，不經過「中心 +- 半徑」。
	// x0 == p.x - r 且 x1 == p.x + r 時與 getRect 逐位元相同。
	rect_min = {
		min(grid.x, max((int)0, (int)(x0 / BLOCK_X))),
		min(grid.y, max((int)0, (int)(y0 / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((x1 + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((y1 + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float3 crossProduct(float3 a, float3 b) {
	float3 result;
	result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

// adopt from gsplat: https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/csrc/forward.cu
inline __device__ glm::mat3 quat_to_rotmat(const glm::vec4 quat) {
	// quat to rotation matrix
	float s = rsqrtf(
		quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
	);
	float w = quat.x * s;
	float x = quat.y * s;
	float y = quat.z * s;
	float z = quat.w * s;

	// glm matrices are column-major
	return glm::mat3(
		1.f - 2.f * (y * y + z * z),
		2.f * (x * y + w * z),
		2.f * (x * z - w * y),
		2.f * (x * y - w * z),
		1.f - 2.f * (x * x + z * z),
		2.f * (y * z + w * x),
		2.f * (x * z + w * y),
		2.f * (y * z - w * x),
		1.f - 2.f * (x * x + y * y)
	);
}


inline __device__ glm::vec4
quat_to_rotmat_vjp(const glm::vec4 quat, const glm::mat3 v_R) {
	float s = rsqrtf(
		quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
	);
	float w = quat.x * s;
	float x = quat.y * s;
	float y = quat.z * s;
	float z = quat.w * s;

	glm::vec4 v_quat;
	// v_R is COLUMN MAJOR
	// w element stored in x field
	v_quat.x =
		2.f * (
				  // v_quat.w = 2.f * (
				  x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
				  z * (v_R[0][1] - v_R[1][0])
			  );
	// x element in y field
	v_quat.y =
		2.f *
		(
			// v_quat.x = 2.f * (
			-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
			z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])
		);
	// y element in z field
	v_quat.z =
		2.f *
		(
			// v_quat.y = 2.f * (
			x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
			z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])
		);
	// z element in w field
	v_quat.w =
		2.f *
		(
			// v_quat.z = 2.f * (
			x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
			2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])
		);
	return v_quat;
}


inline __device__ glm::mat3
scale_to_mat(const float3 scale, const float glob_scale) {
	glm::mat3 S = glm::mat3(1.f);
	S[0][0] = glob_scale * scale.x;
	S[1][1] = glob_scale * scale.y;
	S[2][2] = glob_scale * scale.z;
	return S;
}



#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

// ★ 2026-11-21 純數學化簡 #2（紀錄/實驗分析/39；離線量測 tools/probe/conic_strip_cull_1121.py：skip_box 不跳的 warp 迭代還能再刪 41%）：
//   warp 層級「精確支撐 vs 16x2 像素條」測試，取代外接矩形的鬆弛。支撐＝{rho3d <= tau} ∪ {rho2d <= tau}，tau = 2 ln(255 o)。
//   rho3d <= tau <=> F = p'x^2 + p'y^2 - tau p'z^2 <= 0，p' = (dx q0.x + dy q1.x, dx q0.y + dy q1.y, iz + dx q0.z + dy q1.z)（反單應、錨定）。
//   每列像素中心 y 固定 => F(dx) = A dx^2 + B dx + C；[dx0, dx1] 上有 F <= 0 <=> 端點 <= 0，或（A > 0、頂點在區間內、B^2 >= 4AC）——不需要除法。
//   低通圓盤：rho2d = |xy0 - pix|^2 / FilterSize^2 <= tau <=> 距離^2 <= 0.5 tau。tau 由呼叫端放寬（保守）；F 對 p' 齊二次 => p'z < 0 的另一葉只會多放行（安全）。
__device__ __forceinline__ bool aihStripHit(const float3 q0, const float3 q1, const float3 an, const float tau, const float2 xy0,
	const float sx0, const float sx1, const float sy0, const float sy1)
{
	const float A = q0.x * q0.x + q0.y * q0.y - tau * q0.z * q0.z;
	const float K = q0.x * q1.x + q0.y * q1.y;
	const float L = q1.x * q1.x + q1.y * q1.y;
	const float dx0 = sx0 - an.x, dx1 = sx1 - an.x;
	const float rd2 = 0.5f * tau;
	const float cxn = fminf(fmaxf(xy0.x, sx0), sx1) - xy0.x;
	#pragma unroll
	for (int r = 0; r < 2; r++) {
		const float yrow = r ? sy1 : sy0;
		const float dyd = yrow - xy0.y;
		if (cxn * cxn + dyd * dyd <= rd2) return true;
		const float dy = yrow - an.y;
		const float w0 = dy * q1.z + an.z;
		const float B = 2.f * (dy * K - tau * q0.z * w0);
		const float C = dy * dy * L - tau * w0 * w0;
		if ((A * dx0 + B) * dx0 + C <= 0.f || (A * dx1 + B) * dx1 + C <= 0.f) return true;
		if (A > 0.f && -B >= 2.f * A * dx0 && -B <= 2.f * A * dx1 && B * B >= 4.f * A * C) return true;
	}
	return false;
}

// ★ 11-21 幾何配對遮罩：負責載入這顆面元的那條執行緒，一次算完本 tile 8 個 warp（16x2 像素條）的精確測試，存 8 bit；
//   之後每個 warp 只查自己那一位（取代每個 warp 32 條 lane 重複算同一個測試；也讓沒有配對位元的路徑——完整路徑 forward／GEOM backward——能用）。
//   Ag＝反單應 9 個數（q0, q1, x_mu, y_mu, 1/Z_c）；tau 由呼叫端放寬；pix_min＝tile 左上像素。
__device__ __forceinline__ unsigned aihTileMask(const float* Ag, const float tau, const float2 xy0, const uint2 pix_min)
{
	const float3 q0 = {Ag[0], Ag[1], Ag[2]}, q1 = {Ag[3], Ag[4], Ag[5]}, an = {Ag[6], Ag[7], Ag[8]};
	const float sx0 = (float)pix_min.x + 0.5f, sx1 = sx0 + 15.f;
	unsigned m = 0u;
	#pragma unroll
	for (int w = 0; w < 8; w++) {
		const float sy0 = (float)pix_min.y + 2.f * w + 0.5f;
		if (aihStripHit(q0, q1, an, tau, xy0, sx0, sx1, sy0, sy0 + 1.f)) m |= (1u << w);
	}
	return m;
}
// 放寬的 tau（1%＋0.02，保守）；o <= 1/255 不會進 binning，保險起見回傳極大值（永不跳）
__device__ __forceinline__ float aihTauLoose(const float op)
{
	return (op > 1.0f / 255.0f) ? (2.f * logf(255.f * op) * 1.01f + 0.02f) : 1e30f;
}

// ★ 2026-11-29 固定厚度（使用者 11-03 構想、11-29「低角度露縫直接做」；訓練版）：
//   面元沿法向多一個固定比例的厚度 s_w（= 比例 x 面內較短的尺度，Python 端算），沿射線取 3D 高斯（面內 s_u、s_v、法向 s_w）的
//   最小 Mahalanobis 距離。C＝(Cu, Cv, Cw, s_w)：相機在面元正規化座標（u/s_u、v/s_v、w/s_w）的位置，Python 端算、可微。
//   P0＝(s, 0) 是射線與面元平面的交點，射線方向 d＝P0 − C；min_t |C + t d|^2 = |C x P0|^2 / |P0 − C|^2（Lagrange 恆等式，
//   分子分母都是平方和 => 沒有大數相減）：
//     rho' = [Cw^2 |s|^2 + (Cu s.y − Cv s.x)^2] / [(s.x − Cu)^2 + (s.y − Cv)^2 + Cw^2]
//   s_w -> 0（Cw -> inf）退回 |s|^2；正對面元（Cu = Cv = 0）也是 |s|^2；斜看才變小（看起來沿視線方向變寬）。
//   C.w <= 0（這顆沒有厚度；換塊評測時基準的顆粒）=> 原本的 |s|^2，逐位元相同。s 溢位（側對、p.z≈0）=> NaN => 當作沒有交點（同原本 inf）。
__device__ __forceinline__ float thickRho(const float2 s, const float rho3d, const float4 C)
{
	if (!(C.w > 0.f)) return rho3d;
	const float X = C.x * s.y - C.y * s.x;
	const float du = s.x - C.x, dv = s.y - C.y, w2 = C.z * C.z;
	const float r = (w2 * rho3d + X * X) / (du * du + dv * dv + w2);
	return (r == r) ? r : INFINITY;
}
// d rho' / d s 與 d rho' / d (Cu, Cv, Cw)（rho＝thickRho 的值；只在 C.w > 0 時呼叫）
__device__ __forceinline__ void thickRhoGrad(const float2 s, const float rho, const float4 C, float2& ds, float3& dC)
{
	const float X = C.x * s.y - C.y * s.x;
	const float du = s.x - C.x, dv = s.y - C.y, w2 = C.z * C.z;
	const float iD = 1.f / (du * du + dv * dv + w2);
	ds.x = 2.f * (w2 * s.x - X * C.y - rho * du) * iD;
	ds.y = 2.f * (w2 * s.y + X * C.x - rho * dv) * iD;
	dC.x = 2.f * (X * s.y + rho * du) * iD;
	dC.y = 2.f * (-X * s.x + rho * dv) * iD;
	dC.z = 2.f * C.z * ((s.x * s.x + s.y * s.y) - rho) * iD;
}

#endif