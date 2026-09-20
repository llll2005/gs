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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for
	// "Differentiable Point-Based Radiance Fields for
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ bool computeTransMat(const glm::vec3 &p_world, const glm::vec4 &quat, const glm::vec2 &scale, const float *viewmat, const float4 &intrins, float tan_fovx, float tan_fovy, float* transMat, float3 &normal) {
	// Setup cameras
	// Currently only support ideal pinhole camera
	// but more advanced intrins can be implemented
	const glm::mat3 W = glm::mat3(
		viewmat[0],viewmat[1],viewmat[2],
		viewmat[4],viewmat[5],viewmat[6],
		viewmat[8],viewmat[9],viewmat[10]
	);
	const glm::vec3 cam_pos = glm::vec3(viewmat[12], viewmat[13], viewmat[14]); // camera center
	const glm::mat4 P = glm::mat4(
		intrins.x, 0.0, 0.0, 0.0,
		0.0, intrins.y, 0.0, 0.0,
		intrins.z, intrins.w, 1.0, 1.0,
		0.0, 0.0, 0.0, 0.0
	);

	// Make the geometry of 2D Gaussian as a Homogeneous transformation matrix
	// under the camera view, See Eq. (5) in 2DGS' paper.
	glm::vec3 p_view = W * p_world + cam_pos;
	glm::mat3 R = quat_to_rotmat(quat) * scale_to_mat({scale.x, scale.y, 1.0f}, 1.0f);
	glm::mat3 M = glm::mat3(W * R[0], W * R[1], p_view);
	glm::vec3 tn = W*R[2];
	float cos = glm::dot(-tn, p_view);

#if BACKFACE_CULL
	if (cos == 0.0f) return false;
#endif

#if RENDER_AXUTILITY and DUAL_VISIABLE
	// This means a 2D Gaussian is dual visiable.
	// Experimentally, turning off the dual visiable works eqully.
	float multiplier = cos > 0 ? 1 : -1;
	tn *= multiplier;
#endif
	// projection into screen space, see Eq. (7)
	glm::mat4x3 T = glm::transpose(P * glm::mat3x4(
		glm::vec4(M[0], 0.0),
		glm::vec4(M[1], 0.0),
		glm::vec4(M[2], 1.0)
	));

	transMat[0] = T[0].x;
	transMat[1] = T[0].y;
	transMat[2] = T[0].z;
	transMat[3] = T[1].x;
	transMat[4] = T[1].y;
	transMat[5] = T[1].z;
	transMat[6] = T[2].x;
	transMat[7] = T[2].y;
	transMat[8] = T[2].z;
	normal = {tn.x, tn.y, tn.z};
	return true;
}

// Computing the bounding box of the 2D Gaussian and its center,
// where the center of the bounding box is used to create a low pass filter
// in the image plane
// 精確圓錐外接盒（2026-09-20）。computeAABB 算的是 **r=1** 那條等高線的精確外接盒，呼叫端再乘
// truncated_R —— 那是**線性化**：ray-splat 交點 p = k x l 對像素是仿射的（因為 Tw x Tw = 0），
// 所以 u^2 + v^2 <= R^2 在螢幕上是真正的圓錐曲線，而不同 R 的等高線在透視下**不是等比放大**。
// 精確解同成本：把對偶簽名 (1,1,-1) 換成 (R^2, R^2, -1)。正交極限下自動退化回 R * extent(1)。
//
// ⚠⚠ 這裡回傳的 center 是**該層的**中心，會隨 R 移動，**絕對不可以**寫進 points_xy_image：
//    那個值同時是 renderCUDA 低通核 rho2d 的中心、以及 backward.cu computeAABB 梯度回傳的依據
//    （兩者都按 r=1 的公式）。呼叫端要拿原本的 r=1 中心配上這裡的 [center +- extent] 組不對稱盒。
// ⚠ d = C*_33 ~= -(深度^2)，**負號是常態**；中心與半寬都是 C* 的比值，對整體符號不變
//   => 只有 d == 0 才是真退化（與原版同判準）。
__device__ bool computeAABB_R(const float *transMat, const float R, float2 & center, float2 & extent)
{
	glm::mat4x3 T = glm::mat4x3(
		transMat[0], transMat[1], transMat[2],
		transMat[3], transMat[4], transMat[5],
		transMat[6], transMat[7], transMat[8],
		transMat[6], transMat[7], transMat[8]
	);

	const float R2 = R * R;
	const glm::vec3 sgn = glm::vec3(R2, R2, -1.0f);

	float d = glm::dot(sgn, T[3] * T[3]);
	if (d == 0.0f) return false;

	glm::vec3 f = sgn * (1.0f / d);

	glm::vec3 p = glm::vec3(
		glm::dot(f, T[0] * T[3]),
		glm::dot(f, T[1] * T[3]),
		glm::dot(f, T[2] * T[3]));

	glm::vec3 h0 = p * p -
		glm::vec3(
			glm::dot(f, T[0] * T[0]),
			glm::dot(f, T[1] * T[1]),
			glm::dot(f, T[2] * T[2])
		);

	glm::vec3 h = sqrt(max(glm::vec3(0.0), h0));
	center = {p.x, p.y};
	extent = {h.x, h.y};
	return true;
}

__device__ bool computeAABB(const float *transMat, float2 & center, float2 & extent, int W, int H) {
	glm::mat4x3 T = glm::mat4x3(
		transMat[0], transMat[1], transMat[2],
		transMat[3], transMat[4], transMat[5],
		transMat[6], transMat[7], transMat[8],
		transMat[6], transMat[7], transMat[8]
	);

	float d = glm::dot(glm::vec3(1.0, 1.0, -1.0), T[3] * T[3]);

	if (d == 0.0f) return false;

	glm::vec3 f = glm::vec3(1.0, 1.0, -1.0) * (1.0f / d);

	glm::vec3 p = glm::vec3(
		glm::dot(f, T[0] * T[3]),
		glm::dot(f, T[1] * T[3]),
		glm::dot(f, T[2] * T[3]));
	
	if (p.x < -W/4 || p.x > W*5/4 || p.y < -H/4 || p.y > H*5/4)
		return false;

	glm::vec3 h0 = p * p -
		glm::vec3(
			glm::dot(f, T[0] * T[0]),
			glm::dot(f, T[1] * T[1]),
			glm::dot(f, T[2] * T[2])
		);

	glm::vec3 h = sqrt(max(glm::vec3(0.0), h0)) + glm::vec3(0.0, 0.0, 1e-2);
	center = {p.x, p.y};
	extent = {h.x, h.y};
	return true;
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, const float tan_fovy, const float pp_shifty,
	const float focal_x, const float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	ushort4* rects,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	glm::vec3 p_world = glm::vec3(orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]);
	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	float4 intrins = {focal_x, focal_y, float(W)/2.0, float(H)/2.0 + pp_shifty};
	glm::vec2 scale = scales[idx];
	glm::vec4 quat = rotations[idx];

	const float* transMat;
	bool ok;
	float3 normal;
	if (transMat_precomp != nullptr)
	{
		transMat = transMat_precomp + idx * 9;
	}
	else
	{
		ok = computeTransMat(p_world, quat, scale, viewmatrix, intrins, tan_fovx, tan_fovy, transMats + idx * 9, normal);
		if (!ok) return;
		transMat = transMats + idx * 9;
	}

	//  compute center and extent
	float2 center;
	float2 extent;
	ok = computeAABB(transMat, center, extent, W, H);
	if (!ok) return;

	// add the bounding of countour
#if EXACT_SUPPORT
	// Exact alpha-support bound (2026-07-28). The blend loop already discards any sample with
	// alpha < 1/255 in BOTH passes (forward.cu `if (alpha < 1.0f/255.0f) continue;` and the
	// mirror in backward.cu), so a tile in which this Gaussian's PEAK alpha stays under that
	// threshold contributes exactly nothing to the image and nothing to any gradient -- yet we
	// still allocate a binning entry for it. Solving for where the falloff crosses the
	// threshold gives the radius beyond which the primitive is provably invisible:
	//
	//     alpha = o * exp(-rho/2) >= 1/255   <=>   rho <= 2*ln(255*o)
	//
	// so truncated_R = sqrt(2*ln(255*o)), and any Gaussian with o <= 1/255 never reaches the
	// threshold anywhere and can be dropped from binning outright. The `max(extent, FilterSize)`
	// below already covers the low-pass disc, since the shader takes rho = min(rho3d, rho2d):
	// the support is the UNION of the ellipse and a screen disc of radius FilterSize*sqrt(rho),
	// and both scale with the same truncated_R.
	//
	// NOTE the min() against 3.f. TIGHTBBOX is `#define`d to 0 in auxiliary.h, so the build we
	// actually ship uses a flat 3-sigma box for every primitive regardless of opacity; that, not
	// the TIGHTBBOX formula, is the baseline to stay under. The exact radius EXCEEDS 3 for
	// o > exp(-2.0785) ~= 0.125 (at o=1 it is 3.33, i.e. the flat 3-sigma cut currently discards
	// contribution still worth 2.8/255), so lifting the cap would change output for the better
	// rather than leave it identical. Capping at 3 keeps this a pure, provable memory/speed win
	// with byte-identical renders, which is the thing to measure first; set EXACT_SUPPORT_GROW
	// to lift it and measure the quality side as a separate experiment.
	//
	// Cf. StopThePop (SIGGRAPH 2024) Sec 3.4, which performs the same rejection per tile via an
	// argmax within each tile; this closed form needs no per-tile work. Measured on b12@30k:
	// 58.5% of points fall below 1/255 outright and tile intersections drop ~4.3x, which is the
	// dominant term of the VRAM budget. See 紀錄/論文核對表.md §11.
	if (opacities[idx] <= 1.0f / 255.0f)
		return;
#if EXACT_SUPPORT_RADIUS
	float truncated_R = sqrtf(max(2.f * logf(255.f * opacities[idx]), 0.000001));
#if !EXACT_SUPPORT_GROW
	truncated_R = min(truncated_R, 3.f);
#endif
#else
	float truncated_R = 3.f;      // isolation build: drop-only, radius untouched
#endif
#elif TIGHTBBOX // no use in the paper, but it indeed help speeds.
	// the effective extent is now depended on the opacity of gaussian.
	float truncated_R = sqrtf(max(9.f + logf(opacities[idx]), 0.000001));
#else
	float truncated_R = 3.f;
#endif
	float radius = ceil(truncated_R * max(max(extent.x, extent.y), FilterSize));
#if EXACT_SUPPORT
	// The exact-support radius is a TIGHT bound, so float error in logf plus the ceil() can
	// drop samples sitting exactly on the alpha = 1/255 boundary. Measured without this margin:
	// renders differed by at most 1.5e-3 (0.38 of an 8-bit level) but across up to 25% of
	// pixels -- many marginal samples, not a few large errors. One extra pixel of slack costs
	// at most one ring of tiles and buys back exactness.
	radius += EXACT_SUPPORT_MARGIN_PX;
#if !EXACT_SUPPORT_GROW && !EXACT_CONIC_AABB
	// GROW=0 的語意是「只縮不長」：margin 與浮點誤差都不得讓半徑超過原本的平坦 3-sigma 盒，
	// 否則會在另一個方向上改變輸出（高不透明度顆粒的 truncated_R 已被 cap 在 3）。
	radius = min(radius, ceil(3.f * max(max(extent.x, extent.y), FilterSize)));
#endif
#endif

	uint2 rect_min, rect_max;
#if EXACT_CONIC_AABB
	// ★ 精確圓錐盒（不對稱）。`center`（r=1 的中心）完全不動 => renderCUDA 的低通核 rho2d
	//   與 backward.cu 的梯度回傳都不受影響；改變的只有「哪些 tile 被綁進來」。
	float2 cc, ce;
	if (!computeAABB_R(transMat, truncated_R, cc, ce))
		return;
	const float fs = FilterSize * truncated_R;      // 低通圓盤：||pix - center|| <= FilterSize * R
	float bx0 = min(cc.x - ce.x, center.x - fs) - EXACT_SUPPORT_MARGIN_PX;
	float bx1 = max(cc.x + ce.x, center.x + fs) + EXACT_SUPPORT_MARGIN_PX;
	float by0 = min(cc.y - ce.y, center.y - fs) - EXACT_SUPPORT_MARGIN_PX;
	float by1 = max(cc.y + ce.y, center.y + fs) + EXACT_SUPPORT_MARGIN_PX;
#if EXACT_SUPPORT && !EXACT_SUPPORT_GROW && EXACT_SUPPORT_RADIUS
	{	// 「只縮不長」：夾回**同一套公式**下的平坦 3-sigma 盒（不是線性化的那個）
		float2 c3, e3;
		if (computeAABB_R(transMat, 3.f, c3, e3))
		{
			const float f3 = FilterSize * 3.f;
			bx0 = max(bx0, min(c3.x - e3.x, center.x - f3));
			bx1 = min(bx1, max(c3.x + e3.x, center.x + f3));
			by0 = max(by0, min(c3.y - e3.y, center.y - f3));
			by1 = min(by1, max(c3.y + e3.y, center.y + f3));
		}
	}
#endif
	getRectMinMax(bx0, bx1, by0, by1, rect_min, rect_max, grid);
	// ⚠ `radius`（=> radii）**刻意不改**：Python 端把它餵進 `_max_radii2D`，而那被
	//   screen_size_prune / vpc 的 c = radii^2 / cost_aware_densify 當成「顆粒的螢幕尺寸與成本」。
	//   不對稱盒的覆蓋半徑含「中心位移」，拿去當尺寸會嚴重失真（位移最大 73,491 px）。
	//   => radii 維持線性化的 3-sigma 對稱半徑（下游行為零變化），rects 只管 binning。
	//   radius > 0 恆成立（>= ceil(3 * FilterSize) = 3），所以 radii > 0 <=> 走到這裡 <=> 盒子非空。
#else
	getRect(center, radius, rect_min, rect_max, grid);
#endif
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// compute colors
	if (colors_precomp == nullptr) {
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	depths[idx] = p_view.z;
	radii[idx] = (int)radius;
	points_xy_image[idx] = center;
	// store them in float4
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	// 把 rect 存起來給 duplicateWithKeys 用（原本是拿 center/radii 重算一次）。
	// 線性化路徑存的值與 getRect 算的完全一樣 => 逐位元相同。
	rects[idx] = make_ushort4((unsigned short)rect_min.x, (unsigned short)rect_min.y,
	                          (unsigned short)rect_max.x, (unsigned short)rect_max.y);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ transMats,
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others,
	float* __restrict__ transmittance,
	int* __restrict__ num_covered_pixels,
	bool record_transmittance)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x + 0.5, (float)pix.y + 0.5};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };


#if RENDER_AXUTILITY
	// render axutility ouput
	float D = { 0 };
	float N[3] = {0};
	float dist1 = {0};
	float dist2 = {0};
	float distortion = {0};
	float median_depth = {0};
	float median_weight = {0};
	float median_contributor = {-1};

#endif

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Fisrt compute two homogeneous planes, See Eq. (8)
			float3 Tu = collected_Tu[j];
			float3 Tv = collected_Tv[j];
			float3 Tw = collected_Tw[j];
			float3 k = {-Tu.x + pixf.x * Tw.x, -Tu.y + pixf.x * Tw.y, -Tu.z + pixf.x * Tw.z};
			float3 l = {-Tv.x + pixf.y * Tw.x, -Tv.y + pixf.y * Tw.y, -Tv.z + pixf.y * Tw.z};
			// cross product of two planes is a line (i.e., homogeneous point), See Eq. (10)
			float3 p = crossProduct(k, l);
#if BACKFACE_CULL
			// May hanle this by replacing a low pass filter,
			// but this case is extremely rare.
			if (p.z == 0.0) continue; // there is not intersection
#endif
			// 3d homogeneous point to 2d point on the splat
			float2 s = {p.x / p.z, p.y / p.z};
			// 3d distance. Compute Mahalanobis distance in the canonical splat' space
			float rho3d = (s.x * s.x + s.y * s.y);

			// Add low pass filter according to Botsch et al. [2005],
			// see Eq. (11) from 2DGS paper.
			float2 xy = collected_xy[j];
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			// 2d screen distance
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);
			float rho = min(rho3d, rho2d);

			float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z; // splat depth
			if ((s.x * Tw.x + s.y * Tw.y) + Tw.z < NEAR_PLANE) continue;
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float power = -0.5f * rho;
			// power = -0.5f * 100.f * max(rho - 1, 0.0f);
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			float alpha = min(0.99f, nor_o.w * exp(power));
			if (record_transmittance){
				atomicAdd(&transmittance[collected_id[j]], T * alpha);
				atomicAdd(&num_covered_pixels[collected_id[j]], 1);
			}
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}


#if RENDER_AXUTILITY
			// Render depth distortion map
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			float A = 1-T;
			float mapped_depth = (FAR_PLANE * depth - FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * depth);
			float error = mapped_depth * mapped_depth * A + dist2 - 2 * mapped_depth * dist1;
			distortion += error * alpha * T;

			if (T > 0.5) {
				median_depth = depth;
				median_weight = alpha * T;
				median_contributor = contributor;
			}
			// Render normal map
			for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * alpha * T;

			// Render depth map
			D += depth * alpha * T;
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			dist1 += mapped_depth * alpha * T;
			dist2 += mapped_depth * mapped_depth * alpha * T;
#endif

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

#if RENDER_AXUTILITY
		n_contrib[pix_id + H * W] = median_contributor;
		final_T[pix_id + H * W] = dist1;
		final_T[pix_id + 2 * H * W] = dist2;
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
		out_others[pix_id + MEDIAN_WEIGHT_OFFSET * H * W] = median_weight;
#endif
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* means2D,
	const float* colors,
	const float* transMats,
	const float* depths,
	const float4* normal_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_others,
	float* transmittance,
	int* num_covered_pixels,
	bool record_transmittance)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		means2D,
		colors,
		transMats,
		depths,
		normal_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_others,
		transmittance,
		num_covered_pixels,
		record_transmittance);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, const int H,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy, const float pp_shifty,
	int* radii,
	float2* means2D,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	ushort4* rects,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy, pp_shifty,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		transMats,
		rgb,
		normal_opacity,
		grid,
		tiles_touched,
		rects,
		prefiltered
		);
}
