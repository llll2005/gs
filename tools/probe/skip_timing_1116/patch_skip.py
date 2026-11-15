"""11-16 Fable A-1／A-3 計時版的光柵器補丁：在 cuda_rasterizer/ 目錄裡執行（只給 scratch 副本用，不改 submodule）。
CITYGS_SKIPF（每次 forward／backward 讀）：1＝A-1 forward、2＝A-1 backward、4＝A-3（pair_used 存 8 個 warp 的遮罩；要 bwd_pair_bit）。"""
def patch(P, pairs):
    s = open(P, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (P, s.count(a), a[:80])
        s = s.replace(a, b)
    open(P, "w", encoding="utf-8").write(s)

patch("forward.h", [("	void setTSat(float t);", "	void setTSat(float t);\n	void setSkipFlags(int f);   // ★ 11-16 計時版：1＝A-1 forward（逐 warp 精確支撐盒跳過）、4＝A-3（pair_used 存 8 個 warp 的遮罩）")])
patch("backward.h", [("	void setAlphaMax(float a);", "	void setAlphaMax(float a);\n	void setSkipFlags(int f);   // ★ 11-16 計時版：2＝A-1 backward、4＝A-3 backward（逐 warp 只走自己位元為 1 的配對）")])

F = "forward.cu"
s = open(F, encoding="utf-8").read()
i0 = s.index("renderCUDA(\n\tconst uint2* __restrict__ ranges,")
i1 = s.index("// ★ 2026-10-02 audit（純量測，不參與訓練）")
head, body, tail = s[:i0], s[i0:i1], s[i1:]
def rb(a, b, cnt=1):
    global body
    assert body.count(a) == cnt, (body.count(a), a[:80]); body = body.replace(a, b)
assert head.count("__device__ float g_t_sat = 0.0001f;") == 1
head = head.replace("__device__ float g_t_sat = 0.0001f;", "__device__ float g_t_sat = 0.0001f;\n__device__ int g_skipf = 0;   // ★ 11-16 計時版（見 FORWARD::setSkipFlags）", 1)
rb('''	__shared__ int s_cn1[LEANLIKE ? BLOCK_SIZE : 1];
''', '''	__shared__ int s_cn1[LEANLIKE ? BLOCK_SIZE : 1];
	// ★ 11-16 計時版：A-1（逐 warp 精確支撐盒跳過）與 A-3（pair_used 改存 8 個 warp 的遮罩）
	const bool a1 = LEANLIKE && (g_skipf & 1);
	const bool a3 = rec_pair && (g_skipf & 4);
	__shared__ float4 s_box[LEANLIKE ? BLOCK_SIZE : 1];
	__shared__ unsigned long long s_wm[(MODE == 1) ? BLOCK_SIZE : 1];
	const int wid = block.thread_rank() >> 5;
	const float sx0 = (float)pix_min.x + 0.5f, sx1 = (float)pix_min.x + 15.5f;
	const float sy0 = (float)pix_min.y + 2.f * wid + 0.5f, sy1 = sy0 + 1.f;
''')
rb('''			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
		}
		if (rec_pair) s_used[block.thread_rank()] = 0;''', '''			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
			if (a1) {   // 精確支撐盒：alpha >= 1/255 <=> rho <= 2 ln(255 o)；圓錐盒（R）∪ 低通圓盤（FilterSize R）＋外擴 1 px（浮點保險）
				const float op = normal_opacity[coll_id].w;
				float4 bx = make_float4(-1e30f, 1e30f, -1e30f, 1e30f);
				if (op > 1.0f / 255.0f) {
					const float RE = sqrtf(fmaxf(2.f * logf(255.f * op), 1e-12f));
					float2 cc, ce;
					if (computeAABB_R(transMats + 9 * coll_id, RE, cc, ce)) {
						const float2 xy0 = points_xy_image[coll_id]; const float fs = FilterSize * RE;
						bx = make_float4(fminf(cc.x - ce.x, xy0.x - fs) - 1.f, fmaxf(cc.x + ce.x, xy0.x + fs) + 1.f,
						                 fminf(cc.y - ce.y, xy0.y - fs) - 1.f, fmaxf(cc.y + ce.y, xy0.y + fs) + 1.f);
					}
				}
				s_box[block.thread_rank()] = bx;
			}
		}
		if (rec_pair) s_used[block.thread_rank()] = 0;
		if (a3) s_wm[block.thread_rank()] = 0ull;''')
rb('''				if (!done) do {
					// Keep track of current position in range
					contributor++;
''', '''				if (!done) do {
					// Keep track of current position in range
					contributor++;
					if (a1) { const float4 b = s_box[j];
						if (b.y < sx0 || b.x > sx1 || b.w < sy0 || b.z > sy1) { acc_hit = true; acc_ta = 0.f; break; } }   // 覆蓋數照算（C 逐位元相同）、V 少掉 < T/255 的項
''')
rb('''		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;
''', '''		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;
			if (a1) { const float4 b = s_box[j];
				if (b.y < sx0 || b.x > sx1 || b.w < sy0 || b.z > sy1) { if (rec_acc) atomicAdd(&s_cn1[j], 1); continue; } }
''')
rb('''if (rec_pair) s_used[j] = 1;   // 多個像素同時寫 1 => 良性競爭''',
   '''if (rec_pair) { if (a3) ((volatile unsigned char*)&s_wm[j])[wid] = 1; else s_used[j] = 1; }   // 多個像素同時寫 1 => 良性競爭''', cnt=2)
rb('''			if (rec_pair && range.x + progress < range.y && s_used[block.thread_rank()])
				pair_used[range.x + progress] = 1;''', '''			if (rec_pair && range.x + progress < range.y) {
				if (a3) { const unsigned long long v = s_wm[block.thread_rank()]; unsigned m8 = 0;
					for (int b8 = 0; b8 < 8; b8++) if ((v >> (8 * b8)) & 0xffull) m8 |= (1u << b8);
					if (m8) pair_used[range.x + progress] = (uint8_t)m8; }
				else if (s_used[block.thread_rank()]) pair_used[range.x + progress] = 1;
			}''')
s = head + body + tail
a = '''void FORWARD::setTSat(float t)
{'''
assert s.count(a) == 1
s = s.replace(a, '''void FORWARD::setSkipFlags(int f)
{
	static int cur = -1;
	if (f != cur) { cudaMemcpyToSymbol(g_skipf, &f, sizeof(int)); cur = f; }
}

''' + a, 1)
open(F, "w", encoding="utf-8").write(s)

B = "backward.cu"
s = open(B, encoding="utf-8").read()
def rbk(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (s.count(a), a[:80]); s = s.replace(a, b)
rbk("__device__ float g_alpha_max_b = 0.99f;", '''__device__ float g_alpha_max_b = 0.99f;
__device__ int g_skipf_b = 0;   // ★ 11-16 計時版（見 BACKWARD::setSkipFlags）
// ★ 11-16 計時版：與 forward.cu computeAABB_R 逐字相同（兩個編譯單元各一份）
__device__ static bool computeAABB_R_b(const float *transMat, const float R, float2 & center, float2 & extent)
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
	glm::vec3 p = glm::vec3(glm::dot(f, T[0] * T[3]), glm::dot(f, T[1] * T[3]), glm::dot(f, T[2] * T[3]));
	glm::vec3 h0 = p * p - glm::vec3(glm::dot(f, T[0] * T[0]), glm::dot(f, T[1] * T[1]), glm::dot(f, T[2] * T[2]));
	glm::vec3 h = sqrt(max(glm::vec3(0.0), h0));
	center = {p.x, p.y};
	extent = {h.x, h.y};
	return true;
}''')
rbk('''	__shared__ int s_wcnt[BLOCK_SIZE / 32];
	const bool pb = (pair_used != nullptr);
''', '''	__shared__ int s_wcnt[BLOCK_SIZE / 32];
	const bool pb = (pair_used != nullptr);
	// ★ 11-16 計時版
	const bool a1 = (g_skipf_b & 2) != 0;
	const bool a3 = pb && (g_skipf_b & 4) != 0;
	__shared__ float4 collected_box[BLOCK_SIZE];
	__shared__ unsigned char collected_m[BLOCK_SIZE];
	const int wid = warp.meta_group_rank();
	const float sx0 = (float)pix_min.x + 0.5f, sx1 = (float)pix_min.x + 15.5f;
	const float sy0 = (float)pix_min.y + 2.f * wid + 0.5f, sy1 = sy0 + 1.f;
''')
rbk('''			for (int c = 0; c < C; c++)
				collected_colors[c * BLOCK_SIZE + slot] = colors[coll_id * C + c];
		}''', '''			for (int c = 0; c < C; c++)
				collected_colors[c * BLOCK_SIZE + slot] = colors[coll_id * C + c];
			if (a3) collected_m[slot] = pair_used[range.x + q];
			if (a1) {
				const float op = normal_opacity[coll_id].w;
				float4 bx = make_float4(-1e30f, 1e30f, -1e30f, 1e30f);
				if (op > 1.0f / 255.0f) {
					const float RE = sqrtf(fmaxf(2.f * logf(255.f * op), 1e-12f));
					float2 cc, ce;
					if (computeAABB_R_b(transMats + 9 * coll_id, RE, cc, ce)) {
						const float2 xy0 = points_xy_image[coll_id]; const float fs = FilterSize * RE;
						bx = make_float4(fminf(cc.x - ce.x, xy0.x - fs) - 1.f, fmaxf(cc.x + ce.x, xy0.x + fs) + 1.f,
						                 fminf(cc.y - ce.y, xy0.y - fs) - 1.f, fmaxf(cc.y + ce.y, xy0.y + fs) + 1.f);
					}
				}
				collected_box[slot] = bx;
			}
		}''')
rbk('''			const uint32_t cq = pb ? (uint32_t)collected_q[j] : contributor;
''', '''			const uint32_t cq = pb ? (uint32_t)collected_q[j] : contributor;
			// ★ 11-16 計時版：整個 warp 一致的跳過（位元／盒子對同一個 warp 相同 => 不會在 warp 集體操作前分歧）
			if (a3 && !((collected_m[j] >> wid) & 1)) continue;
			if (a1) { const float4 b = collected_box[j]; if (b.y < sx0 || b.x > sx1 || b.w < sy0 || b.z > sy1) continue; }
''')
rbk('''void BACKWARD::setAlphaMax(float a)
{''', '''void BACKWARD::setSkipFlags(int f)
{
	static int cur = -1;
	if (f != cur) { cudaMemcpyToSymbol(g_skipf_b, &f, sizeof(int)); cur = f; }
}

void BACKWARD::setAlphaMax(float a)
{''')
open(B, "w", encoding="utf-8").write(s)

patch("rasterizer_impl.cu", [
 ('#include <cub/cub.cuh>', '#include <cub/cub.cuh>\n#include <cstdlib>\nstatic int skipf_env() { const char* e = std::getenv("CITYGS_SKIPF"); return e ? std::atoi(e) : 0; }   // ★ 11-16 計時版'),
 ('		CHECK_CUDA(cudaMemset(binningState.pair_used, fmode == 1 ? 0 : 1, num_rendered), debug)',
  '		CHECK_CUDA(cudaMemset(binningState.pair_used, fmode == 1 ? 0 : 0xFF, num_rendered), debug)   // ★ 11-16：非 mode 1 填全 1（A-3 的 8 個 warp 位元都開）'),
 ('	FORWARD::setRecordT((lean & 256) != 0);', '	FORWARD::setSkipFlags(skipf_env());\n	FORWARD::setRecordT((lean & 256) != 0);'),
 ('	CHECK_CUDA(BACKWARD::render(', '	BACKWARD::setSkipFlags(skipf_env());\n	CHECK_CUDA(BACKWARD::render('),
])
print("all patched")
