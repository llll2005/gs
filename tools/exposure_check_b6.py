"""10-15：b6 test 視角（val⊂train）上 best1 與官方 release 的逐幀曝光增益分析（研究總覽 10-15 條）。輸出 outputs/analysis/exposure_b6/。"""
import glob, os, numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def ld(run, n):
    f = glob.glob(f'outputs/lab/{run}/blocks/block_6/test/*/{n}.png.png')[0]
    A = np.asarray(Image.open(f).convert('RGB')).astype(np.float64) / 255; W = A.shape[1] // 2; return A[:, :W], A[:, W:]
names = sorted(os.path.basename(f)[:4] for f in glob.glob('outputs/lab/g44_best1/blocks/block_6/test/*/*.png'))
k = {run: np.array([(lambda g, r: (g * r).sum() / (r * r).sum())(*ld(run, n)) for n in names]) for run in ('g44_best1', 'official_release')}
print('兩模型逐幀增益相關係數 r = %.3f（97 幀）' % np.corrcoef(k['g44_best1'], k['official_release'])[0, 1])
for n in ('1284', '2200'):
    g, r = ld('g44_best1', n); _, ro = ld('official_release', n)
    T = 50; H, W = g.shape[0] // T * T, g.shape[1] // T * T
    tg = g[:H, :W].mean(2).reshape(H // T, T, W // T, T).mean((1, 3)); tr = r[:H, :W].mean(2).reshape(H // T, T, W // T, T).mean((1, 3))
    rat = tr / np.maximum(tg, 1e-3)
    print(f'{n}：50px tile 的 渲染/GT 亮度比 中位 {np.median(rat):.3f}、p10 {np.percentile(rat,10):.3f}、p90 {np.percentile(rat,90):.3f}；比 < 0.9 的 tile {100*(rat<0.9).mean():.0f}%')
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    ax[0, 0].imshow(g); ax[0, 0].set_title(f'{n} GT')
    ax[0, 1].imshow(r); ax[0, 1].set_title('我方 best1（60k 單塊）')
    ax[1, 0].imshow(ro); ax[1, 0].set_title('官方 release（合併，我方光柵器）')
    im = ax[1, 1].imshow(rat, cmap='RdBu', vmin=0.7, vmax=1.3); ax[1, 1].set_title('best1 渲染/GT 亮度比（50px tile；藍＝渲染偏亮、紅＝偏暗）'); plt.colorbar(im, ax=ax[1, 1])
    for a in ax.flat: a.axis('off')
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'DejaVu Sans']
    fig.tight_layout(); fig.savefig(f'outputs/analysis/exposure_b6/{n}.png', dpi=80); plt.close(fig)
print('── 自動曝光假說：場景越暗（模型渲染平均亮度低）=> GT 被提亮越多（增益 > 1）')
for run in ('g44_best1', 'official_release'):
    rm = np.array([ld(run, n)[1].mean() for n in names]); gm = np.array([ld(run, n)[0].mean() for n in names])
    kk = k[run]
    print(f'{run}: corr(增益, 渲染平均亮度) = {np.corrcoef(kk, rm)[0,1]:+.3f}；corr(增益, log 渲染亮度) = {np.corrcoef(kk, np.log(rm))[0,1]:+.3f}；'
          f'渲染最暗 1/4 的幀增益中位 {np.median(kk[rm <= np.percentile(rm,25)]):.3f}、最亮 1/4 {np.median(kk[rm >= np.percentile(rm,75)]):.3f}')
    b = np.polyfit(np.log(rm), np.log(kk), 1)
    print(f'  擬合 log 增益 = {b[0]:+.3f} x log 渲染亮度 + {b[1]:+.3f}  （自動曝光完全補償時斜率 -1、不補償 0）')
print('── 交叉檢驗（場景亮度用另一個模型的渲染，避開「增益＝GT/自己渲染」的機械相關）')
rmb = np.array([ld('g44_best1', n)[1].mean() for n in names]); rmo = np.array([ld('official_release', n)[1].mean() for n in names])
for a, sc, nm in ((k['g44_best1'], rmo, 'best1 增益 vs 官方渲染亮度'), (k['official_release'], rmb, '官方增益 vs best1 渲染亮度')):
    b = np.polyfit(np.log(sc), np.log(a), 1)
    print(f'  {nm}: corr = {np.corrcoef(a, np.log(sc))[0,1]:+.3f}、log-log 斜率 {b[0]:+.3f}')
print('  兩模型渲染亮度彼此的 corr = %.3f' % np.corrcoef(rmb, rmo)[0, 1])
