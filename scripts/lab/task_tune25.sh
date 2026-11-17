#!/bin/bash
# 自主調參（使用者 11-10「開始自主實驗，定時抓 lab 看進度；用 25k 左右，自己調整比例跟一些參數，找到最佳解後排一次 4x4」）
#   底＝新基準（現行配方＋位置 lr x1＋拿掉 v4；_nb_recipe.sh），全長預設 25k。臂＝用 + 串起來的元件（例：g40+lowres、coarse+g40）。
#   對照＝新基準 30k（不重跑）與 A＋dist4；時間只在 [solo] 跑次之間比（coarse 起點另加 coarse 本身攤到每塊約 1 分）。
#   元件：
#     s<NN>    全長 NN k 步（預設 s25）
#     g<NN>    增生期＝全長的 NN%（預設 g50＝現行比例）
#     lowres   前段低解析度：< 全長 1/4 用 1/2、< 1/2 用 3/4，低解析度步不算 dist（11-09 在現行配方上快 12～15%）
#     coarse   起點＝輕量 coarse 裁塊（outputs/lab/coarse_g44r48sh0_crop；task_coarse.sh）
#     sbox     ★ 11-16 A-1 逐 warp 精確支撐盒跳過（光柵器 skip_box；渲染逐位元相同）｜pbw A-3 逐 warp 配對位元＋近平面判準一致（bwd_pair_bit_warp）
#     densenf  ★ 11-16 稠密起點拿掉空中隨機補點、改用凸包外也反投影的校正深度（sfmfill_tri44nf_crop；海面起點臂）
#     coarse5  ★ 11-16 起點＝新底 5 配方的 10k coarse 裁塊（outputs/lab/coarse5_s10g70r48sh0_crop；task_coarse.sh init5／coarse5／crop5）
#     np16     鄰域平面約束每 16 步
#     lr050    位置 lr 再減半（3.2e-5）
#     mvu      v4 判準搬進合併事件（＋BH 用 SfM 間距倍數）
#     mvugrow  同 mvu 但收割期不投票不刪（只在增生期搬）
#     mvugnd   同 mvu 但只動貼地／地面以下的顆粒（離地高的不動）
#     tri      深度換成 SfM 三角形校正的單目（tools/make_tri_depths.py；只留可靠三角形內的像素）——深度支撐閘門、補洞放點跟著用新深度
#     trihf    tri＋補洞放點改成「渲染比校正深度深 2% 就補」（原本 10%）、每次最多 0.5% 顆（原本 0.2%）
#     trist    tri＋單側不能看穿只留「比校正深度深 2% 就罰」那一項，每 8 步（與法向同一步，權重 0.5）
#     trigate  tri＋深度支撐閘門收緊：不要 SfM 豁免、前方 3%、從未貼面 1.5%（原本 5%／5%）
#     tribh    tri＋只擋後面的增生限制：多數有效視角裡落在校正深度表面後 1%～5% 的顆粒，MCMC 幾乎不抽它當母點（不關熵、不刪不搬；
#              SfM 豁免不套用；後 5% 以上＝隔著樓看過去，不算）。規格 紀錄/new_archived/規格與交接/規格_組合臂_表面起點增生限制深度上限_1111.md §2
#     hfd      tri＋需求地圖補貨：補洞事件依落後權重 W(p)（這個像素有多少顏色靠表面後面的顆粒撐）抽像素、新顆粒放在校正深度表面上
#              （＋原本的 α<0.5 洞；可單獨開、不加 loss）。規格 紀錄/new_archived/規格與交接/規格2_表面補貨與落後貢獻計價_1111.md §2
#     vcb      落後貢獻計價進 V/C 剪枝：分數 x max(1 − ρ, 0)，ρ＝貢獻落在校正深度表面後面的比例（光柵器逐顆帳；要有上限圖＝tricap 或 hfd）。規格 2 §3 B1
#     sbp      落後貢獻計價進 MCMC 抽母點：機率 x max(1 − ρ, 0.01)。規格 2 §3 B2
#     bpf      落後計價只算躺平的顆粒（面元法向 z 分量絕對值 > 0.8；牆與斜面的 ρ 一律 0）。11-11 §4 離線檢查：margin 1.5%＋躺平 兩條件都過（`37` §7）
#     tricap<W>[m<N>][g] tri＋逐貢獻的單側深度上限（光柵器 CAP_OFFSET；每一份混合貢獻落在校正深度後面超過 margin 就依混合權重罰）、每 8 步
#              W＝權重（小數點寫 p：tricap0p5）、m<N>＝margin N%（預設 1；小數同樣寫 p：m1p5＝1.5%）、g＝只在增生期（到 densify_until 為止）
#     dense    起點換成稠密初始化（tools/make_tri_dense_init.py：SfM＋補點之外，把可靠像素沿校正深度反投影、多視角一致的體素質心補滿到 0.9 cap）
#     sni      初始化的面元朝向用 SfM 近鄰平面（sfm_normal_init；11-08 在 30k 效果被洗掉，短排程重測）
#     trimask  塊外遮罩換成用校正深度判的那一份（block_masks44_tri；tools/block_mask44.py --depth_dir estimated_depths_tri --depth_raw）
#     cmflat   深度上限只罰躺平的先驗表面（上限遮罩 cap_masks_flat；tools/make_cap_masks.py --kind flat）；要和 tricap<W> 一起用。規格 3 §3
#     r2       兩輪流程（規格 4 §1）：第一輪 0 → S1（預設 0.6 全長）增生到 0.6 S1、顆數 0.7 cap、熵加強（雙向從 S1/6 起、權重 x2）；
#              第二輪 S1 → 全長：增生到 S1＋0.5(全長−S1)、長到 cap、母點只限表面顆粒＋第二輪出生、出生後只准沿平面移動、
#              熵只往 0 推（權重回 0.002）、不剪枝（V/C 只在第一輪）、合併只到 S1
#     r2c<NN>  第一輪顆數 NN% cap（預設 70）｜r2s<NN> S1 佔全長 NN%（預設 60）｜r2free 第二輪不限母點與平面（拆因）｜r2ent0 第二輪熵全關
#              （r2c／r2s／r2free／r2ent0 都要和 r2 一起用）
#     r2w      ★ 11-13 對照：只留 r2 改的窗口與熵（第一輪增生到 0.6 S1、合併只到 S1、熵 x2 雙向從 S1/6 起），**不開第二輪**（round2_from 0、
#              熵的第二輪段也關）、名額 100% cap => 拆「兩輪流程省的時間是窗口還是第二輪」；要和 r2 一起用（寫在哪個位置都一樣，旗標最後才加）
#     r2wl     ★ 11-13 r2w 的重測（r2w 連 S1 之後「熵只往 0、0.002」也關了，不是乾淨對照）：只關第二輪的增生（round2_from 0），
#              S1 之後的弱熵照兩輪流程；名額 100%。要和 r2 一起用、不能和 r2w／r2c 一起用
#              r2w＋r2s<NN>＝不分輪次、窗口依比例縮放：合併到 NN%、增生到 0.6 x NN%、熵雙向從 NN%／6 起（11-13 使用者「分配不同比例的跑次」）
#     規格 5 §2 配方瘦身（每個只關底的一項）：nogate 深度支撐閘門＋增生取樣｜nohf 補洞放點｜noip 增生方向限面內｜nonm 法向一致性｜
#              nonp 不帶 np16（拿掉鄰域平面；只是名字，旗標同不寫 np16）｜nomg 合併整個關掉｜nol2 L1 混 L2｜nopx 點精靈下限｜nohp 刪死顆粒｜
#              mge1 合併改成只併渲染誤差 < 0.1、不設額度、不透明質量守恆（11-08 MMR）
#     v4f      ★ 11-15 v4 只投浮塊票（全部相機、每次事件重投；地下票關掉＝不傷少 SfM 的牆；不開不准往上）——v5 的效果與時間對照
#     v5／v5e<k> ★ 11-15 v5（使用者「v4 改進版、解決時間」）：每步（v5e<k>＝每 k 步）只用當前訓練相機投浮塊票、逐顆累積，事件時用累積票判浮塊、
#              判完累積 x 0.5；新增／搬家的繼承母點的票（sfm_vis_stream；TIDI-GS 式逐步累積證據）。使用者 11-15 viewer：b1 海岸外浮塊只有開 v4 的跑次沒有
#     v5s<k>   ★ 11-15（乙）v5 軟化：母點機率 x max((1-phi)^k, 0.01)，phi＝累積浮塊票／有證據的相機數（取代判到就 x 0.01）
#     rs<k>    ★ 11-15 深夜 校正深度可靠度的斜率級上限放寬到 k（預設 1；4＝SfM 凸包外的外插也用）：b1 海岸外那團在斜視角裡落在凸包外的遠方海面，
#              那裡的校正深度其實準（與海面差 −0.3%）但被判無效 => 閘門投不到票；加校正深度前單目在那裡有值、那團只有約 60 顆
#     ncs<k>   ★ 11-15 深夜 完全沒有幾何約束的顆粒往下沉：閘門視窗內看得到、但從沒碰到可靠先驗也從沒碰到 SfM => 高度上限每個視窗降 k 個像素長度
#              （b1 海岸外那團 100% 沒有 SfM 證據、可靠深度全無效；有 SfM 或可靠深度的地方一律不動）
#     dsf      ★ 11-15（丙）深度支撐閘門的 SfM 備援：先驗深度不可靠／沒有的顆粒改用 SfM 觀測格投空區票（一個閘門、兩種參考；使用者「融進去、只保留一個」）
#     capfc    ★ 11-13 規格 3 §4-3「只補前面」（使用者 11-13「作」；光柵器 depth_cap_grad_mode 3，要重編）：深度上限 B 的梯度只給在上限前面的貢獻、
#              只經由 α（前面變更不透明 => 後面分到的 T 變小）；落在上限後面的貢獻收不到 B 的梯度。loss 與 forward 不變。要和 tricap<W> 一起用
#     mgtrim   ★ 11-13 合併事件照同樣的規則選配對，但不合併：留下兩顆中不透明質量大的那顆（參數、Adam 狀態不動）、小的直接刪（cap 照樣扣）
#              => 與合併同顆數、同時機，只差「併成一顆」還是「刪掉一顆」（使用者「一樣用合併的門檻，分別使用合併跟 trim」）
#     合併借鑑（11-13 使用者「三點採用、記得標來源」；出處 紀錄/機制出處.md）：mgl2 合併後不透明度用 L2 最佳值 o*＝<f,G_m>／<G_m,G_m>｜
#              mgtex 只在 k 近鄰顏色平均絕對偏差 < 0.03 的平滑區域合併（LocoADC 的局部顏色一致性）｜mgrel 併掉的名額不扣、交給 MCMC 搬到需要的地方（merge_prune false）｜
#              mgline 再加線狀合併（燈桿上下段；使用者 11-13）｜mgpx 距離門檻改用像素（最近可見相機裡 MG_PX＝1 像素；px_floor 的量；使用者 11-13）
#     規格 5 §3 加速組合（11-13 使用者「排」）：log0 訓練中不每步記 train/*（只在第 50 步與最後一步記，eval 的「純訓練」計時與底同起訖）｜
#              geo8 dist 每 4 步 -> 每 8 步（權重自動 x 8＝相對 x2，與法向、深度上限落在同一步）｜lowres、s20g70 是現成的
#     noev<k>  ★ 11-13 視線一致性「沒證據」投票＋不准往上，任何階段每 k 步投一次（sfm_vis，不改抽樣：eps 1）｜sink<p> 沒證據就往下沉：每次投票上限再降 p 像素長度
#   TUNE_RESUME_SRC／TUNE_RESUME_NAME／TUNE_RESUME_STEPS：接續模式（使用者 11-13「把機制打開接在已經有的 ckpt 跑幾百步」）——
#     把 outputs/lab/<SRC>/blocks/block_<B> 的終點 ckpt 硬連結到 outputs/lab/<NAME>，從那一步再跑 STEPS 步（Lightning --ckpt_path；不跑圖表資料包）
#   TUNE_PROFILE_STEPS／TUNE_PROFILE_WARMUP／TUNE_PROFILE_NAME：逐段計時模式（task_mech_time.sh MT_TUNE_ARMS 用；規格 5 §1）
#   合併窗口跟著全長縮放（3k ～ 全長 x 0.9；現行配方寫死 27k＝30k 的 90%）；二元熵切雙向（全長／3）與位置 lr 排程由 task_cmp 跟著全長縮放。
# 用法：[solo] bash scripts/lab/task_tune25.sh <塊> <臂>
#       [solo] bash scripts/lab/task_tune25.sh eval <塊> <標籤> <臂>...   （只比跑完的臂；對照新基準 30k、A＋dist4）
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; "") bash "$(dirname "$0")/../_help.sh" "$0"; exit 2 ;; esac
set -u
cd "$(dirname "$0")/../.." || exit 1
source scripts/lab/_common.sh >/dev/null
source scripts/lab/_nb_recipe.sh
SUF=best1maskdistrfed10b4
A_RUN=g44s30ksfm10fssh16di4_$SUF

# 臂字串 => 全域 S（步數）、G（增生 %）、XT（其餘元件）、FAM（家族）、LBL（評估用名稱）
parse() {
  S=25; G=50; XT=()
  local t
  for t in ${1//+/ }; do
    if [[ $t =~ ^s([0-9]+)g([0-9]+)$ ]]; then S=${BASH_REMATCH[1]}; G=${BASH_REMATCH[2]}       # 3x3 的格子寫法 s20g40
    elif [[ $t =~ ^s([0-9]+)$ ]]; then S=${BASH_REMATCH[1]}
    elif [[ $t =~ ^g([0-9]+)$ ]]; then G=${BASH_REMATCH[1]}
    else
      case "$t" in
        lowres|coarse|coarse5|sbox|pbw|densenf|densepl|densepl2|pcs|pcsf|sea3d|seaback|seafront|seanop|np16|lr050|mvu|mvugrow|mvugnd|tri|trihf|trist|trigate|tribh|hfd|vcb|sbp|bpf|dense|sni|trimask|cmflat|r2|r2free|r2ent0|r2w|r2wl) XT+=("$t") ;;
        nogate|nohf|noip|nonm|nonp|nomg|nol2|nopx|nohp|mge1|log0|geo8|capfc|mgtrim|mgl2|mgtex|mgrel|mgline|mgpx|fp16|cskip|pxdel|pxabs|pxdelf|pxabsf) XT+=("$t") ;;
        noev[0-9]*|sink[0-9]*) [[ $t =~ ^(noev|sink)[0-9]+$ ]] || { echo "⛔ 寫法：noev<k>／sink<p>"; return 2; }; XT+=("$t") ;;
        sinkh[0-9]*) [[ $t =~ ^sinkh[0-9]+$ ]] || { echo "⛔ 寫法：sinkh<H>"; return 2; }; XT+=("$t") ;;
        v4f|v5|dsf) XT+=("$t") ;;
        npri[0-9]*) [[ $t =~ ^npri[0-9]+d?$ ]] || { echo "⛔ 寫法：npri<k>[d]（單目法向先驗，權重 0.0125 x k；d＝也套在深度法向上）"; return 2; }; XT+=("$t") ;;
        coarse5c[0-9]*) [[ $t =~ ^coarse5c[0-9]+(f[0-9]+)?$ ]] || { echo "⛔ 寫法：coarse5c<k 步>[f<拆分百分比>]（例 coarse5c30、coarse5c30f50）"; return 2; }; XT+=("$t") ;;
        rs[0-9]) XT+=("$t") ;;
        ncs[0-9]*) [[ $t =~ ^ncs[0-9]+$ ]] || { echo "⛔ 寫法：ncs<k>（完全沒有幾何約束的顆粒每個閘門視窗往下沉 k 像素長度）"; return 2; }; XT+=("$t") ;;
        v5s[0-9]*) [[ $t =~ ^v5s[0-9]+$ ]] || { echo "⛔ 寫法：v5s<k>（軟化：x max((1-phi)^k, 0.01)）"; return 2; }; XT+=("$t") ;;
        v5e[0-9]*) [[ $t =~ ^v5e[0-9]+$ ]] || { echo "⛔ 寫法：v5e<k>（每 k 步投一次）"; return 2; }; XT+=("$t") ;;
        r2c[0-9]*|r2s[0-9]*) [[ $t =~ ^r2[cs][0-9]+$ ]] || { echo "⛔ 寫法：r2c<NN>／r2s<NN>（例 r2c50、r2s50）"; return 2; }; XT+=("$t") ;;
        tricap[0-9]*) [[ $t =~ ^tricap[0-9]+(p[0-9]+)?(m[0-9]+(p[0-9]+)?)?g?$ ]] || { echo "⛔ tricap 寫法：tricap<W>[m<N>][g]（例 tricap2、tricap0p5m3、tricap0p5m1p5、tricap2g）"; return 2; }; XT+=("$t") ;;
        *) echo "⛔ 不認得的元件 $t（s<NN>／g<NN>／s<NN>g<NN>／lowres／coarse／np16／lr050／mvu／mvugrow／mvugnd／tri／trihf／trist／trigate／tribh／hfd／vcb／sbp／bpf／tricap<W>／dense／sni／trimask／cmflat／r2／r2c<NN>／r2s<NN>／r2free／r2ent0／r2w／r2wl／nogate／nohf／noip／nonm／nonp／nomg／nol2／nopx／nohp／mge1／log0／geo8／capfc／mgtrim／mgl2／mgtex／mgrel／mgline／mgpx／fp16／cskip／pxdel／pxabs／pxdelf／pxabsf／noev<k>／sink<p>）"; return 2 ;;
      esac
    fi
  done
  FAM="g44tn_s${S}g${G}$(IFS=; echo "${XT[*]:-}")_"
  LBL="${S}k增生${G}"
  for t in "${XT[@]:-}"; do
    case "$t" in lowres) LBL+="低解析度" ;; coarse) LBL+="coarse起點" ;; coarse5) LBL+="新底5coarse起點" ;; coarse5c*) LBL+="新底5coarse${t#coarse5c}起點" ;; npri*d) _k=${t#npri}; LBL+="單目法向先驗x${_k%d}含深度法向" ;; npri*) LBL+="單目法向先驗x${t#npri}" ;; sbox) LBL+="精確盒跳過" ;; pbw) LBL+="逐warp配對位元" ;; densenf) LBL+="稠密起點不撒空中點" ;; densepl) LBL+="稠密起點平面補點" ;; densepl2) LBL+="稠密起點平面另算名額" ;; sea3d) LBL+="海面物件錨定層" ;; seaback) LBL+="海面寬遮罩上限" ;; seafront) LBL+="海面窄遮罩罰前面" ;; seanop) LBL+="海面格非母點" ;; pcs) LBL+="海面平面上限" ;; pcsf) LBL+="海面與立面平面上限" ;; np16) LBL+="鄰域平面" ;; lr050) LBL+="lr減半" ;; mvu) LBL+="合併地下" ;;
      mvugrow) LBL+="合併地下只增生期" ;; mvugnd) LBL+="合併地下只貼地" ;;
      tri) LBL+="校正深度" ;; trihf) LBL+="校正深度補洞2%" ;; trist) LBL+="校正深度不能看穿2%" ;; trigate) LBL+="校正深度閘門收緊" ;; tribh) LBL+="只擋後面增生" ;; hfd) LBL+="需求地圖補貨" ;; vcb) LBL+="落後計價剪枝" ;; sbp) LBL+="落後計價抽母點" ;; bpf) LBL+="計價只算躺平" ;;
      tricap*) LBL+="逐貢獻深度上限${t#tricap}" ;;
      dense) LBL+="稠密初始化" ;; sni) LBL+="SfM法向起點" ;; trimask) LBL+="校正深度遮罩" ;; cmflat) LBL+="上限只罰躺平" ;; r2) LBL+="兩輪" ;; r2c*) LBL+="第一輪名額${t#r2c}" ;;
      r2s*) LBL+="S1佔${t#r2s}" ;; r2free) LBL+="不限出身" ;; r2ent0) LBL+="第二輪熵關" ;; r2w) LBL+="只留窗口不開第二輪" ;; r2wl) LBL+="只關第二輪增生留後段弱熵" ;;
      nogate) LBL+="拿掉閘門" ;; nohf) LBL+="拿掉補洞" ;; noip) LBL+="拿掉面內增生" ;; nonm) LBL+="拿掉法向" ;; nonp) LBL+="拿掉鄰域平面" ;;
      nomg) LBL+="拿掉合併" ;; nol2) LBL+="拿掉L2" ;; nopx) LBL+="拿掉點精靈下限" ;; nohp) LBL+="拿掉刪死顆粒" ;; mge1) LBL+="合併只併誤差0.1" ;; log0) LBL+="不記log" ;; geo8) LBL+="dist每8步" ;; capfc) LBL+="只補前面" ;; mgtrim) LBL+="合併配對只刪不併" ;; mgl2) LBL+="合併L2不透明度" ;; mgtex) LBL+="合併只在平滑區" ;; mgrel) LBL+="合併名額交給搬家" ;; mgline) LBL+="線狀合併" ;; mgpx) LBL+="合併像素門檻" ;; fp16) LBL+="完整路徑每16步" ;; cskip) LBL+="C區不跑" ;; pxdel) LBL+="點精靈刪除" ;; pxabs) LBL+="點精靈吸收" ;; pxdelf) LBL+="點精靈全解析度才刪" ;; pxabsf) LBL+="點精靈全解析度才吸收" ;; noev*) LBL+="不准往上每${t#noev}步投票" ;; sinkh*) LBL+="只沉離下方${t#sinkh}px以上" ;; sink*) LBL+="沒證據往下沉${t#sink}px" ;; v4f) LBL+="v4只投浮塊票" ;; v5) LBL+="v5每步累積浮塊票" ;; v5e*) LBL+="v5每${t#v5e}步累積浮塊票" ;; v5s*) LBL+="v5軟化k${t#v5s}" ;; ncs*) LBL+="無約束往下沉${t#ncs}px" ;; rs[0-9]) LBL+="校正深度斜率級放寬到${t#rs}" ;; dsf) LBL+="閘門SfM備援" ;; esac
  done
}

if [ "$1" = eval ]; then
  B=${2:?塊}; TAG=${3:?標籤}; shift 3
  # TUNE_BASE／TUNE_BASE_LBL＝評估的基準跑次與名稱（預設新基準 30k；11-10 3x3 改建在新基準＋鄰域平面上 => 基準換成 30k 的鄰域平面臂）
  BASE=${TUNE_BASE:-$NB_RUN}; BLBL=${TUNE_BASE_LBL:-新基準30k}
  R=(); ARGS=("$BLBL=$BASE"); SW=($A_RUN $BASE)
  [ "$BASE" = "$NB_RUN" ] || { ARGS+=(新基準30k=$NB_RUN); SW+=($NB_RUN); }
  ARGS+=(A+dist4=$A_RUN)
  # TUNE_EXTRA="名稱=跑次 ..."：另外帶幾個現成跑次一起比（例：合併地下 v1＝g44s30ksfm10nbmvu_best1maskdistrfed10b4）
  for kv in ${TUNE_EXTRA:-}; do
    [ -d "outputs/${TUNE_ROOT:-lab}/${kv#*=}/blocks/block_$B" ] && { ARGS+=("$kv"); SW+=("${kv#*=}"); } || echo "⚠ 缺 $kv"
  done
  for a in "$@"; do
    parse "$a" || exit 2
    r=$FAM$SUF
    if [ -d outputs/${TUNE_ROOT:-lab}/$r/blocks/block_$B ]; then R+=($r); ARGS+=("$LBL=$r"); else echo "⚠ 缺 $a（$r）"; fi
  done
  # ★ 11-12 TUNE_NOSWAP=1：跳過換塊評測（要載約 4 GB 的合併模型；lab 被外部行程佔卡、共用卡時放不下）＝只算 CPU 那幾項
  if [ "${TUNE_NOSWAP:-0}" = 1 ]; then echo "[eval] TUNE_NOSWAP=1：跳過換塊評測（之後卡空時補）"
  else bash scripts/lab/task_full44.sh swap "$B" "${SW[@]}" "${R[@]}"; fi
  EVAL_ROOT=outputs/${TUNE_ROOT:-lab} EVAL_ONLY=${TUNE_EVAL_ONLY:-time,f1,ground,facade,orient,sea,smallobj} bash scripts/lab/eval_arms_lab.sh "$B" tune_${TAG}_b${B} "${ARGS[@]}"
  exit $?
fi

B=$1; ARM=${2:?臂}
parse "$ARM" || exit 2
ST=$((S * 1000))
XA=(--model.density.init_args.densify_until_iter $((ST * G / 100))
    --model.density.init_args.merge_until_iter $((ST * 9 / 10)))
ENVX=()
R2C=70; R2S=60
for t in "${XT[@]:-}"; do
  case "$t" in r2c*) R2C=${t#r2c} ;; r2s*) R2S=${t#r2s} ;; r2free|r2ent0|r2w|r2wl)
    [[ " ${XT[*]} " =~ \ r2\  ]] || { echo "⛔ $t 要和 r2 一起用"; exit 2; } ;; esac
done
if [[ " ${XT[*]} " =~ \ r2[cs][0-9] ]] && ! [[ " ${XT[*]} " =~ \ r2\  ]]; then echo "⛔ r2c／r2s 要和 r2 一起用"; exit 2; fi
for t in "${XT[@]:-}"; do
  case "$t" in
    lowres) XA+=(--model.train_res_schedule "$((ST / 4)):0.5,$((ST / 2)):0.75" --model.metric.init_args.lambda_dist_lowres 0) ;;
    coarse) CK=outputs/lab/coarse_g44r48sh0_crop/blocks/block_$B/checkpoints/coarse_crop.ckpt
            [ -f "$CK" ] || { echo "⛔ 缺 $CK（task_coarse.sh crop）"; exit 2; }
            XA+=(--model.initialize_from "$CK" --model.overwrite_config false); ENVX+=(CITYGS_KEEP_CFG_RENDERER=1) ;;
    # ★ 11-16 A-1（Fable）：逐 warp 精確支撐盒跳過（forward＋backward；渲染逐位元相同、梯度只差加總順序；本機 lean 步 −7.8～−9%、完整路徑步 −4%）
    sbox)   XA+=(--model.renderer.init_args.skip_box true) ;;
    # ★ 11-16 A-3（Fable）：配對位元細到 8 個 warp（backward 逐 warp 跳）＋backward 近平面判準跟 forward 一樣（改到原本不一致的那些梯度）
    pbw)    XA+=(--model.renderer.init_args.bwd_pair_bit true --model.renderer.init_args.bwd_pair_bit_warp true) ;;
    # ★ 11-16 海面起點臂（Fable）：稠密起點拿掉空中隨機補點、名額改用凸包外也反投影的校正深度補（只動起點；tools/make_tri_dense_init.py --drop_fill --outside_hull）
    # ★ 11-17 平面物件（Fable 第三輪；使用者「作」）：
    #   densepl＝稠密起點拿掉空中隨機補點、改用平面外推深度（海面鋪滿、塔頂補在外推的牆面）反投影補滿（sfmfill_tri44pl_crop）
    #   pcs／pcsf＝深度上限補到平面外推深度的像素（海面／海面＋立面）；需要 tricap 開著（上限本身）
    # ★ 11-17 densepl2＝v2：陸地照原本稠密起點的密度，平面點另外用自己的名額與體素邊長（v1 平面佔掉名額、陸地變稀 => 地下 +0.2～0.5 層）
    densepl2) [[ " ${XT[*]} " =~ \ (dense|densenf|densepl)\  ]] && { echo "⛔ densepl2 不能和其他 dense* 起點一起寫"; exit 2; }
            [ -f "data/matrix_city/aerial/train/block_all/sfmfill_tri44pl2_crop/dup1/block_$B.ply" ] || [ "${CITYGS_DRY:-0}" = 1 ] \
              || { echo "⛔ 缺 sfmfill_tri44pl2_crop/dup1/block_$B.ply（tools/make_tri_dense_init.py --blocks $B --drop_fill --plane_dir estimated_depths_plane --plane_sep --out sfmfill_tri44pl2_crop）"; exit 2; }
            ENVX+=(CITYGS_CROP_INIT=sfmfill_tri44pl2_crop); CITYGS_DUP44=1 ;;
    # ★ 11-17 規格 6（海面物件）：sea3d＝起點用 sfmfill_tri44sea_crop（陸地＝v2、平面點只留立面、海面＝每個海面格一顆鋪磚）＋錨定的海面層，
    #   cap 另外加上鋪磚顆數（海面層有自己的名額）；seaback＝上限改用海面物件的寬遮罩與共識平面；seafront＝窄遮罩上罰比平面淺；seanop＝海面格上離開平面的不當母點
    sea3d)  _SO=data/matrix_city/aerial/train/block_all/sea_object; _SI=data/matrix_city/aerial/train/block_all/sfmfill_tri44sea_crop/dup1
            [[ " ${XT[*]} " =~ \ (dense|densenf|densepl|densepl2)\  ]] && { echo "⛔ sea3d 是自己的起點，不能和其他 dense* 一起寫"; exit 2; }
            [ -f "$_SI/block_$B.ply" ] && [ -f "$_SI/block_${B}_sea.npy" ] && [ -f "$_SO/block_$B.npz" ] || [ "${CITYGS_DRY:-0}" = 1 ] \
              || { echo "⛔ 缺海面物件或起點（tools/make_sea_object.py --blocks $B --masks；tools/make_tri_dense_init.py --blocks $B --drop_fill --plane_dir estimated_depths_plane --plane_sep --drop_outliers --sea_object sea_object --out sfmfill_tri44sea_crop）"; exit 2; }
            ENVX+=(CITYGS_CROP_INIT=sfmfill_tri44sea_crop); CITYGS_DUP44=1
            # ⚠ 11-17 冒煙抓到：lab 的系統 python3 沒有 numpy，原本的 `|| echo 0` 讓海面名額靜默變 0 => 改用 gspl、讀不到就停
            if [ -f "$_SI/block_${B}_sea.npy" ]; then
              SEA_EXTRA=$(conda run -n gspl --no-capture-output python -c "import numpy as np;print(int(np.load('$_SI/block_${B}_sea.npy').sum()))" | tail -1)
              [[ "$SEA_EXTRA" =~ ^[0-9]+$ ]] && [ "$SEA_EXTRA" -gt 0 ] || { echo "⛔ 讀不到海面鋪磚數（$_SI/block_${B}_sea.npy => '$SEA_EXTRA'）"; exit 2; }
            else SEA_EXTRA=0; fi
            XA+=(--model.density.init_args.sea_layer_file "$_SI/block_${B}_sea.npy" --model.density.init_args.sea_layer_obj "$_SO/block_$B.npz") ;;
    seaback) [[ " ${XT[*]} " =~ \ (pcs|pcsf)\  ]] && { echo "⛔ seaback 取代 pcs／pcsf（同一個 depth_cap_plane_dir），不能一起寫"; exit 2; }
            [[ " ${XT[*]} " =~ \ tricap ]] || { echo "⛔ seaback 要和 tricap 一起開"; exit 2; }
            XA+=(--model.metric.init_args.depth_cap_plane_dir data/matrix_city/aerial/train/block_all/sea_object/cap_b$B --model.metric.init_args.depth_cap_plane_kinds sea) ;;
    seafront) XA+=(--model.metric.init_args.sea_front_reg 0.5 --model.metric.init_args.sea_front_margin 0.02 --model.metric.init_args.sea_front_dir data/matrix_city/aerial/train/block_all/sea_object/cap_b$B) ;;
    seanop) XA+=(--model.density.init_args.sea_noparent true --model.density.init_args.sea_layer_obj data/matrix_city/aerial/train/block_all/sea_object/block_$B.npz) ;;
    densepl) [[ " ${XT[*]} " =~ \ dense\  || " ${XT[*]} " =~ \ densenf\  ]] && { echo "⛔ densepl 與 dense／densenf 是不同起點，不能一起寫"; exit 2; }
            [ -f "data/matrix_city/aerial/train/block_all/sfmfill_tri44pl_crop/dup1/block_$B.ply" ] || [ "${CITYGS_DRY:-0}" = 1 ] \
              || { echo "⛔ 缺 sfmfill_tri44pl_crop/dup1/block_$B.ply（tools/make_plane_depth.py --blocks $B；tools/make_tri_dense_init.py --blocks $B --drop_fill --plane_dir estimated_depths_plane --out sfmfill_tri44pl_crop）"; exit 2; }
            ENVX+=(CITYGS_CROP_INIT=sfmfill_tri44pl_crop); CITYGS_DUP44=1 ;;
    pcs|pcsf) _PD=data/matrix_city/aerial/train/block_all/estimated_depths_plane
            [ -d "$_PD" ] || [ "${CITYGS_DRY:-0}" = 1 ] || { echo "⛔ 缺 $_PD（tools/make_plane_depth.py --blocks $B）"; exit 2; }
            [[ " ${XT[*]} " =~ \ tricap ]] || { echo "⛔ $t 要和 tricap（深度上限本身）一起開"; exit 2; }
            XA+=(--model.metric.init_args.depth_cap_plane_dir "$_PD" --model.metric.init_args.depth_cap_plane_kinds "$([ "$t" = pcs ] && echo sea || echo facade,sea)") ;;
    densenf) [[ " ${XT[*]} " =~ \ dense\  ]] && { echo "⛔ densenf 與 dense 是兩種起點，不能一起寫"; exit 2; }
            [ -f "data/matrix_city/aerial/train/block_all/sfmfill_tri44nf_crop/dup1/block_$B.ply" ] || [ "${CITYGS_DRY:-0}" = 1 ] \
              || { echo "⛔ 缺 sfmfill_tri44nf_crop/dup1/block_$B.ply（tools/make_tri_dense_init.py --blocks $B --drop_fill --outside_hull --out sfmfill_tri44nf_crop）"; exit 2; }
            ENVX+=(CITYGS_CROP_INIT=sfmfill_tri44nf_crop); CITYGS_DUP44=1 ;;
    # ★ 11-16 單目法向先驗（Fable H3；使用者「都做」）：完整路徑步加 w(1−|cos(渲染法向, 單目法向)|)，w＝0.0125 x k（同法向一致性的尺度）；
    #   先驗＝tools/make_mono_normals.py（校正深度不可靠、局部平面、深度在本張 SfM 範圍內的立面像素）
    npri*)  _NP=data/matrix_city/aerial/train/block_all/normal_prior_mono
            [ -d "$_NP" ] || [ "${CITYGS_DRY:-0}" = 1 ] || { echo "⛔ 缺 $_NP（tools/make_mono_normals.py --blocks $B）"; exit 2; }
            _k=${t#npri}; _k=${_k%d}
            XA+=(--model.metric.init_args.normal_prior_reg "$(awk -v k="$_k" 'BEGIN{printf "%g", 0.0125 * k}')" --model.metric.init_args.normal_prior_dir "$_NP")
            # ★ 11-17 npri<k>d：先驗也套在渲染深度算出的法向（Fable 第三輪過渡解法：壓掉轉正後的平行第二層）
            [[ $t == *d ]] && XA+=(--model.metric.init_args.normal_prior_depth 1.0) ;;
    # ★ 11-16 coarse 消融（使用者「對 coarse 抽幾個小塊作各階段長度／總長度／參數的消融，時間不超過新底 5」）：
    #   coarse5c<NN>[f<PP>] ＝ coarse NN k 步（CO5_RUN＝coarse5_s<NN>g70r48sh0）、裁塊拆到 PP% cap（預設 85；CO5_CROP_FRAC）
    coarse5c*) [[ $t =~ ^coarse5c([0-9]+)(f([0-9]+))?$ ]]
            CK=outputs/lab/coarse5_s${BASH_REMATCH[1]}g70r48sh0${BASH_REMATCH[3]:+_f${BASH_REMATCH[3]}}_crop/blocks/block_$B/checkpoints/coarse_crop.ckpt
            [ -f "$CK" ] || [ "${CITYGS_DRY:-0}" = 1 ] || { echo "⛔ 缺 $CK（CO5_STEPS／CO5_RUN／CO5_CROP_FRAC 跑 task_coarse.sh coarse5＋crop5）"; exit 2; }
            [[ " ${XT[*]} " =~ \ dense\  ]] && { echo "⛔ coarse 起點與 dense 不能一起寫"; exit 2; }
            XA+=(--model.initialize_from "$CK" --model.overwrite_config false); ENVX+=(CITYGS_KEEP_CFG_RENDERER=1) ;;
    # ★ 11-16 新底 5 版 coarse 起點（task_coarse.sh init5／coarse5／crop5；TUNE_COARSE5＝裁塊跑次名）
    coarse5) CK=outputs/lab/${TUNE_COARSE5:-coarse5_s10g70r48sh0_crop}/blocks/block_$B/checkpoints/coarse_crop.ckpt
            [ -f "$CK" ] || [ "${CITYGS_DRY:-0}" = 1 ] || { echo "⛔ 缺 $CK（task_coarse.sh crop5）"; exit 2; }
            [[ " ${XT[*]} " =~ \ dense\  ]] && { echo "⛔ coarse5 與 dense 是兩種起點，不能一起寫"; exit 2; }
            XA+=(--model.initialize_from "$CK" --model.overwrite_config false); ENVX+=(CITYGS_KEEP_CFG_RENDERER=1) ;;
    np16)   XA+=(--model.metric.init_args.surf_reg 0.01 --model.metric.init_args.surf_orient 1.0 --model.metric.init_args.surf_interval 16
                 --model.metric.init_args.surf_every 400) ;;
    lr050)  XA+=(--model.gaussian.init_args.optimization.means_lr_init 3.2e-05 --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.lr_final 3.2e-07) ;;
    mvu|mvugrow|mvugnd)
            XA+=(--model.density.init_args.merge_vis_under true
                 --model.density.init_args.sfm_vis_behind_max 10 --model.density.init_args.sfm_vis_behind_max_above_nn 16)
            [ "$t" = mvugrow ] && XA+=(--model.density.init_args.merge_vis_under_harvest false)
            [ "$t" = mvugnd ] && XA+=(--model.density.init_args.merge_vis_under_ground true) ;;
    tri|trihf|trist|trigate|tribh|hfd|tricap*)
            _TD=data/matrix_city/aerial/train/block_all/estimated_depths_tri
            [ -d "$_TD" ] && [ -d "${_TD}_rel" ] || { echo "⛔ 缺 $_TD（先跑 tools/make_tri_depths.py）"; exit 2; }
            XA+=(--data.parser.depth_dir estimated_depths_tri --data.parser.depth_scale_name estimated_depth_scales_tri
                 --data.parser.depth_rel_dir estimated_depths_tri_rel)
            [ "$t" = trihf ] && XA+=(--model.density.init_args.hole_fill_margin 0.02 --model.density.init_args.hole_fill_frac 0.005)
            [ "$t" = trist ] && XA+=(--model.metric.init_args.depth_see_through_reg 0.5 --model.metric.init_args.depth_see_through_margin 0.02
                                     --model.metric.init_args.depth_see_through_hole 0 --model.metric.init_args.depth_see_through_interval 8)
            if [[ $t =~ ^tricap([0-9]+)(p([0-9]+))?(m([0-9]+)(p([0-9]+))?)?(g)?$ ]]; then
              _cw=${BASH_REMATCH[1]}; [ -n "${BASH_REMATCH[3]}" ] && _cw="$_cw.${BASH_REMATCH[3]}"
              if [ -n "${BASH_REMATCH[7]}" ]; then                     # m1p5＝1.5%（整數寫法的輸出與 11-11 前相同）
                _cm=$(awk -v a="${BASH_REMATCH[5]}" -v b="${BASH_REMATCH[7]}" 'BEGIN{printf "%g", (a + ("0." b)) / 100}')
              else
                _cm=$(printf '0.%02d' "${BASH_REMATCH[5]:-1}")
              fi
              XA+=(--model.metric.init_args.depth_cap_reg "$_cw" --model.metric.init_args.depth_cap_margin "$_cm" --model.metric.init_args.depth_cap_interval 8)
              [ -n "${BASH_REMATCH[8]}" ] && XA+=(--model.metric.init_args.depth_cap_until_iter $((ST * G / 100)))
            fi
            [ "$t" = hfd ] && XA+=(--model.density.init_args.hole_fill_demand true)
            [ "$t" = tribh ] && XA+=(--model.metric.init_args.depth_support_behind true --model.metric.init_args.depth_support_behind_margin 0.01
                                     --model.metric.init_args.depth_support_behind_max 0.05)
            [ "$t" = trigate ] && XA+=(--model.metric.init_args.depth_support_sfm_radius_nn 0 --model.metric.init_args.depth_support_sfm_radius 0
                                       --model.metric.init_args.depth_support_delta_front 0.03 --model.metric.init_args.depth_support_delta_back 0.015) ;;
    vcb)    XA+=(--model.renderer.init_args.trim_behind_price 1) ;;
    sbp)    XA+=(--model.density.init_args.sample_behind_price 1) ;;
    bpf)    XA+=(--model.renderer.init_args.behind_price_flat 0.8 --model.density.init_args.behind_price_flat 0.8) ;;
    dense)  [ -f "data/matrix_city/aerial/train/block_all/sfmfill_tri44_crop/dup1/block_$B.ply" ] || { echo "⛔ 缺稠密初始化（tools/make_tri_dense_init.py --blocks $B）"; exit 2; }
            ENVX+=(CITYGS_CROP_INIT=sfmfill_tri44_crop); CITYGS_DUP44=1 ;;
    sni)    XA+=(--model.density.init_args.sfm_normal_init true) ;;
    trimask) [ -f "data/matrix_city/aerial/train/block_all/block_masks44_tri/block_$B/stats.json" ] || { echo "⛔ 缺校正深度遮罩 block_masks44_tri/block_$B"; exit 2; }
            ENVX+=(CITYGS_MASK_DIR=block_masks44_tri) ;;
    r2)     S1=$((ST * R2S / 100)); D1=$((S1 * 6 / 10)); D1=$((D1 / 100 * 100)); U2=$((S1 + (ST - S1) / 2)); U2=$((U2 / 100 * 100))
            XA+=(--model.density.init_args.densify_until_iter $D1 --model.density.init_args.merge_until_iter $S1
                 --model.density.init_args.round2_from $S1 --model.density.init_args.round2_densify_until $U2
                 --model.density.init_args.round2_cap_frac "$(awk -v c="$R2C" 'BEGIN{printf "%g", c / 100}')"
                 --model.metric.init_args.opacity_entropy_reg 0.004 --model.metric.init_args.opacity_entropy_both_from_iter $((S1 / 6))
                 --model.metric.init_args.opacity_entropy_both_reg 0.016
                 --model.metric.init_args.opacity_entropy_round2_from $S1 --model.metric.init_args.opacity_entropy_round2_reg 0.002)
            echo "[r2] 兩輪：S1 $S1（${R2S}%）｜第一輪增生到 $D1、顆數 ${R2C}% cap｜第二輪增生 $S1 → $U2｜合併只到 $S1" ;;
    r2free) XA+=(--model.density.init_args.round2_free true) ;;
    # ★ 11-13 規格 5 §2 配方瘦身：每個元件只關底的一項（旗標蓋在 NB／其他元件之後 => 後者勝）
    nogate) XA+=(--model.metric.init_args.depth_support false --model.density.init_args.geom_gate_sampling false) ;;
    nohf)   XA+=(--model.density.init_args.hole_fill_frac 0) ;;
    noip)   XA+=(--model.density.init_args.inplane_offset 0 --model.density.init_args.inplane_hold 0) ;;
    nonm)   XA+=(--model.metric.init_args.lambda_normal 0) ;;
    nonp)   [[ " ${XT[*]} " =~ \ np16\  ]] && { echo "⛔ nonp＝不帶 np16（兩個不能一起寫）"; exit 2; } ;;
    nomg)   XA+=(--model.density.init_args.merge_interval 0) ;;    # 整個合併關掉（merge_prune false 只是改成交給 relocation 搬，合併照做）
    nol2)   XA+=(--model.metric.init_args.rgb_l2_weight 0) ;;
    nopx)   XA+=(--model.density.init_args.px_floor 0) ;;
    nohp)   XA+=(--model.density.init_args.harvest_prune_dead false) ;;
    mge1)   XA+=(--model.density.init_args.merge_render_eps 0.1 --model.density.init_args.merge_opacity mass --model.density.init_args.merge_max_frac 1.0) ;;
    r2ent0) XA+=(--model.metric.init_args.opacity_entropy_round2_reg 0) ;;
    # ★ 11-13 規格 5 §3：log0＝train/* 只在第 0 步與最後一步記（k＝全長−1；0 % k、(全長−1) % k 都是 0）=> 每步不交指標給 Lightning、
    #   進度列的 loss 最多每秒讀一次；eval 的純訓練時間（train/loss 首尾）照樣量得到、與底同一個起訖點
    log0)   XA+=(--model.train_log_interval -50) ;;   # ★ 11-13 改：k＝全長−1 時 Lightning 每 50 步才寫，只剩最後一筆、量不到時間 => 只記第 50 步與最後一步（底的 tfevents 首尾＝49／29999）
    capfc)  [[ " ${XT[*]} " =~ \ tricap[0-9] ]] || { echo "⛔ capfc 要和 tricap<W> 一起用（沒有上限就沒有 B）"; exit 2; }
            XA+=(--model.metric.init_args.depth_cap_grad_mode 3) ;;
    geo8)   XA+=(--model.metric.init_args.lambda_dist_interval 8) ;;
    # ★ 11-14 時間候選（38 §1「量到但規格沒列的候選」；使用者「lab 等太久，抓資料在本機跑」）：
    #   fp16  每 8 步一次的完整路徑步（法向＋深度上限＋閘門＋看穿；佔 21～24%）改每 16 步；四項都是「每 k 步、權重乘 k」=> 平均權重不變，預估約 −10%
    #   cskip lean 訓練 forward 直接略過 C 區（塊外不計分）像素（光柵器工作量 −7%；C 區顆粒少了監督以外的梯度，要看品質）
    fp16)   XA+=(--model.metric.init_args.lambda_normal_interval 16 --model.metric.init_args.depth_support_interval 16
                 --model.metric.init_args.depth_cap_interval 16 --model.metric.init_args.depth_see_through_interval 16) ;;
    cskip)  XA+=(--model.renderer.init_args.skip_unscored_pixels true) ;;
    # ★ 11-14 點精靈刪除／吸收（使用者「撐大嗎，為啥不刪掉或用合併機制併起來」）：取代 px_floor 的「撐大」；每 500 步處理一次
    pxdel)  XA+=(--model.density.init_args.px_floor_mode delete) ;;
    pxabs)  XA+=(--model.density.init_args.px_floor_mode absorb) ;;
    # ★ 11-14 公平版：低解析度期照常撐大，全解析度（新底 3 是第 10,000 步起）才刪／吸收
    pxdelf) XA+=(--model.density.init_args.px_floor_mode delete --model.density.init_args.px_sprite_full_res_only true) ;;
    pxabsf) XA+=(--model.density.init_args.px_floor_mode absorb --model.density.init_args.px_sprite_full_res_only true) ;;
    mgtrim) XA+=(--model.density.init_args.merge_as_trim true) ;;
    mgl2)   XA+=(--model.density.init_args.merge_opacity l2) ;;
    mgtex)  XA+=(--model.density.init_args.merge_tex_tol "${MG_TEX:-0.03}") ;;
    mgrel)  XA+=(--model.density.init_args.merge_prune false) ;;
    mgline) XA+=(--model.density.init_args.merge_line true) ;;
    mgpx)   XA+=(--model.density.init_args.merge_px_k "${MG_PX:-1}") ;;
    noev*)  XA+=(--model.density.init_args.sfm_vis true --model.density.init_args.sfm_vis_cams 0 --model.density.init_args.sfm_vis_cell 8
                 --model.density.init_args.sfm_vis_px 4 --model.density.init_args.sfm_vis_free_frac 0.5 --model.density.init_args.sfm_vis_rep 0.1
                 --model.density.init_args.sfm_vis_eps 1.0 --model.density.init_args.sfm_birth_from_iter 0
                 --model.density.init_args.sfm_vis_noup true --model.density.init_args.sfm_noev_every "${t#noev}") ;;
    # ★ 11-15 v4 只投浮塊票（地下票永不成立：behind_frac 1.01 > 1）／v5 每步用當前訓練相機投浮塊票、逐顆累積（sfm_vis_stream）；都不開不准往上
    v4f|v5|v5e*|v5s*) XA+=(--model.density.init_args.sfm_vis true --model.density.init_args.sfm_vis_cams 0 --model.density.init_args.sfm_vis_cell 8
                 --model.density.init_args.sfm_vis_px 4 --model.density.init_args.sfm_vis_free_frac 0.5 --model.density.init_args.sfm_vis_rep 0.1
                 --model.density.init_args.sfm_vis_behind_frac 1.01 --model.density.init_args.sfm_vis_noup false)
                 case "$t" in v5) XA+=(--model.density.init_args.sfm_vis_stream 1) ;; v5e*) XA+=(--model.density.init_args.sfm_vis_stream "${t#v5e}") ;;
                   v5s*) XA+=(--model.density.init_args.sfm_vis_stream 1 --model.density.init_args.sfm_vis_stream_soft "${t#v5s}") ;; esac ;;
    # ★ 11-15 深夜 校正深度的可靠斜率級上限（預設 1；4＝連 SfM 凸包外的外插也算可靠）。b1 海岸外那團的像素全是斜率級 4（凸包外）
    rs[0-9]) [[ " ${XT[*]} " =~ \ tri ]] || [[ " ${XT[*]} " =~ \ tricap ]] || { echo "⛔ rs<k> 要和校正深度（tri／tricap）一起用"; exit 2; }
            XA+=(--data.parser.depth_rel_slope_max ${t#rs}) ;;
    # ★ 11-15 深夜 完全沒有幾何約束（閘門視窗內看得到、從沒碰到可靠先驗也沒碰到 SfM）的顆粒，每個視窗高度上限往下降 k 像素長度
    ncs*)   XA+=(--model.metric.init_args.depth_support_nocon true --model.density.init_args.sfm_vis_gate true
                 --model.density.init_args.sfm_vis_cell 8 --model.density.init_args.sfm_vis_px 4 --model.density.init_args.sfm_nocon_sink_px "${t#ncs}") ;;
    # ★ 11-15（丙）深度支撐閘門在先驗不可靠的地方改用 SfM 觀測格投票（一個閘門、兩種參考；不開 v4／v5 的抽樣壓制）
    dsf)    XA+=(--model.metric.init_args.depth_support_sfm_free true --model.density.init_args.sfm_vis_gate true
                 --model.density.init_args.sfm_vis_cell 8 --model.density.init_args.sfm_vis_px 4) ;;
    # ★ 11-14 細物保護：只有離下方 SfM 表面 > H 個像素長度的沒證據顆粒會往下沉（海面浮塊中位 504、路燈帶 81、車輛帶 11）
    sinkh*) [[ " ${XT[*]} " =~ \ sink[0-9] ]] || { echo "⛔ sinkh 要和 sink<p> 一起用"; exit 2; }
            XA+=(--model.density.init_args.sfm_noup_sink_min_h_px ${t#sinkh}) ;;
    sink*)  [[ " ${XT[*]} " =~ \ noev[0-9] ]] || { echo "⛔ sink 要和 noev<k> 一起用"; exit 2; }
            XA+=(--model.density.init_args.sfm_noup_sink_px "${t#sink}") ;;   # 權重乘 k 是 metric 自己做的（gs2d_metrics）；(step+1) % 8 與法向、深度上限同一步
    cmflat) [[ " ${XT[*]} " =~ \ tricap[0-9] ]] || { echo "⛔ cmflat 要和 tricap<W> 一起用（沒有上限就沒有遮罩可套）"; exit 2; }
            _CM=data/matrix_city/aerial/train/block_all/cap_masks_flat
            [ -f "$_CM/stats_block_$B.json" ] || { echo "⛔ 缺上限遮罩 $_CM/stats_block_$B.json（tools/make_cap_masks.py --kind flat --blocks $B）"; exit 2; }
            XA+=(--model.metric.init_args.depth_cap_mask_dir "$_CM") ;;
  esac
done
if [[ " ${XT[*]} " =~ \ r2w\  ]]; then   # 放在所有元件之後 => 不管 r2w 寫在 r2 前後，關第二輪的旗標都最後蓋上
  [[ " ${XT[*]} " =~ \ r2c[0-9] ]] && { echo "⛔ r2w 是名額 100% 的單輪對照，不能和 r2c 一起用"; exit 2; }
  XA+=(--model.density.init_args.round2_from 0 --model.metric.init_args.opacity_entropy_round2_from 0)
  echo "[r2w] 只留兩輪流程的窗口與熵、不開第二輪：增生到 $D1、合併到 $S1、熵 x2 雙向從 $((S1 / 6)) 起、cap 100%"
fi
if [[ " ${XT[*]} " =~ \ r2wl\  ]]; then   # 放在所有元件之後
  [[ " ${XT[*]} " =~ \ r2c[0-9] ]] || [[ " ${XT[*]} " =~ \ r2w\  ]] && { echo "⛔ r2wl 不能和 r2c／r2w 一起用（名額 100% 的單輪對照）"; exit 2; }
  XA+=(--model.density.init_args.round2_from 0)
  echo "[r2wl] 只關第二輪的增生：增生到 $D1、合併到 $S1、熵 x2 雙向從 $((S1 / 6)) 起、S1（$S1）之後熵只往 0（0.002）、cap 100%"
fi
CAP=$(python3 -c "import json;print(int(round(1e6 * json.load(open('configs/cap_alloc_sfm_g44.json'))['weights']['$B'])))")
[ -n "${SEA_EXTRA:-}" ] && [ "${SEA_EXTRA:-0}" -gt 0 ] && { echo "[sea3d] cap 另加海面鋪磚 $SEA_EXTRA 顆（海面層自己的名額）：$CAP => $((CAP + SEA_EXTRA))"; CAP=$((CAP + SEA_EXTRA)); }
# ★ 11-13 規格 5 §1 逐段計時模式（task_mech_time.sh 的 MT_TUNE_ARMS 用）：TUNE_PROFILE_STEPS＝訓練到這一步就停（排程仍照全長）、
#   CITYGS_STEP_PROFILE（穩態從 TUNE_PROFILE_WARMUP 步起）、不跑圖表資料包、跑次名＝TUNE_PROFILE_NAME（不進任何家族）
if [ -n "${TUNE_PROFILE_STEPS:-}" ]; then
  ENVX+=(CITYGS_STEP_PROFILE=1 CITYGS_STEP_PROFILE_WARMUP="${TUNE_PROFILE_WARMUP:-1550}" CITYGS_POSTDATA=0
         CITYGS_RUN_NAME="${TUNE_PROFILE_NAME:?TUNE_PROFILE_NAME}")
  XA+=(--trainer.max_steps "$TUNE_PROFILE_STEPS")
  echo "[profile] 逐段計時：訓練到 $TUNE_PROFILE_STEPS 步、穩態從 ${TUNE_PROFILE_WARMUP:-1550} 步起、跑次 $TUNE_PROFILE_NAME"
fi
if [ -n "${TUNE_RESUME_STEPS:-}" ]; then   # ★ 11-13 接續模式
  # ★ 11-14 TUNE_RESUME_SRC_ROOT（來源根，預設 lab）、TUNE_ROOT（輸出根，預設 lab）：本機可以接 lab 抓下來的 ckpt、輸出放 outputs/local
  _src=outputs/${TUNE_RESUME_SRC_ROOT:-lab}/${TUNE_RESUME_SRC:?TUNE_RESUME_SRC}/blocks/block_$B/checkpoints
  _ck=$(ls -t "$_src"/*.ckpt 2>/dev/null | grep -v xyz_rgb | head -1); [ -n "$_ck" ] || { echo "⛔ 缺 $_src 的 ckpt"; exit 2; }
  _st0=$(echo "$_ck" | grep -oE 'step=[0-9]+' | cut -d= -f2)
  _dst=outputs/${TUNE_ROOT:-lab}/${TUNE_RESUME_NAME:?TUNE_RESUME_NAME}/blocks/block_$B/checkpoints; mkdir -p "$_dst"
  [ "${CITYGS_DRY:-0}" = 1 ] || ln -f "$_ck" "$_dst/" 2>/dev/null || cp "$_ck" "$_dst/"
  ENVX+=(CITYGS_RESUME=1 CITYGS_POSTDATA=0 CITYGS_RUN_NAME=${TUNE_ROOT:-lab}/$TUNE_RESUME_NAME)
  XA+=(--trainer.max_steps $((_st0 + TUNE_RESUME_STEPS)))
  echo "[resume] 從 $_ck（step $_st0）接續 $TUNE_RESUME_STEPS 步 => lab/$TUNE_RESUME_NAME"
fi
echo "臂 $ARM => $LBL：全長 $ST、增生到 $((ST * G / 100))、合併到 $((ST * 9 / 10))、cap $CAP、家族 $FAM"
env "${ENVX[@]}" CITYGS_GRID=44 CITYGS_LEAN=1 STEPS=$ST CITYGS_CAP=$CAP CITYGS_DUP44=${CITYGS_DUP44:-2} CITYGS_FAMILY=$FAM \
  bash scripts/lab/task_cmp.sh "$B" $SUF "${NB[@]}" "${XA[@]}"
