from internal.utils.stage_timer import mark  # ★ 11-14 啟動時間拆解
mark("python 啟動（import 前）")
from internal.entrypoints.gspl import cli
mark("import 完成")

if __name__ == "__main__":
    cli()
    mark("cli 結束（訓練與存檔完成，行程即將結束）")
    from internal.utils.stage_timer import write_summary
    write_summary()
