#!/usr/bin/env python
"""掃出「寫了 config 欄位但沒有任何程式碼讀它」的死旗標。

起因（2026-08-26）：`err_guided_densify` 全 repo 只出現在自己的欄位定義與 docstring 裡，
**沒有任何一行程式碼讀它** => `egd_b12` 跑了 9.7 小時測一個不存在的機制。
（意外的好處是它變成同組態重跑，量到真正的噪音底 0.107，見 §11.29。）

做法：解析 `internal/` 底下所有 dataclass 的欄位名，然後在**去掉字串與註解**的原始碼裡
搜尋 `self.config.<name>` / `cfg.<name>` / `config.<name>` / `getattr(..., "<name>", ...)`。
⚠ 必須去掉 docstring，否則「只在 docstring 裡被提到」的死旗標會被誤判為活的。
"""
import ast
import glob
import io
import os
import re
import tokenize

ROOTS = ["internal", "utils", "tools"]


def strip_strings_and_comments(src: str) -> str:
    """把所有字串常量與註解換成空白 —— docstring 裡的提及不算「使用」。"""
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.STRING, tokenize.COMMENT):
                out.append(" ")
            else:
                out.append(tok.string)
            out.append(" ")
    except Exception:
        return src
    return " ".join(out)


def dataclass_fields(path):
    """回傳 [(類別名, 欄位名, 行號)]，只看有型別註記的類別層級指派。"""
    try:
        tree = ast.parse(open(path).read())
    except Exception:
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                out.append((node.name, stmt.target.id, stmt.lineno))
    return out


def main():
    files = [p for r in ROOTS for p in glob.glob(f"{r}/**/*.py", recursive=True)]
    raw = "\n".join(open(p).read() for p in files)
    code = "\n".join(strip_strings_and_comments(open(p).read()) for p in files)
    # ⚠ 2026-08-26 修正：清空字串會把 `getattr(self.config, "add_ratio", 1.05)` 的鍵一起清掉，
    # 於是所有用 getattr 存取的欄位都被誤判成死旗標（add_ratio / blur_split_* 全中）。
    # => 先從**原始碼**抽出所有 getattr 的字串鍵，當成「有被讀」。
    getattr_keys = set(re.findall(r'getattr\s*\([^,]+,\s*["\']([A-Za-z_]\w*)["\']', raw))
    code += " " + " ".join("." + k for k in getattr_keys)

    fields = []
    for p in files:
        for cls, name, ln in dataclass_fields(p):
            fields.append((p, cls, name, ln))

    dead, weak = [], []
    for p, cls, name, ln in fields:
        # 只算「被當成屬性讀」的形式；欄位定義本身（`name : type = ...`）不算
        pats = [rf"\.\s*{re.escape(name)}\b", rf'["\']{re.escape(name)}["\']']
        hits = sum(len(re.findall(pt, code)) for pt in pats)
        # 定義處本身會貢獻 0 次（因為是 `name:` 不是 `.name`），所以 0 = 完全沒被讀
        if hits == 0:
            dead.append((p, cls, name, ln))
        elif hits <= 1:
            weak.append((p, cls, name, ln, hits))

    print(f"掃描 {len(files)} 個檔案、{len(fields)} 個 dataclass 欄位\n")
    print(f"=== ⛔ 完全沒有被讀（死旗標）：{len(dead)} 個 ===")
    for p, cls, name, ln in sorted(dead):
        print(f"  {p}:{ln}  {cls}.{name}")
    print(f"\n=== ⚠ 只被讀 1 次（可能只是轉傳，需人工看）：{len(weak)} 個 ===")
    for p, cls, name, ln, h in sorted(weak):
        print(f"  {p}:{ln}  {cls}.{name}")
    print("""
⚠ 誤判來源：
  - 透過 `**kwargs` / `setattr` / jsonargparse 動態存取的欄位會被誤判為死的
  - 只在**別的檔案**用字串鍵存取的也可能漏
  => 判死之前一定要 `grep -rn '<name>' internal/` 人工確認一次。""")


if __name__ == "__main__":
    main()
