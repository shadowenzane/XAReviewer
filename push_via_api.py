#!/usr/bin/env python3
"""通过 GitHub Git Database API 推送 cpp/ 目录到远端（绕过 HTTPS git 认证限制）。"""
import base64
import json
import subprocess
import sys

REPO = "shadowenzane/XAReviewer"
BRANCH = "main"


def gh_api(method, path, payload=None):
    """调用 gh api，payload 为 JSON 对象时走 --input。"""
    cmd = ["gh", "api", "--method", method, path]
    if payload is not None:
        cmd += ["--input", "-"]
        data = json.dumps(payload)
    else:
        data = None
    r = subprocess.run(cmd, input=data, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"gh api {method} {path} failed:\n{r.stderr}", file=sys.stderr)
        sys.exit(1)
    return json.loads(r.stdout) if r.stdout.strip() else {}


def main():
    # 1. 远端分支当前 commit 与 tree
    ref = gh_api("GET", f"repos/{REPO}/git/ref/heads/{BRANCH}")
    head_sha = ref["object"]["sha"]
    base_commit = gh_api("GET", f"repos/{REPO}/git/commits/{head_sha}")
    base_tree = base_commit["tree"]["sha"]
    print(f"remote HEAD: {head_sha}")

    # 2. 本地 cpp/ 下已跟踪文件列表（含新增的 .gitignore 之外全部内容）
    ls = subprocess.run(["git", "ls-files", "cpp/"], capture_output=True, text=True,
                        cwd="/workspace", check=True)
    files = [f for f in ls.stdout.splitlines() if f.strip()]
    print(f"{len(files)} files to push")

    # 3. 逐文件创建 blob
    tree_entries = []
    for f in files:
        with open(f"/workspace/{f}", "rb") as fp:
            content = base64.b64encode(fp.read()).decode()
        blob = gh_api("POST", f"repos/{REPO}/git/blobs",
                      {"content": content, "encoding": "base64"})
        tree_entries.append({"path": f, "mode": "100644", "type": "blob",
                             "sha": blob["sha"]})
        print(f"  blob ok: {f}")

    # 4. 创建 tree
    tree = gh_api("POST", f"repos/{REPO}/git/trees",
                  {"base_tree": base_tree, "tree": tree_entries})
    print(f"new tree: {tree['sha']}")

    # 5. 创建 commit（消息取本地最新提交）
    msg = subprocess.run(["git", "log", "-1", "--format=%B"], capture_output=True,
                         text=True, cwd="/workspace", check=True).stdout.strip()
    commit = gh_api("POST", f"repos/{REPO}/git/commits",
                    {"message": msg, "tree": tree["sha"], "parents": [head_sha]})
    print(f"new commit: {commit['sha']}")

    # 6. 更新分支引用
    gh_api("PATCH", f"repos/{REPO}/git/refs/heads/{BRANCH}",
           {"sha": commit["sha"], "force": False})
    print("PUSH OK")


if __name__ == "__main__":
    main()
