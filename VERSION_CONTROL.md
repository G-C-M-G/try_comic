# 版本控制说明

项目采用两个 Git 仓库，避免把上游引擎历史和本项目 GUI 混在一起：

- 根目录仓库：GUI、翻译编排、工作台、缓存与文档。
- `manga-image-translator`：上游 MITS 引擎及本项目补丁，独立维护在 `local/try-comic` 分支。

当前引擎基线：上游 `95227a2`，本地补丁提交 `d7dbd60`。根仓库还保存了可移植补丁：`patches/0001-feat-preserve-local-translation-pipeline-customizati.patch`。

大型 ZIP 备份、虚拟环境、模型、缓存、翻译输出和运行期用户数据都由根目录 `.gitignore` 排除；这些文件没有被删除。

日常检查：

```powershell
git status --short --branch
git -C manga-image-translator status --short --branch
```

查看最近版本：

```powershell
git log --oneline --decorate -10
git -C manga-image-translator log --oneline --decorate -10
```

恢复单个受控文件前先看差异：

```powershell
git diff -- path\to\file
git restore -- path\to\file
```

不要用 `git reset --hard` 代替检查差异。根仓库不追踪嵌套引擎目录；修改引擎代码后，需要在 `manga-image-translator` 仓库单独提交。

若以后重新克隆了上游引擎，可从上游基线恢复本项目补丁：

```powershell
git -C manga-image-translator switch -c local/try-comic 95227a2
git -C manga-image-translator am ..\patches\0001-feat-preserve-local-translation-pipeline-customizati.patch
```
