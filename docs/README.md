# SpecForge 文档

我们建议新贡献者从编写文档开始,这有助于您快速了解 SpecForge 代码库。
大多数文档文件位于 `docs/` 文件夹下。

## 文档工作流程

### 安装依赖

```bash
apt-get update && apt-get install -y pandoc parallel retry
pip install -r requirements.txt
```

### 更新文档

在 `docs/` 下的相应子目录中更新您的 Jupyter notebook。如果您添加了新文件,请记得相应地更新 `index.rst`(或相关的 `.rst` 文件)。

- **`pre-commit run --all-files`** 手动运行所有配置的检查,如果可能会应用修复。如果第一次失败,请重新运行以确保完全解决 lint 错误。在创建 Pull Request **之前**,确保您的代码通过所有检查。

```bash
# 1) 编译所有 Jupyter notebooks
make compile  # 此步骤可能需要很长时间(10分钟以上)。如果您能确保添加的文件是正确的,可以考虑跳过此步骤。
make html

# 2) 使用自动构建在本地编译和预览文档
# 这将在文件更改时自动重新构建文档
# 在浏览器中打开显示的端口以查看文档
bash serve.sh

# 2a) 提供文档的替代方式
# 直接使用 make serve
make serve
# 使用自定义端口
PORT=8080 make serve

# 3) 清除 notebook 输出
# nbstripout 会移除 notebook 输出,使您的 PR 保持整洁
pip install nbstripout
find . -name '*.ipynb' -exec nbstripout {} \;

# 4) Pre-commit 检查并创建 PR
# 这些检查通过后,推送您的更改并在您的分支上打开 PR
pre-commit run --all-files
```
---

## 文档风格指南

- 对于常见功能,我们更喜欢使用 **Jupyter Notebooks** 而不是 Markdown,这样所有示例都可以通过我们的文档 CI 流水线执行和验证。对于复杂功能(例如分布式服务),建议使用 Markdown。
- 编写交互式 Jupyter notebook 时请注意文档执行时间。每个交互式 notebook 都将针对每次提交运行和编译,以确保它们可运行,因此应用一些技巧来减少文档编译时间非常重要:
  - 在大多数情况下使用小型模型(例如 `qwen/qwen2.5-0.5b-instruct`)以减少服务器启动时间。
  - 尽可能重用已启动的服务器以减少服务器启动时间。
- 不要使用绝对链接(例如 `https://docs.sglang.ai/get_started/install.html`)。始终优先使用相对链接(例如 `../get_started/install.md`)。
- 参考现有示例学习如何启动服务器、发送查询和其他常见风格。
