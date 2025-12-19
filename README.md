# Equivariant Potentials Collection (等变势能模型合集)

这是一个用于收集和归档各类等变神经网络势能（Equivariant Neural Network Potentials）官方代码实现的仓库。本仓库仅作为容器（Container），通过 **Git Submodule** 管理第三方代码，不包含任何自定义开发代码。

## 📂 当前收录 (Included Models)

所有模型源码均位于 `models/` 目录下：

* **[nequix](https://github.com/atomicarchitects/nequix)**: NequIP 的 JAX 实现版本。
* **[equiformer_v2](https://github.com/atomicarchitects/equiformer_v2)**: 基于 Transformer 的等变网络。
* **[fairchem](https://github.com/facebookresearch/fairchem)**: Meta (Facebook) 的化学 AI 库 (含 OCP)。
* **[reaxnet](https://github.com/reaxnet/reaxnet)**: 包含反应力场的网络。
* **[nequip](https://github.com/mir-group/nequip)**: 原始 PyTorch 版 NequIP。

---

## 🚀 常用操作指南 (Cheatsheet)

### 1. 克隆本仓库 (Clone)

**⚠️ 注意**：因为使用了 Submodule，普通的 clone 命令下载下来的 `models` 目录是空的。

**正确方式：**
```bash
git clone --recursive git@github.com:OutisLi/equiv-zoo
```

**补救方式：**
如果已经普通 clone 了（发现子文件夹为空），请运行：

```bash
git submodule update --init --recursive
```

---

### 2. 更新模型 (Update)

当原作者（如 atomicarchitects 或 facebookresearch）更新了代码，你想把本地的 submodule 同步到最新版：

**一键更新所有模型：**

```bash
# 这会将所有子模块拉取到其远程分支的最新 commit
git submodule update --remote --merge
```

**只更新特定模型（例如只更新 fairchem）：**

```bash
cd models/fairchem
git checkout main    # 确保切换到主分支
git pull origin main # 拉取更新
cd ../..             # 回到根目录
git add models/fairchem
git commit -m "chore: update fairchem to latest version"
```

---

### 3. 添加新模型 (Add New)

如果你发现了新的感兴趣的仓库，想加入到这个合集：

```bash
# 语法: git submodule add <URL> models/<文件夹名>
git submodule add [https://github.com/example/new-model.git](https://github.com/example/new-model.git) models/new-model

# 提交更改
git commit -m "feat: add new-model submodule"
```

---

### 4. 删除模型 (Remove)

如果某个模型不再需要：

```bash
# Git 会自动处理 .gitmodules 和文件删除
git rm models/obsolete-model
git commit -m "chore: remove obsolete-model"
```

---

## ⚠️ 环境依赖说明 (Dependency Warning)

由于不同模型的开发时间跨度和框架不同（PyTorch vs JAX, 不同 CUDA 版本），**请勿尝试在一个 Python 环境中安装所有模型**。

建议为每个模型创建独立的 Conda 环境。

具体安装依赖请参考各 `models/xxx/README.md` 中的官方说明。