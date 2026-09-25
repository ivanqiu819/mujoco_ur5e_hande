# 与朋友一起开发

## 获取项目

私有仓库需要先由仓库所有者在 GitHub 的 Settings → Collaborators 中邀请朋友，朋友接受邀请后再克隆。邀请通过 GitHub 网页完成；本项目不会自动发送邀请。

```bash
git clone https://github.com/ivanqiu819/mujoco_ur5e_hande.git
cd mujoco_ur5e_hande
conda env create -f environment.yml
conda activate mujoco_ur5e
```

已有同名环境时，激活环境后执行 `python -m pip install -r requirements-tested.txt -e .`。模型已包含在仓库中，不需要 Git LFS，也不依赖原作者的外部算法项目。

首次使用 Git，在自己的机器设置提交身份（使用你自己的名字和邮箱）：

```bash
git config --global user.name "你的名字"
git config --global user.email "你的 GitHub 邮箱或隐私邮箱"
```

## 分支、提交与合并

每个功能建一个分支，通过 Pull Request 合并到 `main`：

```bash
git switch main
git pull --ff-only
git switch -c feature/改动名称

# 修改后先查看差异，再明确选择需要提交的文件
git status
git diff
git add src/ur5e_sim/你修改的文件.py configs/你修改的配置.json
git commit -m "描述这个改动解决的问题"
git push -u origin HEAD
```

然后在 GitHub 打开 Pull Request，让朋友检查后合并。下一项工作从最新的 `main` 创建新分支；不要用强制推送覆盖朋友的历史。

## 修改位置与检查

- 新视觉路线写在 `src/ur5e_sim/vision/`，新任务写在 `tasks/`，用户入口写在 `apps/`。
- 相机、插口、速度等参数修改 `configs/`。实体参数变化后运行 `python tools/build_scenes.py`，将生成场景及相关网格一并提交。
- 保留碰撞、快照有效性、歧义、Home 起点和断线恢复检查。
- 功能提交应记录实际测试结果；没有测试的 GUI 行为要注明。

```bash
# 算法与保护测试；无需启动 Viewer
PORT_POSE_FULL_TESTS=1 MPLBACKEND=Agg python -m unittest discover -s tests -v

# 控制回归
python tools/check_control_server.py

# 改动运动/插入流程后，验证同一服务端连续运行两条路线
python tools/check_repeat_insertion.py
```

`outputs/`、缓存、虚拟环境和本机归档位置不提交。保留必要的 `.obj` 网格；它们是模型资源，不能按编译产物忽略。历史验收摘要在 `docs/VALIDATION.md`，原运行输出需要自行重新生成。

仓库默认私有，当前没有添加开源许可证；共享仓库不等于给所有代码和 CAD 资源重新授予开源许可。

参考：[GitHub 分支协作流程](https://docs.github.com/en/get-started/using-github/github-flow)、[GitHub CLI 登录](https://cli.github.com/manual/gh_auth_login)。
