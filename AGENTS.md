# 项目维护说明

- 使用中文。先读 README.md、docs/ARCHITECTURE.md、docs/VALIDATION.md。
- 运行环境为 Conda mujoco_ur5e，MuJoCo 3.14；未激活环境时用 conda run。
- 运行代码仅在 src/ur5e_sim/；apps/ 是薄入口；不得从测试或工具模块导入运行逻辑。
- 旧文件已移到项目外，说明见 docs/ARCHIVE.md；本机位置另记于被忽略的 docs/ARCHIVE.json。不要恢复过时脚本或外部算法目录依赖。
- 几何、相机和任务参数以 configs/ 为准。修改实体参数后用 tools/build_scenes.py 重建；不要只改生成 XML。
- 长度米、四元数 WXYZ、T_A_FROM_B 表示 B 到 A。保留所有碰撞、状态新鲜度、歧义、断线和恢复保护。
- 修改前后运行相关检查。完整验收命令见 README；GUI 最后单独验证，只启动一个 MuJoCo Viewer。
- 记录测试真实结果。最初交接提到的 check_model.py 在接管时缺失，不把替代检查冒充它。
- 当前范围不包含真实机器人、拾取、摩擦夹持、全局规划或力控。
