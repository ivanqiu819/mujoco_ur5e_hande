# UR5e 视觉插入仿真

MuJoCo 中的 UR5e + Robotiq Hand-E，带末端 RGB 相机。支持末端位姿调试，以及用 ArUco／无先验 PnP 定位插座后，携带预夹持插头水平插入 10 mm 并退出。

## GitHub 协作

首次克隆、创建环境和分支协作见 [CONTRIBUTING.md](CONTRIBUTING.md)。`environment.yml` 与 `requirements-tested.txt` 记录已验证的环境；`outputs/` 不在仓库中，运行检查后会重新生成。

## 安装与启动

在本目录执行一次安装；使用已有 `mujoco_ur5e` Conda 环境：

```bash
conda activate mujoco_ur5e
python -m pip install -e .
```

终端 A 启动服务端，自动打开一个 MuJoCo Viewer 和实时 RGB 窗口：

```bash
python apps/server.py
```

终端 B 执行完整插入流程（默认 ArUco）：

```bash
python apps/insert_socket.py
# 切换到 PnP
python apps/insert_socket.py --route pnp
```

每次完整流程从预夹持 Home 开始。正常退出后，会经过本轮记录的观察位姿、抬升位姿返回 Home；出现 `INSERTION_PHASE: home` 后可以直接运行另一条路线，无需重启服务端。整个回程重新经过 IK、碰撞预检和动力学接触检查。

如果你刚运行过旧版本，机器人仍停在孔前，请先重启一次服务端回到 Home，再运行更新后的程序。未知起点不会自动回 Home。

识别、路径或动力学失败会停止。中断插入后，重新连接并显式退出：

```bash
python apps/insert_socket.py --retract-only
```

`--retract-only` 仅退到孔前，不自动执行完整回程；之后重新启动服务端才能开始新的完整流程。需要逐步确认时使用 `--step`，包含最后的返回 Home 步骤。每步按 Enter 继续，输入 `q` 或 Ctrl+C 停止。

## 调试末端位姿

终端 B 改为运行下面的程序；插入程序和调试程序一次只允许一个连接：

```bash
python apps/debug_pose.py
```

常用命令：

```text
pose
status
tcp-rel 0 0 0.005
rotate-rel 0 0 2
tcp 0.30 0.12 0.36
tcp X Y Z QW QX QY QZ
keymode
stop
quit
```

长度为米，旋转增量为度，四元数为 WXYZ。输入 `help` 查看键盘微调和其他命令。普通夹爪调试需启动检测场景：

```bash
python apps/server.py --scene scenes/scene_inspection.xml
```

该场景可以使用 `gripper gap 20`、`gripper open`、`gripper close`。插入场景的插头已固定夹持，禁止改变夹距。

## 固定配置文件

| 文件 | 填写内容 |
|---|---|
| `configs/camera.json` | 相机内参、分辨率、安装关系、预览频率、检测超采样 |
| `configs/port.json` | STL 与单位、端口几何、定位特征、材质、标记安装关系 |
| `configs/insertion.json` | 默认路线、工件布局、插头、观察位姿、距离和速度 |
| `configs/vision/aruco.json` | ArUco 检测与位姿求解参数 |
| `configs/vision/pnp.json` | PnP 特征检测与位姿求解参数 |
| `configs/inspection.json` | 普通检测场景的桌子与工件 |

相对资源路径以配置文件所在目录解析。修改实体、相机或标记安装参数后，重新生成场景并重启服务端：

```bash
python tools/build_scenes.py
```

修改路线及算法参数无需重新生成场景。旧场景与新实体配置不匹配时，插入程序拒绝启动。换用其他 CAD 时必须同步定义特征与碰撞通道；当前碰撞生成器校验矩形孔壁，不支持任意孔型。

### 插入速度

只需修改 `configs/insertion.json` 中的 `insert_speed_m_s`，单位为米/秒。例如 `0.005` 是 5 mm/s，`0.015` 是 15 mm/s。当前配置为 15 mm/s，同时用于插入和退出；每轮启动会打印 `INSERT_SPEED`。

配置和执行器共用 `src/ur5e_sim/control/limits.py` 中的速度校验，当前允许 `0 < 速度 ≤ 0.020 m/s`（20 mm/s）。无需再修改多处代码；普通调速在下一轮任务生效，不必重建场景。首次从旧的 5 mm/s 限速代码升级后，需要重启服务端一次。

该参数是仿真时间内的轨迹峰值上限。分段加减速、实际到位等待和 GUI 运行速度会影响总耗时，整体平均速度会低于设置值。

## 相机与后台运行

RGB 来自服务端状态快照，窗口显示帧号、仿真时间和实际帧率。预览目标 15 Hz；检测使用三倍超采样后缩小的 1280×1072 RGB，图像与拍摄外参绑定。关闭 RGB 窗口不停止机器人。重新打开：

```bash
python apps/preview.py
# 关闭预览窗口
python apps/preview.py --close
```

预览命令使用独立的只读相机通道，不占用运动控制连接。

```bash
python apps/server.py --no-preview   # 仅关闭自动 RGB 窗口
MUJOCO_GL=egl python apps/server.py --headless  # 无图形界面，仍可拍照检测
```

## 验证与目录

```bash
PORT_POSE_FULL_TESTS=1 MPLBACKEND=Agg python -m unittest discover -s tests -v
python tools/check_control_server.py
python tools/check_inspection_scene.py
python tools/check_tcp.py --scene scenes/scene_inspection.xml --offset 0 0 0.01 --check-only
MUJOCO_GL=egl python tools/check_insertion_vision.py
python tools/check_insertion.py
python tools/check_protocol.py
python tools/check_repeat_insertion.py  # 同一服务端连续运行默认 ArUco 和 PnP
# 先关闭其他 Viewer，再验收图形界面
python tools/check_insertion.py --route aruco --viewer --output-dir outputs/gui
```

`apps/` 是用户入口，`src/ur5e_sim/` 是运行模块，`tools/` 是构建和检查工具，`tests/` 是回归测试，`scenes/` 保存可复现的五层场景，`assets/` 保存实际使用的网格。算法已内置，不再依赖旁边的 `port_pose_simulation`。

运行图像、诊断和轨迹集中放在 `outputs/`。旧代码和历史输出已移到项目外；归档位置与清理清单见 [归档说明](docs/ARCHIVE.md)。模块职责、接口和限制见 [架构说明](docs/ARCHITECTURE.md)，实际验收见 [验收记录](docs/VALIDATION.md)。
