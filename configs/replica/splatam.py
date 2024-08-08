import os
from os.path import join as p_join

# scenes = ["room0", "room1", "room2",
#           "office0", "office1", "office2",
#           "office_", "office4"]

scenes = ["room0"]

primary_device="cuda:0"
seed = 0
scene_name = scenes[0]

map_every = 1
keyframe_every = 5
mapping_window_size = 24
tracking_iters = 40
mapping_iters = 60

group_name = "Replica"
run_name = f"{scene_name}_{seed}"

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name, # 设置运行名称
    seed=seed,      # 设置随机种子
    primary_device=primary_device,  # 设置主要设备（如GPU）
    map_every=map_every,            # 设置每隔多少帧进行一次地图更新
    keyframe_every=keyframe_every,  # 设置每隔多少帧保存一个关键帧
    mapping_window_size=mapping_window_size,    # 设置地图更新窗口大小
    report_global_progress_every=500,   # 设置每隔多少帧报告全局进度
    eval_every=5,   # 设置每隔多少帧进行一次评估（在SLAM结束时）
    scene_radius_depth_ratio=3,         # 设置最大第一帧深度与场景半径的比率（用于修剪/密化）
    mean_sq_dist_method="projective",   # 设置高斯尺度的均方距离计算类型，可以是“投影”或“最近邻”
    # 设置高斯分布类型，可以是“各向同性”或“各向异性”
    gaussian_distribution="isotropic", # ["isotropic", "anisotropic"] (Isotropic -> Spherical Covariance, Anisotropic -> Ellipsoidal Covariance)
    report_iter_progress=False, # 是否报告每次迭代的进度
    load_checkpoint=False,      # 是否加载检查点
    checkpoint_time_idx=0,      # 设置检查点时间索引
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=100, # Checkpoint Interval
    use_wandb=True,
    wandb=dict(
        entity="theairlab",
        project="SplaTAM",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        basedir="/home/nerf/datav/Dataset/Replica",
        gradslam_data_cfg="./configs/data/replica.yaml",
        sequence=scene_name,    # 场景名称
        desired_image_height=680,   # 期望的图像高度
        desired_image_width=1200,   # 期望的图像宽度
        start=0,    # 数据集起始帧
        end=-1,     # 数据集结束帧，-1表示直到最后一帧
        stride=1,   # 帧步长
        num_frames=-1,  # 使用的帧数，-1表示使用所有帧
    ),
    tracking=dict(
        use_gt_poses=False, # 是否使用 gt 姿态进行追踪
        forward_prop=True,  # Forward Propagate Poses
        num_iters=tracking_iters,   # 追踪迭代次数
        use_sil_for_loss=True,      # 是否使用轮廓损失
        sil_thres=0.99,             # 轮廓阈值
        use_l1=True,                # 是否使用L1损失
        ignore_outlier_depth_loss=False,    # 是否忽略异常深度损失
        loss_weights=dict(
            im=0.5,     # 图像损失权重
            depth=1.0,  # 深度损失权重
        ),
        lrs=dict(
            means3D=0.0,        # 3D均值学习率
            rgb_colors=0.0,     # RGB颜色学习率
            unnorm_rotations=0.0,   # 非标准化旋转学习率
            logit_opacities=0.0,    # logit不透明度学习率
            log_scales=0.0,         # log尺度学习率
            cam_unnorm_rots=0.0004, # 相机非标准化旋转学习率
            cam_trans=0.002,        # 相机平移学习率
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,    # mapping迭代次数
        add_new_gaussians=True,     # 是否添加新的高斯分布
        sil_thres=0.5, # For Addition of new Gaussians 添加新高斯分布时的轮廓阈值
        use_l1=True,
        use_sil_for_loss=False,     # 是否在损失计算中使用轮廓
        ignore_outlier_depth_loss=False,    # 是否忽略异常深度损失
        loss_weights=dict(
            im=0.5,     # 图像损失权重
            depth=1.0,  # 深度损失权重
        ),
        lrs=dict(
            means3D=0.0001,     # 3D均值学习率
            rgb_colors=0.0025,  # RGB颜色学习率
            unnorm_rotations=0.001, # 非标准化旋转学习率
            logit_opacities=0.05,   # logit不透明度学习率
            log_scales=0.001,       # log尺度学习率
            cam_unnorm_rots=0.0000, # 相机非标准化旋转学习率
            cam_trans=0.0000,       # 相机平移学习率
        ),
        prune_gaussians=True, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=0,      # 在第几次迭代后开始修剪
            remove_big_after=0, # 在第几次迭代后移除大的高斯分布
            stop_after=20,      # 在第几次迭代后停止修剪
            prune_every=20,     # 每隔多少次迭代进行一次修剪
            removal_opacity_threshold=0.005,        # 移除高斯分布的不透明度阈值
            final_removal_opacity_threshold=0.005,  # 最终移除高斯分布的不透明度阈值
            reset_opacities=False,      # 是否重置不透明度
            reset_opacities_every=500,  # 每隔多少次迭代重置不透明度 Doesn't consider iter 0
        ),
        use_gaussian_splatting_densification=False, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=500,        # 在第几次迭代后开始密化
            remove_big_after=3000,  # 在第几次迭代后移除大的高斯分布
            stop_after=5000,        # 在第几次迭代后停止密化
            densify_every=100,      # 每隔多少次迭代进行一次密化
            grad_thresh=0.0002,     # 梯度阈值
            num_to_split_into=2,    # 每个高斯分布分裂成的数量
            removal_opacity_threshold=0.005,        # 移除高斯分布的不透明度阈值
            final_removal_opacity_threshold=0.005,  # 最终移除高斯分布的不透明度阈值
            reset_opacities_every=3000,             # 每隔多少次迭代重置不透明度 Doesn't consider iter 0
        ),
    ),
    viz=dict(
        render_mode='color',        # 渲染模式，可以是'color'（颜色）、'depth'（深度）或'centers'（中心点）
        offset_first_viz_cam=True,  # 将视图相机沿视图方向向后偏移0.5个单位（用于最终重建可视化）
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=True, # Visualize Camera Frustums and Trajectory
        viz_w=600, viz_h=340,           # 可视化宽度、高度
        viz_near=0.01, viz_far=100.0,   # 近、远裁剪面距离
        view_scale=2,   # 视图缩放比例
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=True, # Enter Interactive Mode after Online Recon Viz
    ),
)