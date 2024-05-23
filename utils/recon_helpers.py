import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

def setup_camera(w, h, k, w2c, near=0.01, far=100):
    # 解析相机内参，包括焦距(fx, fy)和中心点(cx, cy)
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    # 将世界到相机的变换矩阵(w2c)转换为PyTorch张量，并移到GPU上，转换为float类型
    w2c = torch.tensor(w2c).cuda().float()
    # 计算相机中心位置，通过取逆变换矩阵的平移部分
    cam_center = torch.inverse(w2c)[:3, 3]
    # 添加一个新维度并调整矩阵的维度顺序以适应后续的矩阵乘法
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    # 构造OpenGL兼容的投影矩阵
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0], # X轴的投影调整
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0], # Y轴的投影调整
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],   # Z轴的深度和透视缩放
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)  # 归一化w坐标
    
    # 计算最终的投影矩阵，将相机的世界变换矩阵与OpenGL投影矩阵结合
    full_proj = w2c.bmm(opengl_proj)
    
    # 初始化相机对象
    cam = Camera(
        image_height=h, # 图像高度
        image_width=w,  # 图像宽度
        tanfovx=w / (2 * fx),   # X方向的视场角的正切值
        tanfovy=h / (2 * fy),   # Y方向的视场角的正切值
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"), # 背景颜色，默认黑色
        scale_modifier=1.0, # 尺度调整因子，默认为1.0
        viewmatrix=w2c,     # 视图矩阵
        projmatrix=full_proj,   # 投影矩阵
        sh_degree=0,            # Spherical Harmonics光照度数，默认为0
        campos=cam_center,      # 相机位置
        prefiltered=False       # 预过滤标志，默认关闭
    )
    return cam  # 返回配置好的相机对象
