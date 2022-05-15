import torch

NOMAP_22_JOINT2IND = {'hips' : 0, 'leftUpLeg' : 1, 'rightUpLeg' : 2, 'spine' : 3, 'leftLeg' : 4, 'rightLeg' : 5,
                'spine1' : 6, 'leftFoot' : 7, 'rightFoot' : 8, 'spine2' : 9, 'leftToeBase' : 10, 'rightToeBase' : 11, 
                'neck' : 12, 'leftShoulder' : 13, 'rightShoulder' : 14, 'head' : 15, 'leftArm' : 16, 'rightArm' : 17,
                'leftForeArm' : 18, 'rightForeArm' : 19, 'leftHand' : 20, 'rightHand' : 21}

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert 3x4 rotation matrix to 4d quaternion vector
    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201
    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.
    Return:
        Tensor: the rotation in quaternion
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`
    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()
    
    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q

def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert quaternion vector to angle axis of rotation.
    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h
    Args:
        quaternion (torch.Tensor): tensor with quaternions.
    Return:
        torch.Tensor: tensor with angle axis of rotation.
    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`
    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis

def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert 3x4 rotation matrix to Rodrigues vector
    Args:
        rotation_matrix (Tensor): rotation matrix.
    Returns:
        Tensor: Rodrigues vector transformation.
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`
    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3,3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32,
                           device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa

def gen_obj_from_motion_seq(root_orient, pose_body, trans, betas, bm, device):
    pred_world_root_orient = torch.tensor(root_orient, device=device)
    T, _ = pred_world_root_orient.size()
    J = len(NOMAP_22_JOINT2IND)
    pred_world_root_orient = rotation_matrix_to_angle_axis(pred_world_root_orient.reshape((T, 3, 3))).reshape((T, 3))
    
    pred_world_pose_body = torch.tensor(pose_body, device=device)
    pred_world_pose_body = rotation_matrix_to_angle_axis(pred_world_pose_body.reshape((T*(J-1), 3, 3))).reshape((T, (J-1)*3))
    pred_world_trans = torch.tensor(trans, device=device)
    betas = torch.tensor(betas, device=device)

    body_pred = bm(
        pose_body=pred_world_pose_body,
        pose_hand=None,
        betas=betas.reshape(1, -1).expand((T, 16)),
        root_orient=pred_world_root_orient,
        trans=pred_world_trans
    )
    pred_smpl_joints = body_pred.Jtr[:, :J]
    human_verts = body_pred.v.data.cpu().numpy()
    human_faces = body_pred.f

    return human_verts, human_faces

def write_to_obj(dest_obj_file, vertices, faces):
    # vertices: N X 3, faces: N X 3
    w_f = open(dest_obj_file, 'w')

    # Write vertices to file 
    for idx in range(vertices.shape[0]):
        w_f.write("v "+str(vertices[idx, 0])+" "+str(vertices[idx, 1])+" "+str(vertices[idx, 2])+"\n")

    # Write faces to file 
    for idx in range(faces.shape[0]):
        w_f.write("f "+str(faces[idx, 0]+1)+" "+str(faces[idx, 1]+1)+" "+str(faces[idx, 2]+1)+"\n")

    w_f.close()

