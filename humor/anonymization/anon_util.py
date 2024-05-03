import torch
from body_model.utils import SMPL_JOINTS

def format_positions_as_original(data, dataset):
    '''
    Takes the SMPL data and transforms the data into the same format as the original data was for direct comparison
    '''
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    CETI_JOINTS = {'hips': 0, 'leftUpLeg': 1, 'rightUpLeg': 4, 'spine': None, 'leftLeg': 2, 'rightLeg': 5,
                   'spine1': None, 'leftFoot': 3, 'rightFoot': 6, 'spine2': 7, 'leftToeBase': None,
                   'rightToeBase': None,
                   'neck': None, 'leftShoulder': 8, 'rightShoulder': 13, 'head': 12, 'leftArm': 9, 'rightArm': 14,
                   'leftForeArm': 10, 'rightForeArm': 15, 'leftHand': 11, 'rightHand': 16}

    # The SMPL JOINT Names in the correct order
    SMPL_JOINTS_NAMES = ['hips', 'leftUpLeg', 'rightUpLeg', 'spine', 'leftLeg', 'rightLeg', 'spine1', 'leftFoot',
                         'rightFoot', 'spine2', 'leftToeBase', 'rightToeBase', 'neck', 'leftShoulder', 'rightShoulder',
                         'head', 'leftArm', 'rightArm', 'leftForeArm', 'rightForeArm', 'leftHand', 'rightHand']


    selected_joints = []
    if dataset == "CeTI-Locomotion":
        for joint in SMPL_JOINTS_NAMES:
            if CETI_JOINTS[joint] is not None:
                selected_joints.append(SMPL_JOINTS[joint])

        selected_joints = torch.tensor(selected_joints).to(device)

    if torch.is_tensor(data):
        result_joints = data.clone().detach().to(device)
    else:
        result_joints = torch.tensor(data).to(device)

    result_joints = result_joints.reshape(-1, len(SMPL_JOINTS), 3)
    result_joints = torch.index_select(result_joints, 1, selected_joints).reshape(-1, len(selected_joints) * 3)
    return result_joints