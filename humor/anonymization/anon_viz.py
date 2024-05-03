import trimesh
import pyrender
import numpy as np
from os.path import join

from PIL import Image


def estimate_root_trans(pose_seq_list):
    pose_trans = 0
    new_pose_list = []
    new_pose_list.append(pose_seq_list[0].reshape(-1, 3))
    for i in range(1, len(pose_seq_list)):
        pose = pose_seq_list[i].reshape(-1, 3)
        prior_pose = pose_seq_list[i - 1].reshape(-1, 3)

        # The 17 point pose is CeTI-Locomotion and has the right foot at a different index
        if pose.shape[0] == 17:
            rigth_foot_motion = abs(pose[5][2] - prior_pose[5][2])
        else:
            rigth_foot_motion = abs(pose[8][2] - prior_pose[8][2])

        pose_trans += rigth_foot_motion

        tmp = np.copy(pose)
        tmp[:,2] += pose_trans
        new_pose_list.append(tmp)

    return np.array(new_pose_list).reshape(len(new_pose_list), -1)


def render_pose_seq(pose_seq_list, path, width=640, height=480, fps=30, center_camera=False, invert_camera=False):


    colors = {
        'pink': [.7, .7, .9],
        'purple': [.9, .7, .7],
        'cyan': [.7, .75, .5],
        'red': [1.0, 0.0, 0.0],

        'green': [.0, 1., .0],
        'yellow': [1., 1., 0],
        'brown': [.5, .7, .7],
        'blue': [.0, .0, 1.],

        'offwhite': [.8, .9, .9],
        'white': [1., 1., 1.],
        'orange': [.5, .65, .9],

        'grey': [.7, .7, .7],
        'black': np.zeros(3),
        'white': np.ones(3),

        'yellowg': [0.83, 1, 0],
    }

    pose_seq_list = np.array(pose_seq_list)

    scene = pyrender.Scene(bg_color=colors['black'], ambient_light=(1, 1, 1))

    # Creates the camera
    # Camera position
    if center_camera:
        tmp = pose_seq_list[0].reshape(len(pose_seq_list[0]), -1, 3)
        cam_pos = [tmp[:,:2,0].mean(), tmp[:,:2,1].mean(), 3.5]
    else:
        cam_pos = [len(pose_seq_list)-1.0, 0.2, 3.0 + len(pose_seq_list)]
    default_cam_pose = np.eye(4)
    default_cam_pose[:3, 3] = np.array(cam_pos)
    # Rotate the camera around the z-axis to make the
    if invert_camera:
        rotate = trimesh.transformations.rotation_matrix(angle=np.radians(180.0), direction=[0, 0, 1], point=cam_pos)
    else:
        rotate = trimesh.transformations.rotation_matrix(angle=np.radians(0.0), direction=[0, 0, 1], point=cam_pos)
    default_cam_pose = np.dot(default_cam_pose, rotate)
    pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    scene.add(pc, pose=default_cam_pose, name='pc-camera')

    # Adds a light
    light = pyrender.PointLight(color=np.ones(3), intensity=1000.0)
    scene.add(light, pose=default_cam_pose)


    # Adds a sphere for every pose
    spheres_list = []

    color_list = [[colors["white"], [0.0,1.0,0.0], [0.0,0.0,1.0]]]

    for k in range(len(pose_seq_list)):
        pose_seq = pose_seq_list[k]
        spheres_list.append([])
        sm = trimesh.creation.uv_sphere(radius=0.025)
        sm.visual.vertex_colors = color_list[k]
        m = pyrender.Mesh.from_trimesh(sm)
        for i in range(len(pose_seq[0]) // 3):
            pose = np.eye(4)
            # Translates the sphere to match the joint position plus an specific offset for the given pose_seq
            pose[:3, 3] = np.array(pose_seq[0][i*3 : i*3+3] + np.array([k*2,0, 0]))

            nlp = pyrender.Node(mesh=m, matrix=pose)
            scene.add_node(nlp)
            spheres_list[k].append(nlp)

    offscreen = True

    if offscreen:
        r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)

        rendered_images = []

        for frame_i in range(len(pose_seq_list[0])):
            # Update the position for all spheres for all sequences
            for k in range(len(pose_seq_list)):
                pose_seq = pose_seq_list[k]
                spheres = spheres_list[k]
                frame = pose_seq[frame_i]

                # Update all spheres for a given sequence frame
                for i in range(len(spheres)):
                    pose = np.eye(4)
                    pose[:3, 3] = frame[i * 3:i * 3 + 3] + np.array([k * 2, 0, 0])
                    scene.set_pose(spheres[i], pose)

            color, depth = r.render(scene)
            color = color.reshape(width * height, 3)
            tmp_im = Image.frombytes("RGB", (width, height), color.tobytes())
            rendered_images.append(tmp_im)

        frame_one = rendered_images[0]
        frame_one.save(path, format="GIF", append_images=rendered_images[1:],
                   save_all=True, duration=1000 / fps, loop=0)



def anon_renders_for_given_sample(sample, dataset, noise_para_list):

    in_path = "../data/"
    out_path = "out/anon_renders/"

    for fitting in ["humor_fitting"]: #"vposer_fitting",
        for anon in ["humor", "vposer", "direct"]:
            for noise_para in noise_para_list:
                sample_in_path = join(in_path, "anon", dataset, fitting, anon, "normal_" + str(noise_para), sample)
                data = np.load(sample_in_path)
                positions = data["positions"]
                file_name = dataset + "_" + fitting + "_" + anon + "_" + noise_para + ".gif"
                render_pose_seq([positions], join(out_path, file_name))


    for noise_para in noise_para_list:
        sample_in_path = join(in_path, "anon", dataset, "original_fitting", "direct", "normal_" + str(noise_para), sample)
        data = np.load(sample_in_path)
        if len(data["positions"].shape) == 3:
            positions = data["positions"][0]
        file_name = dataset + "_original_fitting_direct_" + noise_para + ".gif"
        render_pose_seq([positions], join(out_path, file_name))



if __name__=='__main__':
    import sys, os

    cur_file_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(cur_file_path, '..'))

    #in_path = "/home/simon/DuD/research/pantomime/code/humor/out/horst_fitting/data/original/Horst_Study/S34_0006_Gait.tsv_17169822480"
    #in_path = "/home/simon/DuD/research/pantomime/code/humor/out/ceti_fitting/results_out/sub-K88_task-gaitNormalBackpack_tracksys-RokokoSmartSuitPro1_run-1_step-2_motion.tsv_17151703330"
    #in_path = "/home/simon/DuD/research/pantomime/code/humor/out/horst_fitting/data/original/Horst_Study/S34_0006_Gait.tsv_17174054730"
    #in_path = "/home/simon/DuD/research/pantomime/code/humor/out/horst_fitting/data/original/Horst_Study/S34_0006_Gait.tsv_17174144380"
    #in_path = "/home/simon/DuD/research/pantomime/code/humor/out/horst_fitting/data/original/Horst_Study/S34_0006_Gait.tsv_17174183790"
    in_path = "/home/simon/DuD/research/pantomime/code/humor/out/horst_fitting_reduced/data/original/Horst_Study/S03_0009_Gait.tsv_17191860020"
    in_path = "/home/simon/DuD/research/pantomime/code/humor/out/horst_fitting_reduced/data/original/Horst_Study/S04_0007_Gait.tsv_17192873490"
    #in_path = "/home/simon/DuD/research/pantomime/code/humor/out/horst_fitting_reduced/data/original/Horst_Study/S07_0008_Gait.tsv_17193843500"
    #in_path = "/home/simon/DuD/research/pantomime/code/humor/out/ceti_fitting_only_one/results_out/data/CeTI-Locomotion/derivatives/cut_sequences_only_one/sub-K0_task-gaitFast_tracksys-RokokoSmartSuitPro1_run-1_step-19_motion.tsv_17240564670"
    #in_path = "/home/simon/DuD/research/pantomime/code/humor/out/ceti_fitting_only_one/results_out/data/CeTI-Locomotion/derivatives/cut_sequences_only_one/sub-K0_task-gaitFast_tracksys-RokokoSmartSuitPro1_run-1_step-19_motion.tsv_17240590640"
    #in_path = "/home/simon/DuD/research/pantomime/code/humor/out/ceti_fitting_only_one/results_out/data/CeTI-Locomotion/derivatives/cut_sequences_only_one/sub-K0_task-gaitFast_tracksys-RokokoSmartSuitPro1_run-1_step-19_motion.tsv_17240580900"
    #n_path = "/home/simon/DuD/research/pantomime/code/humor/out/ceti_fitting_only_one/results_out/data/CeTI-Locomotion/derivatives/cut_sequences_only_one/sub-K0_task-gaitFast_tracksys-RokokoSmartSuitPro1_run-1_step-19_motion.tsv_17240661860"
    in_path = "/home/simon/DuD/research/pantomime/code/humor/out/ceti_fitting_only_one/results_out/data/CeTI-Locomotion/derivatives/cut_sequences_only_one/sub-K0_task-gaitFast_tracksys-RokokoSmartSuitPro1_run-1_step-19_motion.tsv_17240673180"

    in_path = "/home/simon/DuD/research/pantomime/code/data/prepared/CeTI-Locomotion/"
    in_path_ = "/home/simon/DuD/research/pantomime/code/data/prepared/Horst-Study_hyper_opt/"
    in_path = "/home/simon/DuD/research/pantomime/code/data/prepared/Horst-Study/"


    #in_path = "/home/simon/DuD/research/pantomime/code/data/anon/CeTI-Locomotion/vposer_fitting/vposer/normal_10.0"


    #sample = "sub-K0.task-gaitNormalBackpack-tracksys-RokokoSmartSuitPro1-run-1-step-8-motion.npz"
    #anon_renders_for_given_sample(sample, "CeTI-Locomotion", ["0.05", "0.1", "0.2", "0.4", "1.0", "5.0", "10.0"])

    sample = "sub-K0.task-gaitFast-tracksys-RokokoSmartSuitPro1-run-1-step-12-motion.npz"
    sample = "S57.0020-Gait.npz"
    #anon_renders_for_given_sample(sample, "CeTI-Locomotion", ["0.05", "0.1", "0.2", "0.4", "0.6", "0.8", "1.5"])

    data = np.load(in_path + sample)
    positions_orig = data["original_positions"]
    positions_vposer = data["vposer_positions"]
    positions_humor = data["humor_positions"]


    #positions_with_trans = estimate_root_trans(np.copy(positions))


    #print(positions.shape)
    #print(positions_with_trans.shape)

    render_pose_seq([positions_orig], "original_positions.gif")
    render_pose_seq([positions_humor, positions_vposer], "fittings.gif")

    '''

    from os.path import join
    from anonymization.prepare_data import get_SMPL_joint_positions


    original = np.load(join(in_path, "observations.npz"))
    fitted_stage2 = np.load(join(in_path, "stage2_results.npz"))
    fitted = np.load(join(in_path, "stage3_results.npz"))

    print(list(original["joints3d"].shape))
    print(list(fitted.keys()))

    tmp = {}
    tmp2 = {}
    for ele in fitted:
        tmp[ele] = fitted[ele]
    for ele in fitted_stage2:
        tmp2[ele] = fitted_stage2[ele]
    fitted = tmp
    fitted_stage2 = tmp2

    fitted["betas"] = np.zeros(fitted["betas"].shape)
    fitted_stage2["betas"] = np.zeros(fitted_stage2["betas"].shape)
    print(fitted_stage2["betas"].shape)


    fitted_positions = get_SMPL_joint_positions(fitted).detach().numpy()
    fitted_positions_stage2 = get_SMPL_joint_positions(fitted_stage2).detach().numpy()

    print(list(fitted_positions.shape))

    #
    render_pose_seq([original["joints3d"].reshape(-1, 66), fitted_positions_stage2, fitted_positions], "fitting_ceti_reduced_markers_test8.gif")
    
    '''
