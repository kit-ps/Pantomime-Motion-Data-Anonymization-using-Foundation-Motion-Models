import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import numpy as np

import torch
from utils.logging import Logger, mkdir
from utils.transforms import rotation_matrix_to_angle_axis, batch_rodrigues

from body_model.utils import SMPL_JOINTS, KEYPT_VERTS, smpl_to_openpose

from body_model.body_model import BodyModel

from utils.torch import load_state

from models.humor_model import HumorModel

from fitting.fitting_utils import load_vposer

from fitting.config import parse_args


J_BODY = len(SMPL_JOINTS)-1 # no root

class MotionAnonymization():
    ''' Anonymize SMPL motion sequence '''

    def __init__(self, device,
                       args, # Config arguments
                       noise_scalar,
                       noise_type,
                       num_betas, # beta size in SMPL model
                       batch_size, # number of sequences to optimize
                       seq_len # length of the sequences
                ):

        B, T = batch_size, seq_len
        self.batch_size = B
        self.seq_len = T
        self.num_betas = num_betas
        self.noise_scalar = noise_scalar
        self.noise_type = noise_type
        self.pose_by_pose = args.pose_by_pose
        self.device = device


        Logger.log('Loading SMPL model from %s...' % (args.smpl))
        body_model_path = args.smpl
        self.fit_gender = body_model_path.split('/')[-2]
        self.body_model = BodyModel(bm_path=body_model_path,
                               num_betas=self.num_betas,
                               batch_size=self.batch_size * T,
                               use_vtx_selector=False).to(device)

        # must always have pose prior to optimize in latent space
        pose_prior, _ = load_vposer(args.vposer)
        self.pose_prior = pose_prior.to(device)

        self.motion_prior = None
        Logger.log('Loading motion prior from %s...' % (args.humor))
        self.motion_prior = HumorModel(in_rot_rep=args.humor_in_rot_rep,
                                  out_rot_rep=args.humor_out_rot_rep,
                                  latent_size=args.humor_latent_size,
                                  model_data_config=args.humor_model_data_config,
                                  steps_in=args.humor_steps_in)
        self.motion_prior.to(device)
        load_state(args.humor, self.motion_prior, map_location=device)



    def run_direct(self, data, attributes, generate_new_positions=False):
        full_seq = {}
        for ele in data:
            full_seq[ele] = data[ele]

        for attr in attributes:
            full_seq[attr] = self.anonymize_noise(data[attr], self.noise_scalar, self.noise_type, self.device,
                                               self.pose_by_pose)

        if generate_new_positions:
            smpl_results, _ = self.smpl_results(full_seq["trans"].float(), full_seq["root_orient"].float(), full_seq["pose_body"].float(),
                                                full_seq["betas"].float())
            full_seq["joints"] = smpl_results['joints3d'].reshape(-1, 22 * 3)

        return full_seq

    def run_vposer(self, data):

        B, T, _ = data["pose_body"].size()

        latent_pose = self.pose_prior.encode(data["pose_body"].reshape(-1, 63).float())

        latent_pose = latent_pose.mean.reshape((B, T, 32))

        latent_pose = self.anonymize_noise(latent_pose, self.noise_scalar, self.noise_type, self.device, self.pose_by_pose)

        latent_pose = latent_pose.reshape((-1, 32))

        anon_pose_body = self.pose_prior.decode(latent_pose, output_type='matrot')

        # Transform back into aa
        anon_pose_body = rotation_matrix_to_angle_axis(anon_pose_body.reshape((B*T*J_BODY, 3, 3)))
        anon_pose_body = anon_pose_body.reshape((B, T, J_BODY*3)).float()

        smpl_results, _ = self.smpl_results(data["trans"], data["root_orient"], anon_pose_body, data["betas"])
        joints = smpl_results['joints3d']

        full_seq = {"pose_body": anon_pose_body,
                    "trans": data["trans"],
                    "root_orient": data["root_orient"],
                    "betas": data["betas"]}

        # Remove the batch size
        for ele in full_seq:
            if len(full_seq[ele].shape) > 2:
                full_seq[ele] = full_seq[ele][0]

        full_seq["joints"] = joints.reshape(-1, 22*3)

        return full_seq


    def run_humor(self, data,
                  data_fps=30,
                  step_length=1,
                  fit_gender='neutral'):

        latent_motion, pre_seq = self.infer_latent_motion(data["trans"], data["root_orient"], data["pose_body"], data["betas"], data_fps=data_fps, full_forward_pass=False)

        pre_seq["root_orient"] = rotation_matrix_to_angle_axis(pre_seq["root_orient"].reshape((-1, 3, 3))).reshape((self.batch_size, self.seq_len, 3))
        pre_seq["pose_body"] = rotation_matrix_to_angle_axis(pre_seq["pose_body"].reshape((-1, 3, 3))).reshape((self.batch_size, self.seq_len, J_BODY*3))

        latent_motion = self.anonymize_noise(latent_motion, self.noise_scalar, self.noise_type, self.device, self.pose_by_pose)

        #TODO Calculate all mini sequences at the same time via batching.

        # TODO: Make an option to use the fitting z for anon
        #if "latent_motion" in data:
        #    latent_motion = data["latent_motion"]

        full_seq = {"pose_body": [],
                    "trans": [],
                    "root_orient": [],
                    "joints": []}

        for i in range(0, latent_motion.size()[1], step_length):
            # Fix step length for the last part of the sequence
            if i + step_length > latent_motion.size()[1]:
                step_length = latent_motion.size()[1] - i

            trans_in = pre_seq["trans"][:, i, :].reshape(self.batch_size, 1, -1)
            root_orient_in = pre_seq["root_orient"][:, i, :].reshape(self.batch_size, 1, -1)
            pose_body_in = pre_seq["pose_body"][:, i, :].reshape(self.batch_size, 1, -1)

            prior_opt_params = (pre_seq["trans_vel"][:, i, :].reshape(self.batch_size, 1, -1),
                                pre_seq["joints_vel"][:, i, :].reshape(self.batch_size, 1, -1),
                                pre_seq["root_orient_vel"][:, i, :].reshape(self.batch_size, 1, -1))

            red_latent_motion = latent_motion[:,i:i+step_length,:].reshape(self.batch_size, step_length, -1)

            tmp = self.rollout_latent_motion(trans_in, root_orient_in, pose_body_in, data["betas"], prior_opt_params,
                                             red_latent_motion, canonicalize_input=False)

            for ele in full_seq:
                full_seq[ele].append(tmp[ele].detach().cpu().numpy()[:,1:1+step_length])

        for ele in full_seq:
            full_seq[ele] = np.concatenate(full_seq[ele], axis=1)[0]

        full_seq["betas"] = data["betas"][0]
        full_seq["joints"] = full_seq["joints"].reshape(-1, 22*3)

        return full_seq



    def anonymize_noise(self, body_pose, scalar, noise_type, device, pose_by_pose=True):
        mean = 0.0
        std = 1.0

        if pose_by_pose:
            vector_size = body_pose.size()
        else:
            vector_size = body_pose.size()[-1]

        if noise_type == "normal":
            noise = np.random.default_rng().normal(mean, std, size=vector_size)
        if noise_type == "laplace":
            noise = np.random.default_rng().laplace(mean, std, size=vector_size)
        if noise_type == "uniform":
            noise = np.random.default_rng().uniform(low=0, high=1, size=vector_size)

        shift = torch.tensor(noise * scalar, device=device, dtype=torch.float)

        body_pose_anon = body_pose + shift

        return body_pose_anon


    def estimate_velocities(self, trans, root_orient, body_pose, betas, data_fps, smpl_results=None):
        '''
        From the SMPL sequence, estimates velocity inputs to the motion prior.

        - trans : root translation
        - root_orient : aa root orientation
        - body_pose
        '''
        B, T, _ = trans.size()
        h = 1.0 / data_fps
        if self.motion_prior.model_data_config in ['smpl+joints', 'smpl+joints+contacts']:
            if smpl_results is None:
                smpl_results, _ = self.smpl_results(trans, root_orient, body_pose, betas)
            trans_vel = self.estimate_linear_velocity(trans, h)
            joints_vel = self.estimate_linear_velocity(smpl_results['joints3d'], h)
            root_orient_mat = batch_rodrigues(root_orient.reshape((-1, 3))).reshape((B, T, 3, 3))
            root_orient_vel = self.estimate_angular_velocity(root_orient_mat, h)
            return trans_vel, joints_vel, root_orient_vel
        else:
            raise NotImplementedError('Only smpl+joints motion prior configuration is supported!')

    def estimate_linear_velocity(self, data_seq, h):
        '''
        Given some batched data sequences of T timesteps in the shape (B, T, ...), estimates
        the velocity for the middle T-2 steps using a second order central difference scheme.
        The first and last frames are with forward and backward first-order 
        differences, respectively
        - h : step size
        '''
        # first steps is forward diff (t+1 - t) / h
        init_vel = (data_seq[:,1:2] - data_seq[:,:1]) / h
        # middle steps are second order (t+1 - t-1) / 2h
        middle_vel = (data_seq[:, 2:] - data_seq[:, 0:-2]) / (2*h)
        # last step is backward diff (t - t-1) / h
        final_vel = (data_seq[:,-1:] - data_seq[:,-2:-1]) / h

        vel_seq = torch.cat([init_vel, middle_vel, final_vel], dim=1)
        return vel_seq

    def estimate_angular_velocity(self, rot_seq, h):
        '''
        Given a batch of sequences of T rotation matrices, estimates angular velocity at T-2 steps.
        Input sequence should be of shape (B, T, ..., 3, 3)
        '''
        # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
        dRdt = self.estimate_linear_velocity(rot_seq, h)
        R = rot_seq
        RT = R.transpose(-1, -2)
        # compute skew-symmetric angular velocity tensor
        w_mat = torch.matmul(dRdt, RT)
        # pull out angular velocity vector
        # average symmetric entries
        w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
        w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
        w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
        w = torch.stack([w_x, w_y, w_z], axis=-1)
        return w

    def infer_latent_motion(self, trans, root_orient, body_pose, betas, data_fps, full_forward_pass=False):
        '''
        By default, gets a sequence of z's from the current SMPL optim params.

        If full_forward_pass is true, in addition to inference, also samples from the posterior and feeds
        through the motion prior decoder to get all terms needed to calculate the ELBO.
        '''
        B, T, _ = trans.size()
        h = 1.0 / data_fps
        latent_motion = None
        if self.motion_prior.model_data_config in ['smpl+joints', 'smpl+joints+contacts']:
            smpl_results, _ = self.smpl_results(trans, root_orient, body_pose, betas)
            trans_vel, joints_vel, root_orient_vel = self.estimate_velocities(trans, root_orient, body_pose, betas, data_fps,
                                                                                smpl_results=smpl_results)

            joints = smpl_results['joints3d']

            # convert rots
            # body pose and root orient are both in aa
            root_orient_in = root_orient
            body_pose_in = body_pose
            if self.motion_prior.in_rot_rep == 'mat' or self.motion_prior.in_rot_rep == '6d':
                root_orient_in = batch_rodrigues(root_orient.reshape(-1, 3)).reshape((B, T, 9))
                body_pose_in = batch_rodrigues(body_pose.reshape(-1, 3)).reshape((B, T, J_BODY*9))
            if self.motion_prior.in_rot_rep == '6d':
                root_orient_in = root_orient_in[:,:,:6]
                body_pose_in = body_pose_in.reshape((B, T, J_BODY, 9))[:,:,:,:6].reshape((B, T, J_BODY*6))
            joints_in = joints.reshape((B, T, -1))
            joints_vel_in = joints_vel.reshape((B, T, -1))

            seq_dict = {
                'trans' : trans,
                'trans_vel' : trans_vel,
                'root_orient' : root_orient_in,
                'root_orient_vel' : root_orient_vel,
                'pose_body' : body_pose_in,
                'joints' : joints_in,
                'joints_vel' : joints_vel_in
            }

            infer_results = self.motion_prior.infer_global_seq(seq_dict, full_forward_pass=False)

            if full_forward_pass:
                # return both the given motion and the one from the forward pass
                # make sure rotations are matrix
                # NOTE: assumes seq_dict is same thing we want to compute loss on - need to change if multiple future steps.
                if self.motion_prior.in_rot_rep != 'mat':
                    seq_dict['trans'] = batch_rodrigues(root_orient.reshape(-1, 3)).reshape((B, T, 9))
                    seq_dict['pose_body'] = batch_rodrigues(body_pose.reshape(-1, 3)).reshape((B, T, J_BODY*9))
                # do not need initial step anymore since output will be T-1
                for k, v in seq_dict.items():
                    seq_dict[k] = v[:,1:]
                for k in infer_results.keys():
                    if k != 'posterior_distrib' and k != 'prior_distrib':
                        infer_results[k] = infer_results[k][:, :, 0] # only want first output step
                infer_results = (seq_dict, infer_results)
            else:
                prior_z, posterior_z = infer_results
                infer_results = posterior_z[0] # mean of the approximate posterior

        else:
            raise NotImplementedError('Only smpl+joints motion prior configuration is supported!')

        return infer_results, seq_dict

    def rollout_latent_motion(self, trans, root_orient, body_pose, betas, prior_opt_params, latent_motion,
                                    return_prior=False,
                                    return_vel=False,
                                    fit_gender='neutral',
                                    use_mean=False,
                                    num_steps=-1,
                                    canonicalize_input=False):
        '''
        Given initial state SMPL parameters and additional prior inputs, rolls out the sequence
        using the encoded latent motion and the motion prior to obtain a full SMPL sequence.

        If latent_motion is None, instead samples num_steps into the future sequence from the prior. if use_mean does this
        using the mean of the prior rather than random samples.

        If canonicalize_input is True, the given initial state is first transformed into the local canonical
        frame before roll out
        '''
        B = trans.size(0)
        is_sampling = latent_motion is None
        Tm1 = num_steps if latent_motion is None else latent_motion.size(1)
        if is_sampling and Tm1 <= 0:
            Logger.log('num_steps must be positive to sample!')
            exit()

        cam_trans = trans
        cam_root_orient = root_orient

        x_past = joints = None
        trans_vel = joints_vel = root_orient_vel = None
        rollout_in_dict = dict()
        if self.motion_prior.model_data_config in ['smpl+joints', 'smpl+joints+contacts']:
            trans_vel, joints_vel, root_orient_vel = prior_opt_params
            smpl_results, _ = self.smpl_results(trans, root_orient, body_pose, betas)
            joints = smpl_results['joints3d']
            # update to correct rotations for input
            root_orient_in = root_orient 
            body_pose_in = body_pose
            if self.motion_prior.in_rot_rep == 'mat' or self.motion_prior.in_rot_rep == '6d':
                root_orient_in = batch_rodrigues(root_orient.reshape(-1, 3)).reshape((B, 1, 9))
                body_pose_in = batch_rodrigues(body_pose.reshape(-1, 3)).reshape((B, 1, J_BODY*9))
            if self.motion_prior.in_rot_rep == '6d':
                root_orient_in = root_orient_in[:,:,:6]
                body_pose_in = body_pose_in.reshape((B, 1, J_BODY, 9))[:,:,:,:6].reshape((B, 1, J_BODY*6))
            joints_in = joints.reshape((B, 1, -1))
            joints_vel_in = joints_vel.reshape((B, 1, -1))

            rollout_in_dict = {
                'trans' : trans,
                'trans_vel' : trans_vel,
                'root_orient' : root_orient_in,
                'root_orient_vel' : root_orient_vel,
                'pose_body' : body_pose_in,
                'joints' : joints_in,
                'joints_vel' : joints_vel_in
            }
        else:
            raise NotImplementedError('Only smpl+joints motion prior configuration is supported!')

        roll_output = self.motion_prior.roll_out(None, rollout_in_dict, Tm1, z_seq=latent_motion, 
                                                  return_prior=return_prior,
                                                  return_z=is_sampling,
                                                  use_mean=use_mean,
                                                  canonicalize_input=canonicalize_input,
                                                  gender=[fit_gender]*B, betas=betas.reshape((B, 1, -1)))

        pred_dict = prior_out = None
        if return_prior:
            pred_dict, prior_out = roll_output
        else:
            pred_dict = roll_output

        out_dict = dict()
        if self.motion_prior.model_data_config in ['smpl+joints', 'smpl+joints+contacts']:
            # copy what we need in correct format and concat with initial state
            trans_out = torch.cat([trans, pred_dict['trans']], dim=1)
            root_orient_out = pred_dict['root_orient']
            root_orient_out = rotation_matrix_to_angle_axis(root_orient_out.reshape((-1, 3, 3))).reshape((B, Tm1, 3))
            root_orient_out = torch.cat([root_orient, root_orient_out], dim=1)
            body_pose_out = pred_dict['pose_body']
            body_pose_out = rotation_matrix_to_angle_axis(body_pose_out.reshape((-1, 3, 3))).reshape((B, Tm1, J_BODY*3))
            body_pose_out = torch.cat([body_pose, body_pose_out], dim=1)
            joints_out = torch.cat([joints, pred_dict['joints'].reshape((B, Tm1, -1, 3))], dim=1)
            out_dict = {
                'trans' : trans_out,
                'root_orient' : root_orient_out,
                'pose_body' : body_pose_out,
                'joints' : joints_out
            }
            if return_vel:
                trans_vel_out = torch.cat([trans_vel, pred_dict['trans_vel']], dim=1)
                out_dict['trans_vel'] = trans_vel_out
                root_orient_vel_out = torch.cat([root_orient_vel, pred_dict['root_orient_vel']], dim=1)
                out_dict['root_orient_vel'] = root_orient_vel_out
                joints_vel_out = torch.cat([joints_vel, pred_dict['joints_vel'].reshape((B, Tm1, -1, 3))], dim=1)
                out_dict['joints_vel'] = joints_vel_out
            if return_prior:
                out_dict['cond_prior'] = prior_out
            if is_sampling:
                out_dict['z'] = pred_dict['z']
        else:
            raise NotImplementedError('Only smpl+joints motion prior configuration is supported!')

        #cam_dict = dict()
        # camera and prior frame are the same if not optimizing floor
        #cam_dict['trans'] = out_dict['trans']
        #cam_dict['root_orient'] = out_dict['root_orient']
        #am_dict['pose_body'] = out_dict['pose_body'] # same for both

        return out_dict#, cam_dict


    def smpl_results(self, trans, root_orient, body_pose, beta):
        '''
        Forward pass of the SMPL model and populates pred_data accordingly with
        joints3d, verts3d, points3d.

        trans : B x T x 3
        root_orient : B x T x 3
        body_pose : B x T x J*3
        beta : B x D
        '''
        B, T, _ = trans.size()
        if T == 1:
            # must expand to use with body model
            trans = trans.expand((self.batch_size, self.seq_len, 3))
            root_orient = root_orient.expand((self.batch_size, self.seq_len, 3))
            body_pose = body_pose.expand((self.batch_size, self.seq_len, J_BODY*3))
        elif T != self.seq_len:
            # raise NotImplementedError('Only supports single or all steps in body model.')
            pad_size = self.seq_len - T
            trans, root_orient, body_pose = self.zero_pad_tensors([trans, root_orient, body_pose], pad_size)

        betas = beta.reshape((self.batch_size, 1, self.num_betas)).expand((self.batch_size, self.seq_len, self.num_betas))
        smpl_body = self.body_model(pose_body=body_pose.reshape((self.batch_size*self.seq_len, -1)),
                                    pose_hand=None, 
                                    betas=betas.reshape((self.batch_size*self.seq_len, -1)),
                                    root_orient=root_orient.reshape((self.batch_size*self.seq_len, -1)),
                                    trans=trans.reshape((self.batch_size*self.seq_len, -1)))
        # body joints
        joints3d = smpl_body.Jtr.reshape((self.batch_size, self.seq_len, -1, 3))[:, :T]
        body_joints3d = joints3d[:,:,:len(SMPL_JOINTS),:]
        added_joints3d = joints3d[:,:,len(SMPL_JOINTS):,:]
        # ALL body vertices
        points3d = smpl_body.v.reshape((self.batch_size, self.seq_len, -1, 3))[:, :T]
        # SELECT body vertices
        verts3d = points3d[:, :T, KEYPT_VERTS, :]

        pred_data = {
            'joints3d' : body_joints3d,
            'points3d' : points3d,
            'verts3d' : verts3d,
            'joints3d_extra' : added_joints3d, # hands and selected OP vertices (if applicable) 
            'faces' : smpl_body.f # always the same, but need it for some losses
        }
        
        return pred_data, smpl_body

    def zero_pad_tensors(self, pad_list, pad_size):
        '''
        Assumes tensors in pad_list are B x T x D and pad temporal dimension
        '''
        B = pad_list[0].size(0)
        new_pad_list = []
        for pad_idx, pad_tensor in enumerate(pad_list):
            padding = torch.zeros((B, pad_size, pad_tensor.size(2))).to(pad_tensor)
            new_pad_list.append(torch.cat([pad_tensor, padding], dim=1))
        return new_pad_list


def main(args, config_file):

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    #Load data
    #data = np.load("/home/simon/DuD/research/pantomime/code/humor/out/ceti_fitting/results_out/sub-K88_task-gaitNormalBackpack_tracksys-RokokoSmartSuitPro1_run-1_step-2_motion.tsv_17151703330/stage3_results.npz")
    #data = np.load("/home/simon/DuD/research/pantomime/code/human_body_prior/support_data/data/CeTI-Locomotion_fitted_new/sub-K88_task-gaitFast_tracksys-RokokoSmartSuitPro1_run-1_step-2_motion.tsv.npz")
    # Contains latent_motion from fitting
    #data = np.load("/home/simon/DuD/research/pantomime/code/humor/out/ceti_fitting/results_out/data/CeTI-Locomotion/derivatives/cut_sequences/sub-K88_task-gaitNormalBackpack_tracksys-RokokoSmartSuitPro1_run-1_step-0_motion.tsv_17156962900/stage3_results.npz")

    input = dict()

    input["pose_body"] = torch.tensor(data["pose_body"].reshape(1, -1, 63)).to(device)
    input["trans"] = torch.tensor(data["trans"].reshape(1, -1, 3)).to(device)
    input["root_orient"] = torch.tensor(data["root_orient"].reshape(1, -1, 3)).to(device)
    input["betas"] = torch.tensor(data["betas"][0].reshape(1, -1)).to(device)
    if "latent_motion" in data:
        input["latent_motion"] = torch.tensor(data["latent_motion"][0].reshape(1, -1, 48)).to(device)
    #input["betas"] = torch.tensor(data["betas"].reshape(1, 16)).to(device)

    motion_anon = MotionAnonymization(device, args, num_betas=input["betas"].size()[1], batch_size=1, seq_len=input["pose_body"].size()[1])
    motion_anon.run_humor(input, data_fps=30)

if __name__=='__main__':
    args = parse_args(sys.argv[1:])
    config_file = sys.argv[1:][0][1:]
    main(args, config_file)