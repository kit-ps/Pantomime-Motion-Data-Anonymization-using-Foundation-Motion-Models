from utils.config import SplitLineParser

def parse_args(argv):
    parser = SplitLineParser(fromfile_prefix_chars='@', allow_abbrev=False)

    # data in options
    parser.add_argument('--data-path', type=str, required=True, help='Path to the data to fit.')
    parser.add_argument('--data-fps', type=int, default=30, help='Sampling rate of the data.')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of sequences to batch together for fitting to data.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help="Shuffles data.")
    parser.set_defaults(shuffle=False)

    parser.add_argument('--data-type', type=str, nargs='+', required=True, choices=['original', 'humor', 'vposer'], default="humor", help='The type of data we are anonymizing.')
    parser.add_argument('--dataset-type', type=str, required=True, choices=['CeTI-Locomotion', 'Horst-Study', 'OUMVLP', 'HuMMan'], default="CeTI-Locomotion", help='The type of data we are anonymizing.')
    parser.add_argument('--anon-type', type=str, nargs='+', required=True, choices=['humor', 'vposer', 'direct'], default=["humor"], help='The type of anonymization we are running on the data.')
    parser.add_argument('--direct-anon-attribute', type=str, required=False, choices=['pose_body', 'positions', 'positions+rotations'], default="pose_body", help='The argument in the data to directly noise.')
    parser.add_argument('--noise-type', type=str, nargs='+', required=True, choices=['normal', 'laplace', 'uniform'], default=["normal"],
                        help='The type of noise to add to the data.')
    parser.add_argument('--noise-scaling', type=float, nargs='+', required=True, default=[1.0], help='The scaling of the applied noise.')
    parser.add_argument('--pose-by-pose', dest='pose_by_pose', action='store_true', help='If a new noise vector should be applied to each pose in a sequence or the same for the entire sequence.')
    parser.set_defaults(pose_by_pose=False)
    parser.add_argument('--remove-betas', dest='remove_betas', action='store_true', help='Configures if the betas are set to zero to remove their influence from the position generation.')
    parser.set_defaults(remove_betas=False)
    parser.add_argument('--remove-root-orient-and-trans', dest='remove_root_orient_and_trans', action='store_true', help='Configures if the root orientation and the root translation are set to zero to remove their influence from the position generation.')
    parser.set_defaults(remove_root_orient_and_trans=False)
    parser.add_argument('--recognition-adaptation', dest='recognition_adaptation', action='store_true', help='Configures if an adaptive anon should be performed to reach a specific recognition goal')
    parser.set_defaults(recognition_adaptation=False)
    parser.add_argument('--recognition-targets', type=float, nargs='+', required=False, default=[1.0], help='The recognition rates to achieve')

    # smpl model path
    parser.add_argument('--smpl', type=str, default='./body_models/smplh/neutral/model.npz', help='Path to SMPL model to use for optimization. Currently only SMPL+H is supported.')
    parser.add_argument('--vposer', type=str, default='./body_models/vposer_v1_0', help='Path to VPoser checkpoint.')

    # Humor path
    parser.add_argument('--humor', type=str, help='Path to HuMoR weights to use as the motion prior.')
    parser.add_argument('--humor-out-rot-rep', type=str, default='aa', choices=['aa', '6d', '9d'], help='Rotation representation to output from the model.')
    parser.add_argument('--humor-in-rot-rep', type=str, default='mat', choices=['aa', '6d', 'mat'], help='Rotation representation to input to the model for the relative full sequence input.')
    parser.add_argument('--humor-latent-size', type=int, default=48, help='Size of the latent feature.')
    parser.add_argument('--humor-model-data-config', type=str, default='smpl+joints+contacts', choices=['smpl+joints', 'smpl+joints+contacts'], help='which state configuration to use for the model')
    parser.add_argument('--humor-steps-in', type=int, default=1, help='Number of input timesteps the prior expects.')

    # options to save/visualize results
    parser.add_argument('--out', type=str, default=None, help='Output path to save fitting results/visualizations to.')

    parser.add_argument('--save-results', dest='save_results', action='store_true', help="Saves final optimized and GT smpl results and observations")
    parser.set_defaults(save_results=False)
    parser.add_argument('--save-stages-results', dest='save_stages_results', action='store_true', help="Saves intermediate optimized results")
    parser.set_defaults(save_stages_results=False)

    known_args, unknown_args = parser.parse_known_args(argv)

    return known_args