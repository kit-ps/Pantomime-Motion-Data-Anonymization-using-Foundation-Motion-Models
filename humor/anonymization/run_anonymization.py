import copy
import sys, os

cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import torch
import time
import numpy as np
from pathlib import Path
from utils.logging import Logger, mkdir
from anonymization.config import parse_args
from anonymization.motion_anonymization import MotionAnonymization
from anonymization.anon_util import format_positions_as_original
from datasets.anonymization_dataset import AnonymizationDataset
from anonymization.verification_experiments import run_verification_for_anon
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
from os.path import join



def regular_anon(args, data_type):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if args.out is not None:
        mkdir(args.out)
        # create logging system
        fit_log_path = os.path.join(args.out, 'anon_' + str(int(time.time())) + '.log')
        Logger.init(fit_log_path)


    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)
    
    num_pools = 4

    run_list = []
    for i in range(num_pools):
        dataset = AnonymizationDataset(
            args.data_path,
            split_size=num_pools,
            split_part=i
        )

        batch_size = 1
        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True,
                                 drop_last=False,
                                 )  # worker_init_fn=lambda _: np.random.seed()
        run_list.append((data_loader, device, data_type, args))

    with mp.Pool(processes=num_pools) as pool:
        pool.map(regular_anon_worker, run_list)


def regular_anon_worker(input_args):
    data_loader, device, data_type, args = input_args
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        meta, data = data
        file_path = meta["file_path"][0]

        prefix = data_type + "_"

        input = dict()

        if data_type == "original":
            if args.dataset_type == "CeTI-Locomotion":
                input["pose_body"] = data[prefix + "rotations"].reshape(1, -1, 17 * 3).to(device).float()
                input["positions"] = data[prefix + "positions"].reshape(1, -1, 17 * 3).to(device).float()
                input["betas"] = torch.zeros(1,10)
            elif args.dataset_type == "Horst-Study":
                input["positions"] = data[prefix + "positions"].to(device).float()
                input["betas"] = torch.zeros(1,10)
                input["pose_body"] = torch.zeros(1, 1, 10)
            elif args.dataset_type == "OUMVLP" or args.dataset_type == "HuMMan":
                input["pose_body"] = data["pose_body"].reshape(1, -1, 63).to(device).float()
                input["trans"] = data["trans"].reshape(1, -1, 3).to(device).float()
                input["root_orient"] = data["root_orient"].reshape(1, -1, 3).to(device).float()
                input["betas"] = torch.zeros(1, 10)#data["betas"][0].reshape(1, -1).to(device).float()


        else:
            input["pose_body"] = data[prefix + "pose_body"].reshape(1, -1, 63).to(device).float()
            input["trans"] = data[prefix + "trans"].reshape(1, -1, 3).to(device).float()
            input["root_orient"] = data[prefix + "root_orient"].reshape(1, -1, 3).to(device).float()
            input["betas"] = data[prefix + "betas"][0].reshape(1, -1).to(device).float()


        if "latent_motion" in data:
            input["latent_motion"] = torch.tensor(data[prefix + "latent_motion"][0].reshape(1, -1, 48)).to(device)

        # We now take the vposer fitting from the humor pipeline as such this step is no longer required
        # The vposer fitting has a beta value for every pose, we simply calculate the mean to only have on per sequence
        #if data_type == "vposer":
        #    input["betas"] = input["betas"].reshape(1, -1, 10).mean(1).to(device)

        # Remove the betas and replace them with an zero tensor to remove the body shape of the person
        if args.remove_betas:
            input["betas"] = torch.zeros(input["betas"].shape).to(device)

        # Remove the root_orient and trans so we dont have to anonymize them
        if args.remove_root_orient_and_trans and data_type != "original":
            input["trans"] = torch.zeros(input["trans"].shape).to(device)
            input["root_orient"] = torch.zeros(input["root_orient"].shape).to(device)
        for anon_type in args.anon_type:
            for noise_type in args.noise_type:
                for noise_scalar in args.noise_scaling:
                    output_path = os.path.join(args.out, prefix + "fitting", anon_type, noise_type + "_" + str(noise_scalar))
                    Path(output_path).mkdir(parents=True, exist_ok=True)

                    # Make a copy of the input to be sure that we do not change it during the anonymization
                    input_tmp = {}
                    for ele in input:
                        input_tmp[ele] = input[ele].clone()

                    # Blocks the prints in the Class Init
                    sys.stdout = open(os.devnull, 'w')
                    motion_anon = MotionAnonymization(device, args, noise_scalar,
                                                      noise_type, num_betas=input_tmp["betas"].size()[1],
                                                      batch_size=1, seq_len=input_tmp["pose_body"].size()[1])
                    sys.stdout = sys.__stdout__
                    if anon_type == "humor":
                        anon_sequence = motion_anon.run_humor(input_tmp, data_fps=args.data_fps)
                    elif anon_type == "vposer":
                        anon_sequence = motion_anon.run_vposer(input_tmp)
                    elif anon_type == "direct":
                        if args.direct_anon_attribute == "positions":
                            if "positions" in input_tmp:
                                anon_sequence = motion_anon.run_direct(input_tmp, ["positions"],
                                                                       generate_new_positions=False)
                            else:
                                anon_sequence = motion_anon.run_direct(input_tmp, ["pose_body"], generate_new_positions=True)
                        elif args.direct_anon_attribute == "positions+rotations":
                            anon_sequence = motion_anon.run_direct(input_tmp, ["positions", "pose_body"], generate_new_positions=False)


                    if "joints" in anon_sequence:
                        anon_sequence["joint_positions_smpl"] = anon_sequence["joints"]
                        del anon_sequence["joints"]
                        if args.dataset_type == "CeTI-Locomotion":
                            anon_sequence["positions"] = format_positions_as_original(anon_sequence["joint_positions_smpl"], "CeTI-Locomotion")
                        else:
                            anon_sequence["positions"] = anon_sequence["joint_positions_smpl"]

                    for ele in anon_sequence:
                        # Remove the batch size from the data
                        if len(anon_sequence[ele].shape) == 3 and anon_sequence[ele].shape[0] == 1:
                            anon_sequence[ele] = anon_sequence[ele].reshape(anon_sequence[ele].shape[1], anon_sequence[ele].shape[2])

                        if torch.is_tensor(anon_sequence[ele]):
                            anon_sequence[ele] = anon_sequence[ele].detach().cpu()

                    np.savez(os.path.join(output_path, file_path), **anon_sequence)


def recognition_adaptation(args, data_type):
    rec_attribute = "positions"
    threshold = 0.02

    # Init the result dict
    results = {}
    for anon_type in args.anon_type:
        results[anon_type] = {}
        for noise_type in args.noise_type:
            results[anon_type][noise_type] = {}
            for target in args.recognition_targets:
                if anon_type == "direct":
                    results[anon_type][noise_type][target] = {"values": [(4, 0), (0, 1)], "best": 4}
                else:
                    results[anon_type][noise_type][target] = {"values": [(20, 0), (0, 1)], "best": 20}



    # Load existing results to not start from scratch
    for anon_type in args.anon_type:
        result_pickle_path = join(args.out, data_type + "_fitting", anon_type, "results.pickle")
        print(result_pickle_path)
        if os.path.isfile(result_pickle_path):
            with (open(result_pickle_path, "rb")) as f:
                prior_results = pickle.load(f)

            sorted_keys = sorted(prior_results.keys(), key=lambda key: float(key.split("_")[-1]))

            for noise_type in args.noise_type:
                for target in args.recognition_targets:
                    for key in sorted_keys:
                        noise = key.split("_")[0]
                        para = float(key.split("_")[1])

                        if noise_type == noise:
                            rec_value = prior_results[key][rec_attribute]["id"]["classification"]
                            results[anon_type][noise_type][target]["values"].append((para, rec_value))


    #Checks what the best performing noise parameters are and if they are close enough to the targets, if not adds new
    #parameters to test to the two lists below
    new_noise_paras = []
    new_anon_types = []
    for anon_type in args.anon_type:
        for noise_type in args.noise_type:
            for target in args.recognition_targets:
                values = results[anon_type][noise_type][target]["values"]
                min_val = min(values, key=lambda tup: abs(tup[1] - target))
                results[anon_type][noise_type][target]["best"] = min_val

                print(anon_type, noise_type, target)
                print(min_val)

                # If the noise gets to close to 0 this indicates that we cannot achieve the rec target even on baseline
                # hence we stop the search
                if min_val[0] < 0.01:
                    continue

                if abs(min_val[1] - target) > threshold:
                    distances = [(val[0], val[1] - target) for val in values]
                    min_pos = min(distances, key=lambda tup: abs(tup[1]) if tup[1] > 0 else 10)
                    min_neg = min(distances, key=lambda tup: abs(tup[1]) if tup[1] < 0 else 10)

                    # Calculate the middle point of the two noise parameters
                    tmp = round( (min_pos[0] + min_neg[0]) / 2, 4)

                    # Only add if not in list
                    if tmp not in new_noise_paras:
                        new_noise_paras.append(tmp)
                        rng = np.random.default_rng()

                        s = rng.normal(0, 0.005)

                        tmp_up = abs(tmp + s)
                        tmp_down = abs(tmp - s)

                        new_noise_paras.append(tmp_up)
                        new_noise_paras.append(tmp_down)
                    if anon_type not in new_anon_types:
                        new_anon_types.append(anon_type)

    if len(new_noise_paras) == 0:
        return results

    print(results)

    #Perform the anonymization with the new parameters
    print("Starting anon for:", new_noise_paras)
    args_copy = copy.deepcopy(args)
    args_copy.noise_scaling = new_noise_paras
    args_copy.anon_type = new_anon_types
    regular_anon(args_copy, data_type)

    #Perform the recognition on the new anon results
    print("Performing verification")
    verification_args = []
    for anon in new_anon_types:
        if data_type == "original" and anon != "direct" and args.dataset_type != "OUMVLP" and args.dataset_type != "HuMMan":
            continue
        verification_args.append(("../data/anon/" + args.dataset_type + "/" + data_type + "_fitting/" + anon + "/",
                     "../data/prepared/" + args.dataset_type + "/", [rec_attribute], True))

    print(verification_args)
    for arg in verification_args:
        run_verification_for_anon(arg)

    return None



def main(args, config_file):

    for data_type in args.data_type:

        # Original data can only be directly anonymized
        args_copy = copy.deepcopy(args)
        if data_type == "original" and args.dataset_type != "OUMVLP" and args.dataset_type != "HuMMan":
            args_copy.anon_type = ["direct"]

        if args_copy.recognition_adaptation:
            results = None
            while results is None:
               results = recognition_adaptation(args_copy, data_type)
            print(results)
        else:
            print("Regular anon")
            from time import perf_counter
            start = perf_counter()
            regular_anon(args_copy, data_type)
            end = perf_counter()
            print(f"Took {end - start} seconds")


if __name__=='__main__':
    args = parse_args(sys.argv[1:])
    config_file = sys.argv[1:][0][1:]
    with torch.no_grad():
        main(args, config_file)