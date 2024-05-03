"""
Performs the verification experiments. Performs classification of identity, sex, and modality unsing an SVM.
"""
import sys, os



cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))
from sklearn.metrics import accuracy_score
from multiprocessing import Pool
from os import listdir
from os.path import isfile, join
import pickle
from data_loader.data_loaders import PantomimeDataLoader
from data_loader.Pantomime_dataset import PantomimeDataset
import torch
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import json



def train(config, dataset):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = PantomimeDataLoader(dataset,
                                      batch_size=config["data_loader"]["args"]["batch_size"],
                                      shuffle=config["data_loader"]["args"]["shuffle"],
                                      validation_split=config["data_loader"]["args"]["validation_split"],
                                      num_workers=config["data_loader"]["args"]["num_workers"])
    valid_data_loader = data_loader.split_validation()

    # updates to number of classes to the one in the dataset and get the right size for the input
    config["arch"]["args"]["frame_size"] = data_loader.dataset.get_frame_size()
    config["arch"]["args"]["num_classes"] = data_loader.dataset.get_num_classes()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss']['type'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                                 config=config,
                                 device=device,
                                 data_loader=data_loader,
                                 valid_data_loader=valid_data_loader,
                                 lr_scheduler=lr_scheduler)

    trainer.train()

    return trainer.mnt_best, trainer.best_model





def classify(dataset_path):



    with open('humor/anonymization/verification_ml/configs/config_EKYT.json', 'r') as f:
        config_file = json.load(f)

    config = ConfigParser(config_file)

    dataset = PantomimeDataset(dataset_path, train=True)

    score, best_model = train(config, dataset)

    dataset.switch_to_test()
    test_data_loader = PantomimeDataLoader(dataset, 1280, validation_split=0.0)

    device, device_ids = prepare_device(config['n_gpu'])
    model = best_model.to(device)
    if len(device_ids) > 1:
        best_model.model = torch.nn.DataParallel(model, device_ids=device_ids)


    best_model.eval()

    predictions = []
    labels = []
    with torch.no_grad():
        
        for batch_idx, (data, target) in enumerate(test_data_loader):
            labels.append(target.detach().cpu().numpy())
            data, target = data.to(device), target.to(device)

            output = best_model(data)

            pred = torch.argmax(output, dim=1)

            predictions.append(pred.detach().cpu().numpy())
            


    score_voting = {}
    for i in range(len(labels[0])):
        if labels[0][i] not in score_voting:
            score_voting[labels[0][i]] = [predictions[0][i]]
        else:
            score_voting[labels[0][i]].append(predictions[0][i])

    counts = 0
    for participant in score_voting:
        most_common = max(set(score_voting[participant]), key=score_voting[participant].count)

        if most_common == participant:
            counts += 1

    voting_score = counts / len(score_voting)

    classification_score = accuracy_score(labels[0], predictions[0])


    return { "classification": classification_score, "predictions": predictions[0],
     "true_labels": labels[0], "classification_voting": voting_score}




def classify_folder(paras):

    folder, dataset_path, dataset_meta_path, data_attributes = paras
    number_of_runs = 10

    results = {}
    for attribute in data_attributes:

        #try:
            results[attribute] = {}

            results[attribute]["id"] = { "classification": 0 , "classification_all": []}
            for _ in range(number_of_runs):
                tmp = classify(dataset_path)
                print(tmp)
                results[attribute]["id"]["classification"] += tmp["classification"]
                results[attribute]["id"]["classification_all"].append(tmp)
            results[attribute]["id"]["classification"] /= number_of_runs

        #except BaseException as error:
        #    print('An exception for attribute {} occurred: {}'.format(attribute, error))

    return folder, results



def run_verification_for_anon(args):
    data_import_dir, meta_path, data_attributes, ignore_existing = args
    print("Running verification for anon")
    print(data_import_dir)
    print(data_attributes)
    folders = [f for f in listdir(data_import_dir) if not isfile(join(data_import_dir, f))]
    #print(folders)

    #SEED = 1234123123
    #np.random.seed(SEED)

    prior_results = {}
    print(data_import_dir + "results_ml.pickle")
    results_pickle_path = join(data_import_dir, "results_ml.pickle")

    # Filter results which already exists
    if ignore_existing:
        if isfile(results_pickle_path):
            with (open(results_pickle_path, "rb")) as f:
                prior_results = pickle.load(f)

            folders = [f for f in folders if f not in prior_results]
            #print(folders)

    print(folders)

    attr_list = []
    for folder in folders:
        print(folder)
        folder_path = join(data_import_dir, folder)

        # Test of flatten works
        attr_list.append((folder, folder_path, meta_path, data_attributes))

    print("Data loaded")
    with Pool(5) as p:
        tmp = p.map(classify_folder, attr_list)

    all_results = prior_results
    for ele in tmp:
        all_results[ele[0]] = ele[1]

    with open(results_pickle_path, "wb") as f:
        print("Saving results at " + results_pickle_path)
        pickle.dump(all_results, f)


def run_verification_experiments(input, attributes):

    if len(input) == 2:

        dataset = input[1]

        args = []
        for fit in ["vposer_fitting", "humor_fitting"]:
            for anon in ["humor", "vposer", "direct"]:
                args.append(("../data/anon/" + dataset + "/" + fit + "/" + anon + "/",
                             "../data/prepared/" + dataset + "/", attributes, True))

        args.append(("../data/anon/" + dataset + "/original_fitting/direct/", "../data/prepared/" + dataset + "/",
                     attributes, True))

        for arg in args:
            run_verification_for_anon(arg)

    elif len(input) == 3:
        dataset = input[1]
        folder = input[2]

        args = []
        if dataset != "HuMMan":
            for fit in ["vposer_fitting", "humor_fitting"]:
                for anon in ["humor", "vposer", "direct"]:
                    args.append(("../data/" + folder + "/" + dataset + "/" + fit + "/" + anon + "/",
                                 "../data/prepared/" + dataset + "/", attributes, False))

            args.append(("../data/" + folder + "/" + dataset + "/original_fitting/direct/", "../data/prepared/" + dataset + "/",
                         attributes, True))
        else:
            for anon in ["humor", "vposer"]:
                args.append(("../data/" + folder + "/" + dataset + "/original_fitting/" + anon + "/",
                             "../data/prepared/" + dataset + "/", attributes, False))


        for arg in args:
            run_verification_for_anon(arg)

    elif len(input) == 4:
        data_import_dir = input[1]
        meta_path = input[2]
        data_attributes = [c for c in input[3].split(',')]
        args = (data_import_dir, meta_path, data_attributes, True)
        print(args)
        run_verification_for_anon(args)
    else:
        print("Missing path or attributes")
        exit()


if __name__ == '__main__':

    # Example call
    # python3 humor/anonymization/verification_experiments.py ../data/anon/CeTI-Locomotion/vposer_fitting/direct/ ../data/prepared/CeTI-Locomotion/ pose_body,positions

    run_verification_experiments(sys.argv, ["positions"])