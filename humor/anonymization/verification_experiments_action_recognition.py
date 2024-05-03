"""
Performs the verification experiments. Performs classification of identity, sex, and modality unsing an SVM.
"""
import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from multiprocessing import Pool
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
from datasets.anonymization_dataset import AnonymizationDataset


# Resamples the give motion sequence to 100 frames and in the end flattens the sequence into one vector
def flatten(data, num_frames=100):

    # Workaround when i forgot to remove the batch from the data
    if len(data.shape) > 2:
        data = data.reshape(data.shape[1], data.shape[2])

    # The betas do not require flatten
    if len(data.shape) == 1:
        return np.array(data)

    new_data = []
    for i in range(len(data[0])):
        dim = data[:,i]

        old_times = np.arange(len(dim))
        step = (old_times[-1] - old_times[0]) / (num_frames)
        new_times = np.arange(old_times[0], old_times[-1], step)

        #Sometimes the rounding gives you an extra step, therefore this makes sure we only have num_frames in total
        new_times = new_times[0:num_frames]

        new_dim = np.interp(new_times, old_times, dim)
        new_data.append(new_dim)

    return np.array(new_data).T.flatten()


def classify(data, labels, split_labels=False):

    test_size = 0.2

    if split_labels:
        # Splits the data based on the split_labels, to have disjunct participants sets for training and testing
        unique_ids = list(set(split_labels))
        ids_train, ids_learn = train_test_split(unique_ids, test_size=test_size)
        X_train, X_test, y_train, y_test = [], [], [], []
        for i in range(len(data)):
            if split_labels[i] in ids_train:
                X_train.append(data[i])
                y_train.append(labels[i])
            else:
                X_test.append(data[i])
                y_test.append(labels[i])

    else:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, stratify=labels)

    clf = SVC(probability=True)
    k_fold_strat = StratifiedKFold(n_splits=2)
    pipe = Pipeline([("scaler", MaxAbsScaler()), ("pca", PCA()), ("clf", clf)])
    cv_results = cross_validate(pipe, X_train, y_train, cv=k_fold_strat, scoring="balanced_accuracy",
                                return_estimator=True)
    est_i = cv_results["test_score"].argmax(axis=0)
    predictions = cv_results["estimator"][est_i].predict(X_test)

    validation_score = cv_results["test_score"][est_i]
    classification_score = accuracy_score(y_test, predictions)

    scores_per_id = {}
    for id in labels:
        tmp = []
        for i in range(len(y_test)):
            if y_test[i] == id:
                tmp.append(predictions[i])

        score = accuracy_score([id] * len(tmp), tmp)
        scores_per_id[id] = score

    return {"validation" : validation_score, "classification": classification_score, "class_per_id": scores_per_id}


def classify_folder(paras):

    folder, dataset, original_dataset, data_attributes = paras
    number_of_runs = 10

    results = {}
    for attribute in data_attributes:

        #try:
            results[attribute] = {}
            data_list = []
            split_labels = []
            labels = []

            print("Classifying modality")
            print(folder)
            for _, sample in enumerate(dataset):
                meta, data = sample
                labels.append(meta["modality"])
                data_list.append(flatten(data[attribute], 100))
                split_labels.append(meta["participant_id"])

            from collections import Counter
            counter = Counter(labels)
            data_list_filtered = []
            split_labels_filtered = []
            labels_filtered = []
            for i in range(len(labels)):
                if counter[labels[i]] >= 25:
                    data_list_filtered.append(data_list[i])
                    split_labels_filtered.append(split_labels[i])
                    labels_filtered.append(labels[i])


            results[attribute]["modality"] = {"classification": 0, "classification_all": []}
            for _ in range(number_of_runs):
                tmp = classify(data_list_filtered, labels_filtered, split_labels=split_labels_filtered)
                results[attribute]["modality"]["classification"] += tmp["classification"]
                results[attribute]["modality"]["classification_all"].append(tmp)
                #results[attribute]["modality"]["classification"] += classify(data_list, labels, split_labels=split_labels)["classification"]

            results[attribute]["modality"]["classification"] /= number_of_runs

        #except BaseException as error:
        #    print('An exception for attribute {} occurred: {}'.format(attribute, error))


    return folder, results



def run_verification_for_anon(args):
    data_import_dir, meta_path, data_attributes, ignore_existing = args
    print(data_import_dir)
    print(data_attributes)
    folders = [f for f in listdir(data_import_dir) if not isfile(join(data_import_dir, f))]

    # Make sure to only use folders which are from the CeTI-Locomotion dataset if not the main folder is CeTI-Locomotion
    if "HuMMan" not in data_import_dir:# or "HuMMan" not in data_import_dir:
        folders = [f for f in folders if "HuMMan" in f] #+ [f for f in folders if "HuMMan" in f]


    prior_results = {}
    results_pickle_path = join(data_import_dir, "results_action_new_reduced.pickle")

    # Filter results which already exists
    if ignore_existing:
        if isfile(results_pickle_path):
            with (open(results_pickle_path, "rb")) as f:
                prior_results = pickle.load(f)

            folders = [f for f in folders if f not in prior_results]
            print(folders)


    # Load all data first to see if something is missing
    if meta_path is not None:
        original_dataset = AnonymizationDataset(meta_path)
    else:
        original_dataset = None
    attr_list = []
    for folder in folders:
        folder_path = join(data_import_dir, folder)
        dataset = AnonymizationDataset(folder_path, meta_path=meta_path)
        # Test of flatten works
        attr_list.append((folder, dataset, original_dataset, data_attributes))

    print("Data loaded")
    with Pool(5) as p:
        tmp = p.map(classify_folder, attr_list)

    all_results = prior_results
    for ele in tmp:
        all_results[ele[0]] = ele[1]

    print(all_results)

    with open(results_pickle_path, "wb") as f:
        pickle.dump(all_results, f)


def run_verification_experiments(input):

    if len(input) == 2:

        dataset = input[1]

        args = []
        for fit in ["vposer_fitting", "humor_fitting"]:
            for anon in ["humor", "vposer", "direct"]:
                args.append(("../data/anon/" + dataset + "/" + fit + "/" + anon + "/",
                             "../data/prepared/" + dataset + "/", ["positions"], True))

        args.append(("../data/anon/" + dataset + "/original_fitting/direct/", "../data/prepared/" + dataset + "/",
                     ["positions"], True))

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
                                 "../data/prepared/" + dataset + "/", ["positions"], True))

            args.append(("../data/" + folder + "/" + dataset + "/original_fitting/direct/", "../data/prepared/" + dataset + "/",
                         ["positions"], True))
        else:
            for anon in ["humor", "vposer", "direct"]:
                args.append(("../data/" + folder + "/" + dataset + "/original_fitting/" + anon + "/",
                             "../data/prepared/" + dataset + "/", ["positions"], True))


        for arg in args:
            run_verification_for_anon(arg)

    elif len(input) == 1:

        attr = ['original_positions', 'vposer_betas', 'vposer_trans', 'vposer_root_orient',
                'vposer_pose_body', 'vposer_latent_pose', 'vposer_latent_motion', 'vposer_positions',
                'vposer_positions_without_trans_betas_root_orient', 'vposer_positions_without_trans_betas_pose_body',
                'vposer_positions_without_trans_root_orient_pose_body',
                'vposer_positions_without_betas_root_orient_pose_body',
                'vposer_positions_without_trans', 'vposer_positions_without_betas', 'vposer_positions_without_root_orient', 'vposer_positions_without_pose_body',
                'humor_betas', 'humor_trans',
                'humor_root_orient', 'humor_pose_body', 'humor_latent_motion', 'humor_latent_pose', 'humor_positions',
                'humor_positions_without_trans_betas_root_orient', 'humor_positions_without_trans_betas_pose_body',
                'humor_positions_without_trans_root_orient_pose_body',
                'humor_positions_without_betas_root_orient_pose_body',
                'humor_positions_without_trans', 'humor_positions_without_betas',
                'humor_positions_without_root_orient', 'humor_positions_without_pose_body']
        attr_reduced = ["original_positions", "vposer_positions", "humor_positions", "vposer_trans", "humor_trans",
                        "vposer_root_orient", "humor_root_orient"]
        attr_reduced_reduced = ["original_positions", "vposer_positions", "humor_positions"]
        args = ("../data/prepared/", None, attr_reduced_reduced, True)
        run_verification_for_anon(args)

    elif len(input) == 4:
        data_import_dir = input[1]
        meta_path = input[2]
        data_attributes = [c for c in input[3].split(',')]
        args = (data_import_dir, meta_path, data_attributes, True)
        run_verification_for_anon(args)
    else:
        print("Missing path or attributes")
        exit()


if __name__ == '__main__':

    # Example call
    # python3 humor/anonymization/verification_experiments.py ../data/anon/CeTI-Locomotion/vposer_fitting/direct/ ../data/prepared/CeTI-Locomotion/ pose_body,positions
    run_verification_experiments(sys.argv)