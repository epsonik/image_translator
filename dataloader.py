import glob
import json

import config_datasets


def split_data(intersection, all_descriptions, train_images, test_images, val_images):
    """
        Split captions to train and test sets  and map image id to the set of captions
    Parameters
    ----------
    all_descriptions: dict
        Dictionary with key-image filename, value- list of captions separated by comma.
        Captions are raw, without any text preprocessing, that is specific for NLP tasks.
    train_images: list
        List of image ids that belongs to train set.
    test_images: list
        List of image ids that belongs to test set.

    Returns
    -------
    train_images_mapping: dict
        Dictionary od images that are in train split, with key-image filename,
         value- list of captions separated by comma. Captions are raw, without any text preprocessing,
         that is specific for NLP tasks.
    test_images_mapping: dict
        Dictionary od images that are in test split, with key-image filename,
         value-list of captions separated by comma.Captions are raw, without any text preprocessing,
          that is specific for NLP tasks.
    """
    train_images_mapping = dict()
    test_images_mapping = dict()
    val_images_mapping = dict()
    for x in list(intersection):
        if x in list(train_images.keys()):
            train_images_mapping[x] = all_descriptions[x]
        if x in list(test_images.keys()):
            test_images_mapping[x] = all_descriptions[x]
        if x in list(val_images.keys()):
            val_images_mapping[x] = all_descriptions[x]
    return train_images_mapping, test_images_mapping, val_images_mapping


def get_dataset_configuration(dataset_name):
    """
        Method to get configuration (path to the images, data splits etc) specific for dataset fe. COCO2017, Flickr8k
    Parameters
    ----------
    dataset_name : str
        Name that explicitly identifies name of the dataset from file config_datasets.py
    Returns
    -------
    config: dict
        Dictionary with configuration parameters fe. language of the dataset or path to the directory with images
    """
    if dataset_name == "flickr8k":
        return config_datasets.config_flickr8k
    elif dataset_name == "flickr8k_polish":
        return config_datasets.config_flickr8k_polish
    elif dataset_name == "flickr30k":
        return config_datasets.config_flickr30k
    elif dataset_name == "flickr30k_polish":
        return config_datasets.config_flickr30k_polish
    elif dataset_name == "aide":
        return config_datasets.config_aide
    elif dataset_name == "coco14":
        return config_datasets.config_coco14
    elif dataset_name == "coco17":
        return config_datasets.config_coco17
    else:
        return Exception("Bad name of dataset")


def load_doc(filename):
    """
    Load docuemnt from specigied "filename"
    Parameters
    ----------
    filename: str
        Path to the file to load
    Returns
    -------
    text: str
        Continous text. Need to be read with separators specific for file
    """
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def load_all_captions_flickr(captions_file_path):
    """
    Load all captions from Flickr type dataset
    Parameters
    ----------
    caption_file_path : str
        Path to the file containing all Flickr dataset captions
    Returns
    -------
    all_captions:dict
        Dictionary with key image filename value- list of captions separated by comma.
        Captions are raw, without any text preprocessing specific for NLP tasks.
    """
    doc = load_doc(captions_file_path)
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # extract filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        # create the list if needed
        if image_id not in mapping:
            mapping[image_id] = list()
        # store description
        mapping[image_id].append(image_desc)
    return mapping


def load_images_coco(configuration):
    """
        Load images from files of COCO dataset
    Parameters
    ----------
    configuration : dict
        Configuration of the datasets used in training from config_datasets.py.
    Returns
    -------
    train_images: dict->{image_filename: global path to the image}
        train split of images
    test_images: dict->{image_filename: global path to the image}
        test split of images
    """
    file_with_images_def = configuration["images_names_file_path"]
    # directory will all images from COCO dataset
    images_folder = configuration["images_dir"]
    info = json.load(open(file_with_images_def))
    train_images_mapping = dict()
    test_images_mapping = dict()
    val_images_mapping = dict()
    restval_images_mapping = dict()
    # iterate over all images from COCO dataset listed in file configuration["images_names_file_path"]
    for ix in range(len(info['images'])):

        img = info['images'][ix]
        # remove from image name .jpg sufix
        image_filename = img['file_path'].rsplit(".", 1)[0]
        # add global path to the image name
        file_path = images_folder + "/" + img['file_path']

        if image_filename.find("/") != -1:
            image_filename = img['file_path'].rsplit("/", 1)[1].rsplit(".", 1)[0]
        # assignment of images from specific split
        # in our case we use train and test dataset
        # all images assigned to restval, train by Karpathy split are assigned to the train split
        # images from val and test are assigned to the test split
        if img['split'] == 'train':
            train_images_mapping[image_filename] = file_path
        elif img['split'] == 'val':
            val_images_mapping[image_filename] = file_path
        elif img['split'] == 'test':
            test_images_mapping[image_filename] = file_path
        elif img['split'] == 'restval':
            train_images_mapping[image_filename] = file_path
    return train_images_mapping, test_images_mapping, val_images_mapping


def load_all_captions_coco(caption_file_path):
    """
    Load all captions from cocodataset
    Parameters
    ----------
    caption_file_path : str
        Path to the file containing COCO dataset captions
    Returns
    -------
    all_captions:dict->{"232332": []
        Dictionary with key image filename value- list of captions separated by comma.
        Captions are raw, without any text preprocessing specific for NLP tasks.
    """
    # open file with coco captions
    imgs = json.load(open(caption_file_path, 'r'))
    imgs = imgs['images']
    all_captions = dict()
    for img in imgs:
        # remove ".jpg" sufix
        image_filename = img['filename'].rsplit(".", 1)[0]
        # create dictionary
        # in the file key "raw" represents captions without any processing
        if image_filename not in all_captions:
            all_captions[image_filename] = list()
        for sent in img['sentences']:
            all_captions[image_filename].append(sent['raw'])

    return all_captions


def load_all_bbox_categories_coco(dataset_configuration):
    """
    Load all bbox categories from cocodataset
    Parameters
    ----------
    caption_file_path : str
        Path to the file containing COCO dataset captions
    Returns
    -------
    all_captions:dict->{"232332": []
        Dictionary with key image filename value- list of captions separated by comma.
        Captions are raw, without any text preprocessing specific for NLP tasks.
    """

    annotations_train_file_path = dataset_configuration["annotations_train_file_path"]
    annotations_test_file_path = dataset_configuration["annotations_test_file_path"]
    all_annotations_by_img_id = dict()
    all_annotations = dict()

    def get_class(coco_data, class_id):
        all_classes_mapping = coco_data['categories']
        for classid in all_classes_mapping:
            if classid['id'] == class_id:
                class_name = classid['name']
                try:
                    return class_name
                except:
                    print("---> Class not found! <---")
                    print(f"Id: {classid}")

    def process(annotations_file_path, split_name):
        with open(annotations_file_path, 'r') as f:
            coco_data = json.load(f)

        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in all_annotations_by_img_id:
                all_annotations_by_img_id[image_id] = list()
            all_annotations_by_img_id[image_id].append(get_class(coco_data, ann["category_id"]))
        print("Number of annotation in {} {}".format(split_name, len(all_annotations_by_img_id)))
        for ix in range(len(coco_data['images'])):
            img = coco_data['images'][ix]
            image_filename = img['file_name'].rsplit(".", 1)[0]
            if image_filename.find("/") != -1:
                image_filename = img['file_path'].rsplit("/", 1)[1].rsplit(".", 1)[0]
            image_id = img["id"]
            if image_id in all_annotations_by_img_id.keys():
                all_annotations[image_filename] = all_annotations_by_img_id[image_id]

    process(annotations_train_file_path, "train")
    process(annotations_test_file_path, "test")
    return all_annotations


def load_images_flickr(images_dir, train_images_file_path, test_images_file_path):
    """Method to map images ids to pictures

    Parameters
    ----------
    images_dir: str
        Path to the directory with all images from  Flickr type dataset
    train_images_file_path
        Path to the file with image names of images from train split
    test_images_file_path
        Path to the file with image names of images from test split
    Returns
    -------
    train_images_mapping: dict->{image_filename: global path to the image}
        train split of images
    test_images_mapping: dict->{image_filename: global path to the image}
        test split of images

    """
    train_images = set(open(train_images_file_path, 'r').read().strip().split('\n'))
    train_images_mapping = dict()
    test_images = set(open(test_images_file_path, 'r').read().strip().split('\n'))
    test_images_mapping = dict()
    # add global paths to the all images in images_dir directory
    all_images = glob.glob(images_dir + '*.jpg')
    for i in all_images:  # img is list of full path names of all images
        image_name = i.split("/")[-1]
        image_id = image_name.split(".")[0]
        if image_name in train_images:  # Check if the image belongs to train set
            train_images_mapping[image_id] = i  # Add it to the dict of train images
        if image_name in test_images:  # Check if the image belongs to test set
            test_images_mapping[image_id] = i  # Add it to the dict of test images
    return train_images_mapping, test_images_mapping


def load_dataset(configuration):
    """General method to load train and test set into framework.
        Framework accepts different types of datasets in test and train split, fe. train - COCO2017, test - Flickr8k.
        Following this assumption, datasets for train and test sets are loaded separately.
        All data for all datasets are loaded (all captions, all images for training and testing) and separated
        accordingly to the splits defined in configurations files. Therefore user can freely mix con
         mix data configurations for training and testing.
    Parameters
    ----------
    configuration : dict
        Configuration of the datasets used in training.

    Returns
    -------
    train: dict->{
                "train_images_mapping_original": dict,
                "test_images_mapping_original": dict,
                "all_captions": dict,
                "train_captions_mapping_original": dict,
                "test_captions_mapping_original": dict
            }
            Dictionary with data assigned to the training split.
    test: dict->{
                "train_images_mapping_original": dict,
                "test_images_mapping_original": dict,
                "all_captions": dict,
                "train_captions_mapping_original": dict,
                "test_captions_mapping_original": dict
            }
            Dictionary with data assigned to the testing split.
    """

    def get_data_for_split(split_name):
        # Load dataset configuration, by the name of the dataset assigned for training/testing
        dataset_configuration = get_dataset_configuration(configuration[split_name]["dataset_name"])
        all_bbox_categories = load_all_bbox_categories_coco(dataset_configuration)
        print("All images with coresponding categories loaded")
        print("Number of images with categories: ", len(all_bbox_categories.keys()))
        # Therefore Flickr and COCO have different file and data structures, to show captions and split of data
        # different methods for loading captions and images are used.
        # Datasets Flickr30k, COCO2017, COCO2014 have the same strucutre of files with captions and split informations.
        if dataset_configuration["data_name"] in ["flickr30k", "coco17", "coco14"]:
            # Load train images and test images and assign them to specific splits
            print("Loading images splits")
            train_images_mapping_original, test_images_mapping_original, val_images_mapping_original = load_images_coco(
                dataset_configuration)
            print("Images splits loaded for {} split".format(split_name))
            print("Number of train images: ", len(train_images_mapping_original))
            print("Number of test images: ", len(test_images_mapping_original))
            print("Number of val images: ", len(val_images_mapping_original))
            # Load all captions from dataset, that is COCO type
            print("Loading all captions")
            all_captions = load_all_captions_coco(dataset_configuration["captions_file_path"])
            print("All captions loaded")
            print("Number of all captions: ", len(all_captions))

        # Datasets Flickr30k, Flickr8k_polish, AIDe, Flickr8k  have the same strucutre of files with captions and split informations.
        if dataset_configuration["data_name"] in ["flickr30k_polish", "flickr8k_polish", "aide", "flickr8k"]:
            return Exception("Not implemented")
            # # Load train images and test images and assign them to specific splits
            # print("Loading images splits")
            # train_images_mapping_original, test_images_mapping_original = load_images_flickr(
            #     dataset_configuration["images_dir"], dataset_configuration[
            #         "train_images_names_file_path"],
            #     dataset_configuration[
            #         "test_images_names_file_path"])
            # print("Images splits loaded")
            # print("Number of train images: ", len(train_images_mapping_original))
            # print("Number of test images: ", len(test_images_mapping_original))
            # # Load all captions from dataset, that is Flickr8k type
            # print("Loading all captions")
            # all_captions = load_all_captions_flickr(dataset_configuration["captions_file_path"])
            # print("All captions loaded")
            # print("Nuber of all captions: ", len(all_captions))
            #
            # all_bbox_categories = load_all_bbox_categories_coco(dataset_configuration)
            # print("All images with coresponding categories loaded")
            # print("Number of images with categories: ", len(all_captions))

        # Assign captions to specific splits
        print("Loading captions splits for {}".format(split_name))
        intersection_categories_captions = list(all_bbox_categories.keys() & all_captions.keys())
        train_captions_mapping_original, test_captions_mapping_original, \
        val_captions_mapping_original = split_data(intersection_categories_captions, all_captions,
                                                   train_images_mapping_original,
                                                   test_images_mapping_original,
                                                   val_images_mapping_original)
        print("Captions splits loaded for {}".format(split_name))
        print("Number of train captions: ", len(train_captions_mapping_original))
        print("Number of test captions: ", len(test_captions_mapping_original))
        print("Number of val captions: ", len(val_captions_mapping_original))

        print("Loading bbox_categories of images splits")
        train_bbox_categories_mapping_original, test_bbox_categories_mapping_original, \
        val_bbox_categories_mapping_original = split_data(intersection_categories_captions, all_bbox_categories,
                                                          train_images_mapping_original,
                                                          test_images_mapping_original,
                                                          val_images_mapping_original)
        print("Categories of bbox  in images loaded")
        print("Number of train images with categories loaded: ", len(train_bbox_categories_mapping_original))
        print("Number of test images with categories loaded: ", len(test_bbox_categories_mapping_original))
        print("Number of val images with categories loaded: ", len(val_bbox_categories_mapping_original))
        return {
            "train": {
                "train_images_mapping_original": train_images_mapping_original,
                "train_captions_mapping_original": train_captions_mapping_original,
                "train_bbox_categories_mapping_original": train_bbox_categories_mapping_original
            },
            "test": {
                "test_images_mapping_original": test_images_mapping_original,
                "test_captions_mapping_original": test_captions_mapping_original,
                "train_bbox_categories_mapping_original": test_bbox_categories_mapping_original
            },
            "val": {
                "val_images_mapping_original": val_images_mapping_original,
                "val_captions_mapping_original": val_captions_mapping_original,
                "train_bbox_categories_mapping_original": val_bbox_categories_mapping_original
            },
            "all_captions": all_captions,
            "all_bbox_categories": all_bbox_categories,
            "language": dataset_configuration['language']
        }

    print("---------------------")
    print("Loading train dataset")
    train = get_data_for_split("train")
    return train, train, train


class DataLoader:
    def __init__(self, configuration):
        print("Loading dataset")
        self.train, self.test, self.val = load_dataset(configuration)
        self.configuration = configuration
