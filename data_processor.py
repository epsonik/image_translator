import pickle
from time import time
import numpy as np

from config_translator import general
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.xception import Xception
from keras.preprocessing import image
import itertools
from keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
import os, codecs
from tqdm import tqdm
from config_translator import glove
import string
import re
from unicodedata import normalize
from keras.preprocessing.sequence import pad_sequences
import json


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_categories():
    categories_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                       'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                       'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                       'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                       'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                       'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                       'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                       'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                       'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush', general["START"], general["STOP"]]
    return categories_list


def get_embedding_matrix(vocab_size, wordtoix, word_embedings_path, embedings_dim):
    """
    Method to represent words from created vocabulary(non repeatable words from all captions in dataset) in the
     multi dimensional vector representation
    Parameters
    ----------
    vocab_size: int
        Number of individual words
    wordtoix:
        Dictionary of individual words in vocabulary with explicit indexes
    word_embedings_path
        Path to te file with embeddings
    embedings_dim: int
        Number of dimensions in the embeddings file.
    Returns
    -------
    embedding_matrix: 2d array
        Matrix, where each row represents coefficients of giwen word from vocabulry to other words.
    """
    embeddings_index = {}
    # From the embeddings matrix get coefficients of particular words and store the in dictionarym by key - words
    f = open(word_embedings_path, encoding="utf-8")
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        import re
        if isfloat(values[1]):
            coefs = np.asarray(values[2:], dtype='float32')
        elif isfloat(values[2]):
            coefs = np.asarray(values[3:], dtype='float32')
        elif isfloat(values[3]):
            coefs = np.asarray(values[4:], dtype='float32')
        elif isfloat(values[4]):
            coefs = np.asarray(values[5:], dtype='float32')
        else:
            coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    # Get 200-dim/100 dense vector for each of the 10000 words in out vocabulary
    embedding_matrix = np.zeros((vocab_size, embedings_dim))
    for word, i in wordtoix.items():
        # if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros
            # 1655,299 199
            embedding_matrix[i] = embedding_vector
    print("Shape of embedding matrix")
    print(embedding_matrix.shape)
    return embedding_matrix


def define_images_feature_model(images_processor):
    """
    Method to define model to encode images.
    Parameters
    ----------
    Returns
    -------
    model_new
        model to encode image
    """

    if images_processor == 'Xception':
        model_images_processor_name = Xception(weights='imagenet')
        from keras.applications.xception import preprocess_input
        print("Used: Xception")
    elif images_processor == 'vgg16':
        model_images_processor_name = VGG16(weights='imagenet')
        from keras.applications.vgg16 import preprocess_input
        print("Used: Vgg16")
    elif images_processor == 'vgg19':
        model_images_processor_name = VGG19(weights="imagenet")
        from keras.applications.vgg16 import preprocess_input
        print("Used: Vgg19")
    elif images_processor == 'resnet152V2':
        model_images_processor_name = ResNet152V2(weights='imagenet')
        from keras.applications.resnet_v2 import preprocess_input
        print("Used: resnet142V2")
    elif images_processor == 'resnet50':
        model_images_processor_name = ResNet50(weights='imagenet')
        from tensorflow.keras.applications.resnet50 import preprocess_input
        print("Used: resnet50")
    elif images_processor == 'denseNet121':
        model_images_processor_name = DenseNet121(weights='imagenet')
        from tensorflow.keras.applications.densenet import preprocess_input
        print("Used: DenseNet121")
    elif images_processor == 'denseNet201':
        model_images_processor_name = DenseNet201(weights='imagenet')
        from tensorflow.keras.applications.densenet import preprocess_input
        print("Used: DenseNet201")
    elif images_processor == 'mobileNet':
        model_images_processor_name = MobileNet(weights='imagenet')
        from tensorflow.keras.applications.mobilenet import preprocess_input
        print("Used: MobileNet")
    elif images_processor == 'mobileNetV2':
        model_images_processor_name = MobileNetV2(weights='imagenet')
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        print("Used: MobileNetV2")
    else:
        model_images_processor_name = InceptionV3(weights='imagenet')
        from keras.applications.inception_v3 import preprocess_input
        print("Used: InceptionV3")
    # Create a new model, by removing the last layer (output layer) from the inception v3
    model_new = Model(model_images_processor_name.input, model_images_processor_name.layers[-2].output)
    return preprocess_input, model_new


def clean_descriptions(captions_mapping):
    """
    Method to:
    *lower all letters
    *remove punctuation
    *remove tokens with numbers
    Parameters
    ----------
    captions_mapping: dict
        Dictionary with keys - image_id and values-list of ground truth captions from training set.

    Returns
    -------
    cleaned_descriptions_mapping: dict
    """
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in captions_mapping.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = desc


def clear(desc_list):
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    clean_pair = list()
    for sentence in desc_list:
        # normalize unicode characters
        sentence = normalize('NFD', sentence).encode('ascii', 'ignore')
        sentence = sentence.decode('UTF-8')
        # tokenize on white space
        sentence = sentence.split()
        # convert to lowercase
        sentence = [word.lower() for word in sentence]
        # remove punctuation from each token
        sentence = [word.translate(table) for word in sentence]
        # remove non-printable chars form each token
        sentence = [re_print.sub('', w) for w in sentence]
        # remove tokens with numbers in them
        sentence = [word for word in sentence if word.isalpha()]
        # store as string
        clean_pair.append(' '.join(sentence))
    return clean_pair


def wrap_captions_in_start_stop(training_captions):
    """
    Method to wrap captions into START and STOP tokens
    Parameters
    ----------
    training_captions : dict
        Dictionary with keys - image_id and values-list od ground truth captions

    Returns
    -------
    train_captions_preprocessed: dict-
            Dictionary with wrapped into START and STOP tokens captions.
    """
    train_captions_preprocessed = dict()
    for image_id in training_captions.keys():
        sentences = training_captions[image_id]
        if image_id not in train_captions_preprocessed:
            train_captions_preprocessed[image_id] = list()
        for sentence in sentences:
            # wrap descriion in START and STOP tokens
            desc = general['START'] + " " + " ".join(sentence) + " " + general['STOP']
            # store
            train_captions_preprocessed[image_id].append(desc)
    return train_captions_preprocessed


def wrap_text_in_start_and_stop(train_dataset):
    bbox_categories_list = []
    output_sentences_list_with_stop = []
    output_sentences_list_with_start = []
    for pair in train_dataset:
        bbox_categories = set(pair["bbox_categories"])
        bbox_categories = ' '.join(map(str, bbox_categories))

        output_sentences = pair["captions"]
        output_sentences = clear(output_sentences)
        for sentence in output_sentences:
            output_sentences_list_with_stop.append(sentence + " " + general['STOP'])
            output_sentences_list_with_start.append(general['START'] + " " + sentence)
            bbox_categories_list.append(bbox_categories)
    print("Number of bbox sentences:", len(bbox_categories_list))
    print("Number of sentences with start: ", len(output_sentences_list_with_start))
    print("Number of sentences with stop: ", len(output_sentences_list_with_stop))
    print("Sample Bboxes")
    print(bbox_categories_list[0:10])
    print("Sample sentences with start")
    print(output_sentences_list_with_start[0:10])
    print("Sample sentences with stop")
    print(output_sentences_list_with_stop[0:10])
    return bbox_categories_list, output_sentences_list_with_start, output_sentences_list_with_stop


def preprocess(image_path, preprocess_input_function, images_processor):
    """
    Method to preprocess images by:
    *resizing to be acceptable by model that encodes images
    *represent in 3D matrix
    Parameters
    ----------
    training_captions : dict
        Dictionary with keys - image_id and values-list od ground truth captions

    Returns
    -------
    train_captions_preprocessed: dict-
            Dictionary with wrapped into START and STOP tokens captions.
    """
    # Convert all the images to size 299x299 as expected by the inception v3 model
    if images_processor == "vgg16" or images_processor == "resnet152V2" or images_processor == "vgg19" or images_processor == "resnet50" or images_processor == "denseNet121" or images_processor == "denseNet201" or images_processor == "mobileNet" or images_processor == "mobileNetV2":
        img = image.load_img(image_path, target_size=(224, 224))
    else:
        img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input_function(x)
    return x


def encode(image_path, images_feature_model, preprocess_input_function, images_processor):
    """
    Function to encode a given image into a vector of size (2048, )
    Parameters
    ----------
    image_path: str
        Path to the image
    images_feature_model:
        Model to predict image feature
    Returns
    -------
    fea_vec:
        Vector that reoresents the image
    """
    image = preprocess(image_path, preprocess_input_function,
                       images_processor)  # resize the image and represent it in numpy 3D array
    fea_vec = images_feature_model.predict(image)  # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])  # reshape from (1, 2048) to (2048, )
    return fea_vec


def preprocess_images(train_images, test_images, configuration):
    """
    Method to preprocess all iamges and store it in unified dict structure.
    Parameters
    ----------
    train_images: dict
        Dictionary with keys - image-id's values - global paths to the images
    test_images: dict
        Dictionary with keys - image-id's values - global paths to the images
    configuration
        Input file with all configurations
    Returns
    -------
    encoded_images_train: dict
        Dctionary with keys - image_id's and values - images encoded as vectors for images from train set
    encoded_images_test: dict
        Dctionary with keys - image_id's and values - images encoded as vectors for images from test set
    """
    # Call the funtion to encode all the train images
    # This will take a while on CPU - Execute this only once
    preprocess_input_function, images_feature_model = define_images_feature_model(configuration["images_processor"])
    word_indexing_path = "./" + configuration["data_name"] + configuration["pickles_dir"]

    def iterate_over_images(images_set, save_path):
        if configuration["encode_images"]:
            start = time()
            encoding_set = {}
            index = 1
            for image_id, image_path in images_set.items():
                encoding_set[image_id] = encode(image_path, images_feature_model, preprocess_input_function,
                                                configuration["images_processor"])
                if index % 100 == 0:
                    print("Processed:")
                    print(index)
                index += 1
            # Save the bottleneck train features to disk
            with open(word_indexing_path + save_path, 'w+b') as encoded_pickle:
                pickle.dump(encoding_set, encoded_pickle)
            print("Encoded images saved under ")
            print(word_indexing_path + configuration["encoded_images_test_path"])
            print("Images encoded")
            print("Time taken in seconds =", time() - start)
        encoding_set = load_encoded(save_path)
        return encoding_set

    def load_encoded(load_path):
        with open(word_indexing_path + load_path, "rb") as encoded_pickle:
            encoded_images_set = pickle.load(encoded_pickle)
        print("Encoded images loaded from: ")
        print(word_indexing_path + configuration["encoded_images_train_path"])
        return encoded_images_set

    encoding_test = iterate_over_images(test_images, configuration["encoded_images_test_path"])

    encoding_train = iterate_over_images(train_images, configuration["encoded_images_train_path"])

    return encoding_train, encoding_test


def get_all_train_text_list(train_captions):
    """
    Method to create a 1D list of all the flattened training captions
    Parameters
    ----------
    train_captions : dict
        Dictionary with keys - image_id and values-list of ground truth captions from training set.

    Returns
    -------
    Flattened list of captions
    """
    return list(itertools.chain(*train_captions.values()))


def get_max_length(captions):
    """
    Calculate the length of the description with the most words.
    Parameters
    ----------
    captions : dict
        List of all captions from set

    Returns
    -------
        Number of the words in longest captions
    """
    a = max(len(sen) for sen in captions)
    return a


def count_words_and_threshold(all_train_captions):
    """
    Count the occurences of words. Return only ones above threshold.
    Parameters
    ----------
    all_train_captions: dict
        Dictionary with keys - image_id and values-list of ground truth captions from training set.
    Returns
    -------
    vocab
        List of non repeatable words.
    """
    word_counts = {}
    nsents = 0
    for sent in all_train_captions:
        nsents += 1
        for w in sent.split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    # Consider only words which occur at least 10 times in the corpus
    vocab = [w for w in word_counts if word_counts[w] >= general["word_count_threshold"]]
    print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))
    return vocab


def ixtowordandbackward(vocab, configuration):
    """
    Method to get a dictionary of words, where keys are words and values are indexes and
    revert this opperations. Dictionary to get word by indexes
    Parameters
    ----------
    vocab: list
        List of non repeatable words
    configuration
    Returns
    -------
    ixtoword
        Dictionary to get word by index
    wordtoix
        Dictionary of words, where keys are words and values are indexes
    """
    word_indexing_path = "./" + configuration["data_name"] + configuration["pickles_dir"]
    if configuration["save_ix_to_word"]:
        ixtoword = {}
        wordtoix = {}
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        with open(word_indexing_path + "/" + configuration["ixtoword_path"], "wb") as encoded_pickle:
            pickle.dump(ixtoword, encoded_pickle)
        with open(word_indexing_path + "/" + configuration["wordtoix_path"], "wb") as encoded_pickle:
            pickle.dump(wordtoix, encoded_pickle)
        return ixtoword, wordtoix

    with open(word_indexing_path + "/" + configuration["ixtoword_path"], "rb") as encoded_pickle:
        ixtoword = pickle.load(encoded_pickle)
    with open(word_indexing_path + "/" + configuration["wordtoix_path"], "rb") as encoded_pickle:
        wordtoix = pickle.load(encoded_pickle)
    return ixtoword, wordtoix


def define_learning_data(data):
    """
    Return the data tha t wil be used in training testing stages
    Parameters
    ----------
    data
        Datasets that are loaded for the training and testing stage.
        From the splits the direct data for training and testing will be excluded.
    Returns
    -------
    train_images_mapping: dict
        Paths to the images used for training.
    train_captions_mapping:dict
        Captions for training  identified by image_id
    test_images_mapping:dict
        Path to the images used for testing.
    test_captions_mapping:dict
        Captions for testing identified by image_id
    data.train["all_captions"]:dict
        All captions from a dataset used for training
    """

    def get_split(split, subset_data):
        if data.configuration[split]['subset_name'] == 'train':
            return subset_data["train"]['train_images_mapping_original'], \
                   subset_data["train"]['train_captions_mapping_original'], \
                   subset_data["train"]['train_bbox_categories_mapping_original']

        if data.configuration[split]['subset_name'] == 'test':
            return subset_data["test"]['test_images_mapping_original'], \
                   subset_data["test"]['test_captions_mapping_original'], \
                   subset_data["test"]['test_bbox_categories_mapping_original']
        if data.configuration[split]['subset_name'] == 'val':
            return subset_data["val"]['val_images_mapping_original'], \
                   subset_data["val"]['val_captions_mapping_original'], \
                   subset_data["val"]['val_bbox_categories_mapping_original']

    train_images_mapping, train_captions_mapping, train_bbox_categories_mapping = get_split("train", data.train)
    test_images_mapping, test_captions_mapping, test_bbox_categories_mapping = get_split("test", data.test)
    val_images_mapping, val_captions_mapping, val_bbox_categories_mapping = get_split("val", data.val)

    return train_images_mapping, train_captions_mapping, train_bbox_categories_mapping, \
           test_images_mapping, test_captions_mapping, test_bbox_categories_mapping, \
           val_images_mapping, val_captions_mapping, val_bbox_categories_mapping, \
           data.train["all_captions"], data.train["all_bbox_categories"]


def load_data(data):
    def get_split(split):
        if data.configuration[split]['subset_name'] == 'train':
            return data.train
        if data.configuration[split]['subset_name'] == 'test':
            return data.test
        if data.configuration[split]['subset_name'] == 'val':
            return data.val

    train = get_split("train")
    test = get_split("test")
    val = get_split("val")
    print("Number of annotation in {}: {}".format("train", len(train)))
    print("Number of annotation in {}: {}".format("test", len(test)))
    print("Number of annotation in {}: {}".format("val", len(val)))
    return train, test, val, data.all


def create_dir_structure(configuration):
    """
    Create directiories to store results of specific steps from processing data during learning process:
    *data_name name of the configurations
    *pickles_dir - directory to store encoded train and test images and dictionaries of words
    *model_save_dir - store weights of trained model and exact model
    *results_directory - directory that cointains files with the results of testing on test data
                         defined in configurations file.
    Parameters
    ----------
    configurations
        File that represents the configiation data specicif for the run
    Returns
    -------
    """
    if not os.path.isdir("./" + configuration["data_name"]):
        os.makedirs("./" + configuration["data_name"])
    if not os.path.isdir("./" + configuration["data_name"] + "/" + configuration["pickles_dir"]):
        os.makedirs("./" + configuration["data_name"] + "/" + configuration["pickles_dir"])
    if not os.path.isdir("./" + configuration["data_name"] + "/" + configuration["model_save_dir"]):
        os.makedirs("./" + configuration["data_name"] + "/" + configuration["model_save_dir"])
    if not os.path.isdir(general["results_directory"]):
        os.makedirs(general["results_directory"])


def define_tokenizer(sentences, filters=''):
    tokenizer = Tokenizer(filters=filters)
    tokenizer.fit_on_texts(sentences)
    return tokenizer


def define_input_tokenizer(sentences, configuration):
    word_indexing_path = "./" + configuration["data_name"] + configuration["pickles_dir"]
    if configuration["save_tokenizer"]:
        tokenizer = define_tokenizer(sentences)
        with open(word_indexing_path + "/" + configuration["input_tokenizer_path"], "wb") as encoded_pickle:
            pickle.dump(define_tokenizer(sentences), encoded_pickle)
        return tokenizer

    with open(word_indexing_path + "/" + configuration["input_tokenizer_path"], "rb") as encoded_pickle:
        tokenizer = pickle.load(encoded_pickle)
    return tokenizer


def define_output_tokenizer(sentences, configuration):
    word_indexing_path = "./" + configuration["data_name"] + configuration["pickles_dir"]
    if configuration["save_tokenizer"]:
        tokenizer = define_tokenizer(sentences)
        with open(word_indexing_path + "/" + configuration["output_tokenizer_path"], "wb") as encoded_pickle:
            pickle.dump(define_tokenizer(sentences), encoded_pickle)
        return tokenizer

    with open(word_indexing_path + "/" + configuration["output_tokenizer_path"], "rb") as encoded_pickle:
        tokenizer = pickle.load(encoded_pickle)
    return tokenizer


def preprocess_data(data):
    create_dir_structure(data.configuration)
    train_dataset, data.test_dataset, val_datatset, all_dataset = load_data(data)

    data.train_bbox_categories_list, \
    data.output_sentences_list_with_start, \
    data.output_sentences_list_with_stop = wrap_text_in_start_and_stop(train_dataset)

    # tokenize the input bounding box categories(input language)
    data.input_tokenizer = define_input_tokenizer(data.train_bbox_categories_list, data.configuration)
    input_integer_seq = data.input_tokenizer.texts_to_sequences(data.train_bbox_categories_list)
    word2idx_inputs = data.input_tokenizer.word_index

    data.max_input_length = get_max_length(input_integer_seq)
    print("Input vocab size: %s" % len(word2idx_inputs))
    print("Length of longest sentence in the input: %s" % data.max_input_length)

    data.output_tokenizer = define_output_tokenizer(
        data.output_sentences_list_with_start + data.output_sentences_list_with_stop, data.configuration)
    # output_integer_seq
    output_with_stop_integer_seq = data.output_tokenizer.texts_to_sequences(data.output_sentences_list_with_stop)
    # output_input_integer_seq
    output_with_start_integer_seq = data.output_tokenizer.texts_to_sequences(data.output_sentences_list_with_start)
    word2idx_outputs = data.output_tokenizer.word_index
    data.num_words_output = len(word2idx_outputs) + 1
    data.max_output_length = get_max_length(output_with_start_integer_seq)
    print("Output vocab size: %s" % len(word2idx_outputs))
    print("Length of longest sentence in the output: %g" % data.max_output_length)

    data.encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=data.max_input_length)
    print("encoder_input_sequences.shape:", data.encoder_input_sequences.shape)
    print("encoder_input_sequences[172]:", data.encoder_input_sequences[7])

    data.decoder_input_sequences = pad_sequences(output_with_start_integer_seq, maxlen=data.max_output_length,
                                                 padding='post')
    print("decoder_input_sequences.shape:", data.decoder_input_sequences.shape)
    print("decoder_input_sequences[172]:", data.decoder_input_sequences[17])

    print("Glove used")
    data.num_words=len(word2idx_inputs) + 1
    data.embedding_matrix_input = get_embedding_matrix(data.num_words, word2idx_inputs,
                                                       glove["word_embedings_path"],
                                                       glove["embedings_dim"])
    return data


def get_all_train_captions_list(train_captions):
    """
    Method to create a 1D list of all the flattened training captions
    Parameters
    ----------
    train_captions : dict
        Dictionary with keys - image_id and values-list of ground truth captions from training set.

    Returns
    -------
    Flattened list of captions
    """
    return list(itertools.chain(*train_captions.values()))


def get_fast_text_embedding_matrix(vocab_size, wordtoix, word_embedings_path, embedings_dim):
    # load embeddings
    print('loading word embeddings...')
    embeddings_index = {}
    f = codecs.open(word_embedings_path, encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('found %s word vectors' % len(embeddings_index))

    # embedding matrix
    print('preparing embedding matrix...')
    words_not_found = []
    embedding_matrix = np.zeros((vocab_size, embedings_dim))
    for word, i in wordtoix.items():
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print(embedding_matrix.shape)

    return embedding_matrix
