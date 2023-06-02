#### Config
data_path = "/home2/data/"
general = {
    "results_directory": "./results",
    "coco-caption_path": "./coco-caption",
    "pl_spacy_model": data_path + 'images/pl_spacy_model',
    "START": 'START',
    "STOP": 'STOP',
    "word_count_threshold": 10
}
glove = {
    "word_embedings_path": data_path + "images/glove/glove.6B.200d.txt",
    "embedings_dim": 199

}
fastText = {
    "word_embedings_path": data_path + "text_models/fastText/wiki-news-300d-1M-subword.vec",
    "embedings_dim": 300

}
# sprawdzić czy w kazdym konfigu poprawnie jest wpisana data_name


config_mixed_coco14_coco14_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "val": {"dataset_name": "coco14", "subset_name": "val"},
    "save_ix_to_word": False,
    "train_model": False,
    "save_model": False,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_glove",
    "text_processor": "glove",
    "continue_training": False
}
