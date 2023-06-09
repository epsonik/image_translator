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
config_flickr8k = {
    "images_dir": data_path + "images/flickr8k/Images/",
    "train_images_names_file_path": data_path + "images/flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt",
    "test_images_names_file_path": data_path + "images/flickr8k/Flickr8k_text/Flickr_8k.testImages.txt",
    "captions_file_path": data_path + "images/flickr8k/Flickr8k_text/Flickr_8k.token.txt",
    "data_name": "flickr8k",
    "language": "eng"
}
config_aide = {
    "images_dir": data_path + "images/flickr8k/Images/",
    "train_images_names_file_path": data_path + "images/aide/aide_text/aide.trainImages.txt",
    "test_images_names_file_path": data_path + "images/aide/aide_text/aide.testImages.txt",
    "captions_file_path": data_path + "images/aide/aide_text/aide.token.txt",
    "data_name": "aide",
    "language": "pl"
}
config_flickr8k_polish = {
    "images_dir": data_path + "images/flickr8k/Images/",
    "train_images_names_file_path": data_path + "images/flickr8k_polish/Flickr8k_polish_text/Flickr_8k_polish.trainImages.txt",
    "test_images_names_file_path": data_path + "images/flickr8k_polish/Flickr8k_polish_text/Flickr_8k_polish.testImages.txt",
    "captions_file_path": data_path + "images/flickr8k_polish/Flickr8k_polish_text/Flickr_8k_polish.token.txt",
    "data_name": "flickr8k_polish",
    "language": "pl"
}
config_flickr30k_polish = {
    "images_dir": data_path + "images/flickr30k/Images/",
    "train_images_path": data_path + "images/flickr30k_polish/Flickr30k_polish_text/Flickr_30k_polish.trainImages.txt",
    "test_images_path": data_path + "images/flickr30k_polish/Flickr30k_polish_text/Flickr_30k_polish.testImages.txt",
    "captions_file_path": data_path + "images/flickr30k_polish/Flickr30k_polish_text/Flickr_30k_polish.token.txt",
    "data_name": "flickr30k_polish",
    "language": "pl"
}
config_flickr30k = {
    "images_dir": data_path + "images/flickr30k/Images",
    "images_names_file_path": data_path + "images/flickr30k/karpathy/f30ktalk.json",
    "captions_file_path": data_path + "images/flickr30k/karpathy/dataset_flickr30k.json",
    "data_name": "flickr30k",
    "language": "eng"
}
config_coco17 = {
    "images_dir": data_path + "images/coco2014",
    "images_names_file_path": data_path + "images/coco2017/annotations/cocotalk.json",
    "captions_file_path": data_path + "images/coco2017/annotations/dataset_coco.json",
    "data_name": "coco17",
    "language": "eng"
}

config_coco14 = {
    "images_dir": data_path + "images/coco2014",
    "images_names_file_path": data_path + "images/coco2014/karpathy/cocotalk.json",
    "captions_file_path": data_path + "images/coco2014/karpathy/dataset_coco.json",
    "annotations_train_file_path": data_path + "images/coco2014/official/annotations/instances_train2014.json",
    "annotations_test_file_path": data_path + "images/coco2014/official/annotations/instances_val2014.json",
    "data_name": "coco14",
    "language": "eng"
}
